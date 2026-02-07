# GPU Network Architecture

## THE GPU IS THE COMPUTER - Network Architecture

### The Reality

Standard network (WiFi, Ethernet) **REQUIRES CPU involvement**. This is not a design choice - it's a hardware limitation:

- NIC hardware delivers interrupts to CPU (hardware architecture)
- Consumer NICs send packets to CPU memory, not GPU
- Apple's kernel network stack is entirely CPU-based
- Even "zero-copy" networking on macOS still requires CPU for packet receive

**Network is UNIQUE - it is the ONLY unavoidable CPU dependency.**

| Operation | CPU Required? | Why |
|-----------|---------------|-----|
| **Network I/O** | **YES** | Hardware limitation - NIC delivers packets to CPU |
| File I/O | NO | GPU-initiated via `MTLIOCommandQueue` / async file loading |
| Process spawning | NO | GPU-native - apps are GPU bytecode, no CPU fork/exec |
| Input handling | NO | GPU processes HID events via ring buffer |
| Rendering | NO | Pure GPU compute + fragment shaders |
| Layout/compositing | NO | GPU constraint solvers and compositors |

**Do not confuse "traditionally done on CPU" with "requires CPU":**
- Process spawning is traditionally CPU (`fork`/`exec`) but our GPU app system spawns apps as GPU bytecode - no CPU involved
- File I/O is traditionally CPU but Metal 3's `MTLIOCommandQueue` enables GPU-initiated storage access
- Only network truly requires CPU because the hardware (NIC) delivers data via CPU interrupts

### Architecture Pattern

```
Internet/Network
       │
       ▼
    Mac's NIC (Hardware)
       │
       ▼
  CPU Kernel (unavoidable - Darwin network stack)
       │
       ▼
  CPU Network Thread (minimal work - just copy)
       │ writes to shared buffer
       ▼
  Ring Buffer (MTLBuffer, StorageModeShared)
       │ GPU polls via atomics
       ▼
  GPU Persistent Kernel
       │
       ▼
  GPU Processing (massively parallel)
```

### Implementation

#### 1. Ring Buffer Structure

```rust
// Rust-side ring buffer definition
#[repr(C)]
pub struct NetworkRingBuffer {
    // Atomic indices for lock-free coordination
    pub write_index: AtomicU32,    // CPU writes, GPU reads
    pub read_index: AtomicU32,     // GPU writes, CPU reads (for flow control)

    // Buffer configuration
    pub buffer_size: u32,          // Number of slots (power of 2)
    pub slot_size: u32,            // Size of each packet slot
    _padding: [u32; 2],            // Align to 16 bytes
}

#[repr(C)]
pub struct NetworkPacketSlot {
    pub length: u32,               // Actual packet length
    pub timestamp: u64,            // Arrival timestamp (nanoseconds)
    pub flags: u32,                // Packet flags (TCP, UDP, etc.)
    pub data: [u8; MAX_PACKET_SIZE], // Packet data (typically 1500 MTU)
}

// Constants
const RING_BUFFER_SLOTS: u32 = 4096;  // Must be power of 2
const MAX_PACKET_SIZE: usize = 2048;   // MTU + headers
```

```metal
// Metal-side ring buffer definition
struct NetworkRingBuffer {
    atomic_uint write_index;       // CPU writes, GPU reads
    atomic_uint read_index;        // GPU writes, CPU reads
    uint buffer_size;              // Number of slots
    uint slot_size;                // Size of each slot
    uint _padding[2];
};

struct NetworkPacketSlot {
    uint length;                   // Packet length
    ulong timestamp;               // Arrival time
    uint flags;                    // Packet type flags
    uchar data[2048];              // Packet data
};
```

#### 2. CPU Network Receive Thread

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::net::UdpSocket;
use metal::{Buffer, Device};

pub struct NetworkReceiver {
    socket: UdpSocket,
    ring_buffer: Arc<Buffer>,      // MTLBuffer, StorageModeShared
    header: *mut NetworkRingBuffer,
    slots: *mut NetworkPacketSlot,
    buffer_mask: u32,              // buffer_size - 1, for fast modulo
}

impl NetworkReceiver {
    pub fn new(device: &Device, port: u16) -> Self {
        // Create shared buffer accessible by both CPU and GPU
        let buffer_size = std::mem::size_of::<NetworkRingBuffer>()
            + (RING_BUFFER_SLOTS as usize * std::mem::size_of::<NetworkPacketSlot>());

        let ring_buffer = device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Initialize header
        let header = ring_buffer.contents() as *mut NetworkRingBuffer;
        unsafe {
            (*header).write_index = AtomicU32::new(0);
            (*header).read_index = AtomicU32::new(0);
            (*header).buffer_size = RING_BUFFER_SLOTS;
            (*header).slot_size = std::mem::size_of::<NetworkPacketSlot>() as u32;
        }

        let slots = unsafe {
            (ring_buffer.contents() as *mut u8)
                .add(std::mem::size_of::<NetworkRingBuffer>()) as *mut NetworkPacketSlot
        };

        let socket = UdpSocket::bind(format!("0.0.0.0:{}", port))
            .expect("Failed to bind socket");
        socket.set_nonblocking(true).expect("Failed to set non-blocking");

        NetworkReceiver {
            socket,
            ring_buffer: Arc::new(ring_buffer),
            header,
            slots,
            buffer_mask: RING_BUFFER_SLOTS - 1,
        }
    }

    /// Run the receive loop - this is the ONLY CPU work for networking
    pub fn receive_loop(&self) {
        let mut packet_buffer = [0u8; MAX_PACKET_SIZE];

        loop {
            // Non-blocking receive
            match self.socket.recv_from(&mut packet_buffer) {
                Ok((length, _addr)) => {
                    self.enqueue_packet(&packet_buffer[..length]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No packet available, yield to avoid spinning
                    std::thread::yield_now();
                }
                Err(e) => {
                    eprintln!("Network error: {}", e);
                }
            }
        }
    }

    fn enqueue_packet(&self, data: &[u8]) {
        unsafe {
            let header = &*self.header;

            // Read current indices
            let write_idx = header.write_index.load(Ordering::Acquire);
            let read_idx = header.read_index.load(Ordering::Acquire);

            // Check if buffer is full (leave one slot empty to distinguish full from empty)
            let next_write = (write_idx + 1) & self.buffer_mask;
            if next_write == read_idx {
                // Buffer full - drop packet (GPU can't keep up)
                // In production, implement backpressure or resize
                return;
            }

            // Write packet to slot
            let slot = &mut *self.slots.add(write_idx as usize);
            slot.length = data.len() as u32;
            slot.timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            slot.flags = 0; // Set based on packet type
            slot.data[..data.len()].copy_from_slice(data);

            // Memory barrier then update write index
            // Release ensures packet data is visible before index update
            header.write_index.store(next_write, Ordering::Release);
        }
    }

    pub fn get_buffer(&self) -> Arc<Buffer> {
        self.ring_buffer.clone()
    }
}
```

#### 3. GPU Polling Kernel

```metal
#include <metal_stdlib>
using namespace metal;

// Packet processing result - written to output buffer
struct ProcessedPacket {
    uint original_index;
    uint result_code;
    float4 computed_data;  // Example: parsed/transformed data
};

// GPU persistent kernel that polls for network packets
kernel void network_poll_kernel(
    device NetworkRingBuffer& ring [[buffer(0)]],
    device NetworkPacketSlot* slots [[buffer(1)]],
    device ProcessedPacket* output [[buffer(2)]],
    device atomic_uint& output_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    // Each thread processes packets in strided fashion
    // Thread 0 gets packets 0, 64, 128, ...
    // Thread 1 gets packets 1, 65, 129, ...

    while (true) {
        // Poll for new packets (acquire semantics)
        uint write_idx = atomic_load_explicit(
            &ring.write_index,
            memory_order_acquire
        );
        uint read_idx = atomic_load_explicit(
            &ring.read_index,
            memory_order_acquire
        );

        // Calculate available packets
        uint available = (write_idx - read_idx) & (ring.buffer_size - 1);

        if (available == 0) {
            // No packets - could yield or continue polling
            // On GPU, we typically just continue (persistent kernel)
            continue;
        }

        // Claim a packet slot using atomic increment
        uint my_slot = atomic_fetch_add_explicit(
            &ring.read_index,
            1,
            memory_order_acq_rel
        ) & (ring.buffer_size - 1);

        // Verify we got a valid slot (race condition check)
        write_idx = atomic_load_explicit(&ring.write_index, memory_order_acquire);
        if (((write_idx - my_slot) & (ring.buffer_size - 1)) == 0) {
            // Slot was empty, another thread got it - retry
            continue;
        }

        // Process the packet
        device NetworkPacketSlot& packet = slots[my_slot];

        // Example processing: parse and transform packet data
        ProcessedPacket result;
        result.original_index = my_slot;
        result.result_code = process_packet(packet, result.computed_data);

        // Write result to output buffer
        uint out_idx = atomic_fetch_add_explicit(&output_count, 1, memory_order_relaxed);
        output[out_idx] = result;
    }
}

// Example packet processing function
uint process_packet(device NetworkPacketSlot& packet, thread float4& output) {
    // This is where GPU's massive parallelism shines
    // Each packet processed independently by a GPU thread

    if (packet.length < 4) {
        return 1; // Error: packet too short
    }

    // Example: interpret first 16 bytes as float4 data
    // In real use: parse protocol, decompress, transform, etc.
    device float* data = (device float*)packet.data;
    output = float4(
        packet.length >= 4 ? data[0] : 0.0,
        packet.length >= 8 ? data[1] : 0.0,
        packet.length >= 12 ? data[2] : 0.0,
        packet.length >= 16 ? data[3] : 0.0
    );

    return 0; // Success
}
```

#### 4. Atomic Synchronization Details

```metal
// Memory ordering for lock-free ring buffer
//
// CPU (producer):
//   1. Write packet data to slot
//   2. memory_order_release on write_index update
//      (ensures packet data visible before index)
//
// GPU (consumer):
//   1. memory_order_acquire on write_index read
//      (ensures we see packet data after seeing index)
//   2. memory_order_acq_rel on read_index update
//      (both acquire previous writes, release our update)

// Alternative: batch processing kernel (better throughput)
kernel void network_batch_kernel(
    device NetworkRingBuffer& ring [[buffer(0)]],
    device NetworkPacketSlot* slots [[buffer(1)]],
    device ProcessedPacket* output [[buffer(2)]],
    device atomic_uint& output_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    // Snapshot current indices
    uint write_idx = atomic_load_explicit(&ring.write_index, memory_order_acquire);
    uint read_idx = atomic_load_explicit(&ring.read_index, memory_order_acquire);

    uint available = (write_idx - read_idx) & (ring.buffer_size - 1);

    // Each thread processes every Nth packet
    for (uint i = tid; i < available; i += threads_per_grid) {
        uint slot_idx = (read_idx + i) & (ring.buffer_size - 1);
        device NetworkPacketSlot& packet = slots[slot_idx];

        ProcessedPacket result;
        result.original_index = slot_idx;
        result.result_code = process_packet(packet, result.computed_data);

        uint out_idx = atomic_fetch_add_explicit(&output_count, 1, memory_order_relaxed);
        output[out_idx] = result;
    }

    // Only thread 0 updates read index after all processing
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_store_explicit(&ring.read_index, write_idx, memory_order_release);
    }
}
```

---

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| CPU involvement | ~1-5% | Packet receive + ring buffer copy only |
| Ring buffer latency | <1μs | Shared memory, atomic coordination |
| GPU processing throughput | 10M+ packets/sec | Depends on packet complexity |
| Memory bandwidth | 400+ GB/s | Apple Silicon unified memory |

#### Latency Breakdown (Single Mac)

```
Packet arrives at NIC           : 0 μs
Darwin kernel processes         : 5-50 μs (unavoidable)
CPU copies to ring buffer       : <1 μs
GPU sees packet (atomic poll)   : <1 μs
GPU processes packet            : 1-100 μs (depends on work)
─────────────────────────────────────────
Total                           : 7-152 μs
```

#### Throughput Scaling

```
1 GPU thread      : ~100K packets/sec
64 GPU threads    : ~6M packets/sec
1024 GPU threads  : ~50M packets/sec (memory bandwidth limited)
4096 GPU threads  : ~100M packets/sec (theoretical max)
```

---

### Why This Is Acceptable

Despite requiring CPU for packet receive, this architecture is acceptable because:

1. **CPU does minimal work**
   - No parsing, no protocol handling, no business logic
   - Just `recv()` → `memcpy()` → `atomic_store()`
   - CPU acts as a "dumb pipe"

2. **GPU does ALL processing**
   - Protocol parsing
   - Data transformation
   - Business logic
   - Response generation

3. **Latency is bounded**
   - Ring buffer provides predictable latency
   - No locks, no context switches in hot path
   - Atomic operations are sub-microsecond

4. **Throughput scales with GPU**
   - More GPU threads = more parallel packet processing
   - CPU is never the bottleneck (it just copies bytes)
   - Memory bandwidth is the limit, not CPU

5. **Architecture is future-proof**
   - If Apple adds GPU-accessible networking, we're ready
   - Clean separation makes swapping network backend easy

---

### Comparison to Traditional Architectures

| Aspect | Traditional | GPU-First (This) |
|--------|-------------|------------------|
| Packet parsing | CPU | GPU |
| Protocol handling | CPU | GPU |
| Data transformation | CPU | GPU |
| Business logic | CPU | GPU |
| Response building | CPU | GPU |
| **CPU work** | **Everything** | **Just recv + copy** |

---

### Future: True GPU Networking

Technologies that could eliminate CPU entirely:

1. **GPUDirect RDMA (NVIDIA/Linux)**
   - NIC writes directly to GPU memory
   - Not available on macOS

2. **Smart NICs**
   - NIC has programmable processor
   - Could DMA directly to unified memory

3. **Apple Silicon Evolution**
   - Future chips might expose NIC to GPU
   - Unified memory makes this architecturally clean

4. **Custom Hardware**
   - FPGA-based NIC with Thunderbolt
   - Direct GPU memory access

Until then, the ring buffer architecture provides the best achievable performance on macOS while keeping CPU involvement to the absolute minimum.
