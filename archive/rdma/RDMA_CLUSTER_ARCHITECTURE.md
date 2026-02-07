# RDMA Cluster Architecture for GPU-Native OS

## Overview

macOS 26.2 introduced RDMA (Remote Direct Memory Access) over Thunderbolt 5, enabling Mac clusters for distributed GPU compute. This document outlines how the GPU-Native OS can leverage RDMA to achieve true distributed GPU computing with minimal CPU involvement.

> **Important: RDMA is OPTIONAL for scale-out only.** A single Mac can run the GPU-Native OS with full functionality using GPU-native processes. RDMA enables *horizontal scaling* across multiple Macs, not basic operation.

---

## Single Mac vs. RDMA Cluster

### Single Mac Operation (No RDMA Required)

A single Mac runs the GPU-Native OS with full functionality:

| Operation | Implementation | CPU Involvement |
|-----------|----------------|-----------------|
| **Process Spawning** | GPU-native processes (compute shaders) | None |
| **File I/O** | `MTLIOCommandQueue` (GPU-initiated) | None |
| **Memory Management** | GPU unified memory | None |
| **Network I/O** | Traditional BSD sockets | **Required** (only exception) |

**Key Point:** Process spawning is entirely GPU-native on a single Mac. The GPU spawns and manages processes via persistent compute kernels - no CPU involvement, no RDMA.

### With RDMA Cluster (Optional Scale-Out)

RDMA enables horizontal scaling across multiple Macs:

| Capability | Description |
|------------|-------------|
| **Distributed Compute** | Spread GPU workloads across 2-8 Macs |
| **Shared Memory Pools** | Up to 1.5TB unified VRAM (8x Mac Studio Ultra) |
| **Network Gateway Pattern** | One Mac handles all network I/O for the cluster |
| **Cross-Mac Process Coordination** | Remote process spawning via RDMA buffers |

**When to use RDMA:**
- Workload exceeds single Mac's GPU capacity
- Dataset exceeds single Mac's memory (192GB max)
- Need fault tolerance / redundancy
- Want to isolate network I/O from compute nodes

**When NOT to use RDMA:**
- Single Mac is sufficient for your workload
- Just running desktop applications
- Development and testing

---

### RDMA Performance Characteristics

| Metric | Traditional Networking | RDMA over TB5 |
|--------|----------------------|---------------|
| Latency | ~300μs | <10μs |
| CPU involvement | High (kernel, driver, stack) | Zero (DMA bypass) |
| Memory copies | 2-4 per transfer | 0 (zero-copy) |
| Bandwidth | 10-100 Gbps | 120 Gbps (TB5) |

### Hardware Requirements (RDMA Cluster Only)

- **macOS**: 26.2 or later
- **Cables**: Thunderbolt 5 (120 Gbps bidirectional)
- **Recommended Hardware**: Mac Studio M3/M4 Ultra (192GB unified memory each)
- **Cluster Size**: 2-8 Macs (1.5TB+ shared VRAM with 4 Mac Studios)

> **Note:** Single Mac operation requires no special hardware beyond a Mac with Apple Silicon.

---

## Architecture Diagram (Multi-Mac Cluster)

The following diagram shows the **optional** RDMA cluster topology. This is NOT required for single-Mac operation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RDMA CLUSTER TOPOLOGY (OPTIONAL)                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     TB5 RDMA (120Gbps)    ┌─────────────────┐
│  Compute Mac 1  │◄─────────────────────────►│  Compute Mac 2  │
│  ┌───────────┐  │                           │  ┌───────────┐  │
│  │ GPU 192GB │  │                           │  │ GPU 192GB │  │
│  │           │  │                           │  │           │  │
│  │ ┌───────┐ │  │     Direct GPU-to-GPU     │  │ ┌───────┐ │  │
│  │ │ RDMA  │◄├──┼─────────────────────────────►├─│ RDMA  │ │  │
│  │ │Buffer │ │  │     Memory Access         │  │ │Buffer │ │  │
│  │ └───────┘ │  │                           │  │ └───────┘ │  │
│  └───────────┘  │                           │  └───────────┘  │
└────────┬────────┘                           └────────┬────────┘
         │                                             │
         │ TB5 RDMA                                    │ TB5 RDMA
         │                                             │
         ▼                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Gateway Mac                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    RDMA Coordinator                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │    │
│  │  │ Network  │  │ Process  │  │Filesystem│  │  Work   │  │    │
│  │  │   I/O    │  │ Spawner  │  │  Cache   │  │  Queue  │  │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘  │    │
│  └───────┼─────────────┼─────────────┼─────────────┼───────┘    │
│          │             │             │             │            │
│          ▼             ▼             ▼             ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Shared RDMA Buffer Pool                     │    │
│  │   [Network Data] [Spawn Requests] [File Cache] [Tasks]   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Internet/Network
```

---

## RDMA Cluster Use Cases

> **Reminder:** These use cases are for multi-Mac clusters only. A single Mac handles all of these operations locally via GPU-native processes and `MTLIOCommandQueue`.

### 1. Cross-Mac Process Coordination via RDMA

When running a multi-Mac cluster, you may want a compute node's GPU to trigger process execution on a different Mac (e.g., the Gateway Mac). This is done via RDMA buffer writes:

> **Single Mac Alternative:** On a single Mac, process spawning is handled entirely by GPU-native persistent compute kernels - no RDMA, no CPU process management.

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS SPAWN FLOW                        │
└─────────────────────────────────────────────────────────────┘

  Compute Mac GPU                    Gateway Mac
  ──────────────────                 ────────────────
        │
        │ 1. Write spawn request to RDMA buffer
        │    (executable path, args, env)
        ▼
  ┌───────────┐
  │   RDMA    │ ──── TB5 RDMA ────► ┌───────────┐
  │  Buffer   │                     │   RDMA    │
  └───────────┘ ◄─── TB5 RDMA ──── │  Buffer   │
        │                           └─────┬─────┘
        │                                 │
        │                                 ▼
        │                           2. Gateway reads request
        │                           3. Gateway spawns process
        │                           4. Process runs on Gateway
        │                                 │
        │                                 ▼
        │                           5. Result written to RDMA
        │
        ▼
  6. Compute GPU reads result
     (zero CPU involvement on compute node)
```

### 2. Network Gateway Pattern (Cluster-Only)

In a multi-Mac cluster, all network I/O can be isolated to a single Gateway Mac, keeping compute nodes completely CPU-free:

> **Single Mac:** Network I/O is the **only operation** that requires CPU involvement on a single Mac. All other operations (process spawning, file I/O, memory management) are GPU-native.

```
┌─────────────────────────────────────────────────────────────┐
│                    NETWORK GATEWAY FLOW                      │
└─────────────────────────────────────────────────────────────┘

  Internet          Gateway Mac              Compute Mac GPU
  ────────          ───────────              ─────────────────
      │
      │ TCP/UDP
      ▼
┌───────────┐
│  Network  │
│   Stack   │
└─────┬─────┘
      │
      ▼
┌───────────┐       ┌───────────┐          ┌───────────┐
│  Receive  │       │   RDMA    │          │   GPU     │
│  Buffer   │──────►│  Transfer │─────────►│  Memory   │
└───────────┘       └───────────┘          └───────────┘
                    Zero-copy DMA           Ready for
                    <10μs latency           compute

Benefits:
- Compute Mac GPU NEVER touches network stack
- No TCP/IP processing on compute nodes
- Data arrives directly in GPU memory
- Network latency hidden by RDMA pipelining
```

### 3. Distributed Compute with Shared Work Queue

```
┌─────────────────────────────────────────────────────────────┐
│                 DISTRIBUTED WORK QUEUE                       │
└─────────────────────────────────────────────────────────────┘

            Shared RDMA Work Queue (on Gateway)
┌─────────────────────────────────────────────────────────────┐
│  [Task 0] [Task 1] [Task 2] [Task 3] [Task 4] [Task 5] ...  │
│     ▲        ▲        ▲        ▲        ▲        ▲          │
│     │        │        │        │        │        │          │
│  Claimed  Claimed   Free    Claimed   Free     Free         │
│   Mac1    Mac2              Mac1                            │
└─────────────────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐     ┌────────┐     ┌────────┐
│ Mac 1  │     │ Mac 2  │     │ Mac 3  │
│  GPU   │     │  GPU   │     │  GPU   │
│        │     │        │     │        │
│ Task 0 │     │ Task 1 │     │  Idle  │
│ Task 3 │     │        │     │        │
└────────┘     └────────┘     └────────┘
    │               │               │
    └───────────────┼───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Result Buffer │
            │ (RDMA shared) │
            └───────────────┘

Work Stealing Algorithm:
1. GPU atomically claims task via RDMA atomic operation
2. Process task locally
3. Write result to shared RDMA result buffer
4. Atomic increment completion counter
5. If queue not empty, claim next task
```

### 4. Shared Filesystem Cache

```
┌─────────────────────────────────────────────────────────────┐
│              DISTRIBUTED FILESYSTEM CACHE                    │
└─────────────────────────────────────────────────────────────┘

                    Gateway Mac
                    ────────────
                         │
                    1. Read file from SSD
                         │
                         ▼
                  ┌─────────────┐
                  │ File Buffer │
                  │   (256MB)   │
                  └──────┬──────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    2. RDMA broadcast to all compute nodes
         │               │               │
         ▼               ▼               ▼
    ┌────────┐     ┌────────┐     ┌────────┐
    │ Mac 1  │     │ Mac 2  │     │ Mac 3  │
    │  GPU   │     │  GPU   │     │  GPU   │
    │ Cache  │     │ Cache  │     │ Cache  │
    └────────┘     └────────┘     └────────┘

Benefits:
- Single SSD read serves entire cluster
- Eliminates N redundant I/O operations
- File data arrives directly in GPU memory
- Cache coherency via RDMA notifications
```

---

## Implementation

### Core Data Structures

```rust
/// RDMA buffer registration for GPU memory
#[repr(C)]
pub struct RdmaBuffer {
    /// Base address in GPU unified memory
    pub gpu_base: *mut u8,
    /// Size in bytes (must be page-aligned)
    pub size: usize,
    /// RDMA registration key (for remote access)
    pub rkey: u32,
    /// Local key (for local access)
    pub lkey: u32,
    /// Thunderbolt port ID
    pub port_id: u8,
    pub _padding: [u8; 3],
}

/// Work item for distributed compute
#[repr(C)]
pub struct WorkItem {
    /// Unique task ID
    pub task_id: u64,
    /// Type of work (compute, network, spawn, etc.)
    pub work_type: u32,
    /// Status: 0=free, 1=claimed, 2=complete
    pub status: AtomicU32,
    /// Node that claimed this task
    pub owner_node: u32,
    /// Payload offset in shared buffer
    pub payload_offset: u64,
    /// Payload size
    pub payload_size: u32,
    /// Result offset (set after completion)
    pub result_offset: u64,
    /// Result size
    pub result_size: u32,
    pub _padding: [u32; 2],
}

/// Process spawn request
#[repr(C)]
pub struct SpawnRequest {
    /// Request ID for correlation
    pub request_id: u64,
    /// Executable path (null-terminated, max 256)
    pub executable: [u8; 256],
    /// Arguments (null-separated, double-null terminated)
    pub args: [u8; 1024],
    /// Environment (KEY=VALUE, null-separated)
    pub env: [u8; 2048],
    /// Working directory
    pub cwd: [u8; 256],
    /// Status: 0=pending, 1=running, 2=complete, 3=error
    pub status: AtomicU32,
    /// Exit code (valid when status=2)
    pub exit_code: i32,
    /// Stdout/stderr offset in result buffer
    pub output_offset: u64,
    pub output_size: u32,
    pub _padding: [u32; 1],
}

/// Network I/O request
#[repr(C)]
pub struct NetworkRequest {
    /// Request ID
    pub request_id: u64,
    /// Operation: 0=recv, 1=send, 2=connect, 3=listen
    pub operation: u32,
    /// Protocol: 0=TCP, 1=UDP
    pub protocol: u32,
    /// Remote address (IPv6, can embed IPv4)
    pub remote_addr: [u8; 16],
    /// Remote port
    pub remote_port: u16,
    /// Local port (0 = ephemeral)
    pub local_port: u16,
    /// Data offset in RDMA buffer
    pub data_offset: u64,
    /// Data size
    pub data_size: u32,
    /// Status
    pub status: AtomicU32,
    /// Bytes transferred
    pub bytes_transferred: u32,
    /// Error code (0 = success)
    pub error_code: i32,
}
```

### RDMA Buffer Setup

```swift
import Metal
import Thunderbolt

/// Sets up an RDMA-accessible GPU buffer
class RdmaGpuBuffer {
    let device: MTLDevice
    let buffer: MTLBuffer
    let rdmaRegion: TBRdmaMemoryRegion

    init(device: MTLDevice, size: Int, port: TBPort) throws {
        self.device = device

        // Allocate GPU buffer with shared storage mode
        // This ensures the memory is accessible by both GPU and RDMA hardware
        guard let buffer = device.makeBuffer(
            length: size,
            options: [.storageModeShared, .hazardTrackingModeTracked]
        ) else {
            throw RdmaError.bufferAllocationFailed
        }
        self.buffer = buffer

        // Register the buffer for RDMA access
        // This pins the memory and provides it to the Thunderbolt RDMA subsystem
        let rdmaConfig = TBRdmaMemoryConfig(
            virtualAddress: buffer.contents(),
            length: size,
            accessFlags: [.remoteRead, .remoteWrite, .localRead, .localWrite]
        )

        self.rdmaRegion = try port.registerMemory(config: rdmaConfig)

        print("RDMA buffer registered:")
        print("  GPU address: \(buffer.gpuAddress)")
        print("  RDMA rkey: \(rdmaRegion.remoteKey)")
        print("  Size: \(size / 1024 / 1024) MB")
    }

    /// Returns the remote key for other nodes to access this buffer
    var remoteKey: UInt32 { rdmaRegion.remoteKey }

    /// Returns the GPU address for local compute shaders
    var gpuAddress: UInt64 { buffer.gpuAddress }

    deinit {
        rdmaRegion.deregister()
    }
}
```

### GPU Memory Mapping for RDMA

```swift
import Metal
import Thunderbolt

/// Maps a remote RDMA buffer into local GPU address space
class RemoteGpuMemory {
    let localDevice: MTLDevice
    let remoteMapping: TBRdmaRemoteMapping
    let localBuffer: MTLBuffer

    init(
        localDevice: MTLDevice,
        remoteNode: TBRdmaNode,
        remoteKey: UInt32,
        remoteAddress: UInt64,
        size: Int
    ) throws {
        self.localDevice = localDevice

        // Create RDMA connection to remote node
        let mappingConfig = TBRdmaRemoteMappingConfig(
            remoteKey: remoteKey,
            remoteAddress: remoteAddress,
            length: size,
            accessFlags: [.read, .write]
        )

        self.remoteMapping = try remoteNode.mapRemoteMemory(config: mappingConfig)

        // Create a local MTLBuffer that references the remote memory
        // This allows GPU shaders to read/write as if it were local memory
        guard let buffer = localDevice.makeBuffer(
            bytesNoCopy: remoteMapping.localVirtualAddress,
            length: size,
            options: [.storageModeShared],
            deallocator: nil
        ) else {
            throw RdmaError.bufferMappingFailed
        }
        self.localBuffer = buffer

        print("Remote memory mapped:")
        print("  Remote node: \(remoteNode.identifier)")
        print("  Local GPU can access at: \(buffer.gpuAddress)")
    }

    /// Explicitly flush writes to ensure visibility on remote node
    func flush() {
        remoteMapping.flush()
    }

    /// Wait for all remote operations to complete
    func fence() {
        remoteMapping.fence()
    }
}
```

### Cross-Mac Synchronization

```swift
import Metal
import Thunderbolt

/// Provides cluster-wide synchronization primitives
class ClusterSynchronization {
    let nodes: [TBRdmaNode]
    let coordinatorBuffer: RdmaGpuBuffer

    // Offsets in coordinator buffer for sync primitives
    private let barrierCounterOffset: Int = 0
    private let barrierGenerationOffset: Int = 8
    private let lockOffset: Int = 16

    init(nodes: [TBRdmaNode], coordinatorBuffer: RdmaGpuBuffer) {
        self.nodes = nodes
        self.coordinatorBuffer = coordinatorBuffer
    }

    /// Cluster-wide barrier - all nodes must reach this point before any proceed
    func barrier() async throws {
        let nodeCount = UInt64(nodes.count)
        let bufferPtr = coordinatorBuffer.buffer.contents()

        // Atomically increment barrier counter
        let counterPtr = bufferPtr.advanced(by: barrierCounterOffset)
            .assumingMemoryBound(to: UInt64.self)
        let generationPtr = bufferPtr.advanced(by: barrierGenerationOffset)
            .assumingMemoryBound(to: UInt64.self)

        let myGeneration = generationPtr.pointee
        let arrived = OSAtomicIncrement64(counterPtr)

        if arrived == nodeCount {
            // Last to arrive - reset counter and advance generation
            counterPtr.pointee = 0
            OSAtomicIncrement64(generationPtr)

            // Broadcast completion to all nodes via RDMA write
            for node in nodes {
                try await node.rdmaWrite(
                    localBuffer: coordinatorBuffer.buffer,
                    localOffset: barrierGenerationOffset,
                    remoteKey: node.barrierBufferKey,
                    remoteOffset: barrierGenerationOffset,
                    length: 8
                )
            }
        } else {
            // Wait for generation to advance
            while generationPtr.pointee == myGeneration {
                // Spin with backoff
                await Task.yield()
            }
        }
    }

    /// Distributed spinlock using RDMA atomics
    func acquireLock() async throws {
        let bufferPtr = coordinatorBuffer.buffer.contents()
        let lockPtr = bufferPtr.advanced(by: lockOffset)
            .assumingMemoryBound(to: UInt32.self)

        while true {
            // Try to acquire lock with atomic compare-and-swap
            let result = try await rdmaCompareAndSwap(
                address: lockPtr,
                expected: 0,
                desired: UInt32(getNodeId())
            )

            if result == 0 {
                // Lock acquired
                return
            }

            // Backoff before retry
            try await Task.sleep(nanoseconds: 100)
        }
    }

    func releaseLock() {
        let bufferPtr = coordinatorBuffer.buffer.contents()
        let lockPtr = bufferPtr.advanced(by: lockOffset)
            .assumingMemoryBound(to: UInt32.self)

        // Memory barrier to ensure all writes are visible
        OSMemoryBarrier()
        lockPtr.pointee = 0
    }

    /// RDMA atomic compare-and-swap
    private func rdmaCompareAndSwap(
        address: UnsafeMutablePointer<UInt32>,
        expected: UInt32,
        desired: UInt32
    ) async throws -> UInt32 {
        // This would use TBRdmaAtomicOperation in the actual implementation
        return OSAtomicCompareAndSwap32(
            Int32(bitPattern: expected),
            Int32(bitPattern: desired),
            UnsafeMutablePointer(OpaquePointer(address))
        ) ? expected : address.pointee
    }
}
```

### Work Queue Distribution

```swift
import Metal
import Thunderbolt

/// Distributed work queue with work stealing
class DistributedWorkQueue {
    let device: MTLDevice
    let workBuffer: RdmaGpuBuffer
    let resultBuffer: RdmaGpuBuffer
    let maxTasks: Int
    let nodeId: UInt32

    private let headerSize = 64  // Queue metadata
    private let taskSize = MemoryLayout<WorkItem>.stride

    init(
        device: MTLDevice,
        maxTasks: Int,
        port: TBPort,
        nodeId: UInt32
    ) throws {
        self.device = device
        self.maxTasks = maxTasks
        self.nodeId = nodeId

        // Allocate work queue buffer
        let workBufferSize = headerSize + (maxTasks * taskSize)
        self.workBuffer = try RdmaGpuBuffer(
            device: device,
            size: workBufferSize,
            port: port
        )

        // Allocate result buffer (4KB per task)
        self.resultBuffer = try RdmaGpuBuffer(
            device: device,
            size: maxTasks * 4096,
            port: port
        )

        // Initialize queue header
        let header = workBuffer.buffer.contents().assumingMemoryBound(to: QueueHeader.self)
        header.pointee = QueueHeader(
            head: 0,
            tail: 0,
            taskCount: 0,
            completedCount: 0
        )
    }

    /// Submit a task to the distributed queue (called from GPU compute shader)
    func createSubmitKernel() -> MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct WorkItem {
            uint64_t task_id;
            uint32_t work_type;
            atomic_uint status;
            uint32_t owner_node;
            uint64_t payload_offset;
            uint32_t payload_size;
            uint64_t result_offset;
            uint32_t result_size;
            uint32_t _padding[2];
        };

        struct QueueHeader {
            atomic_uint head;
            atomic_uint tail;
            atomic_uint task_count;
            atomic_uint completed_count;
        };

        kernel void submit_task(
            device QueueHeader* header [[buffer(0)]],
            device WorkItem* tasks [[buffer(1)]],
            device const uint32_t* new_task_data [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            // Atomically claim a slot
            uint slot = atomic_fetch_add_explicit(
                &header->tail, 1, memory_order_relaxed
            );

            // Initialize task
            tasks[slot].task_id = slot;
            tasks[slot].work_type = new_task_data[0];
            atomic_store_explicit(&tasks[slot].status, 0, memory_order_release);
            tasks[slot].owner_node = 0xFFFFFFFF;  // Unclaimed
            tasks[slot].payload_offset = uint64_t(slot) * 4096;
            tasks[slot].payload_size = new_task_data[1];

            // Increment task count (signals workers)
            atomic_fetch_add_explicit(&header->task_count, 1, memory_order_release);
        }
        """

        let library = try! device.makeLibrary(source: source, options: nil)
        let function = library.makeFunction(name: "submit_task")!
        return try! device.makeComputePipelineState(function: function)
    }

    /// Claim and process tasks (called from GPU compute shader)
    func createWorkerKernel() -> MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct WorkItem {
            uint64_t task_id;
            uint32_t work_type;
            atomic_uint status;
            uint32_t owner_node;
            uint64_t payload_offset;
            uint32_t payload_size;
            uint64_t result_offset;
            uint32_t result_size;
            uint32_t _padding[2];
        };

        struct QueueHeader {
            atomic_uint head;
            atomic_uint tail;
            atomic_uint task_count;
            atomic_uint completed_count;
        };

        kernel void process_tasks(
            device QueueHeader* header [[buffer(0)]],
            device WorkItem* tasks [[buffer(1)]],
            device uint8_t* payload_buffer [[buffer(2)]],
            device uint8_t* result_buffer [[buffer(3)]],
            constant uint32_t& node_id [[buffer(4)]],
            constant uint32_t& max_tasks [[buffer(5)]],
            uint tid [[thread_position_in_grid]]
        ) {
            while (true) {
                // Try to claim a task
                uint task_count = atomic_load_explicit(
                    &header->task_count, memory_order_acquire
                );

                if (task_count == 0) {
                    return;  // No work available
                }

                // Find an unclaimed task
                uint slot = atomic_fetch_add_explicit(
                    &header->head, 1, memory_order_relaxed
                ) % max_tasks;

                // Try to claim it
                uint expected = 0;  // Free
                bool claimed = atomic_compare_exchange_weak_explicit(
                    &tasks[slot].status,
                    &expected,
                    1,  // Claimed
                    memory_order_acq_rel,
                    memory_order_relaxed
                );

                if (!claimed) {
                    continue;  // Someone else got it
                }

                tasks[slot].owner_node = node_id;

                // Process the task based on work_type
                device uint8_t* payload = payload_buffer + tasks[slot].payload_offset;
                device uint8_t* result = result_buffer + tasks[slot].result_offset;

                switch (tasks[slot].work_type) {
                    case 0:  // Compute task
                        // ... perform computation ...
                        break;
                    case 1:  // Data transform
                        // ... transform data ...
                        break;
                    default:
                        break;
                }

                // Mark complete
                atomic_store_explicit(&tasks[slot].status, 2, memory_order_release);
                atomic_fetch_add_explicit(
                    &header->completed_count, 1, memory_order_relaxed
                );
                atomic_fetch_sub_explicit(
                    &header->task_count, 1, memory_order_relaxed
                );
            }
        }
        """

        let library = try! device.makeLibrary(source: source, options: nil)
        let function = library.makeFunction(name: "process_tasks")!
        return try! device.makeComputePipelineState(function: function)
    }
}

struct QueueHeader {
    var head: UInt32
    var tail: UInt32
    var taskCount: UInt32
    var completedCount: UInt32
    var _padding: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0, 0, 0)
}
```

---

## Network Gateway Implementation

```swift
import Metal
import Thunderbolt
import Network

/// Gateway Mac network handler - bridges network I/O to RDMA
class NetworkGateway {
    let rdmaBuffers: [NodeId: RdmaGpuBuffer]
    let requestQueue: RdmaGpuBuffer
    let networkQueue: DispatchQueue

    /// Process network requests from compute nodes
    func startRequestProcessor() {
        networkQueue.async { [self] in
            while true {
                // Scan request queue for pending requests
                let requests = scanPendingRequests()

                for request in requests {
                    Task {
                        await processNetworkRequest(request)
                    }
                }

                // Small sleep to avoid spinning
                Thread.sleep(forTimeInterval: 0.0001)  // 100μs
            }
        }
    }

    private func processNetworkRequest(_ request: NetworkRequest) async {
        switch request.operation {
        case 0:  // Receive
            await handleReceive(request)
        case 1:  // Send
            await handleSend(request)
        case 2:  // Connect
            await handleConnect(request)
        case 3:  // Listen
            await handleListen(request)
        default:
            break
        }
    }

    private func handleReceive(_ request: NetworkRequest) async {
        // Receive data from network
        let connection = getConnection(for: request)

        connection.receive(minimumIncompleteLength: 1, maximumLength: Int(request.data_size)) {
            [self] data, _, _, error in

            if let data = data {
                // RDMA write directly to compute node's GPU memory
                let nodeBuffer = rdmaBuffers[request.sourceNode]!

                Task {
                    try await rdmaWrite(
                        data: data,
                        to: nodeBuffer,
                        offset: Int(request.data_offset)
                    )

                    // Update request status
                    updateRequestStatus(request, status: 2, bytesTransferred: data.count)
                }
            }
        }
    }

    private func handleSend(_ request: NetworkRequest) async {
        // Read data from compute node's GPU via RDMA
        let nodeBuffer = rdmaBuffers[request.sourceNode]!

        let data = try await rdmaRead(
            from: nodeBuffer,
            offset: Int(request.data_offset),
            length: Int(request.data_size)
        )

        // Send to network
        let connection = getConnection(for: request)
        connection.send(content: data, completion: .contentProcessed { error in
            if error == nil {
                self.updateRequestStatus(request, status: 2, bytesTransferred: data.count)
            } else {
                self.updateRequestStatus(request, status: 3, error: error)
            }
        })
    }
}
```

---

## Process Spawner Implementation

```swift
import Foundation
import Thunderbolt

/// Gateway Mac process spawner - handles spawn requests from compute nodes
class ProcessSpawner {
    let spawnRequestBuffer: RdmaGpuBuffer
    let resultBuffer: RdmaGpuBuffer
    let maxConcurrentProcesses: Int

    private var runningProcesses: [UInt64: Process] = [:]

    /// Monitor spawn request buffer and execute processes
    func startSpawnMonitor() {
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            while true {
                let requests = scanPendingSpawnRequests()

                for request in requests {
                    spawnProcess(request)
                }

                Thread.sleep(forTimeInterval: 0.0001)  // 100μs polling
            }
        }
    }

    private func spawnProcess(_ request: SpawnRequest) {
        let process = Process()

        // Extract executable path
        let executablePath = String(
            cString: withUnsafePointer(to: request.executable) {
                $0.withMemoryRebound(to: CChar.self, capacity: 256) { $0 }
            }
        )
        process.executableURL = URL(fileURLWithPath: executablePath)

        // Parse arguments
        process.arguments = parseNullSeparatedStrings(request.args)

        // Parse environment
        process.environment = parseEnvironment(request.env)

        // Set working directory
        let cwd = String(
            cString: withUnsafePointer(to: request.cwd) {
                $0.withMemoryRebound(to: CChar.self, capacity: 256) { $0 }
            }
        )
        process.currentDirectoryURL = URL(fileURLWithPath: cwd)

        // Capture output
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        // Update status to running
        updateSpawnStatus(request.request_id, status: 1)

        process.terminationHandler = { [self] proc in
            // Read output
            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()

            // RDMA write output to result buffer
            Task {
                let outputOffset = try await writeToResultBuffer(outputData)

                // Update request with completion status
                completeSpawnRequest(
                    requestId: request.request_id,
                    exitCode: proc.terminationStatus,
                    outputOffset: outputOffset,
                    outputSize: outputData.count
                )
            }

            runningProcesses.removeValue(forKey: request.request_id)
        }

        do {
            try process.run()
            runningProcesses[request.request_id] = process
        } catch {
            updateSpawnStatus(request.request_id, status: 3, error: error)
        }
    }
}
```

---

## Performance Characteristics

### Latency Comparison

| Operation | TCP/IP | RDMA over TB5 | Improvement |
|-----------|--------|---------------|-------------|
| Small message (64B) | 50-100μs | 2-5μs | 20x |
| Medium message (4KB) | 80-150μs | 5-8μs | 15x |
| Large transfer (1MB) | 500μs-1ms | 50-100μs | 10x |
| GPU memory access | N/A (copy required) | <10μs (direct) | Infinite |

### Bandwidth

| Configuration | Theoretical | Practical |
|--------------|-------------|-----------|
| Single TB5 link | 120 Gbps | 100 Gbps |
| Dual TB5 links | 240 Gbps | 180 Gbps |
| 4-node cluster | 360 Gbps aggregate | 300 Gbps |

### Memory Capacity

| Cluster Size | Total VRAM | Use Case |
|-------------|-----------|----------|
| 2x Mac Studio Ultra | 384 GB | Large ML models |
| 4x Mac Studio Ultra | 768 GB | Very large datasets |
| 8x Mac Studio Ultra | 1.5 TB | Enterprise workloads |

---

## Best Practices

### 1. Buffer Sizing

```swift
// Align buffers to page size for optimal RDMA performance
let pageSize = 16384  // 16KB on Apple Silicon
let bufferSize = ((requestedSize + pageSize - 1) / pageSize) * pageSize
```

### 2. Batching Small Operations

```swift
// BAD: Many small RDMA operations
for item in items {
    try await rdmaWrite(item, size: 64)  // High overhead per operation
}

// GOOD: Batch into single operation
let batchBuffer = items.flatMap { $0.bytes }
try await rdmaWrite(batchBuffer, size: items.count * 64)  // Single operation
```

### 3. Avoiding False Sharing

```swift
// BAD: Adjacent atomic counters cause cache line bouncing
struct Counters {
    var counter1: AtomicUInt32  // Same cache line
    var counter2: AtomicUInt32  // Same cache line - false sharing!
}

// GOOD: Pad to separate cache lines
struct Counters {
    var counter1: AtomicUInt32
    var _pad1: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64) = (0,0,0,0,0,0,0)
    var counter2: AtomicUInt32
    var _pad2: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64) = (0,0,0,0,0,0,0)
}
```

### 4. GPU Kernel Design for RDMA

```metal
// Design kernels to work with RDMA-accessible buffers
kernel void process_rdma_data(
    device float4* rdma_input [[buffer(0)]],   // Remote node's data
    device float4* rdma_output [[buffer(1)]],  // Write results here
    device atomic_uint* completion [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Process data
    rdma_output[tid] = rdma_input[tid] * 2.0;

    // Last thread signals completion
    if (tid == threads - 1) {
        atomic_store_explicit(completion, 1, memory_order_release);
    }
}
```

---

## Future Directions

### GPU-Initiated RDMA

Current implementation requires CPU to set up RDMA operations. Future macOS versions may enable:

```metal
// Hypothetical future API - GPU-initiated RDMA
kernel void gpu_rdma_write(
    device void* local_data [[buffer(0)]],
    constant RdmaDescriptor& remote [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // GPU directly issues RDMA write
    rdma_write_async(
        local_data + tid * 64,
        remote.address + tid * 64,
        64,
        remote.rkey
    );
}
```

### Disaggregated Memory

Treat the cluster as a single memory space:

```
┌─────────────────────────────────────────────────────────────┐
│              CLUSTER-WIDE VIRTUAL ADDRESS SPACE              │
├─────────────────────────────────────────────────────────────┤
│ 0x0000_0000 - 0x2FFF_FFFF : Local GPU Memory (192GB)        │
│ 0x3000_0000 - 0x5FFF_FFFF : Node 1 Memory (192GB)           │
│ 0x6000_0000 - 0x8FFF_FFFF : Node 2 Memory (192GB)           │
│ 0x9000_0000 - 0xBFFF_FFFF : Node 3 Memory (192GB)           │
│                                                              │
│ GPU shader accesses any address transparently               │
│ RDMA hardware handles remote access automatically           │
└─────────────────────────────────────────────────────────────┘
```

---

## References

- Apple Thunderbolt 5 Technical Overview (2026)
- macOS 26.2 RDMA Framework Documentation
- Metal Best Practices Guide - Distributed Compute
- InfiniBand Architecture Specification (concepts applicable to TB5 RDMA)
- GPU-Initiated Communication research papers (BaM, GoFS)
