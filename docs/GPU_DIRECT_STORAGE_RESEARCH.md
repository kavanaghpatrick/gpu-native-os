# GPU Direct Storage and NVMe Integration Research

**Date**: 2026-01-24
**Purpose**: Critical research for understanding GPU-to-storage DMA feasibility for GPU-Native OS project

---

## Executive Summary

**Key Finding**: Direct GPU-to-NVMe DMA is technically possible but **NOT natively supported on Apple Silicon/Metal**. NVIDIA's GPUDirect Storage (GDS) demonstrates the feasibility on x86 systems, and research projects like BaM and AGILE show GPU-initiated storage access is achievable, but Apple has not implemented equivalent functionality for Metal.

**Critical Limitation for Our Project**: Metal compute shaders **cannot directly issue I/O operations**. All storage I/O must be mediated through the CPU using standard macOS APIs (IOKit, CoreFoundation, POSIX). The unified memory architecture partially mitigates this by reducing copy overhead, but true zero-copy GPU-to-NVMe DMA is not available.

**Implications**:
- Must design around CPU-mediated I/O
- Leverage unified memory for efficient CPU-GPU data sharing
- Consider asynchronous I/O patterns to hide latency
- May need to prototype GPU filesystem using CPU staging buffers

---

## 1. GPU Direct Storage (GDS) - NVIDIA Implementation

### What is GPU Direct Storage?

GPU Direct Storage enables **direct DMA transfers between GPU memory and storage devices** (NVMe SSDs, NVMe-oF) without bounce buffers through CPU memory. This eliminates the traditional path where data flows: Storage → CPU Memory → GPU Memory.

**Architecture**:
- **Direct data path**: Storage ↔ GPU memory via PCIe bus
- **Bypass CPU memory**: Eliminates bounce buffer overhead
- **DMA engines**: NVMe controller or storage adapter directly accesses GPU memory
- **cuFile APIs**: High-level interface abstracting low-level I/O complexity

### How It Works

1. **PCIe Direct Access**: Storage devices and GPUs under a common PCIe switch communicate directly with higher bandwidth than routing through CPU
2. **Memory Pinning**: GPU memory buffers registered with storage driver via nvidia-fs.ko kernel module
3. **DMA Operations**: Storage controller performs DMA using GPU memory addresses obtained from callbacks
4. **RDMA Support**: For remote storage, uses GPUDirect RDMA to transfer data over network directly to GPU

### Platform Support

- **NVIDIA GPUs**: Fully supported with cuFile library and nvidia-fs kernel module
- **Linux**: Primary platform (requires O_DIRECT filesystem access)
- **Windows**: Supported with recent GDS versions
- **Requirements**:
  - NVIDIA GPU with CUDA support
  - NVMe SSD or NVMe-oF capable storage
  - PCIe 3.0+ (PCIe atomics optional but not required)

### Performance Benefits

- **Bandwidth**: PCIe bandwidth between GPU-storage often higher than GPU-CPU-storage path
- **Latency**: Most apparent with small transfers (eliminates CPU memcpy overhead)
- **CPU Offload**: Frees CPU cycles and memory bandwidth
- **Scalability**: Multiple GPUs can access storage without CPU bottleneck

### Limitations

- **NVIDIA-specific**: Proprietary to NVIDIA GPUs
- **Kernel module required**: nvidia-fs.ko kernel driver
- **O_DIRECT requirement**: Bypasses page cache (application must manage caching)
- **Limited platform support**: No support for Apple Silicon

**Sources**:
- [NVIDIA GPUDirect Storage Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [What is GPUDirect Storage - Weka](https://www.weka.io/learn/glossary/gpu/what-is-gpudirect-storage/)
- [NVIDIA GPUDirect Storage: 4 Key Features - Cloudian](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/)

---

## 2. Apple Silicon / Metal - No Direct Storage Support

### Metal Capabilities

**What Metal Provides**:
- **Unified memory architecture**: CPU and GPU share physical memory via fabric
- **MTLBuffer**: Shared memory buffers accessible from both CPU and GPU
- **Storage modes**:
  - `.storageModeShared`: Best for unified memory (M-series chips)
  - Both CPU and GPU can read/write without explicit synchronization
- **Compute shaders**: Massively parallel kernel execution
- **Memory coherence**: Automatic via unified memory fabric

**What Metal Does NOT Provide**:
- **Direct I/O from GPU kernels**: No file operations in Metal shaders
- **GPU-initiated storage access**: Cannot issue NVMe commands from GPU
- **Asynchronous I/O APIs for GPU**: No async file I/O primitives in Metal Shading Language
- **Storage DMA to GPU memory**: No equivalent to GPUDirect Storage

### Architecture Comparison

| Feature | NVIDIA GDS | Apple Metal |
|---------|-----------|-------------|
| Direct GPU-NVMe DMA | ✅ Yes | ❌ No |
| Unified Memory | ❌ No (discrete) | ✅ Yes |
| GPU-initiated I/O | ✅ Yes (BaM research) | ❌ No |
| CPU-GPU copy overhead | High (PCIe) | Low (shared memory) |
| Storage APIs in GPU | ✅ cuFile | ❌ None |

### Unified Memory as Partial Mitigation

**Advantages**:
- **Low-copy overhead**: CPU can perform I/O to shared MTLBuffer
- **No PCIe transfers**: Data already in shared memory after CPU reads it
- **High bandwidth**: 120-546 GB/s depending on M4 variant
- **Coherent access**: No explicit synchronization needed

**Pattern for Metal**:
```
1. CPU thread: Async I/O read → MTLBuffer (shared memory)
2. Signal GPU via MTLCommandBuffer
3. GPU kernel: Directly access data in MTLBuffer
4. GPU kernel: Process and write results to MTLBuffer
5. CPU thread: Async I/O write from MTLBuffer → Storage
```

This avoids **copies** but still requires **CPU mediation** for I/O operations.

**Sources**:
- [Metal Overview - Apple Developer](https://developer.apple.com/metal/)
- [Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)
- [Creating and Configuring Metal Buffers](https://medium.com/@ios_guru/creating-and-configuring-metal-buffers-913338be85fd)

---

## 3. Apple Silicon Storage Architecture

### M4 Chip Specifications

**Memory Bandwidth** (unified memory):
- **M4 Base**: 120 GB/s (LPDDR5X-7500)
- **M4 Pro**: 273 GB/s (up to 64GB memory)
- **M4 Max**: 546 GB/s (up to 128GB memory)
  - Binned 32-core variant: 410 GB/s

**I/O Capabilities**:
- **Thunderbolt 5**: Up to 120 Gb/s (15 GB/s) on M4 Pro/Max
- **Thunderbolt 4**: 40 Gb/s (5 GB/s) on M4 base
- **Internal NVMe**: Likely PCIe 4.0 x4 (~7-8 GB/s per lane)

**Architecture**:
- **SoC integration**: CPU, GPU, memory controller, storage controller on single die
- **Unified memory fabric**: High-speed interconnect between all components
- **Storage controller**: Integrated with AES encryption hardware
- **No discrete GPU**: Eliminates PCIe bottleneck between CPU and GPU

### NVMe Controller Specifications

**Apple's Implementation**:
- Integrated NVMe controller in SoC
- Direct connection to unified memory fabric
- Hardware-accelerated AES encryption
- Queue depth and parallel operations: **Not publicly documented**

**Theoretical Capabilities** (based on typical NVMe):
- NVMe 1.4+ support (likely)
- Multiple submission/completion queue pairs
- High queue depth (1K+ entries)
- Parallel I/O operations across multiple namespaces

**Critical Unknown**:
- Can NVMe controller DMA directly to GPU-accessible memory regions?
- Apple does not expose low-level NVMe APIs to userspace
- IOKit provides abstraction but may not allow direct DMA mapping

### Unified Memory Fabric

**How It Works**:
1. **Fabric controller**: Allocates unified memory between CPU, GPU, neural engine, etc.
2. **Coherent access**: All components see consistent memory state
3. **High bandwidth**: Far exceeds typical PCIe CPU-GPU links
4. **Low latency**: On-die interconnect, not external bus

**Storage Integration**:
- Storage controller integrated into fabric
- Can DMA to unified memory
- **Unknown**: Whether GPU-specific memory regions are accessible for NVMe DMA

**Sources**:
- [Apple M4 - Wikipedia](https://en.wikipedia.org/wiki/Apple_M4)
- [Apple M4 specifications - Apple](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
- [Apple vs. Oranges: M-Series SoCs for HPC](https://arxiv.org/html/2502.05317v1)
- [Unified Memory Architecture - MacKeeper](https://mackeeper.com/blog/unified-memory/)

---

## 4. NVMe and GPU Integration Research

### IOKit APIs for GPU Memory Mapping

**IOKit Capabilities**:
- **IOMemoryDescriptor**: Memory mapping abstraction
- **IOBufferMemoryDescriptor**: DMA buffer allocation
- **IOConnectMapMemory**: User-kernel memory mapping
- **vm_map_kernel**: Kernel virtual memory mapping

**IOMMU Support**:
- **Purpose**: Manages DMA address translation and protection
- **Zero-copy I/O**: Can map userspace memory into device DMA address space
- **PCIe peer-to-peer**: Theoretically enables direct device-to-device transfers

**Apple Silicon Specifics**:
- **IOMMU present**: Yes (DART - Device Address Resolution Table)
- **Userspace access**: Extremely limited (sandboxing, no kernel extensions)
- **Metal integration**: No documented APIs for exposing MTLBuffer to NVMe DMA

### Current Limitations

**macOS Restrictions**:
1. **No kernel extensions**: System extensions only (no direct hardware access)
2. **No userspace NVMe drivers**: Unlike Linux (ssd-gpu-dma project)
3. **Sandboxing**: App Sandbox prevents low-level I/O
4. **DriverKit limitations**: Limited hardware access compared to kexts

**What We Cannot Do on macOS**:
- Map MTLBuffer memory regions directly to NVMe controller
- Issue NVMe commands from userspace
- Create custom NVMe driver with GPU DMA support
- Access NVMe BAR (Base Address Register) space

**What We Can Do**:
- Use standard POSIX I/O to shared MTLBuffers
- Leverage unified memory for low-copy overhead
- Use async I/O (dispatch I/O, GCD) for concurrency
- Optimize for Metal's shared memory architecture

**Sources**:
- [IOKit Fundamentals - Apple Developer](https://developer.apple.com/documentation/kernel/iokit_fundamentals)
- [IOMMU Impact - Medium](https://medium.com/@mike.anderson007/the-iommu-impact-i-o-memory-management-units-feb7ea95b819)
- [IOKit - Apple Developer Documentation](https://developer.apple.com/documentation/iokit)

---

## 5. Research: BaM System (GPU-Initiated Storage Access)

### Overview

**BaM** (Big accelerator Memory) is a research system from ASPLOS 2023 that enables **GPU threads to directly initiate NVMe storage accesses** without CPU involvement.

**Key Innovation**: Moves NVMe submission/completion queues and I/O buffers into GPU memory, allowing GPU threads to write directly to NVMe doorbell registers.

### Architecture

**Components**:
1. **GPU-resident NVMe queues**: Submission and completion queues in GPU memory
2. **Software cache**: Fine-grained caching to coalesce requests and reduce I/O amplification
3. **Doorbell writes**: GPU threads write to NVMe controller BAR space to ring doorbell
4. **PCIe peer-to-peer**: Direct GPU-NVMe communication via PCIe

**How It Works**:
```
1. GPU thread: Check software cache for data
2. Cache miss: Thread writes NVMe command to submission queue (in GPU memory)
3. Thread writes to NVMe doorbell register (PCIe MMIO)
4. NVMe controller: Fetches command from GPU memory
5. NVMe controller: DMAs data directly to GPU memory
6. Controller: Writes completion entry to GPU memory
7. GPU thread: Polls completion queue, retrieves data
```

**Key Requirements**:
- GPU memory mappable to NVMe controller DMA
- PCIe BAR space accessible from GPU threads
- Highly concurrent queue management to handle thousands of threads

### Performance Results

- **1.0x - 1.49x speedup** for graph analytics (BFS, connected components)
- **Up to 21.7x hardware cost reduction** vs. keeping data in host memory
- Demonstrates feasibility of GPU-initiated storage access

### Applicability to Metal

**Challenge**: BaM requires capabilities Metal doesn't provide:
- Writing to PCIe device registers (NVMe doorbell) from GPU
- Managing NVMe queues in GPU memory
- Issuing MMIO writes from Metal shaders

**Potential Workaround** (future research):
- Could Apple add Metal APIs for device register access?
- Would require significant security model changes
- Unlikely given Apple's closed ecosystem and sandboxing philosophy

**Sources**:
- [BaM: GPU-Initiated On-Demand High-Throughput Storage Access (arXiv)](https://arxiv.org/abs/2203.04910)
- [BaM - ASPLOS 2023 Proceedings](https://dl.acm.org/doi/10.1145/3575693.3575748)
- [BaM GitHub Implementation](https://github.com/ZaidQureshi/bam)

---

## 6. Research: ssd-gpu-dma (Userspace NVMe Driver)

### Overview

Open-source project enabling **userspace NVMe drivers with CUDA support** for direct GPU-SSD access on Linux.

### Key Features

**Userspace NVMe Driver (libnvm)**:
- Complete NVMe driver implementation in userspace
- Direct device control without kernel involvement
- Zero-copy access (eliminates kernel bounce buffers)

**CUDA Integration**:
- Link libnvm with CUDA programs
- High-performance storage access from CUDA kernels
- PCIe peer-to-peer between NVMe and GPU

**Performance Benefits**:
- Eliminates context switch into kernel
- Zero-copy I/O reduces latency
- Direct mapping of userspace memory to NVMe DMA

### Why This Doesn't Work on macOS

**Linux-Specific Capabilities**:
1. **Userspace I/O (UIO)**: Allows userspace drivers
2. **VFIO**: Safe device pass-through to userspace
3. **Relaxed security model**: Can map device registers to userspace

**macOS Blockers**:
1. **No userspace driver framework**: DriverKit doesn't allow full device control
2. **System Integrity Protection**: Prevents direct hardware access
3. **No kernel extension support**: System extensions are sandboxed
4. **Thunderbolt security**: External devices have limited DMA access

**Implication**: Cannot replicate ssd-gpu-dma approach on macOS/Metal.

**Sources**:
- [ssd-gpu-dma GitHub Repository](https://github.com/enfiskutensykkel/ssd-gpu-dma)
- [ssd-gpu-dma README](https://github.com/enfiskutensykkel/ssd-gpu-dma/blob/master/README.md)

---

## 7. GPU Filesystem Research (GPUfs)

### Overview

**GPUfs** provides POSIX-like file system APIs directly callable from GPU programs, addressing the complexity of traditional GPU I/O patterns.

### Key Challenges Identified

#### 1. Massive Parallelism
- Thousands of threads simultaneously invoking open, close, read, write
- Traditional OS abstractions designed for sequential or modestly parallel access
- Challenge: Efficient synchronization without deadlocks

#### 2. PCIe Bus Atomics
- GPUfs implementation complicated by lack of atomic operations over PCIe
- PCIe-III standard includes atomics but **hardware support is optional**
- **No known hardware currently supports PCIe atomics** (as of GPUfs publication)
- Implication: Cannot use efficient one-sided communication protocols

#### 3. Memory Consistency
- **GPU-CPU memory consistency model** tailored to bulk-synchronous programming
- Traditionally, GPU-CPU communication only at kernel invocation boundaries
- Challenge: RPC protocol must deliver file requests **while kernel is running**
- Requires enforcing consistent updates of CPU-GPU shared memory in both directions

#### 4. Threading and Synchronization
- **Warps run to completion without preemption**
- Spinlocks can cause deadlock (waiting warp blocks execution slot)
- **Threadblocks scheduled non-deterministically**
- Challenge: Implementing reference-count based open/close operations

#### 5. Application Complexity
- Traditional GPU programming requires complex CPU-side code for storage/network
- Turns application development into **low-level systems programming**
- GPUfs aims to abstract this complexity with familiar POSIX APIs

### Relevance to Our Project

**Similar Challenges**:
- We face the same massive parallelism issues (1024 threads in single threadgroup)
- Memory consistency critical for CPU-mediated I/O
- Cannot use spinlocks in Metal (same warp preemption issue)

**GPUfs Solutions We Can Adapt**:
- **RPC-style communication**: CPU daemon servicing GPU I/O requests
- **Software cache**: Reduce I/O requests and amplification
- **Batch operations**: Coalesce multiple thread requests

**Differences**:
- GPUfs targets discrete GPUs (separate memory)
- We have unified memory (simpler consistency model)
- We cannot modify NVMe driver (macOS restriction)

**Sources**:
- [GPUfs: Integrating a File System with GPUs (PDF)](https://iditkeidar.com/wp-content/uploads/files/ftp/GPUfs-TOCS.pdf)
- [GPUfs - ACM Transactions on Computer Systems](https://dl.acm.org/doi/10.1145/2553081)
- [GPUfs - Communications of the ACM](https://cacm.acm.org/research/gpufs/)

---

## 8. GPU Database Systems and Storage DMA

### GMT System (ASPLOS 2024)

**GPU-Orchestrated Memory Tiering**:
- 3-tier hierarchy: GPU memory → Host memory → SSDs
- GPU orchestrates transfers for bandwidth/latency sensitive operations
- Addresses limitation: GPU cannot benefit from host memory as intermediate tier with direct NVMe access

**Key Insight**: Even with GPU-SSD direct access (like GPUDirect), adding host memory as cache improves performance due to lower latency.

### GPU Database Challenges

**Fundamental Limitations**:
1. **Limited GPU memory**: Database size often exceeds GPU VRAM
2. **PCIe bottleneck**: Data transfer between CPU and GPU is slow
3. **Data management complexity**: Requires sophisticated caching and prefetching

**State-of-the-Art Approaches**:
- Software stacks on host CPUs (poor performance)
- Direct SSD access via NVMe queues (lacks host memory benefits)
- Hybrid approaches with tiered caching (best performance)

### Lessons for GPU Filesystem

**Key Takeaways**:
1. **Tiered storage is essential**: GPU memory alone insufficient
2. **Host memory as cache**: Beneficial even with direct GPU-SSD path
3. **Prefetching critical**: Hide latency of slower tiers
4. **Batching reduces overhead**: Coalesce small requests

**Applicability**:
- Our GPU-Native OS will face same challenges at larger scale
- Unified memory simplifies GPU↔host tier (zero-copy)
- Still need efficient SSD tier management
- Consider software cache in shared memory

**Sources**:
- [GMT: GPU Orchestrated Memory Tiering - ASPLOS 2024](https://dl.acm.org/doi/10.1145/3620666.3651353)
- [GPU-Accelerated Database Systems Survey - Springer](https://link.springer.com/chapter/10.1007/978-3-662-45761-0_1)
- [Scaling GPU Databases Beyond GPU Memory - VLDB](https://dl.acm.org/doi/10.14778/3749646.3749710)

---

## 9. Metal and Asynchronous I/O

### Metal's Compute Shader Capabilities

**What Metal Shaders Can Do**:
- Read/write device memory (MTLBuffer, MTLTexture)
- Perform massively parallel computations
- Synchronize within threadgroup (threadgroup_barrier)
- Atomic operations on device memory

**What Metal Shaders Cannot Do**:
- System calls (no file I/O)
- Network operations
- Allocate memory dynamically
- Issue MMIO writes to device registers
- Asynchronous operations (kernel runs to completion)

### Asynchronous I/O Patterns with Metal

**CPU-Side Async I/O** (Required Approach):

**Option 1: Dispatch I/O (GCD)**:
```swift
// CPU thread
let queue = DispatchQueue(label: "io", qos: .userInitiated)
queue.async {
    // Read file into MTLBuffer (shared storage mode)
    let data = try! Data(contentsOf: fileURL)
    data.withUnsafeBytes { ptr in
        memcpy(mtlBuffer.contents(), ptr.baseAddress!, data.count)
    }

    // Signal GPU via MTLCommandBuffer
    dispatchSemaphore.signal()
}

// Metal kernel accesses mtlBuffer directly (shared memory)
```

**Option 2: POSIX AIO**:
```c
// Setup AIO control block
struct aiocb cb;
cb.aio_fildes = fd;
cb.aio_buf = mtlBuffer.contents(); // Shared memory
cb.aio_nbytes = size;
cb.aio_offset = offset;

// Submit async read
aio_read(&cb);

// Poll or wait for completion
while (aio_error(&cb) == EINPROGRESS) { /* wait */ }

// Data now in MTLBuffer, dispatch GPU work
```

**Option 3: io_uring equivalent on macOS**:
- macOS **does not have io_uring**
- Closest equivalent: Dispatch I/O or kqueue + POSIX AIO
- Less efficient than Linux io_uring but viable

### Research: AGILE System

**AGILE** (Lightweight Asynchronous GPU-SSD Integration):
- Enables overlapping computation and I/O at thread level
- Asynchronous data movement from global/pinned memory to shared memory
- **Limitation**: Only async ops GPU→GPU memory, not GPU→SSD

**Key Insight**: Asynchronous I/O models reduce execution time by overlapping compute and communication, but on GPUs, this is limited to memory-to-memory transfers, not device I/O.

### Implications for Our Project

**Pattern to Follow**:
1. CPU async I/O reads → MTLBuffer (shared mode)
2. CPU signals GPU via MTLEvent or semaphore
3. GPU kernel processes data from MTLBuffer
4. GPU signals CPU when done
5. CPU async I/O writes from MTLBuffer → storage

**Advantages**:
- Unified memory eliminates CPU→GPU copy
- Can overlap multiple async I/O operations on CPU
- GPU computation overlaps with I/O on different buffers (double-buffering)

**Limitations**:
- CPU must mediate all I/O (cannot eliminate CPU from path)
- Latency includes CPU scheduling overhead
- Cannot directly issue I/O from Metal kernels

**Sources**:
- [AGILE: Lightweight Asynchronous GPU-SSD Integration](https://arxiv.org/html/2504.19365v3)
- [CUDA Asynchronous Execution Documentation](https://enccs.github.io/OpenACC-CUDA-intermediate/3.02_TaskParallelism/)

---

## 10. Summary: Can We Do GPU-to-Storage DMA on Apple Silicon?

### Direct Answer: **NO** (with current APIs)

**Technical Barriers**:
1. **No Metal APIs for device I/O**: Metal Shading Language cannot issue file operations
2. **No NVMe queue access**: Cannot map NVMe queues to GPU memory
3. **No PCIe BAR access**: Cannot write to device registers from GPU
4. **No userspace drivers**: Cannot create custom NVMe driver like Linux ssd-gpu-dma
5. **Sandboxing**: System Integrity Protection prevents hardware-level access

### What We Have Instead: Unified Memory Architecture

**Advantages Over Discrete GPUs**:
- **No PCIe copy overhead**: Data in shared memory accessible by both CPU and GPU
- **High bandwidth**: 120-546 GB/s depending on M4 variant
- **Coherent access**: No explicit synchronization needed for shared buffers
- **Low latency**: On-die interconnect vs. external PCIe bus

**Realistic Approach**:
```
CPU (async I/O) → Shared Memory (MTLBuffer) ← GPU (compute)
                        ↕
                    NVMe SSD
```

**Performance Characteristics**:
- **I/O latency**: NVMe + CPU scheduling (unavoidable)
- **Data transfer**: Zero-copy if using shared MTLBuffer
- **Bandwidth**: Limited by NVMe speed, not CPU-GPU link
- **Concurrency**: CPU can manage multiple async I/O operations

### Future Possibilities

**What Apple Could Add** (speculative):
1. **Metal I/O APIs**: File operations callable from Metal kernels
2. **GPU-initiated I/O**: Similar to BaM system
3. **Async I/O primitives**: Built into Metal command buffers
4. **Storage tiers in Metal**: Automatic GPU memory → Host memory → SSD management

**Likelihood**: Low in near-term
- Conflicts with sandboxing and security model
- Limited use cases beyond specialized workloads
- Apple's focus is on unified memory, not direct device access

### Recommendations for GPU-Native OS Project

**Design Principles**:
1. **Embrace CPU mediation**: Design for async I/O pattern from start
2. **Leverage unified memory**: Use shared MTLBuffers for all I/O
3. **Optimize CPU path**: Make I/O daemon highly efficient
4. **Software cache**: Reduce I/O requests via caching in shared memory
5. **Prefetching**: Hide latency by predicting access patterns

**Architecture**:
```
┌─────────────────────────────────────────────┐
│          Unified Memory (Shared)            │
│  ┌─────────────┐         ┌──────────────┐  │
│  │ MTLBuffers  │ ← ← ← → │ CPU I/O Daemon│  │
│  │ (GPU access)│         │  (async I/O)  │  │
│  └─────────────┘         └──────────────┘  │
│         ↑                        ↓          │
│    GPU Compute               NVMe SSD       │
│   (Metal kernels)            (storage)      │
└─────────────────────────────────────────────┘
```

**Performance Optimization**:
- Use multiple CPU threads for I/O (parallel reads/writes)
- Double/triple buffering (GPU processes buffer N while CPU loads buffer N+1)
- Batch operations (coalesce small I/O requests)
- Prioritize I/O (use QoS classes in GCD)

**Prototype Strategy**:
1. **Phase 1**: Simple CPU-mediated I/O to shared MTLBuffers
2. **Phase 2**: Async I/O with GCD/Dispatch I/O
3. **Phase 3**: Software cache and prefetching
4. **Phase 4**: Measure and optimize for real workloads

---

## 11. Relevant Research Papers

### GPU Storage Systems

1. **BaM: GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture**
   - Authors: Zaid Qureshi et al.
   - Venue: ASPLOS 2023
   - URL: https://arxiv.org/abs/2203.04910
   - Key Contribution: GPU threads directly access NVMe storage

2. **GMT: GPU Orchestrated Memory Tiering for the Big Data Era**
   - Venue: ASPLOS 2024
   - URL: https://dl.acm.org/doi/10.1145/3620666.3651353
   - Key Contribution: 3-tier memory hierarchy with GPU orchestration

3. **AGILE: Lightweight and Efficient Asynchronous GPU-SSD Integration**
   - URL: https://arxiv.org/html/2504.19365v3
   - Key Contribution: Asynchronous I/O overlapping with GPU compute

4. **GPUfs: Integrating a File System with GPUs**
   - Authors: Mark Silberstein et al.
   - Venue: ACM TOCS
   - URL: https://dl.acm.org/doi/10.1145/2553081
   - Key Contribution: POSIX-like filesystem APIs for GPU programs

### GPU Database Systems

5. **Scaling GPU-Accelerated Databases Beyond GPU Memory Size**
   - Venue: VLDB Endowment 2025
   - URL: https://dl.acm.org/doi/10.14778/3749646.3749710
   - Key Contribution: Memory management for large databases on GPUs

6. **GPU-Accelerated Database Systems: Survey and Open Challenges**
   - Venue: Springer
   - URL: https://link.springer.com/chapter/10.1007/978-3-662-45761-0_1
   - Key Contribution: Comprehensive survey of GPU database architectures

### Apple Silicon Architecture

7. **Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency**
   - Venue: arXiv 2025
   - URL: https://arxiv.org/html/2502.05317v1
   - Key Contribution: Detailed M-series architectural analysis

---

## 12. Key Takeaways for Implementation

### What We Learned

1. **Direct GPU-NVMe DMA is proven technology** (NVIDIA GDS, BaM)
2. **Apple Silicon does not support it** (no Metal APIs, no userspace drivers)
3. **Unified memory is a strong alternative** (zero-copy CPU-GPU sharing)
4. **CPU mediation is unavoidable** on macOS/Metal
5. **Software techniques can mitigate latency** (caching, prefetching, async I/O)

### Critical Questions Answered

**Q: Can Metal compute shaders issue I/O operations?**
A: **No**. Metal shaders cannot perform file I/O, network I/O, or any system calls.

**Q: Can NVMe controllers DMA directly to GPU memory on Apple Silicon?**
A: **Unknown**, but even if technically possible at hardware level, there are **no APIs to enable it** from userspace.

**Q: Does Apple Silicon support anything like GPUDirect Storage?**
A: **No**. Unified memory architecture is Apple's alternative approach.

**Q: Can we use IOKit to map NVMe buffers to GPU memory?**
A: **No**. IOKit doesn't provide APIs for this, and macOS security model prevents it.

### Next Steps for GPU-Native OS Project

1. **Accept CPU mediation**: Design architecture assuming all I/O is CPU-mediated
2. **Optimize the CPU path**: Make I/O daemon as efficient as possible
3. **Prototype with shared MTLBuffers**: Validate zero-copy approach
4. **Implement software cache**: Reduce I/O requests and amplification
5. **Measure real performance**: Determine if CPU-mediated I/O is acceptable for use cases

### Recommended Architecture

```rust
// High-level architecture for GPU-Native OS filesystem

// CPU Thread: I/O Daemon
struct IODaemon {
    // Shared memory ring buffer for I/O requests
    request_queue: MTLBuffer,  // GPU writes requests
    response_queue: MTLBuffer, // CPU writes responses

    // Data buffers in shared memory
    io_buffers: Vec<MTLBuffer>, // Shared storage mode

    // Async I/O management
    io_queue: DispatchQueue,
    pending_ops: HashMap<RequestID, Operation>,
}

// GPU Kernel: Filesystem Operations
kernel void fs_read(
    device IORequest* requests [[buffer(0)]],  // Request queue
    device IOResponse* responses [[buffer(1)]], // Response queue
    device uint8_t* data_buffer [[buffer(2)]],  // Shared data buffer
    uint tid [[thread_position_in_grid]]
) {
    // 1. GPU thread writes read request
    requests[tid] = IORequest {
        .type = READ,
        .path_hash = hash_path(path),
        .offset = offset,
        .length = length,
        .buffer_index = tid,
    };

    // 2. Signal CPU (via atomic flag or MTLEvent)
    atomic_store_explicit(&requests[tid].ready, 1, memory_order_release);

    // 3. Poll for completion (or use MTLEvent for wait)
    while (atomic_load_explicit(&responses[tid].ready, memory_order_acquire) == 0) {
        // Spin or yield (challenge: spinlock can deadlock)
    }

    // 4. Data now available in data_buffer (shared memory)
    process_data(data_buffer + responses[tid].buffer_offset);
}
```

**Challenges to Solve**:
- **Busy-waiting is inefficient**: Need MTLEvent-based signaling
- **Spinlock deadlock**: Cannot have GPU threads spin-wait
- **Request coalescing**: Many threads accessing same file
- **Cache coherence**: Ensure CPU writes visible to GPU

---

## Conclusion

While **true GPU-to-storage DMA is not available on Apple Silicon/Metal**, the **unified memory architecture provides a viable alternative**. By accepting CPU mediation for I/O operations and leveraging shared memory buffers, we can build a GPU-Native filesystem with reasonable performance characteristics.

The key is to **optimize the CPU path** rather than attempt to bypass it entirely. Research from GPUfs, BaM, and GPU database systems provides valuable patterns for managing massive parallelism, caching, and asynchronous I/O that we can adapt to Metal's architecture.

**Final Recommendation**: Proceed with CPU-mediated I/O architecture, prototype with shared MTLBuffers, and measure performance. If I/O becomes a bottleneck, focus optimization on:
1. Software caching in shared memory
2. Prefetching and access pattern prediction
3. Parallel CPU I/O threads
4. Request coalescing and batching

This approach is **pragmatic, implementable with current Metal APIs**, and leverages Apple Silicon's strengths rather than fighting its limitations.
