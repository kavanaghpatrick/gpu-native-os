# GPU-Native Filesystem: Parallel Algorithms Research

Research compilation for proven GPU-parallel algorithms applicable to filesystem operations.
Date: 2026-01-24

---

## 1. String Matching on GPU

### Proven Implementations

**CUSMART (2024-2025)**
- Library with 64 parallelized string matching algorithms
- Performance: 31x to 106x speedup vs sequential CPU
- DRK method: Up to 64 GB/s on 8-character patterns (NVIDIA Tesla K40c)
- Source: [CUSMART Paper](https://link.springer.com/article/10.1631/FITEE.2400091)

**Parallel Approaches**
- **CRK (Cooperative)**: Best for large patterns (>8K chars)
- **DRK (Divide-and-conquer)**: Best for short patterns
- **HRK (Hybrid)**: Combines both approaches
- Source: [Parallel String Matching on GPU](https://escholarship.org/uc/item/2d46g741)

**Aho-Corasick on GPU**
- 8.5-9.5x speedup vs single-thread CPU
- Superior to Boyer-Moore GPU implementations
- Source: [Multipattern String Matching on GPU](https://www.cise.ufl.edu/~sahni/papers/multipatternGPU.pdf)

**Boyer-Moore on GPU**
- Enhanced BM: 10x faster than CPU, 9x faster than multithreaded
- Parallel BM: Up to 45x speedup at maximum
- Sources: [Enhanced Boyer-Moore](https://www.academia.edu/75867665/Accelerating_Enhanced_Boyer_Moore_String_Matching_Algorithm_on_Multicore_GPU_for_Network_Security), [Parallelized Boyer-Moore](https://www.researchgate.net/publication/273912474_Parallelization_and_Performance_Optimization_of_the_Boyer-Moore_Algorithm_on_GPU)

### Regex/Glob Pattern Matching

**HybridSA (2024)**
- 4-60x higher throughput than state-of-art CPU engines
- 4-233x faster than state-of-art GPU engines
- Uses bit parallelism to simulate NFAs on GPU
- Source: [HybridSA Paper](https://dl.acm.org/doi/10.1145/3689771)

**QuickMatch**
- 7x speedup over egrep
- 30x speedup over optimized sequential implementation
- NVIDIA GTX1080 GPU, CUDA
- Source: [QuickMatch Project](https://madhumithasridhara.github.io/QuickMatch/)

**CUDA grep**
- 2-10x faster than grep (depending on workload)
- 68x faster than Perl regex engine
- ~9x speedup when matching large numbers of regexes
- Source: [CUDA grep](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s12/www/competition/bkase.github.com/CUDA-grep/finalreport.html)

**Gregex**
- 48x speedup over traditional CPU implementations
- Up to 16 Gbit/s processing throughput
- Source: [Gregex Paper](https://www.researchgate.net/publication/221027312_Gregex_GPU_Based_High_Speed_Regular_Expression_Matching_Engine)

### Key Challenges
- Irregular memory access patterns reduce efficiency
- Performance bounded by global memory bandwidth
- Poor locality (each line accessed once per regex)
- Source: [CUDA grep challenges](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s12/www/competition/bkase.github.com/CUDA-grep/finalreport.html)

### Filesystem Application
- Fast file search by content (grep-like operations)
- Glob pattern matching for file paths
- Metadata filtering (file type, extension matching)

---

## 2. Parallel Sorting on GPU

### Algorithm Performance Comparison

**Radix Sort** (WINNER for most cases)
- 2-4x faster than other algorithms at large inputs
- O(n) work complexity vs O(n log² n) for bitonic
- Best for: Large datasets (>100K elements), integers/floats
- GPU mergesort: ~100x faster than std::stable_sort on i7 Sandy Bridge
- Sources: [Comparison Study](https://arxiv.org/pdf/1511.03404), [Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)

**Bitonic Sort**
- O(n log² n) work - less efficient than radix
- Dozens of passes vs 4-8 for radix
- Only competitive for small arrays (<100K elements)
- Advantage: In-place, uses half the memory
- Best for: Segmented sorting with small segments
- Sources: [Linebender Sorting Wiki](https://linebender.org/wiki/gpu/sorting/), [Comparison Study](https://arxiv.org/pdf/1511.03404)

**Merge Sort**
- ~50% throughput of radix on large arrays (32-bit keys)
- Advantage: User-defined comparator (supports strings via strcmp)
- Still ~100x faster than CPU std::stable_sort
- Better than radix for large key sizes
- Source: [Modern GPU Mergesort](https://moderngpu.github.io/mergesort.html)

### String vs Integer Performance

**Integers**: Significantly faster
- Can use radix sort (optimal O(n) work)
- Lexicographical order matches integer order
- Source: [Modern GPU Mergesort](https://moderngpu.github.io/mergesort.html)

**Strings**: Require comparison-based sorting
- Must use merge sort with strcmp comparator
- ~50% slower than integer radix sort
- Still massively parallel and faster than CPU
- Source: [Modern GPU Mergesort](https://moderngpu.github.io/mergesort.html)

### Libraries and Benchmarks

**CUB/Thrust vs Hybrid Radix**
- Hybrid radix outperforms CUB for >1.9M keys
- Hybrid superior to Thrust and MGPU merge sort
- Radix: >2x faster than alternatives, >4x GPUSort
- Source: [Memory-Efficient Hybrid Radix](https://arxiv.org/pdf/1611.01137)

**Metal Performance** (limited data)
- ~3G elements/s using subgroup ballot operations
- Some speedup from vanilla wgpu inefficiency
- Source: [Linebender Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)

### Filesystem Application
- Directory listing (sort by name, size, date)
- File extent sorting for defragmentation
- Inode table organization
- Metadata index building

---

## 3. Tree Traversal on GPU

### BFS/DFS Performance

**NVIDIA BFS (Optimal)**
- O(|V| + |E|) work complexity (asymptotically optimal)
- 3.3 billion edges/sec (single GPU)
- 8.3 billion edges/sec (quad-GPU)
- Fine-grained task management + prefix sum
- Source: [High Performance GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-and/BFS%20TR.pdf)

**Hybrid BFS-DFS (HBDFS)**
- Uses precomputed critical frontiers
- Asynchronous between BFS and DFS
- Dynamically selects best method per iteration
- Prevents poor worst-case performance
- Sources: [GPU-Parallel Hybrid Algorithm](https://ieeexplore.ieee.org/document/8334786/), [Hybrid BFS-DFS](https://www.computer.org/csdl/proceedings-article/sitis/2017/4283a450/12OmNyVerZ4)

**Implementation Challenges**
- Irregular memory access patterns
- Data-dependent work distribution
- Barrier synchronization needed between levels
- Load balancing when work varies per thread
- Source: [Parallel BFS on GPU](https://medium.com/@chamudi_sashanka/exploring-parallel-graph-traversal-bfs-dfs-with-openmp-mpi-and-cuda-dbb906979ac9)

### Lock-Free Tree Structures

**Key Findings**
- Lock-based implementations: Poor scalability on GPU
- High lock contention with thousands of threads
- No coherent caches = expensive global memory for lock checks
- ANY locking (even atomic) is detrimental to GPU performance
- Source: [Lock-free Data Structures on GPU](https://www.cse.iitk.ac.in/users/mainakc/pub/icpads2012.pdf)

**Tree Compression Techniques**
- Cleary-Cuckoo hashing enables Cleary compression on GPU
- Fine-grained parallel compression algorithms
- "Braided" B+ tree: Additional pointers for lock-free traversal
- Source: [GPU Tree Database](https://link.springer.com/chapter/10.1007/978-3-031-30823-9_35)

**B-Tree Implementations**
- Fine-grain locks with clever contention reduction
- Support concurrent queries (point, range, successor)
- Support concurrent updates (insert, delete)
- High update throughput despite locks
- Work-stealing deques for load balancing
- Source: [Lock-free Data Structures on GPU](https://www.cse.iitk.ac.in/users/mainakc/pub/icpads2012.pdf)

### Filesystem Application
- Directory tree traversal (BFS for listing)
- Path resolution (DFS for lookups)
- B-tree indexes for metadata
- Extent tree navigation

---

## 4. Hash Tables on GPU

### State-of-the-Art (2024-2025)

**Hive Hash Table (2024-2025)**
- Load factors up to 95%
- 1.5-2x higher throughput than SlabHash/DyCuckoo/WarpCore
- 3.5 billion updates/s
- Nearly 4 billion lookups/s
- Up to 3,853 MOPS (highest among all competitors)
- Warp-cooperative, dynamically resizable
- Source: [Hive Hash Table](https://arxiv.org/pdf/2510.15095)

**WarpSpeed Library (2024-2025)**
- IcebergHT, P2HT, DoubleHT: 21%+ faster insertions
- 2.4-3.8x gains over bucketed cuckoo baseline (BGHT)
- Source: [WarpSpeed Paper](https://arxiv.org/pdf/2509.16407)

**Compact Parallel Hash Tables (Euro-Par 2024)**
- Compactness improves lookups/insertions 10-20%
- Bucketed cuckoo hashing benefits most
- Source: [Compact Parallel Hash Tables](https://link.springer.com/chapter/10.1007/978-3-031-69766-1_16)

### Cuckoo Hashing

**Bucketed Cuckoo**
- 3 hash functions, 16-element buckets
- 1.43 probes per insertion average
- Load factors up to 0.99
- Best-of-class for insertions and queries
- Source: [WarpSpeed Paper](https://arxiv.org/pdf/2509.16407)

**DyCuckoo (2025)**
- Dynamic cuckoo hash table on GPU
- Fine-grained memory control
- Superior efficiency
- Source: [DyCuckoo Paper](https://www.semanticscholar.org/paper/DyCuckoo:-Dynamic-Hash-Tables-on-GPUs-Li-Zhu/cdb591b30610a12c8df0bcfb7549da42abfdac79)

### Perfect Hashing (Static Data)

**GPH (GPU Perfect Hashing)**
- Exactly 1 bucket probe per lookup (guaranteed)
- No collisions for static key sets
- Source: [GPH Paper](https://dl.acm.org/doi/10.1145/3725406)

**BGHT (Bucketed GPU Hash Tables)**
- 1.43 average probe count at 0.99 load factor
- Optimized for static GPU data
- Source: [BGHT GitHub](https://github.com/owensgroup/BGHT)

**Perfect Spatial Hashing**
- Exactly 2 memory accesses per query
- Precomputed for static data
- Ideally suited for parallel SIMD on GPU
- 1.44-2.5 bits per key in practice
- Source: [Perfect Spatial Hashing](https://hhoppe.com/perfecthash.pdf)

### WarpCore vs SlabHash

**WarpCore Performance**
- Up to 1.6 billion inserts/s (GV100)
- Up to 4.3 billion retrievals/s (GV100)
- 8.76x faster than cuDF at ρ=0.8
- 1.64x faster than cuDPP at ρ=0.9
- 2.39x faster than SlabHash at ρ=0.9
- Superior cooperative probing at high load (>70%)
- Most pronounced advantage at >90% load
- Sources: [WarpCore Paper](https://arxiv.org/pdf/2009.07914), [WarpCore IEEE](https://ieeexplore.ieee.org/document/9406635/)

**SlabHash Limitations**
- Pointer-chasing overhead across linked slabs
- Linked-list allocator overhead
- Irregular global memory access patterns
- On par with WarpCore at low densities only
- Source: [WarpCore Paper](https://arxiv.org/pdf/2009.07914)

### Filesystem Application
- Filename to inode lookup
- Directory entry hash tables
- Deduplication hash indexes
- Block hash maps for extent allocation

---

## 5. Compression/Encryption on GPU

### LZ4 Compression

**NVIDIA nvCOMP**
- Official CUDA LZ4 implementation
- 36 GB/s compression (RTX 3090)
- 118 GB/s decompression (RTX 3090)
- vs CPU: ~5 GB/s single core
- Data divided into concurrent blocks
- Block size impacts performance and ratio
- Sources: [nvCOMP GitHub](https://github.com/NVIDIA/nvcomp), [nvCOMP Docs](https://docs.nvidia.com/cuda/nvcomp/lz4.html)

**Recent CPU Improvements** (context for comparison)
- LZ4 1.10 multi-threading: 5.4-8.0x faster
- Decompression: ~60% faster with overlapping
- Source: [LZ4 1.10 Release](https://www.phoronix.com/news/LZ4-1.10-Multi-Threading)

### Zstd Compression

**nvCOMP Zstd Performance**
- Up to 600 GB/s decompression (NVIDIA Blackwell)
- 28% speedup on B40 for large datasets
- 2x faster on small batches
- 40% faster on large batches
- 2-5x less scratch memory required
- 1.2x faster decompression in recent releases
- Chunk size sweet spot: 256KB-16MB
- Sources: [nvCOMP GitHub](https://github.com/NVIDIA/nvcomp), [nvCOMP Release Notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html)

**Algorithm Characteristics**
- Focus on good compression ratios
- Very good decompression performance
- Degradation starts at 256KB chunks
- >16MB chunks not supported
- Source: [nvCOMP Docs](https://docs.nvidia.com/cuda/nvcomp/lz4.html)

### AES Encryption

**CUDA Implementation Benchmarks**

**Peak Performance**
- 605.9 Gbps (NVIDIA Tesla P100-PCIe, bitsliced AES-ECB)
- 580 GB/s (NVIDIA Tesla A100)
- 280 Gbps (NVIDIA GTX 1080 Pascal)
- 214 Gbps (GTX 1080)
- 207 Gbps (GTX TITAN X Maxwell)
- 123 Gbps (GTX 970)
- Sources: [Bitsliced AES on GPU](https://link.springer.com/chapter/10.1007/978-3-319-64701-2_20), [GPU AES Optimization](https://eprint.iacr.org/2021/646.pdf), [GPU AES Implementation](https://arxiv.org/pdf/1902.05234)

**Additional Benchmarks**
- 60 Gbps (NVIDIA Tesla C2050) - 50x faster than Intel i7-920
- 35 Gbps (NVIDIA GeForce GTX285, 16Byte/thread)
- Source: [AES on CUDA GPU](https://www.comp.hkbu.edu.hk/~chxw/papers/AES_2012.pdf)

**Recent Acceleration (A-series/H100)**
- AES-GCM: 1.6x (A-series), 2.6x (H100)
- AES-ECB: 5x (A-series), 10.8x (H100)
- AES-XTS: 2.6x (A-series), 3.5x (H100)
- AES-CTR: 3x (A-series), 5.3x (H100)
- Source: [WolfSSL CUDA AES](https://www.wolfssl.com/accelerating-aes-encryption-with-nvidia-cuda-wolfcrypt-performance-boost/)

**Metal Implementation**
- No specific benchmarks found in research
- Sources focused on CUDA implementations

### Parallel Decompression

**nvCOMP Features**
- Multiple chunks processed in parallel
- Low-level GPU interfaces for concurrency
- Heuristic vs exhaustive kernel config selection
- Heuristic faster for most cases
- Exhaustive worthwhile for repeated similar data
- Source: [nvCOMP GitHub](https://github.com/NVIDIA/nvcomp)

### Filesystem Application
- Transparent file compression (LZ4 for speed)
- Archive compression (Zstd for ratio)
- Encrypted volumes (AES-XTS, AES-CTR)
- Deduplication with compression

---

## 6. Checksums and Integrity

### CRC32 on GPU

**NVIDIA nvCOMP CRC32**
- Generic CRC32 checksum calculation
- Multiple kernel configurations for performance tuning
- Heuristic function for quick config selection
- Exhaustive search for optimal config (slower)
- 2x average speedup vs sequential CPU
- Multiple chunks processed in parallel
- Sources: [nvCOMP CRC32 Docs](https://docs.nvidia.com/cuda/nvcomp/crc32.html), [Parallel CRC32](https://www.researchgate.net/publication/287192702_Design_and_implementation_of_the_CRC-32_checksum_of_parallel_algorithm)

**Implementation Considerations**
- Performance depends on data characteristics
- Kernel config optimization is critical
- Heuristic sufficient for most workloads
- Exhaustive search for batch processing
- Source: [nvCOMP CRC32 Docs](https://docs.nvidia.com/cuda/nvcomp/crc32.html)

### SHA-256 on GPU

**CUDA Performance Benchmarks**

**High-End GPUs**
- 9,531.7 MH/s (NVIDIA RTX 3090, Hashcat, CUDA 11.4)
- Source: [Hashcat RTX 3090 Benchmark](https://gist.github.com/Chick3nman/e4fcee00cb6d82874dace72106d73fef)

**Tesla V100**
- Improved throughput with T=13 launch config
- 80 SMs, 64 threads per SM
- PARSHA-256 beats Intel Xeon Gold 6140
- Source: [Fast Hashing in CUDA](https://nevillewalo.ch/assets/docs/FastHashingInCuda.pdf)

**Historical GPU Comparison**
- AMD Radeon 6970: 323 MH/s
- NVIDIA GTX 570: 105 MH/s
- AMD 3x faster (at that time)
- Source: [AMD vs NVIDIA SHA-256](https://forums.developer.nvidia.com/t/amd-radeon-3x-faster-on-bitcoin-mining-sha-256-hashing-performance/23036)

**Metal Implementation**
- No specific benchmarks found in research
- CUDA implementations dominate literature

### Parallel Integrity Verification

**Scalable GPU-Based Verification (ML Models)**

**GPU Advantages**
- Thousands of parallel compute units for hashing
- Order-of-magnitude improvements vs sequential CPU
- Memory bandwidth: ~1TB/s (GPU) vs ~100GB/s (CPU)
- Can keep pace with model execution (>100GB models)
- Sources: [Scalable GPU Integrity Verification](https://arxiv.org/html/2510.23938)

**Implementation Features**
- GPU-native hash computation (optimized kernels)
- Traditional: SHA-256, SHA-384
- Post-quantum alternatives supported
- Parallel Merkle tree construction
- Real-time verification of multi-GB model shards
- Batch and streaming verification modes
- Source: [Scalable GPU Integrity Verification](https://arxiv.org/html/2510.23938)

**GPU Checkpoint/Restore**
- CRC checksum on GPU buffers
- Calculated before executing kernels
- Overlapped with kernel execution
- Overlapped with checkpoint I/O
- Minimizes overhead
- Source: [ParallelGPUOS](https://arxiv.org/html/2405.12079v1)

**Execution Integrity**
- Vital for GPU kernel correctness and security
- Preserves kernel code integrity
- Guarantees execution flow integrity
- Prevents malicious interference
- Source: [ShadowScope GPU Monitoring](https://arxiv.org/html/2509.00300v2)

### Filesystem Application
- File integrity verification (CRC32, SHA-256)
- Scrubbing for bit rot detection
- Parallel checksum calculation on ingest
- Merkle trees for hierarchical verification
- Real-time integrity checks during read

---

## 7. GPU Filesystem Implementations

### GPUfs (Research System)

**Architecture**
- Thousands of threads invoke open/close/read/write simultaneously
- Addresses massive parallelism, slow sequential execution
- NUMA memory organization
- Source: [GPUfs](https://cacm.acm.org/research/gpufs/)

**Performance**
- 7x faster than 8-core CPU on Linux kernel source
- ~33,000 small files
- Source: [GPUfs](https://cacm.acm.org/research/gpufs/)

**Key Insight**: GPU filesystem must handle concurrent operations from thousands of threads, not just 8-16 cores.

### GPU-Accelerated Metadata Management

**BeeGFS Prototype**
- CPU interacts with filesystem clients
- GPU handles metadata operations in parallel
- >50% faster than CPU-based scheme
- Sources: [GPU Metadata Management](https://jcst.ict.ac.cn/fileup/1000-9000/PDF/2021-1-4-0783.pdf), [GPU Metadata Paper](https://link.springer.com/article/10.1007/s11390-020-0783-9)

**Performance Under Concurrency**
- Superiority strengthens at high concurrency
- HPC systems: millions of parallel threads
- All metadata in GPU memory (no I/O)
- Massive parallel threads for compute power
- Source: [GPU Metadata Management](https://jcst.ict.ac.cn/fileup/1000-9000/PDF/2021-1-4-0783.pdf)

### GPU B-Tree for Filesystems

**Concurrent B+Tree Performance**

**Eirene Framework (2023)**
- Combining-based concurrency control
- Minimizes conflict detection/resolution overhead
- 5% variance (vs 40% STM, 36% Lock)
- Better QoS under load
- Source: [Boosting GPU B+Trees](https://cs.tulane.edu/~lpeng3/papers/PPoPP-23.pdf)

**Key Challenges**
- Conflict detection/resolution complicates logic
- Increases memory accesses
- Execution path divergence
- Performance degradation
- Increased response time variance
- Source: [High Performance GPU B-Tree](https://dl.acm.org/doi/10.1145/3503221.3508419)

**High-Performance GPU B-Tree**
- Concurrent queries: point, range, successor
- Concurrent updates: insert, delete
- Outperforms GPU LSM tree
- Outperforms GPU sorted array
- Source: [Engineering GPU B-Tree](https://dl.acm.org/doi/pdf/10.1145/3293883.3295706)

**Metadata Application**
- >50% faster than CPU for metadata ops
- Scales to millions of concurrent threads
- Source: [GPU Metadata Management](https://jcst.ict.ac.cn/fileup/1000-9000/PDF/2021-1-4-0783.pdf)

---

## 8. Key Takeaways for GPU-Native Filesystem

### What Works Best on GPU

1. **Massive Data Parallelism**
   - Hash tables: 4 billion ops/s (Hive)
   - BFS traversal: 8.3 billion edges/s (quad-GPU)
   - String matching: 64 GB/s (DRK method)
   - Decompression: 600 GB/s (nvCOMP Zstd)

2. **Simple, Regular Access Patterns**
   - Radix sort beats comparison-based sort
   - Perfect hashing beats chained hashing
   - Lock-free beats any locking

3. **Static/Read-Heavy Workloads**
   - Perfect hashing: 1 probe guaranteed
   - Compressed lookup tables
   - Precomputed indexes

### What Struggles on GPU

1. **Lock-Based Concurrency**
   - ANY locking is detrimental
   - Even fine-grained atomic operations
   - No coherent caches = expensive lock checks

2. **Irregular Memory Access**
   - Pointer chasing (SlabHash problem)
   - Tree traversal with dependencies
   - Small random accesses

3. **Sequential Dependencies**
   - Barrier synchronization between BFS levels
   - Path-dependent operations
   - Non-parallelizable workflows

### Recommended Algorithms

| Operation | Algorithm | Throughput | Notes |
|-----------|-----------|------------|-------|
| File search | HybridSA regex | 60x CPU | Bit parallelism NFA |
| Path matching | Aho-Corasick | 9x CPU | Multipattern superior |
| Directory listing | Radix sort | 2-4x others | O(n) work, integers |
| Filename sort | Merge sort | 100x CPU | Supports strings |
| Tree traversal | Hybrid BFS-DFS | 8.3B edges/s | Adaptive method |
| Filename lookup | Hive/WarpCore | 4B lookups/s | Cuckoo hashing |
| Static indexes | Perfect hash | 1 probe | No collisions |
| B-tree index | Eirene framework | 5% variance | Combining-based |
| Compression | nvCOMP LZ4 | 118 GB/s decomp | Block parallelism |
| Archive | nvCOMP Zstd | 600 GB/s decomp | Best ratio |
| Encryption | AES-XTS | 580 GB/s | Volume encryption |
| Checksums | CRC32 | 2x CPU | nvCOMP tunable |
| Integrity | SHA-256 | 9.5 GH/s | Merkle trees |

### Architecture Implications

**Memory Organization**
- Keep hot metadata in GPU memory
- Avoid CPU-GPU transfers
- Use perfect hashing for static data
- Compress cold data

**Concurrency Model**
- Lock-free data structures only
- Combining-based conflict resolution
- Cooperative warp operations
- Avoid barriers when possible

**Access Patterns**
- Coalesced memory access critical
- Batch operations together
- Minimize pointer chasing
- Precompute complex lookups

**Workload Characteristics**
- Async for CPU-GPU coordination
- Thousands of concurrent threads
- Read-heavy with batch updates
- Static data + dynamic overlays

---

## 9. References

### String Matching
- [CUSMART Paper](https://link.springer.com/article/10.1631/FITEE.2400091)
- [Parallel String Matching on GPU](https://escholarship.org/uc/item/2d46g741)
- [Multipattern String Matching on GPU](https://www.cise.ufl.edu/~sahni/papers/multipatternGPU.pdf)
- [HybridSA Paper](https://dl.acm.org/doi/10.1145/3689771)
- [QuickMatch Project](https://madhumithasridhara.github.io/QuickMatch/)
- [CUDA grep](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s12/www/competition/bkase.github.com/CUDA-grep/finalreport.html)
- [Gregex Paper](https://www.researchgate.net/publication/221027312_Gregex_GPU_Based_High_Speed_Regular_Expression_Matching_Engine)

### Sorting
- [Comparison Study](https://arxiv.org/pdf/1511.03404)
- [Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)
- [Linebender Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)
- [Modern GPU Mergesort](https://moderngpu.github.io/mergesort.html)
- [Memory-Efficient Hybrid Radix](https://arxiv.org/pdf/1611.01137)

### Tree Traversal
- [High Performance GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-and/BFS%20TR.pdf)
- [GPU-Parallel Hybrid Algorithm](https://ieeexplore.ieee.org/document/8334786/)
- [Lock-free Data Structures on GPU](https://www.cse.iitk.ac.in/users/mainakc/pub/icpads2012.pdf)
- [GPU Tree Database](https://link.springer.com/chapter/10.1007/978-3-031-30823-9_35)

### Hash Tables
- [Hive Hash Table](https://arxiv.org/pdf/2510.15095)
- [WarpSpeed Paper](https://arxiv.org/pdf/2509.16407)
- [WarpCore Paper](https://arxiv.org/pdf/2009.07914)
- [GPH Paper](https://dl.acm.org/doi/10.1145/3725406)
- [BGHT GitHub](https://github.com/owensgroup/BGHT)
- [Perfect Spatial Hashing](https://hhoppe.com/perfecthash.pdf)

### Compression/Encryption
- [nvCOMP GitHub](https://github.com/NVIDIA/nvcomp)
- [nvCOMP Documentation](https://docs.nvidia.com/cuda/nvcomp/)
- [GPU AES Optimization](https://eprint.iacr.org/2021/646.pdf)
- [Bitsliced AES on GPU](https://link.springer.com/chapter/10.1007/978-3-319-64701-2_20)
- [WolfSSL CUDA AES](https://www.wolfssl.com/accelerating-aes-encryption-with-nvidia-cuda-wolfcrypt-performance-boost/)

### Checksums/Integrity
- [nvCOMP CRC32 Docs](https://docs.nvidia.com/cuda/nvcomp/crc32.html)
- [Hashcat RTX 3090 Benchmark](https://gist.github.com/Chick3nman/e4fcee00cb6d82874dace72106d73fef)
- [Scalable GPU Integrity Verification](https://arxiv.org/html/2510.23938)
- [ParallelGPUOS](https://arxiv.org/html/2405.12079v1)

### GPU Filesystems
- [GPUfs](https://cacm.acm.org/research/gpufs/)
- [GPU Metadata Management](https://jcst.ict.ac.cn/fileup/1000-9000/PDF/2021-1-4-0783.pdf)
- [Boosting GPU B+Trees](https://cs.tulane.edu/~lpeng3/papers/PPoPP-23.pdf)
- [Engineering GPU B-Tree](https://dl.acm.org/doi/pdf/10.1145/3293883.3295706)

### Metal/Apple GPU
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Learn Metal Shader Best Practices](https://developer.apple.com/videos/play/tech-talks/111373/)
