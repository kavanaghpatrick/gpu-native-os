# Complete std Coverage Plan

## THE GPU IS THE COMPUTER

**Goal: Unmodified Rust code runs on GPU. Period.**

User writes:
```rust
use std::collections::HashMap;
use std::fs;
use std::io::Write;

fn main() {
    let data = fs::read_to_string("input.txt").unwrap();
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in data.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    println!("Found {} unique chars", counts.len());
}
```

**This MUST compile and run on GPU.**

---

## Implementation Strategy

For each std item:
- **NATIVE**: Works as-is (re-export from core/alloc)
- **GPU_IMPL**: We provide GPU implementation
- **SHIM**: Thin wrapper that calls GPU equivalent
- **AUTO_TRANSFORM**: Automatically transformed at WASM→GPU stage (user code unchanged)
- **ERROR**: Compile error with helpful message + alternative (only for truly impossible operations)

---

## std::alloc

| Item | Strategy | Notes |
|------|----------|-------|
| `GlobalAlloc` trait | GPU_IMPL | Route to gpu_heap |
| `Layout` | NATIVE | Pure data structure |
| `LayoutError` | NATIVE | Pure error type |
| `alloc()` | GPU_IMPL | `gpu_heap::alloc()` |
| `alloc_zeroed()` | GPU_IMPL | `gpu_heap::alloc_zeroed()` |
| `dealloc()` | GPU_IMPL | `gpu_heap::dealloc()` |
| `realloc()` | GPU_IMPL | `gpu_heap::realloc()` |
| `handle_alloc_error()` | GPU_IMPL | Write to debug buffer, halt |
| `set_alloc_error_hook()` | AUTO_TRANSFORM | Store function pointer in GPU global, call on OOM |

**Implementation:**
```rust
// gpu_std/src/alloc.rs
#[global_allocator]
static GPU_ALLOCATOR: GpuHeapAllocator = GpuHeapAllocator;

struct GpuHeapAllocator;

unsafe impl GlobalAlloc for GpuHeapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        gpu_intrinsic_alloc(layout.size(), layout.align())
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        gpu_intrinsic_dealloc(ptr, layout.size(), layout.align())
    }
    // ...
}
```

---

## std::any

| Item | Strategy | Notes |
|------|----------|-------|
| `Any` trait | AUTO_TRANSFORM | Compile-time TypeId registry in GPU buffer |
| `TypeId` | NATIVE | Compile-time type ID |
| `type_name()` | NATIVE | Compile-time string |
| `type_name_of_val()` | NATIVE | Compile-time string |

---

## std::array

| Item | Strategy | Notes |
|------|----------|-------|
| `[T; N]` | NATIVE | Fixed-size arrays work perfectly |
| `from_fn()` | NATIVE | Compile-time generation |
| `from_ref()` | NATIVE | Reference conversion |
| `from_mut()` | NATIVE | Reference conversion |
| `IntoIter` | GPU_IMPL | Parallel iteration |
| `TryFromSliceError` | NATIVE | Error type |

---

## std::ascii

| Item | Strategy | Notes |
|------|----------|-------|
| `escape_default()` | NATIVE | Pure function |
| `AsciiChar` | NATIVE | u8 wrapper |
| All `is_*` methods | NATIVE | Pure comparisons |
| All `to_*` methods | NATIVE | Pure conversions |

---

## std::backtrace

| Item | Strategy | Notes |
|------|----------|-------|
| `Backtrace` | AUTO_TRANSFORM | Record call sites in debug buffer |
| `BacktraceStatus` | AUTO_TRANSFORM | Simple enum in debug buffer |

---

## std::borrow

| Item | Strategy | Notes |
|------|----------|-------|
| `Borrow` trait | NATIVE | Pure trait |
| `BorrowMut` trait | NATIVE | Pure trait |
| `Cow` | GPU_IMPL | Needs allocator for ToOwned |
| `ToOwned` trait | GPU_IMPL | Uses allocator |

---

## std::boxed

| Item | Strategy | Notes |
|------|----------|-------|
| `Box<T>` | GPU_IMPL | Uses gpu_heap |
| `Box::new()` | GPU_IMPL | Allocate from gpu_heap |
| `Box::pin()` | GPU_IMPL | Allocate pinned |
| `Box::leak()` | GPU_IMPL | Don't deallocate |
| `Box::from_raw()` | GPU_IMPL | Wrap existing allocation |
| `Box::into_raw()` | GPU_IMPL | Unwrap to pointer |

**Implementation:**
```rust
// gpu_std/src/boxed.rs
impl<T> Box<T> {
    pub fn new(x: T) -> Box<T> {
        let ptr = gpu_heap::alloc(Layout::new::<T>()) as *mut T;
        unsafe {
            ptr.write(x);
            Box::from_raw(ptr)
        }
    }
}
```

---

## std::cell

| Item | Strategy | Notes |
|------|----------|-------|
| `Cell<T>` | GPU_IMPL | Atomic wrapper for GPU |
| `RefCell<T>` | AUTO_TRANSFORM | Atomic borrow counter (like AtomicRefCell) |
| `UnsafeCell<T>` | NATIVE | Just removes aliasing restrictions |
| `OnceCell<T>` | GPU_IMPL | Atomic once initialization |
| `LazyCell<T>` | GPU_IMPL | Lazy with atomics |

**GPU Cell Implementation:**
```rust
// gpu_std/src/cell.rs
#[repr(transparent)]
pub struct Cell<T: Copy> {
    value: UnsafeCell<T>,
}

impl<T: Copy> Cell<T> {
    pub fn get(&self) -> T {
        // On GPU, this is just a read - no synchronization needed for Copy types
        unsafe { *self.value.get() }
    }

    pub fn set(&self, val: T) {
        // On GPU, writes to same location from different threads = race
        // User must ensure single-thread access or use atomics
        unsafe { *self.value.get() = val; }
    }
}
```

---

## std::char

| Item | Strategy | Notes |
|------|----------|-------|
| `char` type | NATIVE | u32 internally |
| All `is_*` methods | NATIVE | Pure comparisons |
| All `to_*` methods | NATIVE | Pure conversions |
| `decode_utf16()` | NATIVE | Pure iterator |
| `from_u32()` | NATIVE | Pure conversion |
| `from_digit()` | NATIVE | Pure conversion |
| `CharTryFromError` | NATIVE | Error type |
| `DecodeUtf16Error` | NATIVE | Error type |
| `ParseCharError` | NATIVE | Error type |
| `ToLowercase` | NATIVE | Iterator |
| `ToUppercase` | NATIVE | Iterator |
| `EscapeUnicode` | NATIVE | Iterator |
| `EscapeDefault` | NATIVE | Iterator |
| `EscapeDebug` | NATIVE | Iterator |

---

## std::clone

| Item | Strategy | Notes |
|------|----------|-------|
| `Clone` trait | GPU_IMPL | Deep clone uses allocator |
| `clone()` | GPU_IMPL | May allocate |
| `clone_from()` | GPU_IMPL | May allocate |

---

## std::cmp

| Item | Strategy | Notes |
|------|----------|-------|
| `PartialEq` trait | NATIVE | Pure comparison |
| `Eq` trait | NATIVE | Marker trait |
| `PartialOrd` trait | NATIVE | Pure comparison |
| `Ord` trait | NATIVE | Pure comparison |
| `Ordering` enum | NATIVE | Pure enum |
| `Reverse` | NATIVE | Newtype wrapper |
| `min()` | NATIVE | Pure function |
| `max()` | NATIVE | Pure function |
| `min_by()` | NATIVE | Pure function |
| `max_by()` | NATIVE | Pure function |
| `min_by_key()` | NATIVE | Pure function |
| `max_by_key()` | NATIVE | Pure function |
| `minmax()` | NATIVE | Pure function |
| `minmax_by()` | NATIVE | Pure function |
| `minmax_by_key()` | NATIVE | Pure function |

---

## std::collections

| Item | Strategy | Notes |
|------|----------|-------|
| `HashMap<K,V>` | GPU_IMPL | Cuckoo hash on GPU |
| `HashSet<K>` | GPU_IMPL | HashMap wrapper |
| `BTreeMap<K,V>` | GPU_IMPL | B+ tree or sorted array |
| `BTreeSet<K>` | GPU_IMPL | BTreeMap wrapper |
| `VecDeque<T>` | GPU_IMPL | Ring buffer |
| `LinkedList<T>` | AUTO_TRANSFORM | Arena allocation with indices (no pointers) |
| `BinaryHeap<T>` | GPU_IMPL | Array-based heap |
| `hash_map::*` | GPU_IMPL | All HashMap types |
| `hash_set::*` | GPU_IMPL | All HashSet types |
| `btree_map::*` | GPU_IMPL | All BTreeMap types |
| `btree_set::*` | GPU_IMPL | All BTreeSet types |
| `binary_heap::*` | GPU_IMPL | All BinaryHeap types |
| `vec_deque::*` | GPU_IMPL | All VecDeque types |

**HashMap Implementation:**
```rust
// gpu_std/src/collections/hash_map.rs
// Uses Cuckoo hashing for O(1) guaranteed lookup (no SIMD divergence)

pub struct HashMap<K, V, S = GpuHasher> {
    buckets: GpuVec<Bucket<K, V>>,
    len: u32,
    hasher: S,
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Cuckoo hash insert - O(1) amortized
        let hash1 = self.hash1(&key);
        let hash2 = self.hash2(&key);
        // ... cuckoo insertion logic
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        // O(1) guaranteed - check only 2 locations
        let hash1 = self.hash1(key);
        let hash2 = self.hash2(key);
        if self.buckets[hash1].key == *key { return Some(&self.buckets[hash1].value); }
        if self.buckets[hash2].key == *key { return Some(&self.buckets[hash2].value); }
        None
    }
}
```

---

## std::convert

| Item | Strategy | Notes |
|------|----------|-------|
| `From` trait | NATIVE | Pure conversion |
| `Into` trait | NATIVE | Pure conversion |
| `TryFrom` trait | NATIVE | Fallible conversion |
| `TryInto` trait | NATIVE | Fallible conversion |
| `AsRef` trait | NATIVE | Reference conversion |
| `AsMut` trait | NATIVE | Reference conversion |
| `Infallible` | NATIVE | Never type |
| `identity()` | NATIVE | Identity function |

---

## std::default

| Item | Strategy | Notes |
|------|----------|-------|
| `Default` trait | NATIVE | Default values |
| `default()` | NATIVE | Returns default |

---

## std::env

| Item | Strategy | Notes |
|------|----------|-------|
| `args()` | GPU_IMPL | Read from params buffer |
| `args_os()` | GPU_IMPL | Read from params buffer |
| `current_dir()` | GPU_IMPL | Return from filesystem state |
| `current_exe()` | AUTO_TRANSFORM | Load-time snapshot stored in GPU buffer |
| `home_dir()` | AUTO_TRANSFORM | Load-time snapshot stored in GPU buffer |
| `join_paths()` | NATIVE | Pure string manipulation |
| `remove_var()` | GPU_IMPL | Remove from env buffer |
| `set_current_dir()` | GPU_IMPL | Update filesystem state |
| `set_var()` | GPU_IMPL | Write to env buffer |
| `split_paths()` | NATIVE | Pure string manipulation |
| `temp_dir()` | GPU_IMPL | Return temp path from config |
| `var()` | GPU_IMPL | Read from env buffer |
| `var_os()` | GPU_IMPL | Read from env buffer |
| `vars()` | GPU_IMPL | Iterate env buffer |
| `vars_os()` | GPU_IMPL | Iterate env buffer |
| `VarError` | NATIVE | Error type |
| `JoinPathsError` | NATIVE | Error type |
| `Args` | GPU_IMPL | Iterator over args buffer |
| `ArgsOs` | GPU_IMPL | Iterator over args buffer |
| `Vars` | GPU_IMPL | Iterator over env buffer |
| `VarsOs` | GPU_IMPL | Iterator over env buffer |
| `SplitPaths` | NATIVE | Iterator |

**Implementation:**
```rust
// gpu_std/src/env.rs

// Environment stored in GPU buffer, initialized by CPU before dispatch
static ENV_BUFFER: GpuBuffer<EnvEntry> = ...;
static ARGS_BUFFER: GpuBuffer<GpuString> = ...;

pub fn var(key: &str) -> Result<String, VarError> {
    for entry in ENV_BUFFER.iter() {
        if entry.key == key {
            return Ok(entry.value.clone());
        }
    }
    Err(VarError::NotPresent)
}

pub fn args() -> Args {
    Args { inner: ARGS_BUFFER.iter() }
}
```

---

## std::error

| Item | Strategy | Notes |
|------|----------|-------|
| `Error` trait | GPU_IMPL | Simplified, no Display |
| `Report` | GPU_IMPL | Write formatted error to debug buffer |
| `request_ref()` | AUTO_TRANSFORM | Compile-time TypeId registry lookup |
| `request_value()` | AUTO_TRANSFORM | Compile-time TypeId registry lookup |

**GPU Error Implementation:**
```rust
// gpu_std/src/error.rs

// GPU errors are code-based, not string-based
pub trait Error {
    fn code(&self) -> u32;
    fn source(&self) -> Option<&(dyn Error)> { None }
}

// Macro for defining errors
#[macro_export]
macro_rules! gpu_error {
    ($name:ident { $($variant:ident = $code:expr),* $(,)? }) => {
        #[repr(u32)]
        pub enum $name {
            $($variant = $code,)*
        }

        impl Error for $name {
            fn code(&self) -> u32 { *self as u32 }
        }
    };
}
```

---

## std::f32 / std::f64

| Item | Strategy | Notes |
|------|----------|-------|
| Constants (PI, E, etc.) | NATIVE | Compile-time constants |
| `abs()` | NATIVE | Metal: `abs()` |
| `signum()` | NATIVE | Metal: `sign()` |
| `copysign()` | NATIVE | Metal: `copysign()` |
| `mul_add()` | NATIVE | Metal: `fma()` |
| `div_euclid()` | NATIVE | Pure math |
| `rem_euclid()` | NATIVE | Pure math |
| `powi()` | NATIVE | Metal: `pow()` with int |
| `powf()` | NATIVE | Metal: `pow()` |
| `sqrt()` | NATIVE | Metal: `sqrt()` |
| `exp()` | NATIVE | Metal: `exp()` |
| `exp2()` | NATIVE | Metal: `exp2()` |
| `ln()` | NATIVE | Metal: `log()` |
| `log()` | NATIVE | Metal: `log()` |
| `log2()` | NATIVE | Metal: `log2()` |
| `log10()` | NATIVE | Metal: `log10()` |
| `cbrt()` | GPU_IMPL | `pow(x, 1.0/3.0)` |
| `hypot()` | NATIVE | Metal: `length(float2(x,y))` |
| `sin()` | NATIVE | Metal: `sin()` |
| `cos()` | NATIVE | Metal: `cos()` |
| `tan()` | NATIVE | Metal: `tan()` |
| `asin()` | NATIVE | Metal: `asin()` |
| `acos()` | NATIVE | Metal: `acos()` |
| `atan()` | NATIVE | Metal: `atan()` |
| `atan2()` | NATIVE | Metal: `atan2()` |
| `sin_cos()` | NATIVE | Metal: `sincos()` |
| `exp_m1()` | GPU_IMPL | `exp(x) - 1` |
| `ln_1p()` | GPU_IMPL | `log(1 + x)` |
| `sinh()` | NATIVE | Metal: `sinh()` |
| `cosh()` | NATIVE | Metal: `cosh()` |
| `tanh()` | NATIVE | Metal: `tanh()` |
| `asinh()` | NATIVE | Metal: `asinh()` |
| `acosh()` | NATIVE | Metal: `acosh()` |
| `atanh()` | NATIVE | Metal: `atanh()` |
| `floor()` | NATIVE | Metal: `floor()` |
| `ceil()` | NATIVE | Metal: `ceil()` |
| `round()` | NATIVE | Metal: `round()` |
| `trunc()` | NATIVE | Metal: `trunc()` |
| `fract()` | NATIVE | Metal: `fract()` |
| `is_nan()` | NATIVE | Metal: `isnan()` |
| `is_infinite()` | NATIVE | Metal: `isinf()` |
| `is_finite()` | NATIVE | Metal: `isfinite()` |
| `is_subnormal()` | NATIVE | Bit manipulation |
| `is_normal()` | NATIVE | Bit manipulation |
| `is_sign_positive()` | NATIVE | Metal: `signbit()` |
| `is_sign_negative()` | NATIVE | Metal: `signbit()` |
| `recip()` | NATIVE | `1.0 / x` |
| `to_degrees()` | NATIVE | Metal: `degrees()` |
| `to_radians()` | NATIVE | Metal: `radians()` |
| `max()` | NATIVE | Metal: `max()` |
| `min()` | NATIVE | Metal: `min()` |
| `clamp()` | NATIVE | Metal: `clamp()` |
| `to_bits()` | NATIVE | Metal: `as_type<uint>()` |
| `from_bits()` | NATIVE | Metal: `as_type<float>()` |
| `to_be_bytes()` | NATIVE | Bit manipulation |
| `to_le_bytes()` | NATIVE | Bit manipulation |
| `to_ne_bytes()` | NATIVE | Bit manipulation |
| `from_be_bytes()` | NATIVE | Bit manipulation |
| `from_le_bytes()` | NATIVE | Bit manipulation |
| `from_ne_bytes()` | NATIVE | Bit manipulation |
| `total_cmp()` | NATIVE | Bit comparison |
| `midpoint()` | NATIVE | Pure math |
| `gamma()` | GPU_IMPL | Lanczos approximation |
| `ln_gamma()` | GPU_IMPL | Stirling approximation |

---

## std::ffi

| Item | Strategy | Notes |
|------|----------|-------|
| `CStr` | GPU_IMPL | Null-terminated string view |
| `CString` | GPU_IMPL | Owned null-terminated string |
| `OsStr` | GPU_IMPL | Platform string view |
| `OsString` | GPU_IMPL | Owned platform string |
| `c_char`, `c_int`, etc. | NATIVE | Type aliases |
| `VaList` | AUTO_TRANSFORM | Array-based argument dispatch |
| `FromBytesWithNulError` | NATIVE | Error type |
| `FromBytesUntilNulError` | NATIVE | Error type |
| `FromVecWithNulError` | NATIVE | Error type |
| `IntoStringError` | NATIVE | Error type |
| `NulError` | NATIVE | Error type |

---

## std::fmt

| Item | Strategy | Notes |
|------|----------|-------|
| `format!` | GPU_IMPL | Write to pre-allocated buffer |
| `write!` | GPU_IMPL | Write to buffer impl |
| `print!` | GPU_IMPL | Write to debug buffer |
| `println!` | GPU_IMPL | Write to debug buffer + newline |
| `eprint!` | GPU_IMPL | Write to debug buffer (error level) |
| `eprintln!` | GPU_IMPL | Write to debug buffer + newline |
| `format_args!` | NATIVE | Compile-time format preparation |
| `Debug` trait | GPU_IMPL | Simplified debug output |
| `Display` trait | GPU_IMPL | Simplified display output |
| `Binary` trait | GPU_IMPL | Binary formatting |
| `Octal` trait | GPU_IMPL | Octal formatting |
| `LowerHex` trait | GPU_IMPL | Lowercase hex |
| `UpperHex` trait | GPU_IMPL | Uppercase hex |
| `LowerExp` trait | GPU_IMPL | Scientific notation |
| `UpperExp` trait | GPU_IMPL | Scientific notation |
| `Pointer` trait | GPU_IMPL | Pointer formatting |
| `Write` trait | GPU_IMPL | Core write trait |
| `Formatter` | GPU_IMPL | Simplified formatter |
| `Arguments` | NATIVE | Format args struct |
| `Error` | NATIVE | Format error |
| `Result` | NATIVE | Format result |
| `Alignment` | NATIVE | Alignment enum |
| `DebugStruct` | GPU_IMPL | Struct formatter helper |
| `DebugTuple` | GPU_IMPL | Tuple formatter helper |
| `DebugList` | GPU_IMPL | List formatter helper |
| `DebugSet` | GPU_IMPL | Set formatter helper |
| `DebugMap` | GPU_IMPL | Map formatter helper |

**Formatting Implementation:**
```rust
// gpu_std/src/fmt.rs

pub struct GpuFormatter<'a> {
    buf: &'a mut GpuString,
    // Simplified - no width/precision/fill for now
}

impl<'a> Write for GpuFormatter<'a> {
    fn write_str(&mut self, s: &str) -> Result<(), Error> {
        self.buf.push_str(s);
        Ok(())
    }
}

// Integer formatting
impl Display for i32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        // GPU-native integer to string
        let mut buf = [0u8; 12];
        let len = gpu_itoa(*self, &mut buf);
        f.write_str(unsafe { core::str::from_utf8_unchecked(&buf[..len]) })
    }
}

// Debug buffer for println!
static DEBUG_BUFFER: GpuDebugBuffer = ...;

#[macro_export]
macro_rules! println {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let mut buf = $crate::fmt::GpuString::new();
        write!(&mut buf, $($arg)*).unwrap();
        buf.push('\n');
        $crate::debug::write(&buf);
    }};
}
```

---

## std::fs

| Item | Strategy | Notes |
|------|----------|-------|
| `File` | GPU_IMPL | Handle to GPU filesystem |
| `OpenOptions` | GPU_IMPL | File open configuration |
| `Metadata` | GPU_IMPL | File metadata struct |
| `Permissions` | GPU_IMPL | Permission flags |
| `FileType` | GPU_IMPL | File type enum |
| `DirEntry` | GPU_IMPL | Directory entry |
| `ReadDir` | GPU_IMPL | Directory iterator |
| `DirBuilder` | GPU_IMPL | Directory builder |
| `FileTimes` | GPU_IMPL | File timestamps |
| `canonicalize()` | GPU_IMPL | Path canonicalization |
| `copy()` | GPU_IMPL | File copy |
| `create_dir()` | GPU_IMPL | Create directory |
| `create_dir_all()` | GPU_IMPL | Create directory tree |
| `hard_link()` | GPU_IMPL | Create hard link |
| `metadata()` | GPU_IMPL | Get file metadata |
| `read()` | GPU_IMPL | Read file to Vec |
| `read_dir()` | GPU_IMPL | List directory |
| `read_link()` | GPU_IMPL | Read symlink target |
| `read_to_string()` | GPU_IMPL | Read file to String |
| `remove_dir()` | GPU_IMPL | Remove directory |
| `remove_dir_all()` | GPU_IMPL | Remove directory tree |
| `remove_file()` | GPU_IMPL | Remove file |
| `rename()` | GPU_IMPL | Rename file |
| `set_permissions()` | GPU_IMPL | Set permissions |
| `symlink_metadata()` | GPU_IMPL | Get symlink metadata |
| `write()` | GPU_IMPL | Write bytes to file |
| `exists()` | GPU_IMPL | Check file existence |

**We already have this!** From `src/gpu_os/filesystem.rs`:
```rust
// Existing GPU filesystem with 3M+ path support
// Just need to wrap in std-compatible API

// gpu_std/src/fs.rs
pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let path = path.as_ref();
    let handle = GPU_FILESYSTEM.open(path, OpenOptions::read())?;
    let size = handle.metadata()?.len() as usize;
    let mut buf = Vec::with_capacity(size);
    handle.read_to_end(&mut buf)?;
    Ok(buf)
}

pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
    let bytes = read(path)?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let path = path.as_ref();
    let mut handle = GPU_FILESYSTEM.open(path, OpenOptions::write().create(true))?;
    handle.write_all(contents.as_ref())
}

pub fn read_dir(path: impl AsRef<Path>) -> io::Result<ReadDir> {
    GPU_FILESYSTEM.read_dir(path.as_ref())
}
```

---

## std::future

| Item | Strategy | Notes |
|------|----------|-------|
| `Future` trait | AUTO_TRANSFORM | State machine automatically transformed to parallel work queue dispatch |
| `IntoFuture` trait | AUTO_TRANSFORM | Same |
| `poll_fn()` | AUTO_TRANSFORM | Same |
| `pending()` | AUTO_TRANSFORM | Same |
| `ready()` | AUTO_TRANSFORM | Same |
| `Pending` | AUTO_TRANSFORM | Same |
| `PollFn` | AUTO_TRANSFORM | Same |
| `Ready` | AUTO_TRANSFORM | Same |

**Error message:**
```
error: async/await is not supported on GPU

  Async Rust requires a runtime (Tokio, async-std) that cannot run on GPU.

  GPU programs are inherently parallel. Instead of:

      async fn process(items: Vec<Item>) {
          for item in items {
              process_one(item).await;
          }
      }

  Use parallel dispatch:

      fn process(items: &[Item]) {
          // Each GPU thread processes one item simultaneously
          let tid = gpu::thread_id();
          if tid < items.len() {
              process_one(&items[tid]);
          }
      }
```

---

## std::hash

| Item | Strategy | Notes |
|------|----------|-------|
| `Hash` trait | NATIVE | Pure trait |
| `Hasher` trait | GPU_IMPL | GPU-optimized hasher |
| `BuildHasher` trait | NATIVE | Hasher factory |
| `BuildHasherDefault` | NATIVE | Default hasher builder |
| `SipHasher` | GPU_IMPL | Implement directly or auto-replace with FxHash |
| `DefaultHasher` | GPU_IMPL | FxHash by default |
| `RandomState` | GPU_IMPL | Deterministic seed from dispatch ID |

**GPU Hasher:**
```rust
// gpu_std/src/hash.rs

// FxHash - fast, simple, GPU-friendly
pub struct FxHasher {
    hash: u64,
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.hash = self.hash.rotate_left(5) ^ (*byte as u64);
            self.hash = self.hash.wrapping_mul(0x517cc1b727220a95);
        }
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type DefaultHasher = FxHasher;
```

---

## std::hint

| Item | Strategy | Notes |
|------|----------|-------|
| `black_box()` | NATIVE | Optimization barrier |
| `spin_loop()` | NATIVE | CPU hint (no-op on GPU) |
| `unreachable_unchecked()` | NATIVE | UB hint |
| `assert_unchecked()` | NATIVE | Assumption hint |
| `cold()` | NATIVE | Branch hint |

---

## std::io

| Item | Strategy | Notes |
|------|----------|-------|
| `Read` trait | GPU_IMPL | Read from GPU buffers |
| `Write` trait | GPU_IMPL | Write to GPU buffers |
| `Seek` trait | GPU_IMPL | Seek in GPU buffers |
| `BufRead` trait | GPU_IMPL | Buffered read |
| `stdin()` | GPU_IMPL | Read from input buffer |
| `stdout()` | GPU_IMPL | Write to debug buffer |
| `stderr()` | GPU_IMPL | Write to debug buffer |
| `Stdin` | GPU_IMPL | Input buffer reader |
| `Stdout` | GPU_IMPL | Debug buffer writer |
| `Stderr` | GPU_IMPL | Debug buffer writer |
| `StdinLock` | GPU_IMPL | Locked input reader |
| `StdoutLock` | GPU_IMPL | Locked output writer |
| `StderrLock` | GPU_IMPL | Locked error writer |
| `Error` | GPU_IMPL | I/O error type |
| `ErrorKind` | NATIVE | Error kind enum |
| `Result` | NATIVE | I/O result alias |
| `BufReader` | GPU_IMPL | Buffered reader |
| `BufWriter` | GPU_IMPL | Buffered writer |
| `LineWriter` | GPU_IMPL | Line-buffered writer |
| `Cursor` | GPU_IMPL | In-memory cursor |
| `Take` | GPU_IMPL | Limited reader |
| `Chain` | GPU_IMPL | Chained readers |
| `Empty` | NATIVE | Empty reader |
| `Repeat` | NATIVE | Repeating reader |
| `Sink` | NATIVE | Discarding writer |
| `copy()` | GPU_IMPL | Copy between readers/writers |
| `read_to_string()` | GPU_IMPL | Read all to string |
| `empty()` | NATIVE | Create empty reader |
| `repeat()` | NATIVE | Create repeat reader |
| `sink()` | NATIVE | Create sink writer |
| `SeekFrom` | NATIVE | Seek position enum |
| `IoSlice` | GPU_IMPL | Vectored I/O |
| `IoSliceMut` | GPU_IMPL | Mutable vectored I/O |
| `WriterPanicked` | NATIVE | Error type |
| `IntoInnerError` | NATIVE | Error type |

**Implementation:**
```rust
// gpu_std/src/io.rs

pub struct Stdout {
    buffer: &'static GpuDebugBuffer,
}

impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.buffer.write(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<()> {
        // GPU debug buffer is always flushed
        Ok(())
    }
}

pub fn stdout() -> Stdout {
    Stdout { buffer: &DEBUG_BUFFER }
}

pub fn stdin() -> Stdin {
    Stdin { buffer: &INPUT_BUFFER, pos: 0 }
}
```

---

## std::iter

| Item | Strategy | Notes |
|------|----------|-------|
| `Iterator` trait | GPU_IMPL | Parallel iteration |
| `IntoIterator` trait | NATIVE | Conversion trait |
| `FromIterator` trait | GPU_IMPL | Collect into collections |
| `ExactSizeIterator` trait | NATIVE | Size hint |
| `DoubleEndedIterator` trait | GPU_IMPL | Reverse iteration |
| `FusedIterator` trait | NATIVE | Fused iterator marker |
| `TrustedLen` trait | NATIVE | Trusted length marker |
| `Extend` trait | GPU_IMPL | Extend collections |
| `Sum` trait | GPU_IMPL | Parallel sum |
| `Product` trait | GPU_IMPL | Parallel product |
| All iterator adaptors | GPU_IMPL | Parallel adaptors |
| `empty()` | NATIVE | Empty iterator |
| `once()` | NATIVE | Single-element iterator |
| `once_with()` | NATIVE | Lazy single-element |
| `repeat()` | NATIVE | Repeating iterator |
| `repeat_with()` | NATIVE | Lazy repeat |
| `repeat_n()` | NATIVE | N-element repeat |
| `successors()` | GPU_IMPL | May parallelize |
| `from_fn()` | GPU_IMPL | Generate from function |
| `zip()` | GPU_IMPL | Parallel zip |

**Parallel Iterator:**
```rust
// gpu_std/src/iter.rs

// Iterator methods become parallel operations
impl<I: Iterator> GpuIteratorExt for I {
    fn for_each<F>(self, f: F) where F: Fn(Self::Item) {
        // Each GPU thread processes one element
        let tid = gpu::thread_id();
        let items: Vec<_> = self.collect();  // Materialize
        if tid < items.len() {
            f(items[tid]);
        }
    }

    fn map<B, F>(self, f: F) -> GpuMap<Self, F> {
        GpuMap { iter: self, f }
    }

    fn sum<S>(self) -> S where S: Sum<Self::Item> {
        // Parallel reduction
        gpu::parallel_reduce(self, S::zero(), |a, b| a + b)
    }
}
```

---

## std::marker

| Item | Strategy | Notes |
|------|----------|-------|
| `Copy` trait | NATIVE | Marker trait |
| `Send` trait | NATIVE | Marker (all GPU data is "Send") |
| `Sync` trait | NATIVE | Marker (all GPU data is "Sync") |
| `Sized` trait | NATIVE | Marker trait |
| `Unpin` trait | NATIVE | Marker trait |
| `PhantomData` | NATIVE | Zero-sized marker |
| `PhantomPinned` | NATIVE | Pin marker |
| `Destruct` trait | NATIVE | Destructor marker |
| `Tuple` trait | NATIVE | Tuple marker |

---

## std::mem

| Item | Strategy | Notes |
|------|----------|-------|
| `size_of()` | NATIVE | Compile-time size |
| `size_of_val()` | NATIVE | Runtime size |
| `align_of()` | NATIVE | Compile-time alignment |
| `align_of_val()` | NATIVE | Runtime alignment |
| `needs_drop()` | NATIVE | Drop check |
| `zeroed()` | NATIVE | Zero initialization |
| `uninitialized()` | NATIVE | (Deprecated) |
| `MaybeUninit` | NATIVE | Uninitialized wrapper |
| `transmute()` | NATIVE | Type reinterpretation |
| `transmute_copy()` | NATIVE | Copying transmute |
| `swap()` | NATIVE | Value swap |
| `take()` | NATIVE | Replace with default |
| `replace()` | NATIVE | Replace value |
| `drop()` | GPU_IMPL | No-op on GPU (no destructors) |
| `forget()` | NATIVE | Prevent drop |
| `forget_unsized()` | NATIVE | Unsized forget |
| `ManuallyDrop` | NATIVE | Manual drop wrapper |
| `Discriminant` | NATIVE | Enum discriminant |
| `discriminant()` | NATIVE | Get discriminant |
| `offset_of!()` | NATIVE | Field offset |
| `copy()` | NATIVE | Bitwise copy |
| `copy_nonoverlapping()` | NATIVE | Non-overlapping copy |
| `write()` | NATIVE | Write to pointer |
| `write_bytes()` | NATIVE | Write bytes |
| `write_volatile()` | NATIVE | Volatile write |
| `read()` | NATIVE | Read from pointer |
| `read_unaligned()` | NATIVE | Unaligned read |
| `read_volatile()` | NATIVE | Volatile read |

---

## std::net

| Item | Strategy | Notes |
|------|----------|-------|
| `IpAddr` | NATIVE | IP address enum |
| `Ipv4Addr` | NATIVE | IPv4 address |
| `Ipv6Addr` | NATIVE | IPv6 address |
| `SocketAddr` | NATIVE | Socket address |
| `SocketAddrV4` | NATIVE | IPv4 socket address |
| `SocketAddrV6` | NATIVE | IPv6 socket address |
| `TcpStream` | AUTO_TRANSFORM | Automatically queued to network request buffer, CPU processes after dispatch |
| `TcpListener` | AUTO_TRANSFORM | Automatically queued to network request buffer |
| `UdpSocket` | AUTO_TRANSFORM | Automatically queued to network request buffer |
| `ToSocketAddrs` | GPU_IMPL | For literals only, no DNS |
| `Shutdown` | NATIVE | Shutdown enum |
| `Incoming` | AUTO_TRANSFORM | Works with auto-transformed TcpListener |
| `AddrParseError` | NATIVE | Error type |

**Error message for networking:**
```
error: TCP/UDP networking is not available on GPU

  GPU compute shaders cannot make network connections. Instead:

  1. Queue network requests in an output buffer:

      #[repr(C)]
      struct NetworkRequest {
          dest_ip: [u8; 4],
          dest_port: u16,
          payload_offset: u32,
          payload_len: u32,
      }

  2. Process the queue on CPU after dispatch:

      for req in gpu_output.network_requests() {
          let mut stream = TcpStream::connect(req.dest())?;
          stream.write_all(req.payload())?;
      }

  This pattern allows GPU to prepare network operations that CPU executes.
```

---

## std::num

| Item | Strategy | Notes |
|------|----------|-------|
| `NonZeroU8/16/32/64/128/Usize` | NATIVE | Non-zero integers |
| `NonZeroI8/16/32/64/128/Isize` | NATIVE | Non-zero signed integers |
| `Wrapping<T>` | NATIVE | Wrapping arithmetic |
| `Saturating<T>` | NATIVE | Saturating arithmetic |
| `IntErrorKind` | NATIVE | Parse error kind |
| `ParseIntError` | NATIVE | Integer parse error |
| `ParseFloatError` | NATIVE | Float parse error |
| `TryFromIntError` | NATIVE | Conversion error |
| `FpCategory` | NATIVE | Float category enum |

---

## std::ops

| Item | Strategy | Notes |
|------|----------|-------|
| All arithmetic traits | NATIVE | Add, Sub, Mul, Div, Rem, Neg |
| All bitwise traits | NATIVE | BitAnd, BitOr, BitXor, Shl, Shr, Not |
| All compound assignment | NATIVE | AddAssign, SubAssign, etc. |
| `Deref`/`DerefMut` | NATIVE | Dereference traits |
| `Index`/`IndexMut` | GPU_IMPL | Bounds checking optional |
| `Fn`/`FnMut`/`FnOnce` | GPU_IMPL | Must be inlined |
| `Range*` types | NATIVE | Range structs |
| `RangeBounds` trait | NATIVE | Range trait |
| `Try` trait | NATIVE | ? operator support |
| `FromResidual` trait | NATIVE | ? operator support |
| `Residual` trait | NATIVE | ? operator support |
| `ControlFlow` | NATIVE | Control flow enum |
| `Bound` | NATIVE | Range bound enum |
| `CoerceUnsized` | NATIVE | Coercion marker |
| `DispatchFromDyn` | NATIVE | Dispatch marker |

---

## std::option

| Item | Strategy | Notes |
|------|----------|-------|
| `Option<T>` | NATIVE | Core enum |
| All Option methods | NATIVE | Pure methods |
| `None`, `Some` | NATIVE | Enum variants |
| `IntoIter` | NATIVE | Option iterator |
| `Iter` | NATIVE | Option ref iterator |
| `IterMut` | NATIVE | Option mut iterator |

---

## std::os

| Item | Strategy | Notes |
|------|----------|-------|
| `unix::*` | AUTO_TRANSFORM | Syscall queue to CPU for execution |
| `windows::*` | AUTO_TRANSFORM | Syscall queue to CPU for execution |
| `macos::*` | AUTO_TRANSFORM | Syscall queue to CPU for execution |
| `linux::*` | AUTO_TRANSFORM | Syscall queue to CPU for execution |
| `fd::*` | AUTO_TRANSFORM | Handle table in I/O subsystem |
| `raw::*` | GPU_IMPL | Raw type aliases |

---

## std::panic

| Item | Strategy | Notes |
|------|----------|-------|
| `panic!` | GPU_IMPL | Write to debug buffer, halt |
| `catch_unwind()` | AUTO_TRANSFORM | Error flag propagation with result capture |
| `resume_unwind()` | AUTO_TRANSFORM | Set error flag and halt thread |
| `set_hook()` | AUTO_TRANSFORM | Store function pointer in GPU global |
| `take_hook()` | AUTO_TRANSFORM | Retrieve and clear hook from GPU global |
| `PanicHookInfo` | AUTO_TRANSFORM | Struct populated from panic state buffer |
| `Location` | GPU_IMPL | Compile-time location |
| `AssertUnwindSafe` | NATIVE | Marker type |
| `UnwindSafe` | NATIVE | Marker trait |
| `RefUnwindSafe` | NATIVE | Marker trait |

**GPU Panic Implementation:**
```rust
// gpu_std/src/panic.rs

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    // Write panic info to debug buffer
    let location = info.location().unwrap();
    DEBUG_BUFFER.write_panic(
        location.file(),
        location.line(),
        location.column(),
    );

    // Set panic flag
    PANIC_FLAG.store(1, Ordering::Relaxed);

    // Halt this thread
    gpu::halt();
}

#[macro_export]
macro_rules! panic {
    ($($arg:tt)*) => {{
        $crate::panic::begin_panic(format_args!($($arg)*), $crate::panic::Location::caller());
    }};
}
```

---

## std::path

| Item | Strategy | Notes |
|------|----------|-------|
| `Path` | GPU_IMPL | Path slice type |
| `PathBuf` | GPU_IMPL | Owned path |
| `Components` | GPU_IMPL | Path components iterator |
| `Component` | NATIVE | Component enum |
| `PrefixComponent` | NATIVE | Windows prefix |
| `Prefix` | NATIVE | Prefix enum |
| `Ancestors` | GPU_IMPL | Ancestors iterator |
| `Iter` | GPU_IMPL | Path iterator |
| `Display` | GPU_IMPL | Display wrapper |
| `StripPrefixError` | NATIVE | Error type |
| `MAIN_SEPARATOR` | NATIVE | Path separator |
| `MAIN_SEPARATOR_STR` | NATIVE | Path separator string |
| `is_separator()` | NATIVE | Check separator |
| `absolute()` | GPU_IMPL | Make absolute |

---

## std::pin

| Item | Strategy | Notes |
|------|----------|-------|
| `Pin<P>` | NATIVE | Pin wrapper |
| `pin!` macro | NATIVE | Stack pinning |

---

## std::prelude

| Item | Strategy | Notes |
|------|----------|-------|
| `rust_2021::*` | GPU_IMPL | Re-export GPU std types |
| `rust_2024::*` | GPU_IMPL | Re-export GPU std types |
| `v1::*` | GPU_IMPL | Re-export GPU std types |

---

## std::primitive

| Item | Strategy | Notes |
|------|----------|-------|
| All primitive types | NATIVE | bool, char, f32, f64, i8-i128, u8-u128, isize, usize, str, slice, array, tuple, fn, pointer, reference, unit, never |

---

## std::process

**GPU-NATIVE PROCESS MODEL**

Processes in this system are GPU-native: a "process" is a threadgroup with its own state buffers, not a CPU syscall. This is fundamentally different from traditional OS process spawning.

| Item | Strategy | Notes |
|------|----------|-------|
| `Command` | GPU_IMPL | Lookup bytecode in registry, spawn GPU process |
| `Child` | GPU_IMPL | GpuProcess handle (threadgroup + state buffers) |
| `Output` | GPU_IMPL | Read from process output buffer |
| `Stdio` | GPU_IMPL | GPU ring buffers for stdin/stdout/stderr |
| `ExitStatus` | GPU_IMPL | Atomic status in process state buffer |
| `ExitCode` | GPU_IMPL | Exit code from process state |
| `ChildStdin` | GPU_IMPL | Write end of stdin ring buffer |
| `ChildStdout` | GPU_IMPL | Read end of stdout ring buffer |
| `ChildStderr` | GPU_IMPL | Read end of stderr ring buffer |
| `exit()` | GPU_IMPL | Set exit flag, halt kernel |
| `abort()` | GPU_IMPL | Set abort flag, halt kernel |
| `id()` | GPU_IMPL | Return GPU process ID |

**GPU Process Architecture:**

A GPU process consists of:
1. **Bytecode** - WASM/GPU bytecode loaded from registry
2. **Threadgroup** - Dedicated GPU threads executing the bytecode
3. **State buffers** - Heap, stack, globals isolated per process
4. **I/O buffers** - Ring buffers for stdin/stdout/stderr
5. **Process table entry** - PID, status, parent, children

**Implementation:**
```rust
// gpu_std/src/process.rs

/// GPU process handle - represents a running GPU threadgroup
pub struct Child {
    pid: GpuProcessId,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
}

/// Bytecode registry maps program names to GPU bytecode
static BYTECODE_REGISTRY: GpuHashMap<GpuString, BytecodeHandle> = ...;

/// Process table tracks all running GPU processes
static PROCESS_TABLE: GpuProcessTable = ...;

impl Command {
    pub fn new(program: impl AsRef<str>) -> Command {
        Command {
            program: program.as_ref().into(),
            args: GpuVec::new(),
            env: None,
            stdin: Stdio::inherit(),
            stdout: Stdio::inherit(),
            stderr: Stdio::inherit(),
        }
    }

    pub fn spawn(&mut self) -> io::Result<Child> {
        // 1. Lookup bytecode in registry
        let bytecode = BYTECODE_REGISTRY.get(&self.program)
            .ok_or(io::Error::new(io::ErrorKind::NotFound, "program not found"))?;

        // 2. Allocate process state buffers
        let state = GpuProcessState::new();
        state.set_args(&self.args);
        if let Some(ref env) = self.env {
            state.set_env(env);
        }

        // 3. Allocate I/O ring buffers
        let (stdin_tx, stdin_rx) = gpu_ring_buffer::<u8>(4096);
        let (stdout_tx, stdout_rx) = gpu_ring_buffer::<u8>(4096);
        let (stderr_tx, stderr_rx) = gpu_ring_buffer::<u8>(4096);

        // 4. Register in process table, get PID
        let pid = PROCESS_TABLE.allocate(GpuProcessEntry {
            bytecode: bytecode.clone(),
            state,
            status: AtomicU32::new(PROCESS_RUNNING),
            exit_code: AtomicI32::new(0),
            stdin: stdin_rx,
            stdout: stdout_tx,
            stderr: stderr_tx,
        });

        // 5. Dispatch threadgroup to execute bytecode
        gpu::dispatch_process(pid, bytecode, state);

        Ok(Child {
            pid,
            stdin: Some(ChildStdin { buf: stdin_tx }),
            stdout: Some(ChildStdout { buf: stdout_rx }),
            stderr: Some(ChildStderr { buf: stderr_rx }),
        })
    }
}

impl Child {
    pub fn id(&self) -> u32 {
        self.pid.0
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        // Spin on process status (GPU-native wait)
        loop {
            let status = PROCESS_TABLE.get(self.pid).status.load(Ordering::Acquire);
            if status == PROCESS_EXITED {
                let code = PROCESS_TABLE.get(self.pid).exit_code.load(Ordering::Acquire);
                return Ok(ExitStatus(code));
            }
            gpu::yield_to_scheduler();
        }
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        let status = PROCESS_TABLE.get(self.pid).status.load(Ordering::Acquire);
        if status == PROCESS_EXITED {
            let code = PROCESS_TABLE.get(self.pid).exit_code.load(Ordering::Acquire);
            Ok(Some(ExitStatus(code)))
        } else {
            Ok(None)
        }
    }

    pub fn kill(&mut self) -> io::Result<()> {
        PROCESS_TABLE.get(self.pid).status.store(PROCESS_KILLED, Ordering::Release);
        Ok(())
    }
}

// Process lifecycle constants
const PROCESS_RUNNING: u32 = 1;
const PROCESS_EXITED: u32 = 2;
const PROCESS_KILLED: u32 = 3;
```

**Metal Kernel for Process Dispatch:**
```metal
// Each GPU process runs as a persistent threadgroup
kernel void gpu_process_executor(
    device GpuProcessEntry* process_table [[buffer(0)]],
    device BytecodePool* bytecode_pool [[buffer(1)]],
    uint pid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    device GpuProcessEntry& proc = process_table[pid];

    // Check if this process should run
    if (atomic_load_explicit(&proc.status, memory_order_acquire) != PROCESS_RUNNING) {
        return;
    }

    // Execute bytecode VM
    BytecodeVM vm;
    vm.init(&proc.state, &proc.stdin, &proc.stdout, &proc.stderr);
    vm.load(bytecode_pool, proc.bytecode);

    int exit_code = vm.run();

    // Only thread 0 updates exit status
    if (tid == 0) {
        atomic_store_explicit(&proc.exit_code, exit_code, memory_order_release);
        atomic_store_explicit(&proc.status, PROCESS_EXITED, memory_order_release);
    }
}
```

**Why GPU-Native Processes?**

1. **No CPU involvement** - Process spawning is pure GPU operation
2. **Massive parallelism** - Can run thousands of processes simultaneously
3. **Zero-copy I/O** - stdin/stdout are GPU ring buffers
4. **Unified memory** - All process state in GPU-accessible memory
5. **Sub-microsecond spawn** - No syscall overhead

---

## std::ptr

| Item | Strategy | Notes |
|------|----------|-------|
| `null()` | NATIVE | Null pointer |
| `null_mut()` | NATIVE | Null mut pointer |
| `NonNull` | NATIVE | Non-null wrapper |
| `Alignment` | NATIVE | Alignment type |
| `DynMetadata` | NATIVE | Metadata |
| `Pointee` | NATIVE | Pointee trait |
| `Thin` | NATIVE | Thin trait |
| `addr_eq()` | NATIVE | Address compare |
| `eq()` | NATIVE | Pointer compare |
| `hash()` | NATIVE | Pointer hash |
| `copy()` | NATIVE | Copy bytes |
| `copy_nonoverlapping()` | NATIVE | Non-overlap copy |
| `drop_in_place()` | GPU_IMPL | No-op on GPU |
| `read()` | NATIVE | Read value |
| `read_unaligned()` | NATIVE | Unaligned read |
| `read_volatile()` | NATIVE | Volatile read |
| `replace()` | NATIVE | Replace value |
| `swap()` | NATIVE | Swap values |
| `swap_nonoverlapping()` | NATIVE | Non-overlap swap |
| `write()` | NATIVE | Write value |
| `write_bytes()` | NATIVE | Write bytes |
| `write_unaligned()` | NATIVE | Unaligned write |
| `write_volatile()` | NATIVE | Volatile write |
| `slice_from_raw_parts()` | NATIVE | Create slice |
| `slice_from_raw_parts_mut()` | NATIVE | Create mut slice |
| `from_ref()` | NATIVE | Ref to pointer |
| `from_mut()` | NATIVE | Mut to pointer |
| `addr_of!()` | NATIVE | Address of |
| `addr_of_mut!()` | NATIVE | Mutable address |

---

## std::rc

| Item | Strategy | Notes |
|------|----------|-------|
| `Rc<T>` | AUTO_TRANSFORM | Automatically upgraded to atomic refcount (Arc-equivalent) |
| `Weak<T>` | AUTO_TRANSFORM | Same |

---

## std::result

| Item | Strategy | Notes |
|------|----------|-------|
| `Result<T, E>` | NATIVE | Core enum |
| All Result methods | NATIVE | Pure methods |
| `Ok`, `Err` | NATIVE | Enum variants |
| `IntoIter` | NATIVE | Result iterator |
| `Iter` | NATIVE | Result ref iterator |
| `IterMut` | NATIVE | Result mut iterator |

---

## std::slice

| Item | Strategy | Notes |
|------|----------|-------|
| `[T]` type | GPU_IMPL | Slice type |
| All slice methods | GPU_IMPL | May need bounds checks |
| Iteration methods | GPU_IMPL | Parallel iteration |
| `sort()` | GPU_IMPL | Bitonic or radix sort |
| `sort_by()` | GPU_IMPL | Parallel sort |
| `binary_search()` | GPU_IMPL | Parallel search |
| `from_raw_parts()` | NATIVE | Create from parts |
| `from_raw_parts_mut()` | NATIVE | Create mut from parts |
| `SliceIndex` | NATIVE | Index trait |
| `Iter` | GPU_IMPL | Parallel iterator |
| `IterMut` | GPU_IMPL | Parallel mut iterator |
| `Chunks*` | GPU_IMPL | Chunk iterators |
| `Split*` | GPU_IMPL | Split iterators |
| `Windows` | GPU_IMPL | Window iterator |

---

## std::str

| Item | Strategy | Notes |
|------|----------|-------|
| `str` type | GPU_IMPL | String slice |
| All str methods | GPU_IMPL | String operations |
| `from_utf8()` | NATIVE | UTF-8 validation |
| `from_utf8_unchecked()` | NATIVE | Unchecked UTF-8 |
| `from_utf8_mut()` | NATIVE | Mutable UTF-8 |
| `Utf8Error` | NATIVE | Error type |
| `ParseBoolError` | NATIVE | Error type |
| `Chars` | GPU_IMPL | Char iterator |
| `CharIndices` | GPU_IMPL | Indexed chars |
| `Bytes` | GPU_IMPL | Byte iterator |
| `Lines` | GPU_IMPL | Line iterator |
| `Split*` | GPU_IMPL | Split iterators |
| `Matches` | GPU_IMPL | Match iterator |
| `pattern::*` | GPU_IMPL | Pattern matching |

---

## std::string

| Item | Strategy | Notes |
|------|----------|-------|
| `String` | GPU_IMPL | Uses gpu_heap |
| All String methods | GPU_IMPL | String operations |
| `ToString` trait | GPU_IMPL | Conversion trait |
| `FromStr` trait | NATIVE | Parsing trait |
| `ParseError` | NATIVE | Error type |
| `FromUtf8Error` | NATIVE | Error type |
| `FromUtf16Error` | NATIVE | Error type |
| `Drain` | GPU_IMPL | Drain iterator |

---

## std::sync

| Item | Strategy | Notes |
|------|----------|-------|
| `Arc<T>` | GPU_IMPL | Atomic reference count |
| `Weak<T>` | GPU_IMPL | Weak reference |
| `Mutex<T>` | GPU_IMPL | Spinlock mutex |
| `MutexGuard<T>` | GPU_IMPL | Mutex guard |
| `RwLock<T>` | GPU_IMPL | Reader-writer lock |
| `RwLockReadGuard<T>` | GPU_IMPL | Read guard |
| `RwLockWriteGuard<T>` | GPU_IMPL | Write guard |
| `Condvar` | AUTO_TRANSFORM | Automatically transformed to threadgroup_barrier() |
| `Barrier` | GPU_IMPL | Maps to threadgroup_barrier |
| `BarrierWaitResult` | NATIVE | Barrier result |
| `Once` | GPU_IMPL | Atomic once |
| `OnceLock<T>` | GPU_IMPL | Lazy once |
| `OnceState` | NATIVE | Once state enum |
| `WaitTimeoutResult` | NATIVE | Wait result |
| `mpsc::*` | GPU_IMPL | Ring buffer channels |
| `atomic::*` | NATIVE | All atomic types |
| `PoisonError` | NATIVE | Poison error |
| `TryLockError` | NATIVE | Try lock error |
| `LockResult` | NATIVE | Lock result type |
| `TryLockResult` | NATIVE | Try lock result |
| `LazyLock<T>` | GPU_IMPL | Lazy initialization |
| `Exclusive<T>` | NATIVE | Exclusive wrapper |

**Mutex Implementation:**
```rust
// gpu_std/src/sync/mutex.rs

pub struct Mutex<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

impl<T> Mutex<T> {
    pub fn lock(&self) -> MutexGuard<'_, T> {
        // Spinlock - WARNING: SIMD divergence possible
        while self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            core::hint::spin_loop();
        }
        MutexGuard { mutex: self }
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        if self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            Some(MutexGuard { mutex: self })
        } else {
            None
        }
    }
}

// Warning emitted at compile time:
// "Warning: Mutex on GPU uses spinlock which can cause SIMD divergence.
//  Consider using atomics or threadgroup_barrier() instead."
```

**Channel Implementation:**
```rust
// gpu_std/src/sync/mpsc.rs

pub fn channel<T: Copy>() -> (Sender<T>, Receiver<T>) {
    let buf = GpuRingBuffer::new(256);
    (Sender { buf: buf.clone() }, Receiver { buf })
}

pub struct Sender<T> {
    buf: GpuRingBuffer<T>,
}

impl<T: Copy> Sender<T> {
    pub fn send(&self, val: T) -> Result<(), SendError<T>> {
        self.buf.push(val).map_err(|_| SendError(val))
    }
}

pub struct Receiver<T> {
    buf: GpuRingBuffer<T>,
}

impl<T: Copy> Receiver<T> {
    pub fn recv(&self) -> Result<T, RecvError> {
        self.buf.pop().ok_or(RecvError)
    }

    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.buf.try_pop().ok_or(TryRecvError::Empty)
    }
}
```

---

## std::task

| Item | Strategy | Notes |
|------|----------|-------|
| All items | AUTO_TRANSFORM | Async state machines transformed to parallel work queue dispatch |

---

## std::thread

| Item | Strategy | Notes |
|------|----------|-------|
| `spawn()` | GPU_IMPL | SHIM to parallel dispatch |
| `current()` | GPU_IMPL | Return thread ID wrapper |
| `sleep()` | AUTO_TRANSFORM | Automatically transformed to frame-based timing |
| `sleep_ms()` | AUTO_TRANSFORM | Same |
| `yield_now()` | GPU_IMPL | Maps to barrier |
| `panicking()` | GPU_IMPL | Check panic flag |
| `park()` | AUTO_TRANSFORM | Transformed to frame-based wait |
| `park_timeout()` | AUTO_TRANSFORM | Same |
| `Thread` | GPU_IMPL | Thread handle wrapper |
| `ThreadId` | GPU_IMPL | Thread ID type |
| `JoinHandle<T>` | GPU_IMPL | Completion handle |
| `Result` | NATIVE | Thread result alias |
| `Builder` | GPU_IMPL | Thread configuration |
| `Scope` | GPU_IMPL | Scoped threads |
| `ScopedJoinHandle<T>` | GPU_IMPL | Scoped handle |
| `scope()` | GPU_IMPL | Create scope |
| `LocalKey` | AUTO_TRANSFORM | Thread-indexed buffer (thread_id as array index) |
| `AccessError` | NATIVE | Error type |

**Thread Spawn Implementation:**
```rust
// gpu_std/src/thread.rs

// On GPU, "spawn" means "this work runs in parallel"
// All GPU threads are already running - we just assign work

pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // Allocate result slot
    let result_slot = RESULT_BUFFER.allocate();

    // Schedule work - on GPU this means:
    // "When a thread picks up this work item, execute f"
    WORK_QUEUE.push(WorkItem {
        func: Box::new(move || {
            let result = f();
            RESULT_BUFFER[result_slot] = Some(result);
        }),
        result_slot,
    });

    JoinHandle { result_slot }
}

pub struct JoinHandle<T> {
    result_slot: usize,
}

impl<T> JoinHandle<T> {
    pub fn join(self) -> thread::Result<T> {
        // Wait for completion
        while RESULT_BUFFER[self.result_slot].is_none() {
            gpu::barrier();  // Yield to other work
        }
        Ok(RESULT_BUFFER[self.result_slot].take().unwrap())
    }
}

// current() returns GPU thread info
pub fn current() -> Thread {
    Thread {
        id: ThreadId(gpu::thread_id()),
    }
}
```

---

## std::time

| Item | Strategy | Notes |
|------|----------|-------|
| `Duration` | NATIVE | Pure data type |
| `Instant` | GPU_IMPL | Frame-based instant |
| `SystemTime` | GPU_IMPL | Pre-computed from CPU |
| `UNIX_EPOCH` | GPU_IMPL | Constant |
| `SystemTimeError` | NATIVE | Error type |
| `TryFromFloatSecsError` | NATIVE | Error type |

**Time Implementation:**
```rust
// gpu_std/src/time.rs

// Duration is pure data - works as-is
pub use core::time::Duration;

// Instant uses GPU frame counter
pub struct Instant {
    frame: u32,
    subframe_ns: u32,
}

impl Instant {
    pub fn now() -> Instant {
        Instant {
            frame: GPU_FRAME_COUNTER.load(Ordering::Relaxed),
            subframe_ns: gpu::elapsed_ns(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        let now = Instant::now();
        let frames = now.frame - self.frame;
        let ns = frames as u64 * NS_PER_FRAME + now.subframe_ns as u64;
        Duration::from_nanos(ns)
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        // ...
    }
}

// SystemTime uses CPU-provided timestamp
pub struct SystemTime {
    secs: u64,
    nanos: u32,
}

impl SystemTime {
    pub fn now() -> SystemTime {
        // Read from CPU-provided timestamp buffer
        SystemTime {
            secs: SYSTEM_TIME_BUFFER.secs.load(Ordering::Relaxed),
            nanos: SYSTEM_TIME_BUFFER.nanos.load(Ordering::Relaxed),
        }
    }
}

pub const UNIX_EPOCH: SystemTime = SystemTime { secs: 0, nanos: 0 };
```

---

## std::vec

| Item | Strategy | Notes |
|------|----------|-------|
| `Vec<T>` | GPU_IMPL | Uses gpu_heap |
| All Vec methods | GPU_IMPL | Vector operations |
| `vec![]` macro | GPU_IMPL | Vector initialization |
| `Drain` | GPU_IMPL | Drain iterator |
| `IntoIter` | GPU_IMPL | Consuming iterator |
| `Splice` | GPU_IMPL | Splice iterator |
| `ExtractIf` | GPU_IMPL | Filter drain |

**Vec Implementation:**
```rust
// gpu_std/src/vec.rs

pub struct Vec<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

impl<T> Vec<T> {
    pub fn new() -> Self {
        Vec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        let layout = Layout::array::<T>(cap).unwrap();
        let ptr = gpu_heap::alloc(layout) as *mut T;
        Vec {
            ptr: NonNull::new(ptr).unwrap(),
            len: 0,
            cap,
        }
    }

    pub fn push(&mut self, val: T) {
        if self.len == self.cap {
            self.grow();
        }
        unsafe {
            self.ptr.as_ptr().add(self.len).write(val);
        }
        self.len += 1;
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_layout = Layout::array::<T>(new_cap).unwrap();
        let new_ptr = gpu_heap::realloc(
            self.ptr.as_ptr() as *mut u8,
            Layout::array::<T>(self.cap).unwrap(),
            new_layout.size(),
        ) as *mut T;
        self.ptr = NonNull::new(new_ptr).unwrap();
        self.cap = new_cap;
    }

    // ... rest of Vec implementation
}
```

---

## Summary

### Research Findings

**Key discoveries that enable near-complete GPU coverage:**

1. **MTLIOCommandQueue** - Metal 3 enables GPU-initiated file I/O. The GPU can issue read/write commands directly without CPU involvement, making filesystem operations truly GPU-native.

2. **Network I/O** - Requires CPU mediation (NIC hardware limitation). Network operations are auto-transformed to request queues processed by CPU after dispatch.

3. **Persistent Kernels** - GPU compute kernels can run indefinitely, enabling long-running services and continuous event loops without CPU re-dispatch.

### Total std Items: ~2,500

| Category | NATIVE | GPU_IMPL | AUTO_TRANSFORM | ERROR |
|----------|--------|----------|----------------|-------|
| **Core types** | 95% | 5% | 0% | 0% |
| **Collections** | 0% | 95% | 5% | 0% |
| **I/O** | 10% | 90% | 0% | 0% |
| **Filesystem** | 5% | 90% | 5% | 0% |
| **Formatting** | 30% | 70% | 0% | 0% |
| **Threading** | 5% | 50% | 45% | 0% |
| **Networking** | 50% | 0% | 50% | 0% |
| **Async** | 0% | 0% | 100% | 0% |
| **OS-specific** | 5% | 5% | 90% | 0% |
| **Process** | 5% | 95% | 0% | 0% |
| **Sync primitives** | 10% | 60% | 30% | 0% |

### Implementation Phases

| Phase | Module | Effort | Priority |
|-------|--------|--------|----------|
| **5** | Functions + intrinsics | MEDIUM | **NOW** |
| **6** | alloc (Vec, String, Box) | LOW | CRITICAL |
| **7** | fmt (println!, format!) | MEDIUM | HIGH |
| **8** | Auto-transform (async, Rc, net) | HIGH | CRITICAL |
| **9** | collections (HashMap, etc) | MEDIUM | HIGH |
| **10** | thread (spawn, current) | MEDIUM | HIGH |
| **11** | sync (Mutex, channels) | MEDIUM | MEDIUM |
| **12** | fs (already have base) | LOW | HIGH |
| **13** | io (Read, Write) | MEDIUM | HIGH |
| **14** | time (Instant, Duration) | LOW | MEDIUM |
| **15** | path, env | LOW | MEDIUM |
| **16** | error handling | LOW | HIGH |

### Auto-Transformed Items (User Code Unchanged)

| Item | Transformation |
|------|----------------|
| `async/await` | Parallel work queue dispatch |
| `TcpStream`, `UdpSocket` | Network request queue |
| `Rc<T>`, `Weak<T>` | Atomic refcount (Arc-like) |
| `Condvar` | Threadgroup barrier |
| `thread::sleep()` | Frame-based timing |
| `park()`, `park_timeout()` | Frame-based wait |
| `Future`, async types | State machine to work queue |
| `Any` trait | Compile-time TypeId registry in GPU buffer |
| `Backtrace` | Record call sites in debug buffer |
| `RefCell<T>` | Atomic borrow counter (AtomicRefCell pattern) |
| `LinkedList<T>` | Arena allocation with indices (no pointers) |
| `current_exe()`, `home_dir()` | Load-time snapshot stored in GPU buffer |
| `VaList` | Array-based argument dispatch |
| `unix/windows/macos/linux::*` | Syscall queue to CPU for execution |
| `fd::*` | Handle table in I/O subsystem |
| `catch_unwind()`, panic hooks | Error flag propagation |
| `LocalKey` | Thread-indexed buffer (thread_id as index) |
| `set_alloc_error_hook()` | Function pointer stored in GPU global |

### Items That Truly Cannot Work (ERROR)

With the research findings applied, **nearly zero items require ERROR status**. The remaining limitations are architectural constraints, not hard blockers:

1. **Unbounded recursion** - Limited stack per thread (can be mitigated with explicit stack in heap)
2. **Dynamic library loading at runtime** - Must be compiled in (but static linking covers all use cases)

**Everything else works via NATIVE, GPU_IMPL, or AUTO_TRANSFORM.**

Note: Environment variables (`std::env::var`) work via GPU buffer pre-populated at dispatch time.

### Total Effort Estimate

- **NATIVE (re-exports)**: 0 effort - just re-export from core/alloc
- **GPU_IMPL needed**: ~150 types/functions to implement
- **AUTO_TRANSFORM**: ~30 transformation patterns (compiler/transpiler work)
- **ERROR messages**: ~5 helpful error messages (down from ~50)

With our existing infrastructure (gpu_heap, filesystem, data structures), **most of the hard work is done**. We need to:

1. Create `gpu_std` crate structure
2. Wire allocator intrinsics
3. Wrap existing implementations in std-compatible API
4. Implement GPU-native process model (bytecode registry, process table, threadgroup dispatch)
5. Implement auto-transform patterns in WASM→GPU transpiler
6. Comprehensive testing

### Key Enablers

| Technology | Capability | Impact |
|------------|------------|--------|
| **MTLIOCommandQueue** | GPU-initiated file I/O | Filesystem ops without CPU |
| **GPU-native processes** | Threadgroups as processes | Process spawn without CPU |
| **Persistent kernels** | Indefinite GPU execution | Long-running services |
| **Unified memory** | Zero-copy GPU/CPU access | Process I/O, shared state |
