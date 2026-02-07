//! Type definitions for WASM translator
//!
//! THE GPU IS THE COMPUTER.
//! These types support efficient translation from stack-based WASM to register-based GPU bytecode.

use std::collections::HashMap;

/// Translation error
#[derive(Debug)]
pub enum TranslateError {
    /// WASM parsing error
    Parse(String),
    /// Unsupported WASM feature
    Unsupported(String),
    /// No entry point found
    NoEntryPoint,
    /// Invalid WASM structure
    Invalid(String),
    /// Stack underflow
    StackUnderflow,
    /// Out of registers
    OutOfRegisters,
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslateError::Parse(s) => write!(f, "Parse error: {}", s),
            TranslateError::Unsupported(s) => write!(f, "Unsupported: {}", s),
            TranslateError::NoEntryPoint => write!(f, "No gpu_main entry point found"),
            TranslateError::Invalid(s) => write!(f, "Invalid WASM: {}", s),
            TranslateError::StackUnderflow => write!(f, "Stack underflow"),
            TranslateError::OutOfRegisters => write!(f, "Out of registers"),
        }
    }
}

impl std::error::Error for TranslateError {}

/// Translator configuration
#[derive(Debug, Clone)]
pub struct TranslatorConfig {
    /// Initial memory pages (64KB each)
    pub memory_pages: u32,
    /// Maximum operand stack depth
    pub max_stack_depth: u32,
    /// Base address for globals in state memory
    pub globals_base: u32,
    /// Base address for linear memory in state memory
    pub memory_base: u32,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            memory_pages: 16,        // 16 pages = 1MB max addressable (but fits in smaller state)
            max_stack_depth: 64,
            // Memory layout (relative to state data start after bytecode):
            //   state[0] = return value (bytes 0-15)
            //   state[1-4] = params (bytes 16-79)
            //   state[8-71] = globals (bytes 128-1151, 1KB)
            //   state[72-327] = spill area (bytes 1152-5247, 4KB)
            //   state[328+] = linear memory (bytes 5248+, ~54KB)
            //
            // IMPORTANT: globals_base is a FLOAT4 INDEX (for LD/ST opcodes)
            // IMPORTANT: memory_base is a BYTE OFFSET (for LD4/ST4 opcodes)
            globals_base: 8,         // Globals at state[8] = byte 128 (FLOAT4 INDEX)
            memory_base: 5248,       // Linear memory at byte 5248 (BYTE OFFSET)
        }
    }
}

/// Block kind for control flow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Block,
    Loop,
    If,
    /// Inline function call - Return should jump here, not epilogue
    /// Also records result_count for proper stack handling
    InlineFunction,
}

/// Block context for nested control flow
#[derive(Debug, Clone)]
pub struct BlockContext {
    pub kind: BlockKind,
    pub start_label: Option<usize>,  // For loops
    pub else_label: Option<usize>,   // For if
    pub end_label: usize,
    pub stack_depth: usize,          // Stack depth at block entry
    /// For InlineFunction: number of results the function returns
    pub result_count: usize,
}

/// Function type signature
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// WASM value type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl From<wasmparser::ValType> for ValType {
    fn from(vt: wasmparser::ValType) -> Self {
        match vt {
            wasmparser::ValType::I32 => ValType::I32,
            wasmparser::ValType::I64 => ValType::I64,
            wasmparser::ValType::F32 => ValType::F32,
            wasmparser::ValType::F64 => ValType::F64,
            _ => ValType::I32, // Default for unsupported types
        }
    }
}

/// Parsed WASM function
#[derive(Debug, Clone)]
pub struct WasmFunction {
    pub type_idx: u32,
    pub locals: Vec<ValType>,
    pub code: Vec<u8>,  // Raw WASM bytecode for this function
}

/// GPU Intrinsic type (Phase 5 - Issue #178, Phase 6 - Issue #179, Phase 7 - Issue #180)
/// These map directly to GPU operations, not function calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuIntrinsic {
    /// Returns thread ID (GPU r1)
    ThreadId,
    /// Returns threadgroup size (GPU r2)
    ThreadgroupSize,
    /// Returns frame number (GPU r3)
    Frame,
    /// sin(x) - native GPU operation
    Sin,
    /// cos(x) - native GPU operation
    Cos,
    /// sqrt(x) - native GPU operation
    Sqrt,

    // ═══════════════════════════════════════════════════════════════════════════
    // ALLOCATOR INTRINSICS (Phase 6 - Issue #179)
    // THE GPU IS THE COMPUTER - GPU-resident memory allocator for Rust alloc crate
    // ═══════════════════════════════════════════════════════════════════════════

    /// __rust_alloc(size, align) -> ptr
    Alloc,
    /// __rust_dealloc(ptr, size, align)
    Dealloc,
    /// __rust_realloc(ptr, old_size, new_size, align) -> ptr
    Realloc,
    /// __rust_alloc_zeroed(size, align) -> ptr
    AllocZeroed,

    // ═══════════════════════════════════════════════════════════════════════════
    // DEBUG I/O INTRINSICS (Phase 7 - Issue #180)
    // THE GPU IS THE COMPUTER - debug output via ring buffer
    // Lock-free writes include thread ID for multi-thread debugging
    // ═══════════════════════════════════════════════════════════════════════════

    /// __gpu_debug_i32(value: i32) - debug print i32
    DebugI32,
    /// __gpu_debug_f32(value: f32) - debug print f32
    DebugF32,
    /// __gpu_debug_str(ptr: i32, len: i32) - debug print string
    DebugStr,
    /// __gpu_debug_bool(value: i32) - debug print bool (0=false, non-zero=true)
    DebugBool,
    /// __gpu_debug_newline() - debug newline marker
    DebugNewline,
    /// __gpu_debug_flush() - debug flush marker (indicates output complete)
    DebugFlush,

    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOMATIC CODE TRANSFORMATION INTRINSICS (Phase 8 - Issue #182)
    // THE GPU IS THE COMPUTER - transform CPU patterns to GPU-native equivalents
    // ═══════════════════════════════════════════════════════════════════════════

    /// Work queue push (async/await transformation)
    WorkPush,
    /// Work queue pop (async/await transformation)
    WorkPop,
    /// Threadgroup barrier (Condvar::wait transformation)
    Barrier,
    /// Frame-based timing (thread::sleep transformation)
    FrameWait,
    /// Spinlock acquire (Mutex::lock transformation)
    Spinlock,
    /// Spinlock release (Mutex::unlock transformation)
    Spinunlock,
    /// Atomic increment for Rc::clone
    RcClone,
    /// Atomic decrement for Rc::drop
    RcDrop,
    /// Queue I/O request
    RequestQueue,
    /// Poll I/O request status
    RequestPoll,

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDERING INTRINSICS (Phase 9 - GPU App Framework)
    // THE GPU IS THE COMPUTER - emit graphics primitives from WASM apps
    // ═══════════════════════════════════════════════════════════════════════════

    /// emit_quad(x, y, w, h, color) - emit colored rectangle
    /// color is packed RGBA (0xRRGGBBAA)
    EmitQuad,
    /// get_cursor_x() -> f32 - mouse X position
    GetCursorX,
    /// get_cursor_y() -> f32 - mouse Y position
    GetCursorY,
    /// get_mouse_down() -> i32 - 1 if mouse button pressed, 0 otherwise
    GetMouseDown,
    /// get_time() -> f32 - elapsed time in seconds
    GetTime,
    /// get_screen_width() -> f32 - screen width in pixels
    GetScreenWidth,
    /// get_screen_height() -> f32 - screen height in pixels
    GetScreenHeight,

    // ═══════════════════════════════════════════════════════════════════════════
    // WASI INTRINSICS (Issue #207 - GPU-Native WASI)
    // THE GPU IS THE COMPUTER - WASI system calls implemented on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// fd_write(fd, iovs_ptr, iovs_len, nwritten_ptr) -> errno
    /// Writes to stdout/stderr via debug buffer, returns 0 on success
    WasiFdWrite,
    /// fd_read(fd, iovs_ptr, iovs_len, nread_ptr) -> errno
    /// Always returns EBADF (9) - no input on GPU
    WasiFdRead,
    /// proc_exit(code) - halt execution with exit code
    WasiProcExit,
    /// environ_sizes_get(count_ptr, buf_size_ptr) -> errno
    /// Returns 0 count, 0 size (no environment on GPU)
    WasiEnvironSizesGet,
    /// environ_get(environ_ptr, buf_ptr) -> errno
    /// Returns success (empty environment)
    WasiEnvironGet,
    /// args_sizes_get(count_ptr, buf_size_ptr) -> errno
    /// Returns 0 count, 0 size (no args on GPU)
    WasiArgsSizesGet,
    /// args_get(argv_ptr, buf_ptr) -> errno
    /// Returns success (empty args)
    WasiArgsGet,
    /// clock_time_get(clock_id, precision, time_ptr) -> errno
    /// Returns frame count as nanoseconds
    WasiClockTimeGet,
    /// random_get(buf_ptr, buf_len) -> errno
    /// Uses GPU thread ID as pseudo-random seed
    WasiRandomGet,

    // ═══════════════════════════════════════════════════════════════════════════
    // PANIC HANDLING INTRINSICS (Issue #209 - GPU-Native Panic)
    // THE GPU IS THE COMPUTER - panic handling on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// panic(msg_ptr, msg_len) - write message to debug buffer and halt
    Panic,
    /// unreachable() - halt with unreachable error
    Unreachable,
}

/// Imported function (Phase 5 - Issue #178)
#[derive(Debug, Clone)]
pub struct ImportedFunc {
    /// Module name (e.g., "env")
    pub module: String,
    /// Function name (e.g., "sin")
    pub name: String,
    /// Type index
    pub type_idx: u32,
    /// GPU intrinsic mapping (if applicable)
    pub intrinsic: Option<GpuIntrinsic>,
}

impl ImportedFunc {
    /// Try to map import to GPU intrinsic
    pub fn from_import(module: &str, name: &str, type_idx: u32) -> Self {
        let intrinsic = if module == "env" {
            match name {
                "thread_id" => Some(GpuIntrinsic::ThreadId),
                "threadgroup_size" => Some(GpuIntrinsic::ThreadgroupSize),
                "frame" => Some(GpuIntrinsic::Frame),
                "sin" | "__gpu_sin" => Some(GpuIntrinsic::Sin),
                "cos" | "__gpu_cos" => Some(GpuIntrinsic::Cos),
                "sqrt" | "__gpu_sqrt" => Some(GpuIntrinsic::Sqrt),
                // Phase 6 - Issue #179: Rust allocator intrinsics
                "__rust_alloc" => Some(GpuIntrinsic::Alloc),
                "__rust_dealloc" => Some(GpuIntrinsic::Dealloc),
                "__rust_realloc" => Some(GpuIntrinsic::Realloc),
                "__rust_alloc_zeroed" => Some(GpuIntrinsic::AllocZeroed),
                // Phase 7 - Issue #180: Debug I/O intrinsics
                "__gpu_debug_i32" => Some(GpuIntrinsic::DebugI32),
                "__gpu_debug_f32" => Some(GpuIntrinsic::DebugF32),
                "__gpu_debug_str" => Some(GpuIntrinsic::DebugStr),
                "__gpu_debug_bool" => Some(GpuIntrinsic::DebugBool),
                "__gpu_debug_newline" => Some(GpuIntrinsic::DebugNewline),
                "__gpu_debug_flush" => Some(GpuIntrinsic::DebugFlush),
                // Phase 8 - Issue #182: Automatic code transformation intrinsics
                "__gpu_work_push" => Some(GpuIntrinsic::WorkPush),
                "__gpu_work_pop" => Some(GpuIntrinsic::WorkPop),
                "__gpu_barrier" => Some(GpuIntrinsic::Barrier),
                "__gpu_frame_wait" => Some(GpuIntrinsic::FrameWait),
                "__gpu_spinlock" => Some(GpuIntrinsic::Spinlock),
                "__gpu_spinunlock" => Some(GpuIntrinsic::Spinunlock),
                "__gpu_rc_clone" => Some(GpuIntrinsic::RcClone),
                "__gpu_rc_drop" => Some(GpuIntrinsic::RcDrop),
                "__gpu_request_queue" => Some(GpuIntrinsic::RequestQueue),
                "__gpu_request_poll" => Some(GpuIntrinsic::RequestPoll),
                // Rust std library transformations (demangled names)
                // Note: Real WASM from Rust will have mangled names - these are simplified
                "mutex_lock" | "__sync_mutex_lock" => Some(GpuIntrinsic::Spinlock),
                "mutex_unlock" | "__sync_mutex_unlock" => Some(GpuIntrinsic::Spinunlock),
                "condvar_wait" | "__sync_condvar_wait" => Some(GpuIntrinsic::Barrier),
                "thread_sleep" | "__thread_sleep" => Some(GpuIntrinsic::FrameWait),
                "rc_clone" | "__rc_clone" => Some(GpuIntrinsic::RcClone),
                "rc_drop" | "__rc_drop" => Some(GpuIntrinsic::RcDrop),
                // Phase 9 - Rendering intrinsics for GPU apps
                "emit_quad" | "__gpu_emit_quad" => Some(GpuIntrinsic::EmitQuad),
                "get_cursor_x" | "__gpu_cursor_x" => Some(GpuIntrinsic::GetCursorX),
                "get_cursor_y" | "__gpu_cursor_y" => Some(GpuIntrinsic::GetCursorY),
                "get_mouse_down" | "__gpu_mouse_down" => Some(GpuIntrinsic::GetMouseDown),
                "get_time" | "__gpu_time" => Some(GpuIntrinsic::GetTime),
                "get_screen_width" | "__gpu_screen_width" => Some(GpuIntrinsic::GetScreenWidth),
                "get_screen_height" | "__gpu_screen_height" => Some(GpuIntrinsic::GetScreenHeight),
                // Panic handling (Issue #209)
                "__rust_panic" | "rust_panic" | "__rust_start_panic" => Some(GpuIntrinsic::Panic),
                "abort" | "__abort" => Some(GpuIntrinsic::Unreachable),
                _ => None,
            }
        } else if module == "wasi_snapshot_preview1" {
            // WASI imports (Issue #207 - GPU-Native WASI)
            match name {
                "fd_write" => Some(GpuIntrinsic::WasiFdWrite),
                "fd_read" => Some(GpuIntrinsic::WasiFdRead),
                "proc_exit" => Some(GpuIntrinsic::WasiProcExit),
                "environ_sizes_get" => Some(GpuIntrinsic::WasiEnvironSizesGet),
                "environ_get" => Some(GpuIntrinsic::WasiEnvironGet),
                "args_sizes_get" => Some(GpuIntrinsic::WasiArgsSizesGet),
                "args_get" => Some(GpuIntrinsic::WasiArgsGet),
                "clock_time_get" => Some(GpuIntrinsic::WasiClockTimeGet),
                "random_get" => Some(GpuIntrinsic::WasiRandomGet),
                _ => None,
            }
        } else {
            None
        };

        Self {
            module: module.to_string(),
            name: name.to_string(),
            type_idx,
            intrinsic,
        }
    }
}

/// Data segment - static data to be copied into linear memory at load time
/// Issue #255: WASM Data sections must be parsed and copied to GPU
#[derive(Debug, Clone)]
pub struct DataSegment {
    /// Offset in linear memory where data should be written
    pub offset: u32,
    /// The actual bytes to copy
    pub data: Vec<u8>,
}

/// Global variable with initial value
/// Issue #255: WASM Global sections must be parsed for proper initialization
#[derive(Debug, Clone)]
pub struct GlobalDef {
    /// Type of the global
    pub ty: ValType,
    /// Is this global mutable?
    pub mutable: bool,
    /// Initial value (as i64 to hold any value type)
    pub init_value: i64,
}

/// Parsed WASM module
#[derive(Debug, Default)]
pub struct WasmModule {
    pub types: Vec<FuncType>,
    /// Imported functions (Phase 5 - Issue #178)
    /// Function indices 0..imports.len() are imports
    pub imports: Vec<ImportedFunc>,
    pub functions: Vec<u32>,  // Function index -> type index (for defined functions)
    pub codes: Vec<WasmFunction>,
    pub exports: HashMap<String, u32>,  // Name -> function index
    pub globals: Vec<ValType>,
    pub memory_pages: u32,
    /// Function table for call_indirect (Issue #189)
    /// Maps table_index -> function_index
    /// Built from Element section with active segments
    pub func_table: Vec<Option<u32>>,
    /// Issue #255: Data segments for static data initialization
    pub data_segments: Vec<DataSegment>,
    /// Issue #255: Global definitions with initial values
    pub global_defs: Vec<GlobalDef>,
}

impl WasmModule {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of imported functions
    pub fn import_count(&self) -> u32 {
        self.imports.len() as u32
    }

    /// Check if function index is an import
    pub fn is_import(&self, func_idx: u32) -> bool {
        (func_idx as usize) < self.imports.len()
    }

    /// Get imported function by index
    pub fn get_import(&self, func_idx: u32) -> Option<&ImportedFunc> {
        self.imports.get(func_idx as usize)
    }

    /// Get defined function's type index (adjusting for import count)
    pub fn get_defined_func_type_idx(&self, func_idx: u32) -> Option<u32> {
        let adjusted = func_idx as usize - self.imports.len();
        self.functions.get(adjusted).copied()
    }

    /// Get type index for any function (import or defined)
    /// Used for runtime type checking in call_indirect
    pub fn get_func_type_idx(&self, func_idx: u32) -> Option<u32> {
        if self.is_import(func_idx) {
            Some(self.imports[func_idx as usize].type_idx)
        } else {
            self.get_defined_func_type_idx(func_idx)
        }
    }

    /// Get function type by function index (works for both imports and defined)
    pub fn get_func_type(&self, func_idx: u32) -> Option<&FuncType> {
        let type_idx = if self.is_import(func_idx) {
            self.imports[func_idx as usize].type_idx
        } else {
            *self.functions.get(func_idx as usize - self.imports.len())?
        };
        self.types.get(type_idx as usize)
    }

    /// Find exported function by name
    pub fn find_export(&self, name: &str) -> Option<u32> {
        self.exports.get(name).copied()
    }
}
