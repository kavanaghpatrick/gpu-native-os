# PRD: Comprehensive Bytecode Testing Infrastructure (Issue #190)

## Problem Statement

Our WASM→GPU bytecode translator has **6% test coverage** (7 of 112 opcodes tested). We repeatedly discover bugs through manual debugging rather than automated testing:

- **Denormal flushing**: Integer constants <8M became 0 when stored as float bits
- **PACK2 alignment**: Float2 pair operations corrupted register state
- **Register exhaustion**: Deep stack caused OutOfRegisters without graceful spill
- **Block boundaries**: Register recycling at wrong point caused value corruption

**Cost**: Each bug takes 2-4 hours to diagnose. With proper testing, we'd catch them in seconds.

## Goals

1. **Catch bugs at translation time** - Before GPU execution
2. **Catch bugs at execution time** - Differential testing against reference
3. **Prevent regressions** - Every bug becomes a test case
4. **Enable confident refactoring** - Change code, run tests, ship

## Non-Goals

- Performance optimization (separate concern)
- Visual rendering tests (separate system)
- Fuzzing infrastructure (future phase)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   LAYER 1    │    │   LAYER 2    │    │   LAYER 3    │      │
│  │  Unit Tests  │    │ Integration  │    │ Differential │      │
│  │              │    │    Tests     │    │   Testing    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 TEST HARNESS CORE                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │WasmBuilder  │  │BytecodeExec │  │CpuReference │       │  │
│  │  │             │  │             │  │             │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 DIAGNOSTICS                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │InstrTrace   │  │RegSnapshot  │  │MemoryDump  │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Phase 1: Test Harness Core

#### 1.1 WASM Builder (Enhanced)

```rust
// tests/harness/wasm_builder.rs

/// Build WASM modules programmatically for testing
pub struct WasmBuilder {
    types: Vec<FuncType>,
    imports: Vec<Import>,
    functions: Vec<FuncBody>,
    exports: Vec<Export>,
    memory: Option<MemoryType>,
    globals: Vec<Global>,
}

impl WasmBuilder {
    pub fn new() -> Self { ... }

    /// Add function type signature
    pub fn add_type(&mut self, params: &[ValType], results: &[ValType]) -> u32 { ... }

    /// Add function body with locals
    pub fn add_func(&mut self, type_idx: u32, locals: &[ValType], body: &[u8]) -> u32 { ... }

    /// Export function by name
    pub fn export_func(&mut self, name: &str, func_idx: u32) { ... }

    /// Build final WASM binary
    pub fn build(&self) -> Vec<u8> { ... }

    // Convenience builders for common patterns

    /// Build: fn() -> i32 { body }
    pub fn i32_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[], &[ValType::I32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(i32) -> i32 { body }
    pub fn i32_unary_func(body: &[u8]) -> Vec<u8> { ... }

    /// Build: fn(i32, i32) -> i32 { body }
    pub fn i32_binary_func(body: &[u8]) -> Vec<u8> { ... }

    /// Build: fn() -> i64 { body }
    pub fn i64_func(body: &[u8]) -> Vec<u8> { ... }

    /// Build: fn(i64, i64) -> i64 { body }
    pub fn i64_binary_func(body: &[u8]) -> Vec<u8> { ... }

    /// Build: fn() -> f32 { body }
    pub fn f32_func(body: &[u8]) -> Vec<u8> { ... }

    /// Build: fn() -> f64 { body }
    pub fn f64_func(body: &[u8]) -> Vec<u8> { ... }
}

// WASM instruction helpers (raw bytes)
pub mod wasm_ops {
    // Constants
    pub fn i32_const(val: i32) -> Vec<u8> {
        let mut v = vec![0x41]; // i32.const
        leb128_signed(&mut v, val as i64);
        v
    }

    pub fn i64_const(val: i64) -> Vec<u8> {
        let mut v = vec![0x42]; // i64.const
        leb128_signed(&mut v, val);
        v
    }

    pub fn f32_const(val: f32) -> Vec<u8> {
        let mut v = vec![0x43]; // f32.const
        v.extend_from_slice(&val.to_le_bytes());
        v
    }

    pub fn f64_const(val: f64) -> Vec<u8> {
        let mut v = vec![0x44]; // f64.const
        v.extend_from_slice(&val.to_le_bytes());
        v
    }

    // Arithmetic (i32)
    pub const I32_ADD: u8 = 0x6A;
    pub const I32_SUB: u8 = 0x6B;
    pub const I32_MUL: u8 = 0x6C;
    pub const I32_DIV_S: u8 = 0x6D;
    pub const I32_DIV_U: u8 = 0x6E;
    pub const I32_REM_S: u8 = 0x6F;
    pub const I32_REM_U: u8 = 0x70;

    // Bitwise (i32)
    pub const I32_AND: u8 = 0x71;
    pub const I32_OR: u8 = 0x72;
    pub const I32_XOR: u8 = 0x73;
    pub const I32_SHL: u8 = 0x74;
    pub const I32_SHR_S: u8 = 0x75;
    pub const I32_SHR_U: u8 = 0x76;
    pub const I32_ROTL: u8 = 0x77;
    pub const I32_ROTR: u8 = 0x78;

    // Comparison (i32)
    pub const I32_EQZ: u8 = 0x45;
    pub const I32_EQ: u8 = 0x46;
    pub const I32_NE: u8 = 0x47;
    pub const I32_LT_S: u8 = 0x48;
    pub const I32_LT_U: u8 = 0x49;
    pub const I32_GT_S: u8 = 0x4A;
    pub const I32_GT_U: u8 = 0x4B;
    pub const I32_LE_S: u8 = 0x4C;
    pub const I32_LE_U: u8 = 0x4D;
    pub const I32_GE_S: u8 = 0x4E;
    pub const I32_GE_U: u8 = 0x4F;

    // i64 operations
    pub const I64_ADD: u8 = 0x7C;
    pub const I64_SUB: u8 = 0x7D;
    pub const I64_MUL: u8 = 0x7E;
    pub const I64_DIV_S: u8 = 0x7F;
    pub const I64_DIV_U: u8 = 0x80;
    pub const I64_REM_S: u8 = 0x81;
    pub const I64_REM_U: u8 = 0x82;
    pub const I64_AND: u8 = 0x83;
    pub const I64_OR: u8 = 0x84;
    pub const I64_XOR: u8 = 0x85;
    pub const I64_SHL: u8 = 0x86;
    pub const I64_SHR_S: u8 = 0x87;
    pub const I64_SHR_U: u8 = 0x88;

    // f32 operations
    pub const F32_ADD: u8 = 0x92;
    pub const F32_SUB: u8 = 0x93;
    pub const F32_MUL: u8 = 0x94;
    pub const F32_DIV: u8 = 0x95;
    pub const F32_SQRT: u8 = 0x91;

    // f64 operations
    pub const F64_ADD: u8 = 0xA0;
    pub const F64_SUB: u8 = 0xA1;
    pub const F64_MUL: u8 = 0xA2;
    pub const F64_DIV: u8 = 0xA3;
    pub const F64_SQRT: u8 = 0x9F;

    // Conversions
    pub const I32_WRAP_I64: u8 = 0xA7;
    pub const I64_EXTEND_I32_S: u8 = 0xAC;
    pub const I64_EXTEND_I32_U: u8 = 0xAD;
    pub const F32_CONVERT_I32_S: u8 = 0xB2;
    pub const F32_CONVERT_I32_U: u8 = 0xB3;
    pub const F64_CONVERT_I32_S: u8 = 0xB7;
    pub const F64_CONVERT_I32_U: u8 = 0xB8;

    // Control flow
    pub const END: u8 = 0x0B;
    pub const BLOCK: u8 = 0x02;
    pub const LOOP: u8 = 0x03;
    pub const IF: u8 = 0x04;
    pub const ELSE: u8 = 0x05;
    pub const BR: u8 = 0x0C;
    pub const BR_IF: u8 = 0x0D;

    // Local/Global
    pub const LOCAL_GET: u8 = 0x20;
    pub const LOCAL_SET: u8 = 0x21;
    pub const LOCAL_TEE: u8 = 0x22;
    pub const GLOBAL_GET: u8 = 0x23;
    pub const GLOBAL_SET: u8 = 0x24;

    // Memory
    pub const I32_LOAD: u8 = 0x28;
    pub const I64_LOAD: u8 = 0x29;
    pub const F32_LOAD: u8 = 0x2A;
    pub const F64_LOAD: u8 = 0x2B;
    pub const I32_STORE: u8 = 0x36;
    pub const I64_STORE: u8 = 0x37;
    pub const F32_STORE: u8 = 0x38;
    pub const F64_STORE: u8 = 0x39;
}
```

#### 1.2 Bytecode Executor

```rust
// tests/harness/executor.rs

/// Execute bytecode on GPU and return result
pub struct BytecodeExecutor {
    device: metal::Device,
    system: GpuAppSystem,
}

impl BytecodeExecutor {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device")?;
        let system = GpuAppSystem::new(&device)?;
        Ok(Self { device, system })
    }

    /// Translate WASM and execute on GPU, return i32 result
    pub fn run_wasm_i32(&mut self, wasm: &[u8]) -> Result<i32, ExecutionError> {
        let translator = WasmTranslator::new(TranslatorConfig::default());
        let bytecode = translator.translate(wasm)?;
        self.run_bytecode_i32(&bytecode)
    }

    /// Execute raw bytecode, return i32 result
    pub fn run_bytecode_i32(&mut self, bytecode: &[u8]) -> Result<i32, ExecutionError> {
        let slot = self.system.launch_by_type(app_type::BYTECODE)?;
        self.system.write_app_state(slot, bytecode);
        self.system.run_frame();
        self.system.read_bytecode_result(slot)
            .ok_or(ExecutionError::NoResult)
    }

    /// Execute and return i64 result (read as two i32s)
    pub fn run_wasm_i64(&mut self, wasm: &[u8]) -> Result<i64, ExecutionError> {
        let translator = WasmTranslator::new(TranslatorConfig::default());
        let bytecode = translator.translate(wasm)?;
        self.run_bytecode_i64(&bytecode)
    }

    pub fn run_bytecode_i64(&mut self, bytecode: &[u8]) -> Result<i64, ExecutionError> {
        let slot = self.system.launch_by_type(app_type::BYTECODE)?;
        self.system.write_app_state(slot, bytecode);
        self.system.run_frame();

        // i64 stored as (lo, hi) in state[0], state[1]
        let lo = self.system.read_bytecode_result_at(slot, 0)? as u32;
        let hi = self.system.read_bytecode_result_at(slot, 1)? as u32;
        Ok(((hi as i64) << 32) | (lo as i64))
    }

    /// Execute and return f32 result
    pub fn run_wasm_f32(&mut self, wasm: &[u8]) -> Result<f32, ExecutionError> {
        let result_bits = self.run_wasm_i32(wasm)? as u32;
        Ok(f32::from_bits(result_bits))
    }

    /// Execute and return f64 result
    pub fn run_wasm_f64(&mut self, wasm: &[u8]) -> Result<f64, ExecutionError> {
        let result_bits = self.run_wasm_i64(wasm)? as u64;
        Ok(f64::from_bits(result_bits))
    }

    /// Get bytecode without executing (for inspection)
    pub fn translate_only(&self, wasm: &[u8]) -> Result<Vec<u8>, TranslateError> {
        let translator = WasmTranslator::new(TranslatorConfig::default());
        translator.translate(wasm)
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    TranslationFailed(TranslateError),
    LaunchFailed(String),
    NoResult,
    Timeout,
}
```

#### 1.3 CPU Reference Implementation

```rust
// tests/harness/cpu_reference.rs

/// Execute WASM on wasmtime for reference comparison
pub struct CpuReference {
    engine: wasmtime::Engine,
}

impl CpuReference {
    pub fn new() -> Self {
        Self { engine: wasmtime::Engine::default() }
    }

    /// Execute WASM and return i32 result
    pub fn run_i32(&self, wasm: &[u8]) -> Result<i32, String> {
        let module = wasmtime::Module::new(&self.engine, wasm)
            .map_err(|e| format!("Module creation failed: {e}"))?;
        let mut store = wasmtime::Store::new(&self.engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[])
            .map_err(|e| format!("Instantiation failed: {e}"))?;

        // Try common entry point names
        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Ok(func) = instance.get_typed_func::<(), i32>(&mut store, name) {
                return func.call(&mut store, ())
                    .map_err(|e| format!("Call failed: {e}"));
            }
        }
        Err("No entry point found".to_string())
    }

    /// Execute and return i64 result
    pub fn run_i64(&self, wasm: &[u8]) -> Result<i64, String> { ... }

    /// Execute and return f32 result
    pub fn run_f32(&self, wasm: &[u8]) -> Result<f32, String> { ... }

    /// Execute and return f64 result
    pub fn run_f64(&self, wasm: &[u8]) -> Result<f64, String> { ... }
}
```

#### 1.4 Differential Testing Framework

```rust
// tests/harness/differential.rs

/// Compare GPU execution against CPU reference
pub struct DifferentialTester {
    gpu: BytecodeExecutor,
    cpu: CpuReference,
}

impl DifferentialTester {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            gpu: BytecodeExecutor::new()?,
            cpu: CpuReference::new(),
        })
    }

    /// Assert GPU and CPU produce same i32 result
    pub fn assert_same_i32(&mut self, wasm: &[u8], context: &str) {
        let cpu_result = self.cpu.run_i32(wasm)
            .expect(&format!("CPU execution failed: {context}"));
        let gpu_result = self.gpu.run_wasm_i32(wasm)
            .expect(&format!("GPU execution failed: {context}"));

        assert_eq!(
            cpu_result, gpu_result,
            "Mismatch for {context}: CPU={cpu_result}, GPU={gpu_result}"
        );
    }

    /// Assert GPU and CPU produce same i64 result
    pub fn assert_same_i64(&mut self, wasm: &[u8], context: &str) { ... }

    /// Assert GPU and CPU produce same f32 result (with epsilon)
    pub fn assert_same_f32(&mut self, wasm: &[u8], epsilon: f32, context: &str) {
        let cpu_result = self.cpu.run_f32(wasm)
            .expect(&format!("CPU execution failed: {context}"));
        let gpu_result = self.gpu.run_wasm_f32(wasm)
            .expect(&format!("GPU execution failed: {context}"));

        let diff = (cpu_result - gpu_result).abs();
        assert!(
            diff <= epsilon,
            "Mismatch for {context}: CPU={cpu_result}, GPU={gpu_result}, diff={diff}"
        );
    }

    /// Assert GPU and CPU produce same f64 result (with epsilon)
    pub fn assert_same_f64(&mut self, wasm: &[u8], epsilon: f64, context: &str) { ... }
}

/// Convenience macro for differential testing
#[macro_export]
macro_rules! assert_wasm_eq {
    ($wasm:expr, i32) => {{
        let mut tester = DifferentialTester::new().unwrap();
        tester.assert_same_i32($wasm, stringify!($wasm));
    }};
    ($wasm:expr, i64) => {{
        let mut tester = DifferentialTester::new().unwrap();
        tester.assert_same_i64($wasm, stringify!($wasm));
    }};
    ($wasm:expr, f32, $eps:expr) => {{
        let mut tester = DifferentialTester::new().unwrap();
        tester.assert_same_f32($wasm, $eps, stringify!($wasm));
    }};
}
```

---

### Phase 2: Opcode Unit Tests

#### 2.1 Test Structure

```rust
// tests/opcodes/mod.rs

mod i32_arithmetic;
mod i32_bitwise;
mod i32_comparison;
mod i64_arithmetic;
mod i64_bitwise;
mod i64_comparison;
mod f32_arithmetic;
mod f32_comparison;
mod f64_arithmetic;
mod conversions;
mod memory;
mod control_flow;
mod atomics;

/// Standard edge case values for i32 testing
pub const I32_EDGE_CASES: &[i32] = &[
    0,
    1,
    -1,
    2,
    -2,
    7,           // Small positive
    -7,          // Small negative
    64,          // Denormal boundary
    127,         // Max i8
    -128,        // Min i8
    255,         // Max u8
    256,         // Just above u8
    1000,        // Common value
    32767,       // Max i16
    -32768,      // Min i16
    65535,       // Max u16
    8_388_607,   // Max safe integer in f32 mantissa (2^23 - 1)
    8_388_608,   // DENORMAL BOUNDARY - 2^23
    16_777_215,  // Max precise f32 integer (2^24 - 1)
    i32::MAX,    // 2147483647
    i32::MIN,    // -2147483648
    i32::MAX - 1,
    i32::MIN + 1,
];

/// Standard edge case values for i64 testing
pub const I64_EDGE_CASES: &[i64] = &[
    0,
    1,
    -1,
    i32::MAX as i64,
    i32::MIN as i64,
    i32::MAX as i64 + 1,  // Just beyond i32
    i32::MIN as i64 - 1,
    u32::MAX as i64,      // Max u32
    u32::MAX as i64 + 1,
    i64::MAX,
    i64::MIN,
    i64::MAX - 1,
    i64::MIN + 1,
    0x0000_0001_0000_0000_i64,  // Bit 32 set
    0x8000_0000_0000_0000_u64 as i64,  // Sign bit only
];

/// Standard edge case values for f32 testing
pub const F32_EDGE_CASES: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    f32::MIN_POSITIVE,     // Smallest positive normal
    f32::EPSILON,          // Smallest difference from 1.0
    f32::MAX,
    f32::MIN,              // Most negative
    f32::INFINITY,
    f32::NEG_INFINITY,
    // Note: NaN handling may differ between CPU and GPU
];

/// Standard edge case values for f64 testing
pub const F64_EDGE_CASES: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    f64::MIN_POSITIVE,
    f64::EPSILON,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
];
```

#### 2.2 i32 Arithmetic Tests

```rust
// tests/opcodes/i32_arithmetic.rs

use crate::harness::*;
use crate::opcodes::I32_EDGE_CASES;

/// Test i32.add with all edge case combinations
#[test]
fn test_i32_add_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(a),
                wasm_ops::i32_const(b),
                wasm_ops::I32_ADD,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.add({a}, {b})"));
        }
    }
}

/// Test i32.sub with overflow cases
#[test]
fn test_i32_sub_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(a),
                wasm_ops::i32_const(b),
                wasm_ops::I32_SUB,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.sub({a}, {b})"));
        }
    }
}

/// Test i32.mul with overflow wrapping
#[test]
fn test_i32_mul_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(a),
                wasm_ops::i32_const(b),
                wasm_ops::I32_MUL,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.mul({a}, {b})"));
        }
    }
}

/// Test i32.div_s with special cases (div by zero, MIN/-1)
#[test]
fn test_i32_div_s_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    // Skip division by zero (undefined behavior)
    let divisors: Vec<i32> = I32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I32_EDGE_CASES {
        for &b in &divisors {
            // Skip MIN / -1 (overflow, undefined in WASM)
            if a == i32::MIN && b == -1 {
                continue;
            }

            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(a),
                wasm_ops::i32_const(b),
                wasm_ops::I32_DIV_S,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.div_s({a}, {b})"));
        }
    }
}

/// Test i32.div_u (unsigned division)
#[test]
fn test_i32_div_u_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    let divisors: Vec<i32> = I32_EDGE_CASES.iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();

    for &a in I32_EDGE_CASES {
        for &b in &divisors {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(a),
                wasm_ops::i32_const(b),
                wasm_ops::I32_DIV_U,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.div_u({a}, {b})"));
        }
    }
}

/// Test i32.rem_s (signed remainder)
#[test]
fn test_i32_rem_s_exhaustive() {
    // Similar to div_s with same special cases
    ...
}

/// Test i32.rem_u (unsigned remainder)
#[test]
fn test_i32_rem_u_exhaustive() {
    ...
}
```

#### 2.3 i32 Bitwise Tests

```rust
// tests/opcodes/i32_bitwise.rs

/// Test i32.and
#[test]
fn test_i32_and_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    for &a in I32_EDGE_CASES {
        for &b in I32_EDGE_CASES {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(a),
                wasm_ops::i32_const(b),
                wasm_ops::I32_AND,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.and({a:#x}, {b:#x})"));
        }
    }
}

/// Test i32.shl with shift amounts 0-31 and edge cases
#[test]
fn test_i32_shl_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    let shift_amounts = [0, 1, 7, 8, 15, 16, 23, 24, 31, 32, 33, 63, 64];

    for &val in I32_EDGE_CASES {
        for &shift in &shift_amounts {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(val),
                wasm_ops::i32_const(shift),
                wasm_ops::I32_SHL,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.shl({val:#x}, {shift})"));
        }
    }
}

/// Test i32.shr_s (arithmetic shift right)
#[test]
fn test_i32_shr_s_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    let shift_amounts = [0, 1, 7, 8, 15, 16, 23, 24, 31, 32, 33];

    for &val in I32_EDGE_CASES {
        for &shift in &shift_amounts {
            let wasm = WasmBuilder::i32_binary_func(&[
                wasm_ops::i32_const(val),
                wasm_ops::i32_const(shift),
                wasm_ops::I32_SHR_S,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i32(&wasm, &format!("i32.shr_s({val:#x}, {shift})"));
        }
    }
}

/// Test i32.shr_u (logical shift right)
#[test]
fn test_i32_shr_u_exhaustive() {
    ...
}

/// Test i32.rotl (rotate left)
#[test]
fn test_i32_rotl_exhaustive() {
    ...
}

/// Test i32.rotr (rotate right)
#[test]
fn test_i32_rotr_exhaustive() {
    ...
}

/// Test i32.clz (count leading zeros)
#[test]
fn test_i32_clz_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    for &val in I32_EDGE_CASES {
        let wasm = WasmBuilder::i32_unary_func(&[
            wasm_ops::i32_const(val),
            0x67, // i32.clz opcode
            wasm_ops::END,
        ].concat());

        tester.assert_same_i32(&wasm, &format!("i32.clz({val:#x})"));
    }
}
```

#### 2.4 i64 Tests (Critical - Currently 0% Coverage)

```rust
// tests/opcodes/i64_arithmetic.rs

use crate::harness::*;
use crate::opcodes::I64_EDGE_CASES;

/// Test i64.add with 64-bit precision
#[test]
fn test_i64_add_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    for &a in I64_EDGE_CASES {
        for &b in I64_EDGE_CASES {
            let wasm = WasmBuilder::i64_binary_func(&[
                wasm_ops::i64_const(a),
                wasm_ops::i64_const(b),
                wasm_ops::I64_ADD,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i64(&wasm, &format!("i64.add({a}, {b})"));
        }
    }
}

/// Test i64.mul - particularly important for overflow
#[test]
fn test_i64_mul_exhaustive() {
    let mut tester = DifferentialTester::new().unwrap();

    // Specifically test 32-bit overflow scenarios
    let critical_values: &[i64] = &[
        0,
        1,
        -1,
        0x7FFF_FFFF,         // Max i32
        0x8000_0000,         // Min i32 (as u32)
        0xFFFF_FFFF,         // Max u32
        0x1_0000_0000,       // 2^32
        0x7FFF_FFFF_FFFF_FFFF,  // Max i64
    ];

    for &a in critical_values {
        for &b in critical_values {
            let wasm = WasmBuilder::i64_binary_func(&[
                wasm_ops::i64_const(a),
                wasm_ops::i64_const(b),
                wasm_ops::I64_MUL,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i64(&wasm, &format!("i64.mul({a:#x}, {b:#x})"));
        }
    }
}

/// Test i64.shl with shift amounts that cross 32-bit boundary
#[test]
fn test_i64_shl_cross_boundary() {
    let mut tester = DifferentialTester::new().unwrap();

    let shift_amounts = [0, 1, 31, 32, 33, 63, 64, 65];

    for &val in I64_EDGE_CASES {
        for &shift in &shift_amounts {
            let wasm = WasmBuilder::i64_binary_func(&[
                wasm_ops::i64_const(val),
                wasm_ops::i64_const(shift),
                wasm_ops::I64_SHL,
                wasm_ops::END,
            ].concat());

            tester.assert_same_i64(&wasm, &format!("i64.shl({val:#x}, {shift})"));
        }
    }
}

/// Test i64 wrap to i32 (truncation)
#[test]
fn test_i32_wrap_i64() {
    let mut tester = DifferentialTester::new().unwrap();

    for &val in I64_EDGE_CASES {
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i64_const(val),
            wasm_ops::I32_WRAP_I64,
            wasm_ops::END,
        ].concat());

        tester.assert_same_i32(&wasm, &format!("i32.wrap_i64({val:#x})"));
    }
}

/// Test i64 extend from i32 (signed)
#[test]
fn test_i64_extend_i32_s() {
    let mut tester = DifferentialTester::new().unwrap();

    for &val in I32_EDGE_CASES {
        let wasm = WasmBuilder::i64_func(&[
            wasm_ops::i32_const(val),
            wasm_ops::I64_EXTEND_I32_S,
            wasm_ops::END,
        ].concat());

        tester.assert_same_i64(&wasm, &format!("i64.extend_i32_s({val})"));
    }
}
```

#### 2.5 Denormal-Specific Tests

```rust
// tests/opcodes/denormal.rs

/// Test constants in the denormal danger zone (< 2^23)
#[test]
fn test_denormal_constants() {
    let mut exec = BytecodeExecutor::new().unwrap();

    // These values, when stored as float bits, become denormals
    let denormal_danger = [
        1, 2, 3, 4, 5, 6, 7, 8,
        15, 16, 17,
        31, 32, 33,
        63, 64, 65,
        100, 104, 127, 128, 255, 256,
        1000, 4096, 8191, 8192,
        1_000_000, 2_000_000, 4_000_000,
        8_388_607,  // Max before 2^23
        8_388_608,  // Exactly 2^23 - should be fine
    ];

    for &val in &denormal_danger {
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(val),
            wasm_ops::END,
        ].concat());

        let result = exec.run_wasm_i32(&wasm)
            .expect(&format!("Execution failed for constant {val}"));

        assert_eq!(
            result, val,
            "Denormal corruption: expected {val}, got {result}"
        );
    }
}

/// Test addresses in denormal range survive load/store
#[test]
fn test_denormal_addresses() {
    let mut exec = BytecodeExecutor::new().unwrap();

    // Store value at address, then load it back
    for addr in [4, 8, 16, 64, 100, 104, 128, 256] {
        let wasm = WasmBuilder::i32_func(&[
            // Store 42 at address
            wasm_ops::i32_const(addr),      // address
            wasm_ops::i32_const(42),         // value
            wasm_ops::I32_STORE,
            0x02, 0x00,                       // align=4, offset=0

            // Load from address
            wasm_ops::i32_const(addr),
            wasm_ops::I32_LOAD,
            0x02, 0x00,

            wasm_ops::END,
        ].concat());

        let result = exec.run_wasm_i32(&wasm)
            .expect(&format!("Execution failed for address {addr}"));

        assert_eq!(
            result, 42,
            "Denormal address corruption at {addr}: got {result}"
        );
    }
}
```

---

### Phase 3: Integration Tests

#### 3.1 Real Rust Programs

```rust
// tests/integration/real_programs.rs

/// Test that our translator matches wasmtime for real Rust programs
#[test]
fn test_fibonacci_wasm() {
    let wasm = include_bytes!("../../test_programs/fibonacci.wasm");
    assert_wasm_eq!(wasm, i32);
}

#[test]
fn test_bubble_sort_wasm() {
    let wasm = include_bytes!("../../test_programs/bubble_sort.wasm");
    assert_wasm_eq!(wasm, i32);
}

#[test]
fn test_xxhash_wasm() {
    let wasm = include_bytes!("../../test_programs/xxhash.wasm");
    assert_wasm_eq!(wasm, i64);
}

/// Test programs that use i64 heavily
#[test]
fn test_fnv_hash_wasm() {
    // FNV-1a uses 64-bit constants
    let wasm = include_bytes!("../../test_programs/fnv_hash.wasm");
    assert_wasm_eq!(wasm, i64);
}
```

#### 3.2 Control Flow Depth Tests

```rust
// tests/integration/control_flow.rs

/// Test deeply nested blocks (potential register exhaustion)
#[test]
fn test_nested_blocks_5_deep() {
    let mut body = vec![];

    // Push 5 nested blocks
    for _ in 0..5 {
        body.push(wasm_ops::BLOCK);
        body.push(0x40); // void block
    }

    // Return value from innermost block
    body.extend_from_slice(&wasm_ops::i32_const(42));

    // Close all blocks
    for _ in 0..5 {
        body.push(wasm_ops::END);
    }
    body.push(wasm_ops::END); // function end

    let wasm = WasmBuilder::i32_func(&body);

    let mut exec = BytecodeExecutor::new().unwrap();
    let result = exec.run_wasm_i32(&wasm).unwrap();
    assert_eq!(result, 42);
}

/// Test nested loops with accumulator
#[test]
fn test_nested_loops_3_deep() {
    // for i in 0..3 { for j in 0..3 { for k in 0..3 { sum += 1; } } }
    // Expected: 27

    let wasm = include_bytes!("../../test_programs/nested_loops.wasm");

    let mut tester = DifferentialTester::new().unwrap();
    tester.assert_same_i32(&wasm, "nested_loops_3x3x3");
}

/// Test if/else with different return types on branches
#[test]
fn test_if_else_return_paths() {
    // if (cond) { return 1 } else { return 2 }
    let body = [
        wasm_ops::i32_const(1),    // condition (true)
        wasm_ops::IF,
        0x7F,                       // result type: i32
        wasm_ops::i32_const(100),
        wasm_ops::ELSE,
        wasm_ops::i32_const(200),
        wasm_ops::END,
        wasm_ops::END,
    ].concat();

    let wasm = WasmBuilder::i32_func(&body);
    assert_wasm_eq!(&wasm, i32);
}
```

#### 3.3 Register Pressure Tests

```rust
// tests/integration/register_pressure.rs

/// Force spill by using more than 20 temp values
#[test]
fn test_register_spill_triggered() {
    let mut body = vec![];

    // Push 25 constants (more than 20 temp registers)
    for i in 0..25 {
        body.extend_from_slice(&wasm_ops::i32_const(i + 1));
    }

    // Sum them all (pop 24, keep running sum)
    for _ in 0..24 {
        body.push(wasm_ops::I32_ADD);
    }

    body.push(wasm_ops::END);

    let wasm = WasmBuilder::i32_func(&body);

    let mut tester = DifferentialTester::new().unwrap();
    tester.assert_same_i32(&wasm, "sum_of_1_to_25");

    // Expected: 1+2+3+...+25 = 325
}

/// Verify spill doesn't corrupt values
#[test]
fn test_spill_value_preservation() {
    let mut body = vec![];

    // Push specific pattern that's easy to verify
    let values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1024, 2048, 4096, 8192, 16384, 32768,
                  65536, 131072, 262144, 524288, 1048576];

    for &v in &values {
        body.extend_from_slice(&wasm_ops::i32_const(v));
    }

    // XOR them all together
    for _ in 0..values.len()-1 {
        body.push(wasm_ops::I32_XOR);
    }

    body.push(wasm_ops::END);

    let wasm = WasmBuilder::i32_func(&body);

    let mut tester = DifferentialTester::new().unwrap();
    tester.assert_same_i32(&wasm, "xor_spill_test");
}
```

---

### Phase 4: Diagnostics

#### 4.1 Instruction Tracing

```rust
// src/gpu_os/gpu_app_system.rs (add to Metal shader)

/*
Add debug tracing buffer to bytecode VM:

kernel void bytecode_vm(
    device BytecodeInst* code [[buffer(0)]],
    device float4* regs [[buffer(1)]],
    device uint* state [[buffer(2)]],
    device DebugTrace* trace [[buffer(10)]],  // NEW
    uint tid [[thread_position_in_grid]]
) {
    uint trace_idx = 0;

    while (pc < code_size && iterations < MAX_ITERATIONS) {
        BytecodeInst inst = code[pc];

        // Record instruction execution
        if (trace_idx < MAX_TRACE) {
            trace[trace_idx].pc = pc;
            trace[trace_idx].opcode = inst.opcode;
            trace[trace_idx].dst = inst.dst;
            trace[trace_idx].s1 = inst.s1;
            trace[trace_idx].s2 = inst.s2;
            trace[trace_idx].regs_before = regs[inst.dst];
            trace_idx++;
        }

        // Execute instruction...

        // Record result
        if (trace_idx > 0) {
            trace[trace_idx - 1].regs_after = regs[inst.dst];
        }
    }
}

struct DebugTrace {
    uint pc;
    uint opcode;
    uint dst;
    uint s1;
    uint s2;
    float4 regs_before;
    float4 regs_after;
};
*/

/// Rust side: read and format trace
pub fn dump_execution_trace(trace_buffer: &metal::Buffer) -> String {
    let mut output = String::new();
    unsafe {
        let traces = trace_buffer.contents() as *const DebugTrace;
        for i in 0..MAX_TRACE {
            let t = *traces.add(i);
            if t.opcode == 0 && t.pc == 0 { break; }

            writeln!(output,
                "[{:04}] {:02X} r{} = op(r{}, r{})  before={:?} after={:?}",
                t.pc, t.opcode, t.dst, t.s1, t.s2,
                t.regs_before, t.regs_after
            ).unwrap();
        }
    }
    output
}
```

#### 4.2 Bytecode Validator

```rust
// wasm_translator/src/validator.rs

/// Validate bytecode before execution
pub fn validate_bytecode(bytecode: &[u8]) -> Result<(), ValidationError> {
    let header = BytecodeHeader::from_bytes(&bytecode[..16])?;

    let code_size = header.code_size as usize;
    let inst_size = 8; // sizeof(BytecodeInst)

    if bytecode.len() < 16 + code_size * inst_size {
        return Err(ValidationError::TruncatedCode);
    }

    for i in 0..code_size {
        let offset = 16 + i * inst_size;
        let inst = BytecodeInst::from_bytes(&bytecode[offset..offset+inst_size])?;

        // Check register indices
        if inst.dst > 31 || inst.s1 > 31 || inst.s2 > 31 {
            return Err(ValidationError::InvalidRegister {
                pc: i, dst: inst.dst, s1: inst.s1, s2: inst.s2
            });
        }

        // Check jump targets
        if inst.opcode == OP_JMP || inst.opcode == OP_JZ || inst.opcode == OP_JNZ {
            let target = inst.imm_bits as usize;
            if target >= code_size {
                return Err(ValidationError::JumpOutOfBounds {
                    pc: i, target
                });
            }
        }

        // Check opcode is known
        if !is_valid_opcode(inst.opcode) {
            return Err(ValidationError::UnknownOpcode {
                pc: i, opcode: inst.opcode
            });
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum ValidationError {
    TruncatedCode,
    InvalidRegister { pc: usize, dst: u8, s1: u8, s2: u8 },
    JumpOutOfBounds { pc: usize, target: usize },
    UnknownOpcode { pc: usize, opcode: u8 },
}
```

---

## File Structure

```
tests/
├── harness/
│   ├── mod.rs              # Export all harness components
│   ├── wasm_builder.rs     # WASM binary construction
│   ├── executor.rs         # GPU bytecode execution
│   ├── cpu_reference.rs    # wasmtime reference
│   └── differential.rs     # Comparison framework
├── opcodes/
│   ├── mod.rs              # Edge case constants
│   ├── i32_arithmetic.rs   # add, sub, mul, div, rem
│   ├── i32_bitwise.rs      # and, or, xor, shl, shr, rot
│   ├── i32_comparison.rs   # eq, ne, lt, gt, le, ge
│   ├── i64_arithmetic.rs   # 64-bit math
│   ├── i64_bitwise.rs      # 64-bit bit ops
│   ├── i64_comparison.rs   # 64-bit comparisons
│   ├── f32_arithmetic.rs   # float math
│   ├── f64_arithmetic.rs   # double math
│   ├── conversions.rs      # type conversions
│   ├── memory.rs           # load/store
│   ├── denormal.rs         # denormal edge cases
│   └── control_flow.rs     # jumps, branches
├── integration/
│   ├── real_programs.rs    # Compiled Rust programs
│   ├── control_flow.rs     # Nesting depth tests
│   └── register_pressure.rs # Spill tests
└── test_comprehensive.rs   # Main test runner
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Opcode coverage | 6% | 95% |
| i32 ops tested | 3/18 | 18/18 |
| i64 ops tested | 0/21 | 21/21 |
| f32 ops tested | 0/13 | 13/13 |
| f64 ops tested | 0/6 | 6/6 |
| Integration tests | 0 | 20+ |
| Bug detection time | Hours | Seconds |

---

## Implementation Order

### Week 1: Foundation
1. [ ] Create `tests/harness/` module structure
2. [ ] Implement `WasmBuilder` with all convenience methods
3. [ ] Implement `BytecodeExecutor` with i32/i64/f32/f64 variants
4. [ ] Add wasmtime dependency, implement `CpuReference`
5. [ ] Implement `DifferentialTester` and `assert_wasm_eq!` macro

### Week 2: i32 Coverage
6. [ ] i32 arithmetic tests (add, sub, mul, div_s, div_u, rem_s, rem_u)
7. [ ] i32 bitwise tests (and, or, xor, shl, shr_s, shr_u, rotl, rotr, clz)
8. [ ] i32 comparison tests (eq, ne, lt_s, lt_u, gt_s, gt_u, le_s, le_u, ge_s, ge_u, eqz)
9. [ ] Denormal constant tests
10. [ ] Denormal address tests

### Week 3: i64 Coverage (Critical Gap)
11. [ ] i64 arithmetic tests
12. [ ] i64 bitwise tests
13. [ ] i64 comparison tests
14. [ ] i64 conversion tests (wrap, extend_s, extend_u)
15. [ ] i64 edge cases (32-bit boundary crossing)

### Week 4: Float & Integration
16. [ ] f32 arithmetic and comparison tests
17. [ ] f64 arithmetic tests
18. [ ] Float conversion tests
19. [ ] Register pressure / spill tests
20. [ ] Nested control flow tests
21. [ ] Real Rust program integration tests

---

## Dependencies

```toml
# Cargo.toml additions

[dev-dependencies]
wasmtime = "19.0"  # CPU reference execution
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| wasmtime version mismatch | Pin exact version, document WASM spec compliance |
| GPU timeout during test | Add iteration limit, smaller test cases |
| Float NaN handling differs | Document CPU/GPU NaN differences, skip NaN tests |
| Test suite too slow | Parallelize with `cargo test -j N`, use `--release` |
