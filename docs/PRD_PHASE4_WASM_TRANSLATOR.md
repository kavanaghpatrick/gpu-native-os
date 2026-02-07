# PRD Phase 4: WASM to GPU Bytecode Translator

## THE GPU IS THE COMPUTER

**Issue**: Compile Real Rust via WebAssembly
**Phase**: 4 of 5
**Duration**: 4 weeks
**Depends On**: Phase 1-3 (Integer Ops, Atomics, DSL Macro)
**Enables**: Running compiled `no_std` Rust crates on GPU

---

## Problem Statement

The DSL macro (Phase 3) enables Rust-like syntax but is limited to what we explicitly support. To run real Rust code (including `no_std` crates), we need a compilation path from Rust to GPU bytecode.

**Solution**: Use WebAssembly as an intermediate representation.

```
Rust Source → rustc → WASM → Our Translator → GPU Bytecode
```

This leverages rustc's existing WASM backend while giving us full control over GPU execution.

---

## Why WASM?

| Alternative | Pros | Cons |
|-------------|------|------|
| **LLVM IR** | Full optimization | ~1000 instruction types, complex |
| **MIR** | Rust-specific | Unstable, tightly coupled to rustc |
| **WASM** | 172 instructions, well-specified | Extra translation step |
| **SPIR-V** | GPU-native | Tied to Vulkan semantics |

WASM wins because:
1. **Simple**: 172 instructions in MVP
2. **Stable**: W3C standard, won't change unexpectedly
3. **Rust-native**: `rustc` targets WASM directly
4. **Proven**: Projects like wasm2spirv, wasm-gpu exist

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPILATION PIPELINE                          │
│                                                                  │
│  1. Rust Source                                                  │
│     │                                                            │
│     ▼ rustc --target wasm32-unknown-unknown                     │
│                                                                  │
│  2. WASM Binary (.wasm)                                          │
│     │                                                            │
│     ▼ wasmparser crate                                          │
│                                                                  │
│  3. WASM IR (our internal representation)                        │
│     │                                                            │
│     ▼ Type analysis, stack→register conversion                  │
│                                                                  │
│  4. GPU Bytecode                                                 │
│     │                                                            │
│     ▼ BytecodeVM execution                                      │
│                                                                  │
│  5. GPU Execution                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Design

### WASM Instruction Categories

| Category | Count | GPU Support |
|----------|-------|-------------|
| Control flow | 12 | Full |
| Variable access | 8 | Full |
| Memory | 27 | Partial (no grow) |
| Numeric i32 | 35 | Full (via Phase 1) |
| Numeric i64 | 35 | Partial (emulated) |
| Numeric f32 | 20 | Full |
| Numeric f64 | 20 | Partial (demoted to f32) |
| Conversion | 30 | Full |

### WASM to Bytecode Mapping

```rust
// In wasm_translator/src/translate.rs

fn translate_instruction(&mut self, inst: &Instruction) -> Result<(), Error> {
    match inst {
        // ═══════════════════════════════════════════════════════════════
        // CONTROL FLOW
        // ═══════════════════════════════════════════════════════════════

        Instruction::Unreachable => {
            self.emit.halt();  // GPU can't trap, just halt
        }

        Instruction::Nop => {
            self.emit.nop();
        }

        Instruction::Block { blockty } => {
            let label = self.new_label("block_end");
            self.block_stack.push(BlockContext {
                kind: BlockKind::Block,
                end_label: label,
                result_type: *blockty,
            });
        }

        Instruction::Loop { blockty } => {
            let start_label = self.new_label("loop_start");
            let end_label = self.new_label("loop_end");
            self.emit.label(&start_label);
            self.block_stack.push(BlockContext {
                kind: BlockKind::Loop,
                start_label: Some(start_label),
                end_label,
                result_type: *blockty,
            });
        }

        Instruction::If { blockty } => {
            let cond = self.pop_operand();
            let else_label = self.new_label("else");
            let end_label = self.new_label("endif");

            self.emit.jz(cond, &else_label);
            self.block_stack.push(BlockContext {
                kind: BlockKind::If,
                else_label: Some(else_label),
                end_label,
                result_type: *blockty,
            });
        }

        Instruction::Else => {
            let ctx = self.block_stack.last_mut().unwrap();
            self.emit.jmp(&ctx.end_label);
            self.emit.label(ctx.else_label.as_ref().unwrap());
            ctx.else_label = None;  // Mark else as visited
        }

        Instruction::End => {
            let ctx = self.block_stack.pop().unwrap();
            if let Some(else_label) = ctx.else_label {
                // If block without else - emit else label
                self.emit.label(&else_label);
            }
            self.emit.label(&ctx.end_label);
        }

        Instruction::Br { relative_depth } => {
            let target = self.get_branch_target(*relative_depth);
            self.emit.jmp(&target);
        }

        Instruction::BrIf { relative_depth } => {
            let cond = self.pop_operand();
            let target = self.get_branch_target(*relative_depth);
            self.emit.jnz(cond, &target);
        }

        Instruction::BrTable { targets, default_target } => {
            // Jump table
            let index = self.pop_operand();
            for (i, target) in targets.iter().enumerate() {
                let cmp = self.alloc_temp();
                self.emit.loadi_int(cmp, i as i32);
                self.emit.int_eq(cmp, index, cmp);
                let label = self.get_branch_target(*target);
                self.emit.jnz(cmp, &label);
            }
            let default_label = self.get_branch_target(*default_target);
            self.emit.jmp(&default_label);
        }

        Instruction::Return => {
            // Copy return value to r4 if any
            if let Some(ret) = self.return_value_reg {
                let val = self.pop_operand();
                self.emit.mov(4, val);  // Return value in r4
            }
            self.emit.ret();
        }

        Instruction::Call { function_index } => {
            self.translate_call(*function_index)?;
        }

        Instruction::CallIndirect { type_index, table_index } => {
            // Function pointer call
            let idx = self.pop_operand();
            // Load function address from table
            // ... complex, needs function table support
            todo!("call_indirect");
        }

        // ═══════════════════════════════════════════════════════════════
        // LOCAL/GLOBAL VARIABLES
        // ═══════════════════════════════════════════════════════════════

        Instruction::LocalGet { local_index } => {
            let reg = self.local_to_reg(*local_index);
            self.push_operand(reg);
        }

        Instruction::LocalSet { local_index } => {
            let val = self.pop_operand();
            let reg = self.local_to_reg(*local_index);
            self.emit.mov(reg, val);
        }

        Instruction::LocalTee { local_index } => {
            let val = self.peek_operand();
            let reg = self.local_to_reg(*local_index);
            self.emit.mov(reg, val);
        }

        Instruction::GlobalGet { global_index } => {
            // Globals stored in state memory
            let addr = self.global_base + *global_index;
            let dst = self.alloc_temp();
            self.emit.loadi_uint(30, addr);
            self.emit.ld(dst, 30, 0);
            self.push_operand(dst);
        }

        Instruction::GlobalSet { global_index } => {
            let val = self.pop_operand();
            let addr = self.global_base + *global_index;
            self.emit.loadi_uint(30, addr);
            self.emit.st(val, 30, 0);
        }

        // ═══════════════════════════════════════════════════════════════
        // MEMORY OPERATIONS
        // ═══════════════════════════════════════════════════════════════

        Instruction::I32Load { memarg } => {
            let addr = self.pop_operand();
            let dst = self.alloc_temp();
            // Add offset
            if memarg.offset > 0 {
                self.emit.loadi_uint(30, memarg.offset as u32);
                self.emit.int_add(addr, addr, 30);
            }
            // Convert byte address to float4 index (divide by 16)
            self.emit.loadi_uint(30, 4);
            self.emit.shr_u(addr, addr, 30);
            self.emit.ld(dst, addr, 0);
            self.push_operand(dst);
        }

        Instruction::I32Store { memarg } => {
            let val = self.pop_operand();
            let addr = self.pop_operand();
            if memarg.offset > 0 {
                self.emit.loadi_uint(30, memarg.offset as u32);
                self.emit.int_add(addr, addr, 30);
            }
            self.emit.loadi_uint(30, 4);
            self.emit.shr_u(addr, addr, 30);
            self.emit.st(val, addr, 0);
        }

        Instruction::MemorySize { .. } => {
            // Return fixed memory size
            let dst = self.alloc_temp();
            self.emit.loadi_uint(dst, self.memory_pages);
            self.push_operand(dst);
        }

        Instruction::MemoryGrow { .. } => {
            // GPU can't grow memory - return -1 (failure)
            self.pop_operand();  // Discard requested pages
            let dst = self.alloc_temp();
            self.emit.loadi_int(dst, -1);
            self.push_operand(dst);
        }

        // ═══════════════════════════════════════════════════════════════
        // I32 ARITHMETIC
        // ═══════════════════════════════════════════════════════════════

        Instruction::I32Const { value } => {
            let dst = self.alloc_temp();
            self.emit.loadi_int(dst, *value);
            self.push_operand(dst);
        }

        Instruction::I32Add => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_add(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Sub => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_sub(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Mul => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_mul(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32DivS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_div_s(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32DivU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_div_u(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32RemS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_rem_s(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32RemU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_rem_u(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32And => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.bit_and(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Or => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.bit_or(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Xor => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.bit_xor(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Shl => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.shl(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32ShrS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.shr_s(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32ShrU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.shr_u(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Rotl => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.rotl(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Rotr => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.rotr(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Clz => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.clz(dst, a);
            self.push_operand(dst);
        }

        // ═══════════════════════════════════════════════════════════════
        // I32 COMPARISON
        // ═══════════════════════════════════════════════════════════════

        Instruction::I32Eqz => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.loadi_int(30, 0);
            self.emit.int_eq(dst, a, 30);
            self.push_operand(dst);
        }

        Instruction::I32Eq => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_eq(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32Ne => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_ne(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32LtS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_lt_s(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32LtU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_lt_u(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32LeS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_le_s(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32LeU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_le_u(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::I32GtS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_lt_s(dst, b, a);  // Swap operands
            self.push_operand(dst);
        }

        Instruction::I32GtU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_lt_u(dst, b, a);
            self.push_operand(dst);
        }

        Instruction::I32GeS => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_le_s(dst, b, a);
            self.push_operand(dst);
        }

        Instruction::I32GeU => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_le_u(dst, b, a);
            self.push_operand(dst);
        }

        // ═══════════════════════════════════════════════════════════════
        // F32 OPERATIONS
        // ═══════════════════════════════════════════════════════════════

        Instruction::F32Const { value } => {
            let dst = self.alloc_temp();
            self.emit.loadi(dst, *value);
            self.push_operand(dst);
        }

        Instruction::F32Add => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.add(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::F32Sub => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.sub(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::F32Mul => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.mul(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::F32Div => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.div(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::F32Sqrt => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.sqrt(dst, a);
            self.push_operand(dst);
        }

        Instruction::F32Abs => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.abs(dst, a);
            self.push_operand(dst);
        }

        Instruction::F32Neg => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.neg(dst, a);
            self.push_operand(dst);
        }

        Instruction::F32Min => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.min(dst, a, b);
            self.push_operand(dst);
        }

        Instruction::F32Max => {
            let b = self.pop_operand();
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.max(dst, a, b);
            self.push_operand(dst);
        }

        // ═══════════════════════════════════════════════════════════════
        // CONVERSIONS
        // ═══════════════════════════════════════════════════════════════

        Instruction::I32TruncF32S => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.f_to_int(dst, a);
            self.push_operand(dst);
        }

        Instruction::I32TruncF32U => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.f_to_uint(dst, a);
            self.push_operand(dst);
        }

        Instruction::F32ConvertI32S => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.int_to_f(dst, a);
            self.push_operand(dst);
        }

        Instruction::F32ConvertI32U => {
            let a = self.pop_operand();
            let dst = self.alloc_temp();
            self.emit.uint_to_f(dst, a);
            self.push_operand(dst);
        }

        Instruction::I32ReinterpretF32 => {
            // No-op: bits stay the same, just type changes
            // Our registers hold bits, not typed values
        }

        Instruction::F32ReinterpretI32 => {
            // No-op
        }

        // ═══════════════════════════════════════════════════════════════
        // I64 - Emulated or error
        // ═══════════════════════════════════════════════════════════════

        inst if is_i64_instruction(inst) => {
            return Err(Error::Unsupported("i64 operations not supported"));
        }

        _ => {
            return Err(Error::Unsupported(format!("instruction: {:?}", inst)));
        }
    }

    Ok(())
}
```

### Stack Machine to Register Conversion

WASM is stack-based. Our bytecode is register-based. The translation:

```rust
// WASM stack operations map to register operations

struct OperandStack {
    stack: Vec<u8>,      // Register numbers
    next_temp: u8,       // Next available temp register (8-23)
}

impl OperandStack {
    fn push(&mut self, reg: u8) {
        self.stack.push(reg);
    }

    fn pop(&mut self) -> u8 {
        self.stack.pop().expect("stack underflow")
    }

    fn peek(&self) -> u8 {
        *self.stack.last().expect("stack underflow")
    }

    fn alloc_temp(&mut self) -> u8 {
        let reg = self.next_temp;
        self.next_temp += 1;
        if self.next_temp > 23 {
            panic!("out of temp registers - need spilling");
        }
        reg
    }
}

// Example translation:
//
// WASM:                      GPU Bytecode:
// i32.const 10               loadi_int r8, 10
// i32.const 20               loadi_int r9, 20
// i32.add                    int_add r10, r8, r9
//
// Stack: [10] → [10, 20] → [30]
// Regs:  r8=10, r9=20, r10=30
```

### Rust Compilation Setup

```toml
# Example GPU-targeted crate: gpu_app/Cargo.toml

[package]
name = "gpu_app"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
# No std dependencies!

[profile.release]
opt-level = "z"       # Optimize for size
lto = true            # Link-time optimization
panic = "abort"       # No unwinding
```

```rust
// gpu_app/src/lib.rs
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// GPU entry point
#[no_mangle]
pub extern "C" fn gpu_main(state: *mut f32, tid: u32) {
    unsafe {
        let idx = tid as usize * 4;
        let x = *state.add(idx);
        let y = *state.add(idx + 1);
        *state.add(idx) = x + 1.0;
        *state.add(idx + 1) = y + 1.0;
    }
}
```

```bash
# Build to WASM
cargo build --target wasm32-unknown-unknown --release

# Translate to GPU bytecode
wasm2gpu target/wasm32-unknown-unknown/release/gpu_app.wasm -o gpu_app.gpubc
```

### API for Translation

```rust
// In wasm_translator/src/lib.rs

pub struct WasmTranslator {
    config: TranslatorConfig,
}

pub struct TranslatorConfig {
    pub memory_pages: u32,      // Initial memory (64KB pages)
    pub stack_size: u32,        // Max operand stack depth
    pub inline_threshold: u32,  // Inline functions smaller than this
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            memory_pages: 16,       // 1MB
            stack_size: 256,
            inline_threshold: 64,
        }
    }
}

impl WasmTranslator {
    pub fn new(config: TranslatorConfig) -> Self {
        Self { config }
    }

    /// Translate WASM binary to GPU bytecode
    pub fn translate(&self, wasm_bytes: &[u8]) -> Result<Vec<u64>, Error> {
        let parser = wasmparser::Parser::new(0);
        let mut module = WasmModule::new();

        // Parse WASM
        for payload in parser.parse_all(wasm_bytes) {
            match payload? {
                Payload::TypeSection(reader) => {
                    module.parse_types(reader)?;
                }
                Payload::FunctionSection(reader) => {
                    module.parse_functions(reader)?;
                }
                Payload::CodeSectionEntry(body) => {
                    module.parse_code(body)?;
                }
                Payload::ExportSection(reader) => {
                    module.parse_exports(reader)?;
                }
                _ => {}
            }
        }

        // Find entry point
        let entry = module.find_export("gpu_main")
            .ok_or(Error::NoEntryPoint)?;

        // Translate
        let mut emitter = BytecodeEmitter::new();
        let mut ctx = TranslationContext::new(&module, &mut emitter, &self.config);
        ctx.translate_function(entry)?;

        Ok(emitter.finish())
    }
}
```

---

## Test Cases

### Test File: `tests/test_phase4_wasm_translator.rs`

```rust
//! Phase 4: WASM Translator Tests
//!
//! THE GPU IS THE COMPUTER.
//! Compile real Rust to GPU via WASM.

use metal::Device;
use wasm_translator::WasmTranslator;
use rust_experiment::gpu_os::bytecode_vm::BytecodeVM;

// Helper to compile Rust to WASM bytes (would use actual rustc in practice)
fn compile_to_wasm(rust_code: &str) -> Vec<u8> {
    // In real implementation: invoke rustc
    // For tests: use pre-compiled WASM or WAT
    todo!()
}

#[test]
fn test_simple_addition() {
    // WASM equivalent of: fn add(a: i32, b: i32) -> i32 { a + b }
    let wasm = wat::parse_str(r#"
        (module
            (func (export "gpu_main") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
            )
        )
    "#).unwrap();

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(&wasm).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    // Set up arguments: r4=10, r5=20
    vm.set_register_int(4, 10);
    vm.set_register_int(5, 20);

    vm.load_program(&bytecode);
    vm.execute(&device);

    // Result in r4
    assert_eq!(vm.read_register_int(4), 30);
}

#[test]
fn test_loop_sum() {
    // WASM equivalent of: fn sum_to_n(n: i32) -> i32 { (0..=n).sum() }
    let wasm = wat::parse_str(r#"
        (module
            (func (export "gpu_main") (param $n i32) (result i32)
                (local $i i32)
                (local $sum i32)
                (local.set $i (i32.const 0))
                (local.set $sum (i32.const 0))
                (block $break
                    (loop $continue
                        ;; if i > n, break
                        (br_if $break (i32.gt_s (local.get $i) (local.get $n)))
                        ;; sum += i
                        (local.set $sum (i32.add (local.get $sum) (local.get $i)))
                        ;; i++
                        (local.set $i (i32.add (local.get $i) (i32.const 1)))
                        ;; continue
                        (br $continue)
                    )
                )
                (local.get $sum)
            )
        )
    "#).unwrap();

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(&wasm).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.set_register_int(4, 10);  // n = 10

    vm.load_program(&bytecode);
    vm.execute(&device);

    assert_eq!(vm.read_register_int(4), 55);  // 0+1+2+...+10 = 55
}

#[test]
fn test_memory_access() {
    // WASM with linear memory access
    let wasm = wat::parse_str(r#"
        (module
            (memory (export "memory") 1)
            (func (export "gpu_main")
                ;; Store 42 at address 0
                (i32.store (i32.const 0) (i32.const 42))
                ;; Load it back
                (i32.load (i32.const 0))
                ;; Store at address 4
                (i32.store (i32.const 4) (i32.const 0))
            )
        )
    "#).unwrap();

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(&wasm).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&bytecode);
    vm.execute(&device);

    // Check memory
    assert_eq!(vm.read_state_int(0), 42);
    assert_eq!(vm.read_state_int(1), 42);  // Loaded and stored
}

#[test]
fn test_float_operations() {
    let wasm = wat::parse_str(r#"
        (module
            (func (export "gpu_main") (param f32 f32) (result f32)
                local.get 0
                local.get 1
                f32.add
                f32.sqrt
            )
        )
    "#).unwrap();

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(&wasm).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.set_register_float(4, 9.0);
    vm.set_register_float(5, 16.0);

    vm.load_program(&bytecode);
    vm.execute(&device);

    // sqrt(9 + 16) = sqrt(25) = 5
    assert!((vm.read_register_float(4) - 5.0).abs() < 0.001);
}

#[test]
fn test_branching() {
    // if-else
    let wasm = wat::parse_str(r#"
        (module
            (func (export "gpu_main") (param $x i32) (result i32)
                (if (result i32) (i32.gt_s (local.get $x) (i32.const 0))
                    (then (i32.const 1))
                    (else (i32.const -1))
                )
            )
        )
    "#).unwrap();

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(&wasm).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    // Test positive
    vm.set_register_int(4, 5);
    vm.load_program(&bytecode);
    vm.execute(&device);
    assert_eq!(vm.read_register_int(4), 1);

    // Test negative
    vm.set_register_int(4, -5);
    vm.execute(&device);
    assert_eq!(vm.read_register_int(4), -1);
}

#[test]
fn test_function_call() {
    // Internal function call
    let wasm = wat::parse_str(r#"
        (module
            (func $double (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.mul
            )
            (func (export "gpu_main") (param i32) (result i32)
                local.get 0
                call $double
            )
        )
    "#).unwrap();

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(&wasm).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.set_register_int(4, 21);

    vm.load_program(&bytecode);
    vm.execute(&device);

    assert_eq!(vm.read_register_int(4), 42);
}

#[test]
fn test_real_rust_compilation() {
    // This test requires actual Rust→WASM compilation
    // Compile: gpu_test_crate with wasm32-unknown-unknown target

    let wasm_bytes = include_bytes!("../test_data/simple_rust.wasm");

    let translator = WasmTranslator::new(Default::default());
    let bytecode = translator.translate(wasm_bytes).expect("translation failed");

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&bytecode);
    vm.execute(&device);

    // Verify expected behavior
}
```

---

## Unsupported Features

| WASM Feature | Why Unsupported | Alternative |
|--------------|-----------------|-------------|
| `memory.grow` | GPU can't allocate | Pre-allocate max size |
| `i64` operations | No native 64-bit | Use i32 or emulate |
| `f64` operations | Limited value | Demote to f32 |
| `call_indirect` | Complex | Inline or limit |
| Threads/atomics | Different model | Use our Phase 2 atomics |
| SIMD | Different ISA | Use our vector ops |
| Exceptions | GPU can't trap | Return error codes |

---

## Success Criteria

1. **Core WASM instructions translate** correctly
2. **Real `no_std` Rust compiles** to functioning GPU bytecode
3. **Performance within 2x** of hand-written bytecode
4. **Clear error messages** for unsupported features

---

## Files to Create

| File | Purpose |
|------|---------|
| `wasm_translator/Cargo.toml` | Crate definition |
| `wasm_translator/src/lib.rs` | Public API |
| `wasm_translator/src/parser.rs` | WASM parsing via wasmparser |
| `wasm_translator/src/translate.rs` | Instruction translation |
| `wasm_translator/src/stack.rs` | Stack→register conversion |
| `wasm_translator/src/emit.rs` | Bytecode emission |
| `tests/test_phase4_wasm_translator.rs` | Tests |
| `test_data/simple_rust.wasm` | Pre-compiled test WASM |

---

## Next Phase

**Phase 5: I/O Command Queue Integration** - Non-blocking file access from GPU apps.
