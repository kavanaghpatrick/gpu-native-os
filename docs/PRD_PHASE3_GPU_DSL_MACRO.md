# PRD Phase 3: GPU Kernel DSL Macro

## THE GPU IS THE COMPUTER

**Issue**: Rust Proc Macro for GPU Bytecode Generation
**Phase**: 3 of 5
**Duration**: 4 weeks
**Depends On**: Phase 1 (Integer Ops), Phase 2 (Atomic Ops)
**Enables**: Writing GPU apps in Rust-like syntax

---

## Problem Statement

Currently, GPU apps are written by:
1. Manual `BytecodeAssembler` calls
2. Hand-managing register allocation
3. Manually computing jump offsets

This is error-prone and unproductive. We need a Rust-like DSL that compiles to bytecode.

---

## Goals

1. Create `gpu_kernel!` proc macro that compiles to bytecode
2. Support Rust-like syntax: let bindings, if/else, loops, function calls
3. Automatic register allocation
4. Type inference for int vs float operations
5. Compile-time errors for unsupported patterns

---

## Non-Goals

- Full Rust language support (that's Phase 4 via WASM)
- Dynamic memory allocation
- Recursion
- Closures with captures
- Trait objects

---

## Technical Design

### Syntax Overview

```rust
use gpu_dsl::gpu_kernel;

gpu_kernel! {
    fn particle_update(dt: f32) {
        let tid = TID;  // Built-in: thread ID
        let count = STATE[0].x as u32;

        for i in (tid * 16)..(tid * 16 + 16) {
            if i >= count { break; }

            let pos = STATE[i * 2];
            let vel = STATE[i * 2 + 1];

            // Physics
            let new_vel = f4(vel.x, vel.y + GRAVITY * dt, vel.z, vel.w);
            let new_pos = f4(
                pos.x + new_vel.x * dt,
                pos.y + new_vel.y * dt,
                pos.z,
                pos.w
            );

            STATE[i * 2] = new_pos;
            STATE[i * 2 + 1] = new_vel;

            // Emit quad
            emit_quad(new_pos.xy(), f2(4.0, 4.0), STATE[i * 2 + 1], 0.5);
        }
    }
}
```

### Built-in Constants and Functions

```rust
// Thread/Frame info (read-only)
TID         // Thread ID (uint)
TGSIZE      // Threadgroup size
FRAME       // Current frame number
TIME        // Time in seconds (float)
SCREEN      // Screen dimensions (float2)

// State access
STATE[idx]              // Read float4 from state
STATE[idx] = val        // Write float4 to state
ATOMIC[idx]             // Read atomic uint
ATOMIC[idx] += val      // Atomic add
ATOMIC[idx].cas(e, d)   // Compare-and-swap

// Graphics
emit_quad(pos, size, color, depth)
emit_vertex(pos, color, depth)

// Math (float)
f2(x, y)
f4(x, y, z, w)
sin(x), cos(x), sqrt(x), abs(x), min(a, b), max(a, b)
dot(a, b), length(v), normalize(v)

// Type conversion
x as u32    // float to uint
x as i32    // float to int
x as f32    // int/uint to float
```

### Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MACRO EXPANSION                               │
│                                                                  │
│  1. Parse TokenStream into AST                                   │
│     - Function signature                                         │
│     - Variable declarations (let)                                │
│     - Expressions (arithmetic, comparison)                       │
│     - Statements (if, for, while, break)                        │
│     - Function calls (built-ins)                                 │
│                                                                  │
│  2. Type inference                                               │
│     - Infer f32/u32/i32 from usage                              │
│     - Insert implicit conversions                                │
│                                                                  │
│  3. Register allocation                                          │
│     - Linear scan allocator                                      │
│     - Spill to state memory if needed (r24-r31 are callee-save) │
│                                                                  │
│  4. Bytecode emission                                            │
│     - Generate BytecodeAssembler calls                           │
│     - Resolve jump targets                                       │
│                                                                  │
│  5. Output: Rust code that builds bytecode                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### AST Representation

```rust
// In gpu_dsl_macro/src/ast.rs

pub enum Type {
    F32,
    U32,
    I32,
    F32x2,
    F32x4,
    Bool,
    Void,
}

pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub body: Vec<Statement>,
}

pub enum Statement {
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    Assign {
        target: AssignTarget,
        value: Expr,
    },
    If {
        condition: Expr,
        then_body: Vec<Statement>,
        else_body: Option<Vec<Statement>>,
    },
    For {
        var: String,
        range: Range,
        body: Vec<Statement>,
    },
    While {
        condition: Expr,
        body: Vec<Statement>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
    Expr(Expr),  // Expression statement (e.g., function call)
}

pub enum AssignTarget {
    Variable(String),
    StateIndex(Expr),
    AtomicIndex(Expr),
}

pub enum Expr {
    Literal(Literal),
    Variable(String),
    Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Unary { op: UnaryOp, operand: Box<Expr> },
    Call { func: String, args: Vec<Expr> },
    Index { base: Box<Expr>, index: Box<Expr> },
    Field { base: Box<Expr>, field: String },
    Cast { expr: Box<Expr>, ty: Type },
    StateRead(Box<Expr>),
    AtomicRead(Box<Expr>),
}

pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr,
}
```

### Register Allocator

```rust
// In gpu_dsl_macro/src/regalloc.rs

pub struct RegisterAllocator {
    // r0-r3: Reserved (special)
    // r4-r7: Arguments
    // r8-r23: Temporaries (16 available)
    // r24-r31: Callee-saved

    free_temps: Vec<u8>,      // Available temp registers
    var_to_reg: HashMap<String, u8>,
    spill_slots: Vec<String>, // Variables spilled to state memory
    next_spill: u32,
}

impl RegisterAllocator {
    pub fn new() -> Self {
        Self {
            free_temps: (8..=23).rev().collect(),  // r8-r23
            var_to_reg: HashMap::new(),
            spill_slots: Vec::new(),
            next_spill: 0xF000,  // Spill area in state memory
        }
    }

    pub fn allocate(&mut self, var: &str) -> RegisterOrSpill {
        if let Some(reg) = self.free_temps.pop() {
            self.var_to_reg.insert(var.to_string(), reg);
            RegisterOrSpill::Register(reg)
        } else {
            // Spill to state memory
            let slot = self.next_spill;
            self.next_spill += 1;
            self.spill_slots.push(var.to_string());
            RegisterOrSpill::Spill(slot)
        }
    }

    pub fn get(&self, var: &str) -> Option<RegisterOrSpill> {
        if let Some(&reg) = self.var_to_reg.get(var) {
            Some(RegisterOrSpill::Register(reg))
        } else if let Some(idx) = self.spill_slots.iter().position(|v| v == var) {
            Some(RegisterOrSpill::Spill(0xF000 + idx as u32))
        } else {
            None
        }
    }

    pub fn free(&mut self, var: &str) {
        if let Some(reg) = self.var_to_reg.remove(var) {
            self.free_temps.push(reg);
        }
    }
}

pub enum RegisterOrSpill {
    Register(u8),
    Spill(u32),
}
```

### Code Generator

```rust
// In gpu_dsl_macro/src/codegen.rs

pub struct CodeGenerator {
    output: Vec<TokenTree>,
    regalloc: RegisterAllocator,
    loop_stack: Vec<LoopContext>,
    label_counter: u32,
}

struct LoopContext {
    break_label: String,
    continue_label: String,
}

impl CodeGenerator {
    pub fn generate(&mut self, func: &Function) -> TokenStream {
        // Emit function wrapper
        let name = &func.name;
        let mut tokens = quote! {
            pub fn #name() -> Vec<u64> {
                let mut asm = BytecodeAssembler::new();
        };

        // Generate body
        for stmt in &func.body {
            self.gen_statement(stmt, &mut tokens);
        }

        // Emit halt and return
        tokens.extend(quote! {
                asm.halt();
                asm.finish()
            }
        });

        tokens
    }

    fn gen_statement(&mut self, stmt: &Statement, out: &mut TokenStream) {
        match stmt {
            Statement::Let { name, value, .. } => {
                let reg = self.regalloc.allocate(name);
                self.gen_expr_into(value, reg, out);
            }

            Statement::Assign { target, value } => {
                match target {
                    AssignTarget::Variable(name) => {
                        let reg = self.regalloc.get(name).unwrap();
                        self.gen_expr_into(value, reg, out);
                    }
                    AssignTarget::StateIndex(idx) => {
                        let val_reg = self.gen_expr_temp(value, out);
                        let idx_reg = self.gen_expr_temp(idx, out);
                        out.extend(quote! {
                            asm.st(#val_reg, #idx_reg, 0);
                        });
                    }
                    AssignTarget::AtomicIndex(idx) => {
                        let val_reg = self.gen_expr_temp(value, out);
                        let idx_reg = self.gen_expr_temp(idx, out);
                        out.extend(quote! {
                            asm.atomic_store(#val_reg, #idx_reg);
                        });
                    }
                }
            }

            Statement::If { condition, then_body, else_body } => {
                let else_label = self.new_label("else");
                let end_label = self.new_label("endif");

                // Evaluate condition
                let cond_reg = self.gen_expr_temp(condition, out);

                if else_body.is_some() {
                    out.extend(quote! {
                        asm.jz(#cond_reg, #else_label);
                    });
                } else {
                    out.extend(quote! {
                        asm.jz(#cond_reg, #end_label);
                    });
                }

                // Then block
                for stmt in then_body {
                    self.gen_statement(stmt, out);
                }

                if let Some(else_stmts) = else_body {
                    out.extend(quote! {
                        asm.jmp(#end_label);
                        asm.label(#else_label);
                    });

                    for stmt in else_stmts {
                        self.gen_statement(stmt, out);
                    }
                }

                out.extend(quote! {
                    asm.label(#end_label);
                });
            }

            Statement::For { var, range, body } => {
                let loop_label = self.new_label("for");
                let end_label = self.new_label("endfor");

                // Initialize loop variable
                let var_reg = self.regalloc.allocate(var);
                self.gen_expr_into(&range.start, var_reg, out);

                // Loop limit
                let limit_reg = self.gen_expr_temp(&range.end, out);

                self.loop_stack.push(LoopContext {
                    break_label: end_label.clone(),
                    continue_label: loop_label.clone(),
                });

                out.extend(quote! {
                    asm.label(#loop_label);
                });

                // Check condition
                let cmp_reg = self.regalloc.allocate_temp();
                out.extend(quote! {
                    asm.int_lt_u(#cmp_reg, #var_reg, #limit_reg);
                    asm.jz(#cmp_reg, #end_label);
                });

                // Body
                for stmt in body {
                    self.gen_statement(stmt, out);
                }

                // Increment
                out.extend(quote! {
                    asm.loadi_int(30, 1);
                    asm.int_add(#var_reg, #var_reg, 30);
                    asm.jmp(#loop_label);
                    asm.label(#end_label);
                });

                self.loop_stack.pop();
                self.regalloc.free(var);
            }

            Statement::Break => {
                let ctx = self.loop_stack.last().expect("break outside loop");
                let label = &ctx.break_label;
                out.extend(quote! {
                    asm.jmp(#label);
                });
            }

            Statement::Continue => {
                let ctx = self.loop_stack.last().expect("continue outside loop");
                let label = &ctx.continue_label;
                out.extend(quote! {
                    asm.jmp(#label);
                });
            }

            Statement::Expr(expr) => {
                // For side-effect expressions like function calls
                self.gen_expr_temp(expr, out);
            }

            _ => {}
        }
    }

    fn gen_expr_into(&mut self, expr: &Expr, dst: RegisterOrSpill, out: &mut TokenStream) {
        match expr {
            Expr::Literal(Literal::Float(f)) => {
                let f = *f as f32;
                out.extend(quote! {
                    asm.loadi(#dst, #f);
                });
            }

            Expr::Literal(Literal::Int(i)) => {
                let i = *i as i32;
                out.extend(quote! {
                    asm.loadi_int(#dst, #i);
                });
            }

            Expr::Variable(name) if name == "TID" => {
                out.extend(quote! {
                    asm.get_tid(#dst);
                });
            }

            Expr::Variable(name) => {
                let src = self.regalloc.get(name).expect("undefined variable");
                if src != dst {
                    out.extend(quote! {
                        asm.mov(#dst, #src);
                    });
                }
            }

            Expr::Binary { op, left, right } => {
                let left_reg = self.gen_expr_temp(left, out);
                let right_reg = self.gen_expr_temp(right, out);

                let op_method = match op {
                    BinOp::Add => quote! { add },
                    BinOp::Sub => quote! { sub },
                    BinOp::Mul => quote! { mul },
                    BinOp::Div => quote! { div },
                    BinOp::Lt => quote! { lt },
                    BinOp::Le => quote! { le },
                    BinOp::Gt => quote! { gt },
                    BinOp::Ge => quote! { ge },
                    BinOp::Eq => quote! { eq },
                    BinOp::Ne => quote! { ne },
                    BinOp::BitAnd => quote! { bit_and },
                    BinOp::BitOr => quote! { bit_or },
                    BinOp::BitXor => quote! { bit_xor },
                    BinOp::Shl => quote! { shl },
                    BinOp::Shr => quote! { shr_u },
                    _ => panic!("unsupported op"),
                };

                out.extend(quote! {
                    asm.#op_method(#dst, #left_reg, #right_reg);
                });
            }

            Expr::Call { func, args } => {
                self.gen_builtin_call(func, args, dst, out);
            }

            Expr::StateRead(idx) => {
                let idx_reg = self.gen_expr_temp(idx, out);
                out.extend(quote! {
                    asm.ld(#dst, #idx_reg, 0);
                });
            }

            Expr::AtomicRead(idx) => {
                let idx_reg = self.gen_expr_temp(idx, out);
                out.extend(quote! {
                    asm.atomic_load(#dst, #idx_reg);
                });
            }

            Expr::Cast { expr, ty } => {
                let src = self.gen_expr_temp(expr, out);
                match ty {
                    Type::F32 => out.extend(quote! { asm.int_to_f(#dst, #src); }),
                    Type::U32 => out.extend(quote! { asm.f_to_uint(#dst, #src); }),
                    Type::I32 => out.extend(quote! { asm.f_to_int(#dst, #src); }),
                    _ => panic!("unsupported cast"),
                }
            }

            _ => panic!("unsupported expr"),
        }
    }

    fn gen_builtin_call(&mut self, func: &str, args: &[Expr], dst: RegisterOrSpill, out: &mut TokenStream) {
        match func {
            "f4" => {
                // f4(x, y, z, w) -> construct float4
                let x = self.gen_expr_temp(&args[0], out);
                let y = self.gen_expr_temp(&args[1], out);
                let z = self.gen_expr_temp(&args[2], out);
                let w = self.gen_expr_temp(&args[3], out);
                out.extend(quote! {
                    asm.pack_f4(#dst, #x, #y, #z, #w);
                });
            }

            "emit_quad" => {
                // emit_quad(pos, size, color, depth)
                let pos = self.gen_expr_temp(&args[0], out);
                let size = self.gen_expr_temp(&args[1], out);
                let color = self.gen_expr_temp(&args[2], out);
                let depth = self.gen_expr_temp(&args[3], out);
                out.extend(quote! {
                    asm.emit_quad(#pos, #size, #color, #depth);
                });
            }

            "sin" | "cos" | "sqrt" | "abs" => {
                let arg = self.gen_expr_temp(&args[0], out);
                let op = match func {
                    "sin" => quote! { sin },
                    "cos" => quote! { cos },
                    "sqrt" => quote! { sqrt },
                    "abs" => quote! { abs },
                    _ => unreachable!(),
                };
                out.extend(quote! {
                    asm.#op(#dst, #arg);
                });
            }

            "min" | "max" => {
                let a = self.gen_expr_temp(&args[0], out);
                let b = self.gen_expr_temp(&args[1], out);
                let op = if func == "min" { quote! { min } } else { quote! { max } };
                out.extend(quote! {
                    asm.#op(#dst, #a, #b);
                });
            }

            "atomic_add" => {
                let addr = self.gen_expr_temp(&args[0], out);
                let val = self.gen_expr_temp(&args[1], out);
                out.extend(quote! {
                    asm.atomic_add(#dst, #addr, #val);
                });
            }

            _ => panic!("unknown builtin: {}", func),
        }
    }

    fn new_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("{}_{}", prefix, self.label_counter)
    }
}
```

### Macro Entry Point

```rust
// In gpu_dsl_macro/src/lib.rs

use proc_macro::TokenStream;
use syn::parse_macro_input;

mod ast;
mod parser;
mod regalloc;
mod codegen;

#[proc_macro]
pub fn gpu_kernel(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ast::Function);

    let mut codegen = codegen::CodeGenerator::new();
    let output = codegen.generate(&parsed);

    output.into()
}
```

---

## Example Compilation

### Input

```rust
gpu_kernel! {
    fn simple_add() {
        let a = 10;
        let b = 20;
        let c = a + b;
        STATE[0] = f4(c as f32, 0.0, 0.0, 1.0);
    }
}
```

### Output (Generated Rust)

```rust
pub fn simple_add() -> Vec<u64> {
    let mut asm = BytecodeAssembler::new();

    // let a = 10
    asm.loadi_int(8, 10);

    // let b = 20
    asm.loadi_int(9, 20);

    // let c = a + b
    asm.int_add(10, 8, 9);

    // STATE[0] = f4(c as f32, 0.0, 0.0, 1.0)
    asm.int_to_f(11, 10);      // c as f32
    asm.loadi(12, 0.0);
    asm.loadi(13, 0.0);
    asm.loadi(14, 1.0);
    asm.pack_f4(15, 11, 12, 13, 14);
    asm.loadi_uint(16, 0);     // index 0
    asm.st(15, 16, 0);

    asm.halt();
    asm.finish()
}
```

---

## Test Cases

### Test File: `tests/test_phase3_gpu_dsl.rs`

```rust
//! Phase 3: GPU DSL Macro Tests
//!
//! THE GPU IS THE COMPUTER.
//! Write GPU apps in Rust-like syntax.

use metal::Device;
use gpu_dsl::gpu_kernel;
use rust_experiment::gpu_os::bytecode_vm::BytecodeVM;

#[test]
fn test_simple_arithmetic() {
    gpu_kernel! {
        fn add_test() {
            let a = 10.0;
            let b = 20.0;
            STATE[0] = f4(a + b, a - b, a * b, a / b);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&add_test());
    vm.execute(&device);

    let result = vm.read_state_f4(0);
    assert_eq!(result, [30.0, -10.0, 200.0, 0.5]);
}

#[test]
fn test_integer_ops() {
    gpu_kernel! {
        fn int_test() {
            let a = 100;
            let b = 7;
            let sum = a + b;
            let diff = a - b;
            let prod = a * b;
            let quot = a / b;
            let rem = a % b;

            STATE[0] = f4(sum as f32, diff as f32, prod as f32, quot as f32);
            STATE[1] = f4(rem as f32, 0.0, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&int_test());
    vm.execute(&device);

    let r0 = vm.read_state_f4(0);
    let r1 = vm.read_state_f4(1);

    assert_eq!(r0[0], 107.0);  // sum
    assert_eq!(r0[1], 93.0);   // diff
    assert_eq!(r0[2], 700.0);  // prod
    assert_eq!(r0[3], 14.0);   // quot
    assert_eq!(r1[0], 2.0);    // rem
}

#[test]
fn test_if_else() {
    gpu_kernel! {
        fn if_test() {
            let x = 10;
            let result;

            if x > 5 {
                result = 100;
            } else {
                result = 200;
            }

            STATE[0] = f4(result as f32, 0.0, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&if_test());
    vm.execute(&device);

    let result = vm.read_state_f4(0);
    assert_eq!(result[0], 100.0);  // x > 5, so result = 100
}

#[test]
fn test_for_loop() {
    gpu_kernel! {
        fn loop_test() {
            let sum = 0;
            for i in 1..11 {
                sum = sum + i;
            }
            STATE[0] = f4(sum as f32, 0.0, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&loop_test());
    vm.execute(&device);

    let result = vm.read_state_f4(0);
    assert_eq!(result[0], 55.0);  // 1+2+3+...+10 = 55
}

#[test]
fn test_while_loop() {
    gpu_kernel! {
        fn while_test() {
            let i = 0;
            let sum = 0;
            while i < 5 {
                sum = sum + i;
                i = i + 1;
            }
            STATE[0] = f4(sum as f32, i as f32, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&while_test());
    vm.execute(&device);

    let result = vm.read_state_f4(0);
    assert_eq!(result[0], 10.0);  // 0+1+2+3+4 = 10
    assert_eq!(result[1], 5.0);   // i ended at 5
}

#[test]
fn test_break_continue() {
    gpu_kernel! {
        fn break_test() {
            let sum = 0;
            for i in 0..100 {
                if i >= 5 {
                    break;
                }
                sum = sum + i;
            }
            STATE[0] = f4(sum as f32, 0.0, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&break_test());
    vm.execute(&device);

    let result = vm.read_state_f4(0);
    assert_eq!(result[0], 10.0);  // 0+1+2+3+4 = 10 (broke at 5)
}

#[test]
fn test_atomic_operations() {
    gpu_kernel! {
        fn atomic_test() {
            // Initialize atomic counter
            ATOMIC[0] = 0;

            // Increment 5 times
            for _ in 0..5 {
                atomic_add(0, 1);
            }

            // Read final value
            let count = ATOMIC[0];
            STATE[0] = f4(count as f32, 0.0, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&atomic_test());
    vm.execute(&device);

    let result = vm.read_state_f4(0);
    assert_eq!(result[0], 5.0);
}

#[test]
fn test_math_functions() {
    gpu_kernel! {
        fn math_test() {
            let x = 1.0;
            let s = sin(x);
            let c = cos(x);
            let sq = sqrt(4.0);
            let ab = abs(-5.0);
            let mn = min(3.0, 7.0);
            let mx = max(3.0, 7.0);

            STATE[0] = f4(s, c, sq, ab);
            STATE[1] = f4(mn, mx, 0.0, 0.0);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&math_test());
    vm.execute(&device);

    let r0 = vm.read_state_f4(0);
    let r1 = vm.read_state_f4(1);

    assert!((r0[0] - 0.841).abs() < 0.01);  // sin(1)
    assert!((r0[1] - 0.540).abs() < 0.01);  // cos(1)
    assert_eq!(r0[2], 2.0);                  // sqrt(4)
    assert_eq!(r0[3], 5.0);                  // abs(-5)
    assert_eq!(r1[0], 3.0);                  // min(3, 7)
    assert_eq!(r1[1], 7.0);                  // max(3, 7)
}

#[test]
fn test_emit_quad() {
    gpu_kernel! {
        fn quad_test() {
            emit_quad(f4(100.0, 100.0, 0.0, 0.0), f4(50.0, 50.0, 0.0, 0.0),
                      f4(1.0, 0.0, 0.0, 1.0), 0.5);
        }
    }

    let device = Device::system_default().expect("No Metal device");
    let vm = BytecodeVM::new(&device).expect("Failed to create VM");

    vm.load_program(&quad_test());
    vm.execute(&device);

    // Check vertex count
    let vertex_count = vm.read_vertex_count();
    assert_eq!(vertex_count, 6);  // 2 triangles = 6 vertices
}
```

---

## Error Messages

The macro should provide helpful compile-time errors:

```rust
// Error: undefined variable
gpu_kernel! {
    fn test() {
        let x = y + 1;  // error: undefined variable `y`
    }
}

// Error: type mismatch
gpu_kernel! {
    fn test() {
        let x = 5;       // inferred as int
        let y = x + 1.0; // error: cannot add int and float
    }
}

// Error: unsupported feature
gpu_kernel! {
    fn test() {
        let v = vec![1, 2, 3];  // error: Vec not supported on GPU
    }
}

// Error: break outside loop
gpu_kernel! {
    fn test() {
        break;  // error: break outside of loop
    }
}
```

---

## Success Criteria

1. **All test kernels compile** and produce correct bytecode
2. **Bytecode executes correctly** on GPU VM
3. **Compile-time errors** for unsupported patterns
4. **Register allocation** works for kernels with >16 variables
5. **No runtime crashes** from generated bytecode

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Macro expansion time | < 100ms for 1000-line kernel |
| Generated bytecode size | < 2x hand-written bytecode |
| Runtime performance | Equal to hand-written bytecode |

---

## Files to Create

| File | Purpose |
|------|---------|
| `gpu_dsl_macro/Cargo.toml` | Proc macro crate |
| `gpu_dsl_macro/src/lib.rs` | Macro entry point |
| `gpu_dsl_macro/src/ast.rs` | AST definitions |
| `gpu_dsl_macro/src/parser.rs` | Syn-based parser |
| `gpu_dsl_macro/src/regalloc.rs` | Register allocator |
| `gpu_dsl_macro/src/codegen.rs` | Code generator |
| `gpu_dsl/Cargo.toml` | User-facing crate |
| `gpu_dsl/src/lib.rs` | Re-exports and prelude |
| `tests/test_phase3_gpu_dsl.rs` | Test file |

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| Runtime parsing | Slow, error-prone | Compile-time macro |
| Dynamic register alloc | Can fail at runtime | Static analysis |
| Generating Metal source | Bypasses our VM | Generate bytecode |
| Allowing recursion | GPU can't do it | Compile-time error |

---

## Next Phase

**Phase 4: WASM→Bytecode Translator** - Compile real `no_std` Rust via WASM.
