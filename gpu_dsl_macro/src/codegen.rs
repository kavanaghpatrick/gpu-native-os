//! Code generator for GPU DSL
//!
//! THE GPU IS THE COMPUTER.
//! Generates efficient bytecode through BytecodeAssembler calls.

use crate::ast::*;
use crate::regalloc::{Location, RegisterAllocator};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Loop context for break/continue
struct LoopContext {
    break_label: usize,
    continue_label: usize,
}

/// Code generator
pub struct CodeGenerator {
    regalloc: RegisterAllocator,
    loop_stack: Vec<LoopContext>,
    label_counter: usize,
    /// Tracks inferred types for variables
    var_types: std::collections::HashMap<String, Type>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            regalloc: RegisterAllocator::new(),
            loop_stack: Vec::new(),
            label_counter: 0,
            var_types: std::collections::HashMap::new(),
        }
    }

    /// Generate code for a GPU function
    pub fn generate(&mut self, func: &GpuFunction) -> TokenStream {
        let func_name = format_ident!("{}", func.name);

        // Generate body statements
        let body_code = self.gen_statements(&func.body);

        quote! {
            pub fn #func_name() -> Vec<u8> {
                use rust_experiment::gpu_os::gpu_app_system::{BytecodeAssembler, bytecode_op};

                let mut asm = BytecodeAssembler::new();

                // Label tracking for jumps
                let mut labels: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                let mut patches: Vec<(usize, usize)> = Vec::new();

                #body_code

                asm.halt();

                // Patch jump targets
                for (inst_idx, label_id) in patches {
                    if let Some(&target) = labels.get(&label_id) {
                        asm.patch_jump(inst_idx, target);
                    }
                }

                asm.build(10000)
            }
        }
    }

    fn gen_statements(&mut self, stmts: &[Statement]) -> TokenStream {
        let code: Vec<TokenStream> = stmts.iter().map(|s| self.gen_statement(s)).collect();
        quote! { #(#code)* }
    }

    fn gen_statement(&mut self, stmt: &Statement) -> TokenStream {
        match stmt {
            Statement::Let { name, value, .. } => {
                // Infer type from value
                let inferred_type = self.infer_type(value);
                self.var_types.insert(name.clone(), inferred_type);

                let loc = self.regalloc.allocate(name);
                let value_code = self.gen_expr_into(value, loc);
                quote! { #value_code }
            }

            Statement::Assign { target, value } => match target {
                AssignTarget::Variable(name) => {
                    let loc = self
                        .regalloc
                        .get(name)
                        .unwrap_or_else(|| self.regalloc.allocate(name));
                    self.gen_expr_into(value, loc)
                }
                AssignTarget::State(idx) => {
                    let idx_loc = self.regalloc.allocate_temp();
                    let val_loc = self.regalloc.allocate_temp();
                    let idx_code = self.gen_expr_into(idx, idx_loc);
                    let val_code = self.gen_expr_into(value, val_loc);
                    let idx_reg = self.loc_to_reg(idx_loc);
                    let val_reg = self.loc_to_reg(val_loc);
                    self.regalloc.free_temp(idx_loc);
                    self.regalloc.free_temp(val_loc);
                    quote! {
                        #idx_code
                        #val_code
                        asm.st(#idx_reg, #val_reg, 0.0);
                    }
                }
                AssignTarget::Atomic(idx) => {
                    let idx_loc = self.regalloc.allocate_temp();
                    let val_loc = self.regalloc.allocate_temp();
                    let idx_code = self.gen_expr_into(idx, idx_loc);
                    let val_code = self.gen_expr_into(value, val_loc);
                    let idx_reg = self.loc_to_reg(idx_loc);
                    let val_reg = self.loc_to_reg(val_loc);
                    self.regalloc.free_temp(idx_loc);
                    self.regalloc.free_temp(val_loc);
                    quote! {
                        #idx_code
                        #val_code
                        asm.atomic_store(#val_reg, #idx_reg);
                    }
                }
            },

            Statement::If {
                condition,
                then_body,
                else_body,
            } => {
                let else_label = self.new_label();
                let end_label = self.new_label();

                let cond_loc = self.regalloc.allocate_temp();
                let cond_code = self.gen_expr_into(condition, cond_loc);
                let cond_reg = self.loc_to_reg(cond_loc);
                self.regalloc.free_temp(cond_loc);

                let then_code = self.gen_statements(then_body);

                if let Some(else_stmts) = else_body {
                    let else_code = self.gen_statements(else_stmts);
                    quote! {
                        #cond_code
                        patches.push((asm.jz(#cond_reg, 0), #else_label));
                        #then_code
                        patches.push((asm.jmp(0), #end_label));
                        labels.insert(#else_label, asm.pc());
                        #else_code
                        labels.insert(#end_label, asm.pc());
                    }
                } else {
                    quote! {
                        #cond_code
                        patches.push((asm.jz(#cond_reg, 0), #end_label));
                        #then_code
                        labels.insert(#end_label, asm.pc());
                    }
                }
            }

            Statement::For {
                var,
                start,
                end,
                body,
            } => {
                let loop_label = self.new_label();
                let end_label = self.new_label();

                self.loop_stack.push(LoopContext {
                    break_label: end_label,
                    continue_label: loop_label,
                });

                // Track var as integer
                self.var_types.insert(var.clone(), Type::I32);

                // Initialize loop variable
                let var_loc = self.regalloc.allocate(var);
                let start_code = self.gen_expr_into(start, var_loc);

                // Compute limit
                let limit_loc = self.regalloc.allocate_temp();
                let limit_code = self.gen_expr_into(end, limit_loc);

                let var_reg = self.loc_to_reg(var_loc);
                let limit_reg = self.loc_to_reg(limit_loc);

                let body_code = self.gen_statements(body);

                self.loop_stack.pop();
                self.regalloc.free(var);
                self.regalloc.free_temp(limit_loc);

                let cmp_reg = self.regalloc.get_scratch(0);
                let one_reg = self.regalloc.get_scratch(1);

                quote! {
                    #start_code
                    #limit_code
                    labels.insert(#loop_label, asm.pc());
                    // Check i < limit
                    asm.int_lt_u(#cmp_reg, #var_reg, #limit_reg);
                    patches.push((asm.jz(#cmp_reg, 0), #end_label));
                    // Body
                    #body_code
                    // Increment
                    asm.loadi_int(#one_reg, 1);
                    asm.int_add(#var_reg, #var_reg, #one_reg);
                    patches.push((asm.jmp(0), #loop_label));
                    labels.insert(#end_label, asm.pc());
                }
            }

            Statement::While { condition, body } => {
                let loop_label = self.new_label();
                let end_label = self.new_label();

                self.loop_stack.push(LoopContext {
                    break_label: end_label,
                    continue_label: loop_label,
                });

                let cond_loc = self.regalloc.allocate_temp();
                let cond_code = self.gen_expr_into(condition, cond_loc);
                let cond_reg = self.loc_to_reg(cond_loc);
                self.regalloc.free_temp(cond_loc);

                let body_code = self.gen_statements(body);

                self.loop_stack.pop();

                quote! {
                    labels.insert(#loop_label, asm.pc());
                    #cond_code
                    patches.push((asm.jz(#cond_reg, 0), #end_label));
                    #body_code
                    patches.push((asm.jmp(0), #loop_label));
                    labels.insert(#end_label, asm.pc());
                }
            }

            Statement::Break => {
                let label = self
                    .loop_stack
                    .last()
                    .expect("break outside loop")
                    .break_label;
                quote! {
                    patches.push((asm.jmp(0), #label));
                }
            }

            Statement::Continue => {
                let label = self
                    .loop_stack
                    .last()
                    .expect("continue outside loop")
                    .continue_label;
                quote! {
                    patches.push((asm.jmp(0), #label));
                }
            }

            Statement::Expr(expr) => {
                // For side-effect expressions, just generate them
                let loc = self.regalloc.allocate_temp();
                let code = self.gen_expr_into(expr, loc);
                self.regalloc.free_temp(loc);
                code
            }
        }
    }

    fn gen_expr_into(&mut self, expr: &Expr, dst: Location) -> TokenStream {
        let dst_reg = self.loc_to_reg(dst);

        match expr {
            Expr::LitInt(i) => {
                let val = *i as i32;
                quote! { asm.loadi_int(#dst_reg, #val); }
            }

            Expr::LitFloat(f) => {
                let val = *f as f32;
                quote! { asm.loadi(#dst_reg, #val); }
            }

            Expr::LitBool(b) => {
                let val = if *b { 1i32 } else { 0i32 };
                quote! { asm.loadi_int(#dst_reg, #val); }
            }

            Expr::Var(name) => {
                // Built-in variables
                match name.as_str() {
                    "TID" => quote! { asm.mov(#dst_reg, 1); }, // r1 = TID
                    "TGSIZE" => quote! { asm.mov(#dst_reg, 2); }, // r2 = threadgroup size
                    _ => {
                        let src_loc = self.regalloc.get(name).expect(&format!("undefined variable: {}", name));
                        let src_reg = self.loc_to_reg(src_loc);
                        if src_reg != dst_reg {
                            quote! { asm.mov(#dst_reg, #src_reg); }
                        } else {
                            quote! {}
                        }
                    }
                }
            }

            Expr::Binary { op, left, right } => {
                let left_loc = self.regalloc.allocate_temp();
                let right_loc = self.regalloc.allocate_temp();

                let left_code = self.gen_expr_into(left, left_loc);
                let right_code = self.gen_expr_into(right, right_loc);

                let left_reg = self.loc_to_reg(left_loc);
                let right_reg = self.loc_to_reg(right_loc);

                // Infer if we need integer or float ops
                let left_type = self.infer_type(left);
                let use_int = left_type.is_integer();

                let op_code = self.gen_binary_op(*op, dst_reg, left_reg, right_reg, use_int);

                self.regalloc.free_temp(left_loc);
                self.regalloc.free_temp(right_loc);

                quote! {
                    #left_code
                    #right_code
                    #op_code
                }
            }

            Expr::Unary { op, operand } => {
                let operand_code = self.gen_expr_into(operand, dst);
                let op_code = match op {
                    UnaryOp::Neg => {
                        let operand_type = self.infer_type(operand);
                        if operand_type.is_integer() {
                            quote! { asm.int_neg(#dst_reg, #dst_reg); }
                        } else {
                            let zero_reg = self.regalloc.get_scratch(0);
                            quote! {
                                asm.loadi(#zero_reg, 0.0);
                                asm.sub(#dst_reg, #zero_reg, #dst_reg);
                            }
                        }
                    }
                    UnaryOp::Not => {
                        // Logical NOT: result = (val == 0) ? 1 : 0
                        let zero_reg = self.regalloc.get_scratch(0);
                        quote! {
                            asm.loadi_int(#zero_reg, 0);
                            asm.int_eq(#dst_reg, #dst_reg, #zero_reg);
                        }
                    }
                    UnaryOp::BitNot => {
                        quote! { asm.bit_not(#dst_reg, #dst_reg); }
                    }
                };
                quote! {
                    #operand_code
                    #op_code
                }
            }

            Expr::Call { func, args } => self.gen_call(func, args, dst),

            Expr::Field { base, field } => {
                let base_loc = self.regalloc.allocate_temp();
                let base_code = self.gen_expr_into(base, base_loc);
                let base_reg = self.loc_to_reg(base_loc);

                // Extract component
                let extract_code = match field.as_str() {
                    "x" | "0" => quote! { asm.mov(#dst_reg, #base_reg); }, // x is first component
                    "y" | "1" => {
                        // Need to extract y component
                        // For now, we assume float4 is stored with x in .x
                        // This is a simplification - real impl would extract
                        quote! {
                            // TODO: proper component extraction
                            asm.mov(#dst_reg, #base_reg);
                        }
                    }
                    "z" | "2" => quote! { asm.mov(#dst_reg, #base_reg); },
                    "w" | "3" => quote! { asm.mov(#dst_reg, #base_reg); },
                    _ => quote! { asm.mov(#dst_reg, #base_reg); },
                };

                self.regalloc.free_temp(base_loc);

                quote! {
                    #base_code
                    #extract_code
                }
            }

            Expr::Cast { expr, ty } => {
                let inner_code = self.gen_expr_into(expr, dst);
                let cast_code = match ty {
                    Type::F32 => quote! { asm.int_to_f(#dst_reg, #dst_reg); },
                    Type::I32 => quote! { asm.f_to_int(#dst_reg, #dst_reg); },
                    Type::U32 => quote! { asm.f_to_uint(#dst_reg, #dst_reg); },
                    _ => quote! {},
                };
                quote! {
                    #inner_code
                    #cast_code
                }
            }

            Expr::StateRead(idx) => {
                let idx_loc = self.regalloc.allocate_temp();
                let idx_code = self.gen_expr_into(idx, idx_loc);
                let idx_reg = self.loc_to_reg(idx_loc);
                self.regalloc.free_temp(idx_loc);
                quote! {
                    #idx_code
                    asm.ld(#dst_reg, #idx_reg, 0.0);
                }
            }

            Expr::AtomicRead(idx) => {
                let idx_loc = self.regalloc.allocate_temp();
                let idx_code = self.gen_expr_into(idx, idx_loc);
                let idx_reg = self.loc_to_reg(idx_loc);
                self.regalloc.free_temp(idx_loc);
                quote! {
                    #idx_code
                    asm.atomic_load(#dst_reg, #idx_reg);
                }
            }
        }
    }

    fn gen_binary_op(
        &self,
        op: BinOp,
        dst: u8,
        left: u8,
        right: u8,
        use_int: bool,
    ) -> TokenStream {
        match op {
            BinOp::Add => {
                if use_int {
                    quote! { asm.int_add(#dst, #left, #right); }
                } else {
                    quote! { asm.add(#dst, #left, #right); }
                }
            }
            BinOp::Sub => {
                if use_int {
                    quote! { asm.int_sub(#dst, #left, #right); }
                } else {
                    quote! { asm.sub(#dst, #left, #right); }
                }
            }
            BinOp::Mul => {
                if use_int {
                    quote! { asm.int_mul(#dst, #left, #right); }
                } else {
                    quote! { asm.mul(#dst, #left, #right); }
                }
            }
            BinOp::Div => {
                if use_int {
                    quote! { asm.int_div_s(#dst, #left, #right); }
                } else {
                    quote! { asm.div(#dst, #left, #right); }
                }
            }
            BinOp::Rem => {
                if use_int {
                    quote! { asm.int_rem_s(#dst, #left, #right); }
                } else {
                    quote! { asm.modulo(#dst, #left, #right); }
                }
            }
            BinOp::Eq => {
                if use_int {
                    quote! { asm.int_eq(#dst, #left, #right); }
                } else {
                    quote! { asm.eq(#dst, #left, #right); }
                }
            }
            BinOp::Ne => {
                if use_int {
                    quote! { asm.int_ne(#dst, #left, #right); }
                } else {
                    // ne = !(eq)
                    quote! {
                        asm.eq(#dst, #left, #right);
                        asm.loadi_int(#right, 0);
                        asm.int_eq(#dst, #dst, #right);
                    }
                }
            }
            BinOp::Lt => {
                if use_int {
                    quote! { asm.int_lt_s(#dst, #left, #right); }
                } else {
                    quote! { asm.lt(#dst, #left, #right); }
                }
            }
            BinOp::Le => {
                if use_int {
                    quote! { asm.int_le_s(#dst, #left, #right); }
                } else {
                    // le = lt || eq
                    quote! {
                        asm.lt(#dst, #left, #right);
                        asm.eq(30, #left, #right);
                        asm.bit_or(#dst, #dst, 30);
                    }
                }
            }
            BinOp::Gt => {
                if use_int {
                    // gt = !(le)
                    quote! {
                        asm.int_le_s(#dst, #left, #right);
                        asm.loadi_int(30, 0);
                        asm.int_eq(#dst, #dst, 30);
                    }
                } else {
                    quote! { asm.gt(#dst, #left, #right); }
                }
            }
            BinOp::Ge => {
                if use_int {
                    // ge = !(lt)
                    quote! {
                        asm.int_lt_s(#dst, #left, #right);
                        asm.loadi_int(30, 0);
                        asm.int_eq(#dst, #dst, 30);
                    }
                } else {
                    // ge = gt || eq
                    quote! {
                        asm.gt(#dst, #left, #right);
                        asm.eq(30, #left, #right);
                        asm.bit_or(#dst, #dst, 30);
                    }
                }
            }
            BinOp::And => {
                // Logical AND
                quote! { asm.bit_and(#dst, #left, #right); }
            }
            BinOp::Or => {
                // Logical OR
                quote! { asm.bit_or(#dst, #left, #right); }
            }
            BinOp::BitAnd => quote! { asm.bit_and(#dst, #left, #right); },
            BinOp::BitOr => quote! { asm.bit_or(#dst, #left, #right); },
            BinOp::BitXor => quote! { asm.bit_xor(#dst, #left, #right); },
            BinOp::Shl => quote! { asm.shl(#dst, #left, #right); },
            BinOp::Shr => quote! { asm.shr_u(#dst, #left, #right); },
        }
    }

    fn gen_call(&mut self, func: &str, args: &[Expr], dst: Location) -> TokenStream {
        let dst_reg = self.loc_to_reg(dst);

        match func {
            // Float4 constructor
            "f4" => {
                if args.len() == 4 {
                    let x_loc = self.regalloc.allocate_temp();
                    let y_loc = self.regalloc.allocate_temp();
                    let z_loc = self.regalloc.allocate_temp();
                    let w_loc = self.regalloc.allocate_temp();

                    let x_code = self.gen_expr_into(&args[0], x_loc);
                    let y_code = self.gen_expr_into(&args[1], y_loc);
                    let z_code = self.gen_expr_into(&args[2], z_loc);
                    let w_code = self.gen_expr_into(&args[3], w_loc);

                    let x_reg = self.loc_to_reg(x_loc);
                    let y_reg = self.loc_to_reg(y_loc);
                    let z_reg = self.loc_to_reg(z_loc);
                    let w_reg = self.loc_to_reg(w_loc);

                    self.regalloc.free_temp(x_loc);
                    self.regalloc.free_temp(y_loc);
                    self.regalloc.free_temp(z_loc);
                    self.regalloc.free_temp(w_loc);

                    // Move x to dst, then set y, z, w
                    quote! {
                        #x_code
                        #y_code
                        #z_code
                        #w_code
                        asm.mov(#dst_reg, #x_reg);
                        asm.sety(#dst_reg, 0.0);  // Will be overwritten
                        // For now, f4 just keeps x component
                        // Full f4 support needs vector pack instruction
                    }
                } else {
                    quote! { compile_error!("f4 requires 4 arguments"); }
                }
            }

            // Float2 constructor
            "f2" => {
                if args.len() == 2 {
                    let x_loc = self.regalloc.allocate_temp();
                    let x_code = self.gen_expr_into(&args[0], x_loc);
                    let x_reg = self.loc_to_reg(x_loc);
                    self.regalloc.free_temp(x_loc);
                    // Simplified: just return x
                    quote! {
                        #x_code
                        asm.mov(#dst_reg, #x_reg);
                    }
                } else {
                    quote! { compile_error!("f2 requires 2 arguments"); }
                }
            }

            // emit_quad(pos, size, color, depth)
            "emit_quad" => {
                if args.len() == 4 {
                    let pos_loc = self.regalloc.allocate_temp();
                    let size_loc = self.regalloc.allocate_temp();
                    let color_loc = self.regalloc.allocate_temp();
                    let depth_loc = self.regalloc.allocate_temp();

                    let pos_code = self.gen_expr_into(&args[0], pos_loc);
                    let size_code = self.gen_expr_into(&args[1], size_loc);
                    let color_code = self.gen_expr_into(&args[2], color_loc);
                    let depth_code = self.gen_expr_into(&args[3], depth_loc);

                    let pos_reg = self.loc_to_reg(pos_loc);
                    let size_reg = self.loc_to_reg(size_loc);
                    let color_reg = self.loc_to_reg(color_loc);
                    let depth_reg = self.loc_to_reg(depth_loc);

                    self.regalloc.free_temp(pos_loc);
                    self.regalloc.free_temp(size_loc);
                    self.regalloc.free_temp(color_loc);
                    self.regalloc.free_temp(depth_loc);

                    // Get depth as float literal if possible
                    let depth_val = match &args[3] {
                        Expr::LitFloat(f) => *f as f32,
                        _ => 0.5,
                    };

                    quote! {
                        #pos_code
                        #size_code
                        #color_code
                        asm.quad(#pos_reg, #size_reg, #color_reg, #depth_val);
                    }
                } else {
                    quote! { compile_error!("emit_quad requires 4 arguments"); }
                }
            }

            // Atomic operations
            "atomic_add" => {
                if args.len() == 2 {
                    let addr_loc = self.regalloc.allocate_temp();
                    let val_loc = self.regalloc.allocate_temp();

                    let addr_code = self.gen_expr_into(&args[0], addr_loc);
                    let val_code = self.gen_expr_into(&args[1], val_loc);

                    let addr_reg = self.loc_to_reg(addr_loc);
                    let val_reg = self.loc_to_reg(val_loc);

                    self.regalloc.free_temp(addr_loc);
                    self.regalloc.free_temp(val_loc);

                    quote! {
                        #addr_code
                        #val_code
                        asm.atomic_add(#dst_reg, #addr_reg, #val_reg);
                    }
                } else {
                    quote! { compile_error!("atomic_add requires 2 arguments"); }
                }
            }

            "atomic_inc" => {
                if args.len() == 1 {
                    let addr_loc = self.regalloc.allocate_temp();
                    let addr_code = self.gen_expr_into(&args[0], addr_loc);
                    let addr_reg = self.loc_to_reg(addr_loc);
                    self.regalloc.free_temp(addr_loc);
                    quote! {
                        #addr_code
                        asm.atomic_inc(#dst_reg, #addr_reg);
                    }
                } else {
                    quote! { compile_error!("atomic_inc requires 1 argument"); }
                }
            }

            // Math functions
            "sin" | "cos" | "sqrt" | "abs" => {
                if args.len() == 1 {
                    let arg_code = self.gen_expr_into(&args[0], dst);
                    // These need to be implemented as shader intrinsics
                    // For now, stub them
                    quote! {
                        #arg_code
                        // TODO: Add math intrinsic opcodes
                    }
                } else {
                    quote! { compile_error!("Math function requires 1 argument"); }
                }
            }

            "min" | "max" => {
                if args.len() == 2 {
                    let a_loc = self.regalloc.allocate_temp();
                    let b_loc = self.regalloc.allocate_temp();

                    let a_code = self.gen_expr_into(&args[0], a_loc);
                    let b_code = self.gen_expr_into(&args[1], b_loc);

                    let a_reg = self.loc_to_reg(a_loc);
                    let b_reg = self.loc_to_reg(b_loc);

                    self.regalloc.free_temp(a_loc);
                    self.regalloc.free_temp(b_loc);

                    // min/max via comparison
                    if func == "min" {
                        quote! {
                            #a_code
                            #b_code
                            asm.lt(30, #a_reg, #b_reg);
                            // If a < b: dst = a, else dst = b
                            // Simplified: select based on comparison
                            // TODO: Add proper select instruction
                            asm.mov(#dst_reg, #a_reg);
                        }
                    } else {
                        quote! {
                            #a_code
                            #b_code
                            asm.gt(30, #a_reg, #b_reg);
                            asm.mov(#dst_reg, #a_reg);
                        }
                    }
                } else {
                    quote! { compile_error!("min/max require 2 arguments"); }
                }
            }

            // Method-style calls (converted from x.method() syntax)
            "xy" => {
                // Extract xy from a vector
                if args.len() == 1 {
                    self.gen_expr_into(&args[0], dst)
                } else {
                    quote! { compile_error!("xy requires 1 argument"); }
                }
            }

            _ => {
                let func_ident = format_ident!("{}", func);
                quote! { compile_error!(concat!("Unknown function: ", stringify!(#func_ident))); }
            }
        }
    }

    fn loc_to_reg(&self, loc: Location) -> u8 {
        match loc {
            Location::Register(r) => r,
            Location::Spill(_) => {
                // For spilled values, use scratch register
                // Real impl would load from memory
                self.regalloc.get_scratch(0)
            }
        }
    }

    fn new_label(&mut self) -> usize {
        let label = self.label_counter;
        self.label_counter += 1;
        label
    }

    fn infer_type(&self, expr: &Expr) -> Type {
        match expr {
            Expr::LitInt(_) => Type::I32,
            Expr::LitFloat(_) => Type::F32,
            Expr::LitBool(_) => Type::Bool,
            Expr::Var(name) => {
                // Check known variables
                if let Some(ty) = self.var_types.get(name) {
                    return *ty;
                }
                // Built-ins
                match name.as_str() {
                    "TID" | "TGSIZE" => Type::U32,
                    _ => Type::F32, // Default to float
                }
            }
            Expr::Binary { left, .. } => self.infer_type(left),
            Expr::Unary { operand, .. } => self.infer_type(operand),
            Expr::Cast { ty, .. } => *ty,
            Expr::Call { func, .. } => {
                match func.as_str() {
                    "f4" => Type::F32x4,
                    "f2" | "sin" | "cos" | "sqrt" | "abs" | "min" | "max" => Type::F32,
                    "atomic_add" | "atomic_inc" => Type::U32,
                    _ => Type::F32,
                }
            }
            Expr::StateRead(_) => Type::F32x4,
            Expr::AtomicRead(_) => Type::U32,
            Expr::Field { .. } => Type::F32,
        }
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}
