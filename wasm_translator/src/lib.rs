//! WASM to GPU Bytecode Translator
//!
//! THE GPU IS THE COMPUTER.
//!
//! This crate translates WebAssembly binaries to GPU bytecode that can run
//! on our GPU bytecode VM. This enables compiling real `no_std` Rust code
//! via the standard rustc→WASM pipeline.
//!
//! # Example
//!
//! ```ignore
//! use wasm_translator::WasmTranslator;
//!
//! // Compile Rust to WASM: cargo build --target wasm32-unknown-unknown
//! let wasm_bytes = std::fs::read("my_app.wasm")?;
//!
//! // Translate to GPU bytecode
//! let translator = WasmTranslator::new(Default::default());
//! let bytecode = translator.translate(&wasm_bytes)?;
//!
//! // Load into GPU VM and execute
//! ```
//!
//! # Supported WASM Features
//!
//! - i32 operations: full support via Phase 1 integer ops
//! - f32 operations: full support
//! - Control flow: block, loop, if/else, br, br_if
//! - Local/global variables
//! - Linear memory access
//!
//! # Unsupported Features
//!
//! - i64 operations (no native 64-bit on GPU)
//! - f64 operations (demoted to f32)
//! - memory.grow (GPU can't allocate)
//! - function calls (inlined or error)
//! - SIMD, threads, exceptions

mod emit;
mod stack;
mod translate;
mod types;

pub use types::{TranslateError, TranslatorConfig, WasmModule, GpuIntrinsic, ImportedFunc};

use translate::TranslationContext;
use wasmparser::{Parser, Payload, CompositeInnerType, ElementKind, ElementItems, ConstExpr};

/// WASM to GPU bytecode translator
pub struct WasmTranslator {
    config: TranslatorConfig,
}

impl WasmTranslator {
    /// Create a new translator with the given configuration
    pub fn new(config: TranslatorConfig) -> Self {
        Self { config }
    }

    /// Translate WASM binary to GPU bytecode
    pub fn translate(&self, wasm_bytes: &[u8]) -> Result<Vec<u8>, TranslateError> {
        let mut module = WasmModule::new();
        let mut code_bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
        let mut type_indices: Vec<u32> = Vec::new();

        // Parse WASM
        for payload in Parser::new(0).parse_all(wasm_bytes) {
            let payload = payload.map_err(|e| TranslateError::Parse(e.to_string()))?;

            match payload {
                Payload::TypeSection(reader) => {
                    for rec_group in reader {
                        let rec_group = rec_group.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        for subtype in rec_group.types() {
                            if let CompositeInnerType::Func(func_type) = &subtype.composite_type.inner {
                                let params: Vec<types::ValType> = func_type.params()
                                    .iter()
                                    .map(|v| convert_val_type(*v))
                                    .collect();
                                let results: Vec<types::ValType> = func_type.results()
                                    .iter()
                                    .map(|v| convert_val_type(*v))
                                    .collect();
                                module.types.push(types::FuncType { params, results });
                            }
                        }
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // IMPORT SECTION (Phase 5 - Issue #178)
                // Parse function imports for GPU intrinsics
                // ═══════════════════════════════════════════════════════════════
                Payload::ImportSection(reader) => {
                    for import in reader {
                        let import = import.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        if let wasmparser::TypeRef::Func(type_idx) = import.ty {
                            eprintln!("[IMPORT] {}::{} -> type {}", import.module, import.name, type_idx);
                            let imported = ImportedFunc::from_import(
                                import.module,
                                import.name,
                                type_idx,
                            );
                            module.imports.push(imported);
                        }
                    }
                }

                Payload::FunctionSection(reader) => {
                    for func in reader {
                        let type_idx = func.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        type_indices.push(type_idx);
                    }
                }

                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        if let wasmparser::ExternalKind::Func = export.kind {
                            module.exports.insert(export.name.to_string(), export.index);
                        }
                    }
                }

                Payload::CodeSectionStart { .. } => {}

                Payload::CodeSectionEntry(body) => {
                    // Store function index -> type index mapping
                    let func_idx = code_bodies.len();
                    if func_idx < type_indices.len() {
                        module.functions.push(type_indices[func_idx]);
                    }
                    code_bodies.push(body);
                }

                // ═══════════════════════════════════════════════════════════════
                // TABLE SECTION (Issue #189 - call_indirect support)
                // Parse table definitions to know table sizes
                // ═══════════════════════════════════════════════════════════════
                Payload::TableSection(reader) => {
                    for table in reader {
                        let table = table.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        // Initialize table with the specified minimum size
                        let min_size = table.ty.initial as usize;
                        if module.func_table.len() < min_size {
                            module.func_table.resize(min_size, None);
                        }
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // ELEMENT SECTION (Issue #189 - call_indirect support)
                // Parse element segments to populate function table
                // ═══════════════════════════════════════════════════════════════
                Payload::ElementSection(reader) => {
                    for element in reader {
                        let element = element.map_err(|e| TranslateError::Parse(e.to_string()))?;

                        // Only process active segments (those that initialize table)
                        if let ElementKind::Active { table_index, offset_expr } = element.kind {
                            // We only support table 0 for now
                            let table_idx = table_index.unwrap_or(0);
                            if table_idx != 0 {
                                continue;
                            }

                            // Evaluate constant offset expression
                            let offset = eval_const_expr(&offset_expr)?;

                            // Issue #261 fix: Limit table size to prevent OOM on malformed WASM
                            const MAX_TABLE_SIZE: usize = 65536;

                            // Extract function indices from the element items
                            match element.items {
                                ElementItems::Functions(funcs) => {
                                    for (i, func_result) in funcs.into_iter().enumerate() {
                                        let func_idx = func_result.map_err(|e| TranslateError::Parse(e.to_string()))?;
                                        let table_slot = offset as usize + i;
                                        // Issue #261: Bounds check before resize
                                        if table_slot >= MAX_TABLE_SIZE {
                                            return Err(TranslateError::Invalid(
                                                format!("Element offset {} exceeds maximum table size {}", table_slot, MAX_TABLE_SIZE)
                                            ));
                                        }
                                        // Ensure table is large enough
                                        if module.func_table.len() <= table_slot {
                                            module.func_table.resize(table_slot + 1, None);
                                        }
                                        module.func_table[table_slot] = Some(func_idx);
                                    }
                                }
                                ElementItems::Expressions(_, exprs) => {
                                    // Expression elements (ref.func ...) - less common
                                    for (i, expr_result) in exprs.into_iter().enumerate() {
                                        let expr = expr_result.map_err(|e| TranslateError::Parse(e.to_string()))?;
                                        if let Some(func_idx) = eval_ref_func_expr(&expr) {
                                            let table_slot = offset as usize + i;
                                            // Issue #261: Bounds check before resize
                                            if table_slot >= MAX_TABLE_SIZE {
                                                return Err(TranslateError::Invalid(
                                                    format!("Element offset {} exceeds maximum table size {}", table_slot, MAX_TABLE_SIZE)
                                                ));
                                            }
                                            if module.func_table.len() <= table_slot {
                                                module.func_table.resize(table_slot + 1, None);
                                            }
                                            module.func_table[table_slot] = Some(func_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // MEMORY SECTION (Issue #255)
                // Parse memory limits to validate WASM fits in GPU allocation
                // ═══════════════════════════════════════════════════════════════
                Payload::MemorySection(reader) => {
                    for memory in reader {
                        let memory = memory.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        // Store minimum pages required
                        module.memory_pages = memory.initial as u32;
                        eprintln!("[MEMORY] min_pages={}, max_pages={:?}",
                            memory.initial, memory.maximum);
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // GLOBAL SECTION (Issue #255)
                // Parse global definitions and initial values
                // ═══════════════════════════════════════════════════════════════
                Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        let ty = convert_val_type(global.ty.content_type);
                        let mutable = global.ty.mutable;
                        let init_value = eval_const_expr_i64(&global.init_expr)?;
                        eprintln!("[GLOBAL] type={:?}, mutable={}, init={}", ty, mutable, init_value);
                        module.global_defs.push(types::GlobalDef { ty, mutable, init_value });
                        module.globals.push(ty);
                    }
                }

                // ═══════════════════════════════════════════════════════════════
                // DATA SECTION (Issue #255)
                // Parse data segments for static data initialization
                // ═══════════════════════════════════════════════════════════════
                Payload::DataSection(reader) => {
                    for data in reader {
                        let data = data.map_err(|e| TranslateError::Parse(e.to_string()))?;
                        match data.kind {
                            wasmparser::DataKind::Active { memory_index, offset_expr } => {
                                if memory_index != 0 {
                                    continue; // Only support memory 0
                                }
                                let offset = eval_const_expr(&offset_expr)? as u32;
                                let bytes = data.data.to_vec();
                                eprintln!("[DATA] offset={}, len={}", offset, bytes.len());
                                module.data_segments.push(types::DataSegment { offset, data: bytes });
                            }
                            wasmparser::DataKind::Passive => {
                                // Passive segments need memory.init to copy - skip for now
                                eprintln!("[DATA] passive segment (skipped)");
                            }
                        }
                    }
                }

                _ => {}
            }
        }

        // Find entry point
        // Standard names first - developer writes normal Rust, we handle GPU
        let entry_idx = module.exports.get("main")
            .or_else(|| module.exports.get("_start"))
            .or_else(|| module.exports.get("gpu_main"))  // Legacy fallback
            .copied()
            .ok_or(TranslateError::NoEntryPoint)?;

        // Adjust for imports: entry_idx is absolute, code_bodies is for defined functions only
        let import_count = module.import_count();
        let defined_func_idx = entry_idx.checked_sub(import_count)
            .ok_or_else(|| TranslateError::Invalid("cannot call import as entry point".into()))?;

        // Get function type
        let type_idx = *module.functions.get(defined_func_idx as usize)
            .ok_or(TranslateError::Invalid("entry function not found".into()))?;
        let func_type = module.types.get(type_idx as usize)
            .ok_or(TranslateError::Invalid("function type not found".into()))?;

        // Get code body
        let body = code_bodies.get(defined_func_idx as usize)
            .ok_or(TranslateError::Invalid("function code not found".into()))?;

        // Parse locals
        let mut local_count = 0u32;
        let locals_reader = body.get_locals_reader()
            .map_err(|e| TranslateError::Parse(e.to_string()))?;
        for local in locals_reader {
            let (count, _ty) = local.map_err(|e| TranslateError::Parse(e.to_string()))?;
            local_count += count;
        }

        // ═══════════════════════════════════════════════════════════════════════════
        // PHASE 5: Pre-parse all function bodies for inlining support
        // THE GPU IS THE COMPUTER - all functions become GPU code
        // Also compute local counts for each function for proper inlining
        // ═══════════════════════════════════════════════════════════════════════════
        let mut function_ops: Vec<Vec<wasmparser::Operator<'_>>> = Vec::with_capacity(code_bodies.len());
        let mut function_local_counts: Vec<u32> = Vec::with_capacity(code_bodies.len());
        for code_body in &code_bodies {
            // Parse operators
            let ops_reader = code_body.get_operators_reader()
                .map_err(|e| TranslateError::Parse(e.to_string()))?;
            let mut ops = Vec::new();
            for op_result in ops_reader {
                let op = op_result.map_err(|e| TranslateError::Parse(e.to_string()))?;
                ops.push(op);
            }
            function_ops.push(ops);

            // Parse locals count
            let mut func_local_count = 0u32;
            let locals_reader = code_body.get_locals_reader()
                .map_err(|e| TranslateError::Parse(e.to_string()))?;
            for local in locals_reader {
                let (count, _ty) = local.map_err(|e| TranslateError::Parse(e.to_string()))?;
                func_local_count += count;
            }
            function_local_counts.push(func_local_count);
        }

        // Create translation context with module and function ops for inlining
        let param_count = func_type.params.len() as u32;
        let mut ctx = TranslationContext::new(param_count, local_count, &self.config)
            .with_module(&module, &function_ops, &function_local_counts);

        // ═══════════════════════════════════════════════════════════════════════════
        // PROLOGUE: Load parameters from state buffer
        // THE GPU IS THE COMPUTER - parameters come from GPU memory, not CPU
        //
        // Calling convention:
        //   state[0] = reserved for return value
        //   state[1] = param 0 (loaded into r4)
        //   state[2] = param 1 (loaded into r5)
        //   state[3] = param 2 (loaded into r6)
        //   state[4] = param 3 (loaded into r7)
        // ═══════════════════════════════════════════════════════════════════════════
        for i in 0..param_count.min(4) {
            let param_reg = 4 + i as u8;  // r4, r5, r6, r7
            let state_idx = 1 + i;        // state[1], state[2], state[3], state[4]
            // Load param from state[state_idx] into param_reg
            // Issue #213 fix: Use loadi (float VALUE) instead of loadi_uint (integer BITS)
            // Small integers as float bits are denormalized and get flushed to zero!
            ctx.emit.loadi(30, state_idx as f32);  // r30 = state index as float
            ctx.emit.ld(param_reg, 30, 0.0);       // param_reg = state[r30]
        }

        // ═══════════════════════════════════════════════════════════════════════════
        // CRITICAL FIX: Initialize ALL WASM globals from parsed GlobalSection (Issue #255)
        // THE GPU IS THE COMPUTER - WASM expects proper global initialization
        //
        // Previously only __stack_pointer (global 0) was hardcoded to 8192.
        // Now we initialize ALL globals from their actual WASM init values,
        // with special handling for __stack_pointer to fit our GPU memory model.
        //
        // WASM linear memory layout in our VM:
        //   - memory_base = 5248 bytes (where linear memory starts)
        //   - Available space ~64KB per app
        //   - Stack grows DOWN from initial pointer
        // ═══════════════════════════════════════════════════════════════════════════
        for (idx, global_def) in module.global_defs.iter().enumerate() {
            let global_offset = self.config.globals_base + (idx as u32);

            // Special case: __stack_pointer (usually global 0) needs adjustment
            // WASM typically sets it to 1MB, but we only have 64KB
            let init_value = if idx == 0 && global_def.init_value > 65536 {
                // Likely __stack_pointer - cap to our available space
                8192u32  // 8KB stack space
            } else {
                // Use actual init value (truncated to u32 for GPU storage)
                global_def.init_value as u32
            };

            ctx.emit.loadi_uint(29, init_value);  // r29 = init value
            ctx.emit.loadi_uint(30, global_offset);  // r30 = global slot
            ctx.emit.st(30, 29, 0.0);  // state[global_offset] = init_value
        }

        // If no globals were defined, still initialize stack pointer
        if module.global_defs.is_empty() {
            let stack_init_value: u32 = 8192;
            ctx.emit.loadi_uint(29, stack_init_value);
            ctx.emit.loadi_uint(30, self.config.globals_base);
            ctx.emit.st(30, 29, 0.0);
        }

        // ═══════════════════════════════════════════════════════════════════════════
        // DATA SEGMENT INITIALIZATION (Issue #255)
        // Copy static data from WASM data sections into linear memory
        // This handles string literals, initialized arrays, etc.
        // ═══════════════════════════════════════════════════════════════════════════
        for segment in &module.data_segments {
            // Generate bytecode to initialize data at segment.offset
            // For each 4-byte chunk, generate: loadi_uint + st
            let memory_base = self.config.memory_base;
            for (i, chunk) in segment.data.chunks(4).enumerate() {
                let addr = segment.offset + (i as u32 * 4);
                let value = match chunk.len() {
                    1 => chunk[0] as u32,
                    2 => u16::from_le_bytes([chunk[0], chunk[1]]) as u32,
                    3 => chunk[0] as u32 | (chunk[1] as u32) << 8 | (chunk[2] as u32) << 16,
                    4 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    _ => 0,
                };
                // Store value at memory_base + addr
                ctx.emit.loadi_uint(29, value);  // r29 = value (preserve integer bits!)
                // Calculate absolute address: (memory_base + addr) / 16 for float4 index
                let float4_idx = (memory_base + addr) / 16;
                // Issue #213 fix: Use loadi (float VALUE) for addresses, not loadi_uint (integer BITS)
                ctx.emit.loadi(30, float4_idx as f32);  // r30 = index as float
                ctx.emit.st(30, 29, 0.0);  // state[index] = value
            }
        }

        // Parse and translate operators from the entry function
        let entry_ops = &function_ops[defined_func_idx as usize];
        if std::env::var("TRACE_ENTRY").is_ok() {
            eprintln!("[DEBUG ENTRY] Entry function has {} ops", entry_ops.len());
            for (i, op) in entry_ops.iter().enumerate() {
                eprintln!("[DEBUG ENTRY] op[{}]: {:?}", i, op);
            }
        }
        for (i, op) in entry_ops.iter().enumerate() {
            if let Err(e) = ctx.translate_operator(op) {
                eprintln!("[TRANSLATE ERROR] Failed at instruction {}: {:?}", i, op);
                eprintln!("[TRANSLATE ERROR] Stack depth: {}", ctx.stack.depth());
                return Err(e);
            }
        }
        // ═══════════════════════════════════════════════════════════════════════════
        // EPILOGUE: Store return value to state[0], then halt
        // THE GPU IS THE COMPUTER - results go to GPU memory
        //
        // All return paths (explicit Return or implicit function end) jump here.
        // Return value is in r4 (set by Return handler or final expression).
        // ═══════════════════════════════════════════════════════════════════════════
        ctx.emit.define_label(ctx.epilogue_label);

        if !func_type.results.is_empty() {
            // Return value should be in r4 (convention from Return handler)
            // Or on the operand stack if implicit return
            let ret_reg = if let Ok(reg) = ctx.stack.pop() {
                // Move to r4 for consistency
                ctx.emit.mov(4, reg);
                4
            } else {
                4  // Return handler already put it in r4
            };

            // Store to state[3] (after allocator header at state[0-2])
            // State layout: SlabAllocator[0-2] (48 bytes) | result[3] (16 bytes) | heap[4+]
            // Issue #213 fix: Use loadi (float VALUE 3.0) instead of loadi_uint (integer BITS 0x3)
            // LD/ST now use uint(float) conversion, so we need float values not bit patterns
            //
            // Note: i32 values are now stored as float VALUES by LOADI_INT (Issue #213 fix).
            // No conversion needed here - ret_reg already contains a normalized float value.

            ctx.emit.loadi(30, 3.0);  // r30 = 3.0 (state index as float)
            ctx.emit.st(30, ret_reg, 0.0);  // state[3] = ret_reg (already float value)
        }

        // Emit final halt
        ctx.emit.halt();

        Ok(ctx.finish())
    }

    /// Translate WAT (text format) to GPU bytecode
    #[cfg(feature = "wat")]
    pub fn translate_wat(&self, wat: &str) -> Result<Vec<u8>, TranslateError> {
        let wasm = wat::parse_str(wat)
            .map_err(|e| TranslateError::Parse(e.to_string()))?;
        self.translate(&wasm)
    }
}

/// Convert wasmparser ValType to our ValType
fn convert_val_type(v: wasmparser::ValType) -> types::ValType {
    match v {
        wasmparser::ValType::I32 => types::ValType::I32,
        wasmparser::ValType::I64 => types::ValType::I64,
        wasmparser::ValType::F32 => types::ValType::F32,
        wasmparser::ValType::F64 => types::ValType::F64,
        _ => types::ValType::I32, // Default for ref types
    }
}

impl Default for WasmTranslator {
    fn default() -> Self {
        Self::new(TranslatorConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANT EXPRESSION EVALUATION (Issue #189 - call_indirect support)
// THE GPU IS THE COMPUTER - evaluate WASM const expressions at compile time
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluate a constant expression to get an i32 value
/// Used for element segment offsets (e.g., `(i32.const 1)`)
fn eval_const_expr(expr: &ConstExpr) -> Result<i32, TranslateError> {
    use wasmparser::Operator;

    let mut reader = expr.get_operators_reader();
    while let Ok(op) = reader.read() {
        match op {
            Operator::I32Const { value } => return Ok(value),
            Operator::End => break,
            _ => {}
        }
    }
    // Default to 0 if no i32.const found
    Ok(0)
}

/// Evaluate a constant expression to an i64 value (Issue #255)
/// Supports i32, i64, f32, f64 constant instructions
fn eval_const_expr_i64(expr: &ConstExpr) -> Result<i64, TranslateError> {
    use wasmparser::Operator;

    let mut reader = expr.get_operators_reader();
    while let Ok(op) = reader.read() {
        match op {
            Operator::I32Const { value } => return Ok(value as i64),
            Operator::I64Const { value } => return Ok(value),
            Operator::F32Const { value } => return Ok(value.bits() as i64),
            Operator::F64Const { value } => return Ok(value.bits() as i64),
            Operator::End => break,
            _ => {}
        }
    }
    // Default to 0 if no const found
    Ok(0)
}

/// Evaluate a ref.func expression to get the function index
/// Used for element expressions (e.g., `(ref.func $foo)`)
fn eval_ref_func_expr(expr: &ConstExpr) -> Option<u32> {
    use wasmparser::Operator;

    let mut reader = expr.get_operators_reader();
    while let Ok(op) = reader.read() {
        match op {
            Operator::RefFunc { function_index } => return Some(function_index),
            Operator::End => break,
            _ => {}
        }
    }
    None
}
