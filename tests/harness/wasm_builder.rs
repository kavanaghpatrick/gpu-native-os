//! WASM binary builder for testing
//!
//! Provides programmatic construction of WASM modules for opcode testing.

/// WASM opcodes as raw bytes
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

    // i32 Arithmetic
    pub const I32_ADD: u8 = 0x6A;
    pub const I32_SUB: u8 = 0x6B;
    pub const I32_MUL: u8 = 0x6C;
    pub const I32_DIV_S: u8 = 0x6D;
    pub const I32_DIV_U: u8 = 0x6E;
    pub const I32_REM_S: u8 = 0x6F;
    pub const I32_REM_U: u8 = 0x70;

    // i32 Bitwise
    pub const I32_AND: u8 = 0x71;
    pub const I32_OR: u8 = 0x72;
    pub const I32_XOR: u8 = 0x73;
    pub const I32_SHL: u8 = 0x74;
    pub const I32_SHR_S: u8 = 0x75;
    pub const I32_SHR_U: u8 = 0x76;
    pub const I32_ROTL: u8 = 0x77;
    pub const I32_ROTR: u8 = 0x78;

    // i32 Unary
    pub const I32_CLZ: u8 = 0x67;
    pub const I32_CTZ: u8 = 0x68;
    pub const I32_POPCNT: u8 = 0x69;

    // i32 Comparison
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

    // i64 Arithmetic
    pub const I64_ADD: u8 = 0x7C;
    pub const I64_SUB: u8 = 0x7D;
    pub const I64_MUL: u8 = 0x7E;
    pub const I64_DIV_S: u8 = 0x7F;
    pub const I64_DIV_U: u8 = 0x80;
    pub const I64_REM_S: u8 = 0x81;
    pub const I64_REM_U: u8 = 0x82;

    // i64 Bitwise
    pub const I64_AND: u8 = 0x83;
    pub const I64_OR: u8 = 0x84;
    pub const I64_XOR: u8 = 0x85;
    pub const I64_SHL: u8 = 0x86;
    pub const I64_SHR_S: u8 = 0x87;
    pub const I64_SHR_U: u8 = 0x88;
    pub const I64_ROTL: u8 = 0x89;
    pub const I64_ROTR: u8 = 0x8A;

    // i64 Unary
    pub const I64_CLZ: u8 = 0x79;
    pub const I64_CTZ: u8 = 0x7A;
    pub const I64_POPCNT: u8 = 0x7B;

    // i64 Comparison
    pub const I64_EQZ: u8 = 0x50;
    pub const I64_EQ: u8 = 0x51;
    pub const I64_NE: u8 = 0x52;
    pub const I64_LT_S: u8 = 0x53;
    pub const I64_LT_U: u8 = 0x54;
    pub const I64_GT_S: u8 = 0x55;
    pub const I64_GT_U: u8 = 0x56;
    pub const I64_LE_S: u8 = 0x57;
    pub const I64_LE_U: u8 = 0x58;
    pub const I64_GE_S: u8 = 0x59;
    pub const I64_GE_U: u8 = 0x5A;

    // f32 Arithmetic
    pub const F32_ABS: u8 = 0x8B;
    pub const F32_NEG: u8 = 0x8C;
    pub const F32_CEIL: u8 = 0x8D;
    pub const F32_FLOOR: u8 = 0x8E;
    pub const F32_TRUNC: u8 = 0x8F;
    pub const F32_NEAREST: u8 = 0x90;
    pub const F32_SQRT: u8 = 0x91;
    pub const F32_ADD: u8 = 0x92;
    pub const F32_SUB: u8 = 0x93;
    pub const F32_MUL: u8 = 0x94;
    pub const F32_DIV: u8 = 0x95;
    pub const F32_MIN: u8 = 0x96;
    pub const F32_MAX: u8 = 0x97;
    pub const F32_COPYSIGN: u8 = 0x98;

    // f32 Comparison
    pub const F32_EQ: u8 = 0x5B;
    pub const F32_NE: u8 = 0x5C;
    pub const F32_LT: u8 = 0x5D;
    pub const F32_GT: u8 = 0x5E;
    pub const F32_LE: u8 = 0x5F;
    pub const F32_GE: u8 = 0x60;

    // f64 Arithmetic
    pub const F64_ABS: u8 = 0x99;
    pub const F64_NEG: u8 = 0x9A;
    pub const F64_CEIL: u8 = 0x9B;
    pub const F64_FLOOR: u8 = 0x9C;
    pub const F64_TRUNC: u8 = 0x9D;
    pub const F64_NEAREST: u8 = 0x9E;
    pub const F64_SQRT: u8 = 0x9F;
    pub const F64_ADD: u8 = 0xA0;
    pub const F64_SUB: u8 = 0xA1;
    pub const F64_MUL: u8 = 0xA2;
    pub const F64_DIV: u8 = 0xA3;
    pub const F64_MIN: u8 = 0xA4;
    pub const F64_MAX: u8 = 0xA5;
    pub const F64_COPYSIGN: u8 = 0xA6;

    // f64 Comparison
    pub const F64_EQ: u8 = 0x61;
    pub const F64_NE: u8 = 0x62;
    pub const F64_LT: u8 = 0x63;
    pub const F64_GT: u8 = 0x64;
    pub const F64_LE: u8 = 0x65;
    pub const F64_GE: u8 = 0x66;

    // Conversions
    pub const I32_WRAP_I64: u8 = 0xA7;
    pub const I32_TRUNC_F32_S: u8 = 0xA8;
    pub const I32_TRUNC_F32_U: u8 = 0xA9;
    pub const I32_TRUNC_F64_S: u8 = 0xAA;
    pub const I32_TRUNC_F64_U: u8 = 0xAB;
    pub const I64_EXTEND_I32_S: u8 = 0xAC;
    pub const I64_EXTEND_I32_U: u8 = 0xAD;
    pub const I64_TRUNC_F32_S: u8 = 0xAE;
    pub const I64_TRUNC_F32_U: u8 = 0xAF;
    pub const I64_TRUNC_F64_S: u8 = 0xB0;
    pub const I64_TRUNC_F64_U: u8 = 0xB1;
    pub const F32_CONVERT_I32_S: u8 = 0xB2;
    pub const F32_CONVERT_I32_U: u8 = 0xB3;
    pub const F32_CONVERT_I64_S: u8 = 0xB4;
    pub const F32_CONVERT_I64_U: u8 = 0xB5;
    pub const F32_DEMOTE_F64: u8 = 0xB6;
    pub const F64_CONVERT_I32_S: u8 = 0xB7;
    pub const F64_CONVERT_I32_U: u8 = 0xB8;
    pub const F64_CONVERT_I64_S: u8 = 0xB9;
    pub const F64_CONVERT_I64_U: u8 = 0xBA;
    pub const F64_PROMOTE_F32: u8 = 0xBB;

    // Reinterpret
    pub const I32_REINTERPRET_F32: u8 = 0xBC;
    pub const I64_REINTERPRET_F64: u8 = 0xBD;
    pub const F32_REINTERPRET_I32: u8 = 0xBE;
    pub const F64_REINTERPRET_I64: u8 = 0xBF;

    // Control flow
    pub const UNREACHABLE: u8 = 0x00;
    pub const NOP: u8 = 0x01;
    pub const BLOCK: u8 = 0x02;
    pub const LOOP: u8 = 0x03;
    pub const IF: u8 = 0x04;
    pub const ELSE: u8 = 0x05;
    pub const END: u8 = 0x0B;
    pub const BR: u8 = 0x0C;
    pub const BR_IF: u8 = 0x0D;
    pub const BR_TABLE: u8 = 0x0E;
    pub const RETURN: u8 = 0x0F;
    pub const CALL: u8 = 0x10;
    pub const CALL_INDIRECT: u8 = 0x11;

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
    pub const I32_LOAD8_S: u8 = 0x2C;
    pub const I32_LOAD8_U: u8 = 0x2D;
    pub const I32_LOAD16_S: u8 = 0x2E;
    pub const I32_LOAD16_U: u8 = 0x2F;
    pub const I64_LOAD8_S: u8 = 0x30;
    pub const I64_LOAD8_U: u8 = 0x31;
    pub const I64_LOAD16_S: u8 = 0x32;
    pub const I64_LOAD16_U: u8 = 0x33;
    pub const I64_LOAD32_S: u8 = 0x34;
    pub const I64_LOAD32_U: u8 = 0x35;
    pub const I32_STORE: u8 = 0x36;
    pub const I64_STORE: u8 = 0x37;
    pub const F32_STORE: u8 = 0x38;
    pub const F64_STORE: u8 = 0x39;
    pub const I32_STORE8: u8 = 0x3A;
    pub const I32_STORE16: u8 = 0x3B;
    pub const I64_STORE8: u8 = 0x3C;
    pub const I64_STORE16: u8 = 0x3D;
    pub const I64_STORE32: u8 = 0x3E;
    pub const MEMORY_SIZE: u8 = 0x3F;
    pub const MEMORY_GROW: u8 = 0x40;

    // Drop/Select
    pub const DROP: u8 = 0x1A;
    pub const SELECT: u8 = 0x1B;

    /// Encode signed LEB128
    pub fn leb128_signed(buf: &mut Vec<u8>, mut val: i64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            let more = !((val == 0 && byte & 0x40 == 0) || (val == -1 && byte & 0x40 != 0));
            if more {
                buf.push(byte | 0x80);
            } else {
                buf.push(byte);
                break;
            }
        }
    }

    /// Encode unsigned LEB128
    pub fn leb128_unsigned(buf: &mut Vec<u8>, mut val: u64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                buf.push(byte | 0x80);
            } else {
                buf.push(byte);
                break;
            }
        }
    }
}

/// WASM value types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValType {
    I32 = 0x7F,
    I64 = 0x7E,
    F32 = 0x7D,
    F64 = 0x7C,
}

/// Build WASM modules programmatically
pub struct WasmBuilder {
    types: Vec<(Vec<ValType>, Vec<ValType>)>,  // (params, results)
    functions: Vec<(u32, Vec<ValType>, Vec<u8>)>,  // (type_idx, locals, body)
    exports: Vec<(String, u32)>,  // (name, func_idx)
    memory: Option<(u32, Option<u32>)>,  // (min, max)
}

impl WasmBuilder {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            functions: Vec::new(),
            exports: Vec::new(),
            memory: None,
        }
    }

    /// Add a function type signature, returns type index
    pub fn add_type(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
        let idx = self.types.len() as u32;
        self.types.push((params.to_vec(), results.to_vec()));
        idx
    }

    /// Add a function with given type, locals, and body
    pub fn add_func(&mut self, type_idx: u32, locals: &[ValType], body: &[u8]) -> u32 {
        let idx = self.functions.len() as u32;
        self.functions.push((type_idx, locals.to_vec(), body.to_vec()));
        idx
    }

    /// Export a function by name
    pub fn export_func(&mut self, name: &str, func_idx: u32) {
        self.exports.push((name.to_string(), func_idx));
    }

    /// Add memory (min pages, optional max pages)
    pub fn add_memory(&mut self, min: u32, max: Option<u32>) {
        self.memory = Some((min, max));
    }

    /// Build the final WASM binary
    pub fn build(&self) -> Vec<u8> {
        let mut wasm = Vec::new();

        // Magic number and version
        wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D]); // \0asm
        wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

        // Type section (section 1)
        if !self.types.is_empty() {
            let mut type_section = Vec::new();
            wasm_ops::leb128_unsigned(&mut type_section, self.types.len() as u64);
            for (params, results) in &self.types {
                type_section.push(0x60); // func type
                wasm_ops::leb128_unsigned(&mut type_section, params.len() as u64);
                for p in params {
                    type_section.push(*p as u8);
                }
                wasm_ops::leb128_unsigned(&mut type_section, results.len() as u64);
                for r in results {
                    type_section.push(*r as u8);
                }
            }
            wasm.push(0x01); // section id
            wasm_ops::leb128_unsigned(&mut wasm, type_section.len() as u64);
            wasm.extend(type_section);
        }

        // Function section (section 3) - declares function signatures
        if !self.functions.is_empty() {
            let mut func_section = Vec::new();
            wasm_ops::leb128_unsigned(&mut func_section, self.functions.len() as u64);
            for (type_idx, _, _) in &self.functions {
                wasm_ops::leb128_unsigned(&mut func_section, *type_idx as u64);
            }
            wasm.push(0x03); // section id
            wasm_ops::leb128_unsigned(&mut wasm, func_section.len() as u64);
            wasm.extend(func_section);
        }

        // Memory section (section 5)
        if let Some((min, max)) = self.memory {
            let mut mem_section = Vec::new();
            wasm_ops::leb128_unsigned(&mut mem_section, 1); // 1 memory
            if let Some(max_val) = max {
                mem_section.push(0x01); // has max
                wasm_ops::leb128_unsigned(&mut mem_section, min as u64);
                wasm_ops::leb128_unsigned(&mut mem_section, max_val as u64);
            } else {
                mem_section.push(0x00); // no max
                wasm_ops::leb128_unsigned(&mut mem_section, min as u64);
            }
            wasm.push(0x05); // section id
            wasm_ops::leb128_unsigned(&mut wasm, mem_section.len() as u64);
            wasm.extend(mem_section);
        }

        // Export section (section 7)
        if !self.exports.is_empty() {
            let mut export_section = Vec::new();
            wasm_ops::leb128_unsigned(&mut export_section, self.exports.len() as u64);
            for (name, func_idx) in &self.exports {
                wasm_ops::leb128_unsigned(&mut export_section, name.len() as u64);
                export_section.extend(name.as_bytes());
                export_section.push(0x00); // func export
                wasm_ops::leb128_unsigned(&mut export_section, *func_idx as u64);
            }
            wasm.push(0x07); // section id
            wasm_ops::leb128_unsigned(&mut wasm, export_section.len() as u64);
            wasm.extend(export_section);
        }

        // Code section (section 10) - function bodies
        if !self.functions.is_empty() {
            let mut code_section = Vec::new();
            wasm_ops::leb128_unsigned(&mut code_section, self.functions.len() as u64);

            for (_, locals, body) in &self.functions {
                let mut func_body = Vec::new();

                // Locals: group consecutive same-type locals
                if locals.is_empty() {
                    wasm_ops::leb128_unsigned(&mut func_body, 0);
                } else {
                    let mut groups: Vec<(u32, ValType)> = Vec::new();
                    for local in locals {
                        if let Some(last) = groups.last_mut() {
                            if last.1 == *local {
                                last.0 += 1;
                                continue;
                            }
                        }
                        groups.push((1, *local));
                    }
                    wasm_ops::leb128_unsigned(&mut func_body, groups.len() as u64);
                    for (count, ty) in groups {
                        wasm_ops::leb128_unsigned(&mut func_body, count as u64);
                        func_body.push(ty as u8);
                    }
                }

                // Body
                func_body.extend(body);

                // Write func body with size
                wasm_ops::leb128_unsigned(&mut code_section, func_body.len() as u64);
                code_section.extend(func_body);
            }

            wasm.push(0x0A); // section id
            wasm_ops::leb128_unsigned(&mut wasm, code_section.len() as u64);
            wasm.extend(code_section);
        }

        wasm
    }

    // ============ Convenience builders ============

    /// Build: fn() -> i32 { body }
    pub fn i32_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[], &[ValType::I32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(i32) -> i32 { body }
    pub fn i32_unary_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[ValType::I32], &[ValType::I32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(i32, i32) -> i32 { body }
    pub fn i32_binary_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[ValType::I32, ValType::I32], &[ValType::I32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn() -> i64 { body }
    pub fn i64_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[], &[ValType::I64]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(i64) -> i64 { body }
    pub fn i64_unary_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[ValType::I64], &[ValType::I64]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(i64, i64) -> i64 { body }
    pub fn i64_binary_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[ValType::I64, ValType::I64], &[ValType::I64]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn() -> f32 { body }
    pub fn f32_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[], &[ValType::F32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(f32, f32) -> f32 { body }
    pub fn f32_binary_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[ValType::F32, ValType::F32], &[ValType::F32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn() -> f64 { body }
    pub fn f64_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[], &[ValType::F64]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn(f64, f64) -> f64 { body }
    pub fn f64_binary_func(body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[ValType::F64, ValType::F64], &[ValType::F64]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build: fn() -> i32 { body } with memory
    pub fn i32_func_with_memory(body: &[u8], pages: u32) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        b.add_memory(pages, None);
        let ty = b.add_type(&[], &[ValType::I32]);
        let f = b.add_func(ty, &[], body);
        b.export_func("main", f);
        b.build()
    }

    /// Build function with locals: fn() -> i32 { locals; body }
    pub fn i32_func_with_locals(locals: &[ValType], body: &[u8]) -> Vec<u8> {
        let mut b = WasmBuilder::new();
        let ty = b.add_type(&[], &[ValType::I32]);
        let f = b.add_func(ty, locals, body);
        b.export_func("main", f);
        b.build()
    }
}

impl Default for WasmBuilder {
    fn default() -> Self {
        Self::new()
    }
}
