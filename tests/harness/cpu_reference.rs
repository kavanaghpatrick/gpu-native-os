//! CPU Reference Implementation using wasmtime
//!
//! Execute WASM on CPU for comparison against GPU results.

use wasmtime::{Engine, Module, Store, Instance, Val};
use std::fmt;

/// Errors from CPU reference execution
#[derive(Debug)]
pub enum CpuError {
    ModuleCreation(String),
    Instantiation(String),
    NoEntryPoint,
    CallFailed(String),
    WrongResultType,
}

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuError::ModuleCreation(e) => write!(f, "Module creation failed: {}", e),
            CpuError::Instantiation(e) => write!(f, "Instantiation failed: {}", e),
            CpuError::NoEntryPoint => write!(f, "No entry point found (tried: main, _start, gpu_main)"),
            CpuError::CallFailed(e) => write!(f, "Function call failed: {}", e),
            CpuError::WrongResultType => write!(f, "Function returned wrong type"),
        }
    }
}

impl std::error::Error for CpuError {}

/// Execute WASM on wasmtime for reference comparison
pub struct CpuReference {
    engine: Engine,
}

impl CpuReference {
    pub fn new() -> Self {
        Self {
            engine: Engine::default()
        }
    }

    /// Execute WASM and return i32 result
    pub fn run_i32(&self, wasm: &[u8]) -> Result<i32, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        // Try common entry point names
        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let mut results = [Val::I32(0)];
                func.call(&mut store, &[], &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::I32(v) => return Ok(v),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM with i32 arguments and return i32 result
    pub fn run_i32_with_args(&self, wasm: &[u8], args: &[i32]) -> Result<i32, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let params: Vec<Val> = args.iter().map(|&v| Val::I32(v)).collect();
                let mut results = [Val::I32(0)];
                func.call(&mut store, &params, &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::I32(v) => return Ok(v),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM and return i64 result
    pub fn run_i64(&self, wasm: &[u8]) -> Result<i64, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let mut results = [Val::I64(0)];
                func.call(&mut store, &[], &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::I64(v) => return Ok(v),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM with i64 arguments and return i64 result
    pub fn run_i64_with_args(&self, wasm: &[u8], args: &[i64]) -> Result<i64, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let params: Vec<Val> = args.iter().map(|&v| Val::I64(v)).collect();
                let mut results = [Val::I64(0)];
                func.call(&mut store, &params, &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::I64(v) => return Ok(v),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM and return f32 result
    pub fn run_f32(&self, wasm: &[u8]) -> Result<f32, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let mut results = [Val::F32(0)];
                func.call(&mut store, &[], &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::F32(bits) => return Ok(f32::from_bits(bits)),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM with f32 arguments and return f32 result
    pub fn run_f32_with_args(&self, wasm: &[u8], args: &[f32]) -> Result<f32, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let params: Vec<Val> = args.iter().map(|&v| Val::F32(v.to_bits())).collect();
                let mut results = [Val::F32(0)];
                func.call(&mut store, &params, &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::F32(bits) => return Ok(f32::from_bits(bits)),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM and return f64 result
    pub fn run_f64(&self, wasm: &[u8]) -> Result<f64, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let mut results = [Val::F64(0)];
                func.call(&mut store, &[], &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::F64(bits) => return Ok(f64::from_bits(bits)),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Execute WASM with f64 arguments and return f64 result
    pub fn run_f64_with_args(&self, wasm: &[u8], args: &[f64]) -> Result<f64, CpuError> {
        let (mut store, instance) = self.instantiate(wasm)?;

        for name in ["main", "_start", "gpu_main", "__main"] {
            if let Some(func) = instance.get_func(&mut store, name) {
                let params: Vec<Val> = args.iter().map(|&v| Val::F64(v.to_bits())).collect();
                let mut results = [Val::F64(0)];
                func.call(&mut store, &params, &mut results)
                    .map_err(|e| CpuError::CallFailed(e.to_string()))?;

                match results[0] {
                    Val::F64(bits) => return Ok(f64::from_bits(bits)),
                    _ => return Err(CpuError::WrongResultType),
                }
            }
        }

        Err(CpuError::NoEntryPoint)
    }

    /// Create a module and instance from WASM bytes
    fn instantiate(&self, wasm: &[u8]) -> Result<(Store<()>, Instance), CpuError> {
        let module = Module::new(&self.engine, wasm)
            .map_err(|e| CpuError::ModuleCreation(e.to_string()))?;

        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| CpuError::Instantiation(e.to_string()))?;

        Ok((store, instance))
    }
}

impl Default for CpuReference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::wasm_builder::{WasmBuilder, wasm_ops};

    #[test]
    fn test_cpu_reference_creation() {
        let _cpu = CpuReference::new();
    }

    #[test]
    fn test_cpu_simple_i32() {
        let cpu = CpuReference::new();
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(42)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        let result = cpu.run_i32(&wasm).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_cpu_i32_add() {
        let cpu = CpuReference::new();
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(10)[..].to_vec(),
            wasm_ops::i32_const(32)[..].to_vec(),
            vec![wasm_ops::I32_ADD],
            vec![wasm_ops::END],
        ].concat());

        let result = cpu.run_i32(&wasm).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_cpu_i64_const() {
        let cpu = CpuReference::new();
        let wasm = WasmBuilder::i64_func(&[
            wasm_ops::i64_const(0x123456789ABCDEF0_i64)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        let result = cpu.run_i64(&wasm).unwrap();
        assert_eq!(result, 0x123456789ABCDEF0_i64);
    }
}
