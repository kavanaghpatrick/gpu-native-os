//! GPU Bytecode Executor for testing
//!
//! Execute WASM/bytecode on GPU and return results.

use metal::Device;
use wasm_translator::{WasmTranslator, TranslatorConfig, TranslateError};
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type};
use std::fmt;

/// Errors that can occur during execution
#[derive(Debug)]
pub enum ExecutionError {
    NoMetalDevice,
    SystemCreationFailed(String),
    TranslationFailed(TranslateError),
    LaunchFailed(String),
    NoResult,
    Timeout,
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionError::NoMetalDevice => write!(f, "No Metal device available"),
            ExecutionError::SystemCreationFailed(e) => write!(f, "System creation failed: {}", e),
            ExecutionError::TranslationFailed(e) => write!(f, "Translation failed: {:?}", e),
            ExecutionError::LaunchFailed(e) => write!(f, "Launch failed: {}", e),
            ExecutionError::NoResult => write!(f, "No result returned"),
            ExecutionError::Timeout => write!(f, "Execution timed out"),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<TranslateError> for ExecutionError {
    fn from(e: TranslateError) -> Self {
        ExecutionError::TranslationFailed(e)
    }
}

/// Execute bytecode on GPU and return results
pub struct BytecodeExecutor {
    device: Device,
    system: GpuAppSystem,
}

impl BytecodeExecutor {
    /// Create a new executor
    pub fn new() -> Result<Self, ExecutionError> {
        let device = Device::system_default()
            .ok_or(ExecutionError::NoMetalDevice)?;
        let mut system = GpuAppSystem::new(&device)
            .map_err(|e| ExecutionError::SystemCreationFailed(e))?;
        system.set_use_parallel_megakernel(true);
        Ok(Self { device, system })
    }

    /// Translate WASM and execute on GPU, return i32 result
    pub fn run_wasm_i32(&mut self, wasm: &[u8]) -> Result<i32, ExecutionError> {
        let bytecode = self.translate(wasm)?;
        self.run_bytecode_i32(&bytecode)
    }

    /// Execute raw bytecode, return i32 result
    pub fn run_bytecode_i32(&mut self, bytecode: &[u8]) -> Result<i32, ExecutionError> {
        let slot = self.system.launch_by_type(app_type::BYTECODE)
            .ok_or_else(|| ExecutionError::LaunchFailed("Failed to launch bytecode app".into()))?;
        self.system.write_app_state(slot, bytecode);
        self.system.run_frame();
        let result = self.system.read_bytecode_result(slot)
            .ok_or(ExecutionError::NoResult);

        // Close the app to free the slot for reuse
        self.system.close_app(slot);

        result
    }

    /// Translate WASM and execute on GPU, return i64 result
    pub fn run_wasm_i64(&mut self, wasm: &[u8]) -> Result<i64, ExecutionError> {
        let bytecode = self.translate(wasm)?;
        self.run_bytecode_i64(&bytecode)
    }

    /// Execute raw bytecode, return i64 result
    /// Reads both lo and hi 32-bit words from state[0].xy to reconstruct the i64
    pub fn run_bytecode_i64(&mut self, bytecode: &[u8]) -> Result<i64, ExecutionError> {
        let slot = self.system.launch_by_type(app_type::BYTECODE)
            .ok_or_else(|| ExecutionError::LaunchFailed("Failed to launch bytecode app".into()))?;
        self.system.write_app_state(slot, bytecode);
        self.system.run_frame();

        // Read the full i64 result (lo and hi words from state[0].xy)
        let result = self.system.read_bytecode_result_i64(slot)
            .ok_or(ExecutionError::NoResult)?;

        // Close the app to free the slot for reuse
        self.system.close_app(slot);

        Ok(result)
    }

    /// Translate WASM and execute on GPU, return f32 result
    pub fn run_wasm_f32(&mut self, wasm: &[u8]) -> Result<f32, ExecutionError> {
        let result_bits = self.run_wasm_i32(wasm)? as u32;
        Ok(f32::from_bits(result_bits))
    }

    /// Execute raw bytecode, return f32 result (via i32 bits - for backward compat)
    /// WARNING: This interprets the result as bits, use run_bytecode_f32_direct for float ops
    pub fn run_bytecode_f32(&mut self, bytecode: &[u8]) -> Result<f32, ExecutionError> {
        let result_bits = self.run_bytecode_i32(bytecode)? as u32;
        Ok(f32::from_bits(result_bits))
    }

    /// Execute raw bytecode, return f32 result directly (not as bits)
    /// USE THIS for float operations (LN, EXP, SIN, COS, etc.)
    pub fn run_bytecode_f32_direct(&mut self, bytecode: &[u8]) -> Result<f32, ExecutionError> {
        let slot = self.system.launch_by_type(app_type::BYTECODE)
            .ok_or_else(|| ExecutionError::LaunchFailed("Failed to launch bytecode app".into()))?;
        self.system.write_app_state(slot, bytecode);
        self.system.run_frame();
        let result = self.system.read_bytecode_result_f32_direct(slot)
            .ok_or(ExecutionError::NoResult);

        self.system.close_app(slot);
        result
    }

    /// Translate WASM and execute on GPU, return f64 result
    /// THE GPU IS THE COMPUTER - Metal doesn't support native f64, so we use double-single:
    /// value = hi + lo where hi is stored in .x and lo in .y (Issue #27)
    /// This provides ~47 bits of mantissa precision (vs 52 for native f64)
    pub fn run_wasm_f64(&mut self, wasm: &[u8]) -> Result<f64, ExecutionError> {
        let bytecode = self.translate(wasm)?;
        self.run_bytecode_f64(&bytecode)
    }

    /// Execute raw bytecode, return f64 result using double-single format
    /// THE GPU IS THE COMPUTER - reads hi from .x, lo from .y, returns hi + lo
    pub fn run_bytecode_f64(&mut self, bytecode: &[u8]) -> Result<f64, ExecutionError> {
        let slot = self.system.launch_by_type(app_type::BYTECODE)
            .ok_or_else(|| ExecutionError::LaunchFailed("Failed to launch bytecode app".into()))?;
        self.system.write_app_state(slot, bytecode);
        self.system.run_frame();

        // Read the f64 result in double-single format (hi + lo)
        let result = self.system.read_bytecode_result_f64(slot)
            .ok_or(ExecutionError::NoResult)?;

        self.system.close_app(slot);
        Ok(result)
    }

    /// Translate WASM to bytecode without executing (for inspection)
    pub fn translate(&self, wasm: &[u8]) -> Result<Vec<u8>, ExecutionError> {
        let translator = WasmTranslator::new(TranslatorConfig::default());
        translator.translate(wasm).map_err(ExecutionError::from)
    }

    /// Get bytecode statistics
    pub fn bytecode_stats(bytecode: &[u8]) -> BytecodeStats {
        if bytecode.len() < 16 {
            return BytecodeStats::default();
        }
        let code_size = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
        BytecodeStats {
            total_bytes: bytecode.len(),
            instruction_count: code_size as usize,
            header_bytes: 16,
            code_bytes: code_size as usize * 8,
        }
    }
}

/// Statistics about generated bytecode
#[derive(Debug, Default)]
pub struct BytecodeStats {
    pub total_bytes: usize,
    pub instruction_count: usize,
    pub header_bytes: usize,
    pub code_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::wasm_builder::{WasmBuilder, wasm_ops};

    #[test]
    fn test_executor_creation() {
        let exec = BytecodeExecutor::new();
        assert!(exec.is_ok(), "Should create executor on macOS with Metal");
    }

    #[test]
    fn test_simple_i32_const() {
        let mut exec = BytecodeExecutor::new().unwrap();
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(42)[..].to_vec(),
            vec![wasm_ops::END],
        ].concat());

        let result = exec.run_wasm_i32(&wasm).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_simple_i32_add() {
        let mut exec = BytecodeExecutor::new().unwrap();
        let wasm = WasmBuilder::i32_func(&[
            wasm_ops::i32_const(10)[..].to_vec(),
            wasm_ops::i32_const(32)[..].to_vec(),
            vec![wasm_ops::I32_ADD],
            vec![wasm_ops::END],
        ].concat());

        let result = exec.run_wasm_i32(&wasm).unwrap();
        assert_eq!(result, 42);
    }
}
