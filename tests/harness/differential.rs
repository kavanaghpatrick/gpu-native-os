//! Differential Testing Framework
//!
//! Compare GPU execution against CPU reference to catch bugs.

use crate::harness::executor::{BytecodeExecutor, ExecutionError};
use crate::harness::cpu_reference::{CpuReference, CpuError};
use std::fmt;

/// Errors from differential testing
#[derive(Debug)]
pub enum DiffError {
    CpuError(CpuError),
    GpuError(ExecutionError),
    Mismatch { cpu: String, gpu: String, context: String },
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffError::CpuError(e) => write!(f, "CPU error: {}", e),
            DiffError::GpuError(e) => write!(f, "GPU error: {}", e),
            DiffError::Mismatch { cpu, gpu, context } =>
                write!(f, "Mismatch for {}: CPU={}, GPU={}", context, cpu, gpu),
        }
    }
}

impl std::error::Error for DiffError {}

/// Compare GPU execution against CPU reference
pub struct DifferentialTester {
    gpu: BytecodeExecutor,
    cpu: CpuReference,
}

impl DifferentialTester {
    /// Create a new differential tester
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            gpu: BytecodeExecutor::new()?,
            cpu: CpuReference::new(),
        })
    }

    /// Assert GPU and CPU produce same i32 result
    pub fn assert_same_i32(&mut self, wasm: &[u8], context: &str) {
        let cpu_result = self.cpu.run_i32(wasm)
            .unwrap_or_else(|e| panic!("CPU execution failed for {}: {}", context, e));
        let gpu_result = self.gpu.run_wasm_i32(wasm)
            .unwrap_or_else(|e| panic!("GPU execution failed for {}: {}", context, e));

        assert_eq!(
            cpu_result, gpu_result,
            "Mismatch for {}: CPU={}, GPU={}",
            context, cpu_result, gpu_result
        );
    }

    /// Assert GPU and CPU produce same i32 result, returning Result instead of panicking
    pub fn check_same_i32(&mut self, wasm: &[u8], context: &str) -> Result<i32, DiffError> {
        let cpu_result = self.cpu.run_i32(wasm)
            .map_err(DiffError::CpuError)?;
        let gpu_result = self.gpu.run_wasm_i32(wasm)
            .map_err(DiffError::GpuError)?;

        if cpu_result != gpu_result {
            return Err(DiffError::Mismatch {
                cpu: cpu_result.to_string(),
                gpu: gpu_result.to_string(),
                context: context.to_string(),
            });
        }

        Ok(cpu_result)
    }

    /// Assert GPU and CPU produce same i64 result
    pub fn assert_same_i64(&mut self, wasm: &[u8], context: &str) {
        let cpu_result = self.cpu.run_i64(wasm)
            .unwrap_or_else(|e| panic!("CPU execution failed for {}: {}", context, e));
        let gpu_result = self.gpu.run_wasm_i64(wasm)
            .unwrap_or_else(|e| panic!("GPU execution failed for {}: {}", context, e));

        assert_eq!(
            cpu_result, gpu_result,
            "Mismatch for {}: CPU={} (0x{:016X}), GPU={} (0x{:016X})",
            context, cpu_result, cpu_result as u64, gpu_result, gpu_result as u64
        );
    }

    /// Assert GPU and CPU produce same i64 result, returning Result
    pub fn check_same_i64(&mut self, wasm: &[u8], context: &str) -> Result<i64, DiffError> {
        let cpu_result = self.cpu.run_i64(wasm)
            .map_err(DiffError::CpuError)?;
        let gpu_result = self.gpu.run_wasm_i64(wasm)
            .map_err(DiffError::GpuError)?;

        if cpu_result != gpu_result {
            return Err(DiffError::Mismatch {
                cpu: format!("{} (0x{:016X})", cpu_result, cpu_result as u64),
                gpu: format!("{} (0x{:016X})", gpu_result, gpu_result as u64),
                context: context.to_string(),
            });
        }

        Ok(cpu_result)
    }

    /// Assert GPU and CPU produce same f32 result (with epsilon tolerance)
    pub fn assert_same_f32(&mut self, wasm: &[u8], epsilon: f32, context: &str) {
        let cpu_result = self.cpu.run_f32(wasm)
            .unwrap_or_else(|e| panic!("CPU execution failed for {}: {}", context, e));
        let gpu_result = self.gpu.run_wasm_f32(wasm)
            .unwrap_or_else(|e| panic!("GPU execution failed for {}: {}", context, e));

        // Handle NaN specially
        if cpu_result.is_nan() && gpu_result.is_nan() {
            return; // Both NaN is acceptable
        }

        // Handle infinity
        if cpu_result.is_infinite() && gpu_result.is_infinite() {
            assert_eq!(
                cpu_result.is_sign_positive(), gpu_result.is_sign_positive(),
                "Mismatch for {}: CPU={}, GPU={} (infinity sign differs)",
                context, cpu_result, gpu_result
            );
            return;
        }

        let diff = (cpu_result - gpu_result).abs();
        assert!(
            diff <= epsilon,
            "Mismatch for {}: CPU={}, GPU={}, diff={} (epsilon={})",
            context, cpu_result, gpu_result, diff, epsilon
        );
    }

    /// Assert GPU and CPU produce same f32 result, returning Result
    pub fn check_same_f32(&mut self, wasm: &[u8], epsilon: f32, context: &str) -> Result<f32, DiffError> {
        let cpu_result = self.cpu.run_f32(wasm)
            .map_err(DiffError::CpuError)?;
        let gpu_result = self.gpu.run_wasm_f32(wasm)
            .map_err(DiffError::GpuError)?;

        // Handle NaN
        if cpu_result.is_nan() && gpu_result.is_nan() {
            return Ok(cpu_result);
        }

        // Handle infinity
        if cpu_result.is_infinite() && gpu_result.is_infinite() {
            if cpu_result.is_sign_positive() != gpu_result.is_sign_positive() {
                return Err(DiffError::Mismatch {
                    cpu: cpu_result.to_string(),
                    gpu: gpu_result.to_string(),
                    context: context.to_string(),
                });
            }
            return Ok(cpu_result);
        }

        let diff = (cpu_result - gpu_result).abs();
        if diff > epsilon {
            return Err(DiffError::Mismatch {
                cpu: cpu_result.to_string(),
                gpu: gpu_result.to_string(),
                context: context.to_string(),
            });
        }

        Ok(cpu_result)
    }

    /// Assert GPU and CPU produce same f64 result (with epsilon tolerance)
    pub fn assert_same_f64(&mut self, wasm: &[u8], epsilon: f64, context: &str) {
        let cpu_result = self.cpu.run_f64(wasm)
            .unwrap_or_else(|e| panic!("CPU execution failed for {}: {}", context, e));
        let gpu_result = self.gpu.run_wasm_f64(wasm)
            .unwrap_or_else(|e| panic!("GPU execution failed for {}: {}", context, e));

        // Handle NaN
        if cpu_result.is_nan() && gpu_result.is_nan() {
            return;
        }

        // Handle infinity
        if cpu_result.is_infinite() && gpu_result.is_infinite() {
            assert_eq!(
                cpu_result.is_sign_positive(), gpu_result.is_sign_positive(),
                "Mismatch for {}: CPU={}, GPU={} (infinity sign differs)",
                context, cpu_result, gpu_result
            );
            return;
        }

        let diff = (cpu_result - gpu_result).abs();
        // Use relative error for large values (double-single uses f32 internally)
        // Absolute epsilon for small values, relative epsilon for large values
        let max_abs = cpu_result.abs().max(gpu_result.abs());
        let effective_epsilon = if max_abs > 1.0 {
            epsilon * max_abs  // Relative epsilon scaled by magnitude
        } else {
            epsilon  // Absolute epsilon for small values
        };
        assert!(
            diff <= effective_epsilon,
            "Mismatch for {}: CPU={}, GPU={}, diff={} (epsilon={}, effective={})",
            context, cpu_result, gpu_result, diff, epsilon, effective_epsilon
        );
    }

    /// Assert GPU and CPU produce same f64 result, returning Result
    pub fn check_same_f64(&mut self, wasm: &[u8], epsilon: f64, context: &str) -> Result<f64, DiffError> {
        let cpu_result = self.cpu.run_f64(wasm)
            .map_err(DiffError::CpuError)?;
        let gpu_result = self.gpu.run_wasm_f64(wasm)
            .map_err(DiffError::GpuError)?;

        // Handle NaN
        if cpu_result.is_nan() && gpu_result.is_nan() {
            return Ok(cpu_result);
        }

        // Handle infinity
        if cpu_result.is_infinite() && gpu_result.is_infinite() {
            if cpu_result.is_sign_positive() != gpu_result.is_sign_positive() {
                return Err(DiffError::Mismatch {
                    cpu: cpu_result.to_string(),
                    gpu: gpu_result.to_string(),
                    context: context.to_string(),
                });
            }
            return Ok(cpu_result);
        }

        let diff = (cpu_result - gpu_result).abs();
        // Use relative error for large values (double-single uses f32 internally)
        let max_abs = cpu_result.abs().max(gpu_result.abs());
        let effective_epsilon = if max_abs > 1.0 {
            epsilon * max_abs
        } else {
            epsilon
        };
        if diff > effective_epsilon {
            return Err(DiffError::Mismatch {
                cpu: cpu_result.to_string(),
                gpu: gpu_result.to_string(),
                context: context.to_string(),
            });
        }

        Ok(cpu_result)
    }

    /// Run a batch of i32 tests, collecting all failures
    pub fn batch_i32<F>(&mut self, test_cases: impl Iterator<Item = (Vec<u8>, String)>, mut on_failure: F)
    where
        F: FnMut(&str, DiffError),
    {
        for (wasm, context) in test_cases {
            if let Err(e) = self.check_same_i32(&wasm, &context) {
                on_failure(&context, e);
            }
        }
    }

    /// Run a batch of i64 tests, collecting all failures
    pub fn batch_i64<F>(&mut self, test_cases: impl Iterator<Item = (Vec<u8>, String)>, mut on_failure: F)
    where
        F: FnMut(&str, DiffError),
    {
        for (wasm, context) in test_cases {
            if let Err(e) = self.check_same_i64(&wasm, &context) {
                on_failure(&context, e);
            }
        }
    }
}

/// Convenience macro for differential testing
#[macro_export]
macro_rules! assert_wasm_eq {
    ($wasm:expr, i32) => {{
        let mut tester = $crate::harness::differential::DifferentialTester::new().unwrap();
        tester.assert_same_i32($wasm, stringify!($wasm));
    }};
    ($wasm:expr, i64) => {{
        let mut tester = $crate::harness::differential::DifferentialTester::new().unwrap();
        tester.assert_same_i64($wasm, stringify!($wasm));
    }};
    ($wasm:expr, f32) => {{
        let mut tester = $crate::harness::differential::DifferentialTester::new().unwrap();
        tester.assert_same_f32($wasm, 1e-6, stringify!($wasm));
    }};
    ($wasm:expr, f32, $eps:expr) => {{
        let mut tester = $crate::harness::differential::DifferentialTester::new().unwrap();
        tester.assert_same_f32($wasm, $eps, stringify!($wasm));
    }};
    ($wasm:expr, f64) => {{
        let mut tester = $crate::harness::differential::DifferentialTester::new().unwrap();
        tester.assert_same_f64($wasm, 1e-10, stringify!($wasm));
    }};
    ($wasm:expr, f64, $eps:expr) => {{
        let mut tester = $crate::harness::differential::DifferentialTester::new().unwrap();
        tester.assert_same_f64($wasm, $eps, stringify!($wasm));
    }};
}
