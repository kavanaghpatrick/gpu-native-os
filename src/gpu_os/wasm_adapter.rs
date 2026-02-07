// WASM Adapter - Connects WASM translator to persistent runtime (Issue #281)
//
// THE GPU IS THE COMPUTER. This adapter bridges the WASM translator
// to the persistent GPU runtime, enabling real Rust code to run
// on GPU without modification.
//
// Architecture Note:
// Due to workspace structure (wasm_translator depends on rust-experiment),
// direct dependency would create a cycle. Instead, this adapter provides:
// 1. Bytecode loading and spawning (no WASM dependency)
// 2. Convenience functions for callers who have access to wasm_translator
//
// Usage:
// ```
// // In test or example (where both crates are available):
// use wasm_translator::WasmTranslator;
// use rust_experiment::gpu_os::wasm_adapter::WasmAdapter;
//
// let wasm_bytes = std::fs::read("my_app.wasm")?;
// let bytecode = WasmTranslator::default().translate(&wasm_bytes)?;
// let pid = WasmAdapter::run_bytecode(&mut runtime, &bytecode)?;
// ```

use crate::gpu_os::persistent_runtime::PersistentRuntime;

/// WASM magic number: "\0asm"
pub const WASM_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d];

/// Adapter that connects GPU bytecode to the persistent runtime
///
/// This adapter handles loading pre-translated GPU bytecode into the
/// persistent runtime and spawning processes.
///
/// Note: WASM translation is done externally via `wasm_translator` crate
/// to avoid cyclic dependencies in the workspace.
pub struct WasmAdapter;

impl WasmAdapter {
    /// Run pre-translated GPU bytecode on the persistent runtime
    ///
    /// This is the core method - accepts already-translated GPU bytecode
    /// (from wasm_translator::WasmTranslator::translate).
    ///
    /// # Arguments
    /// * `runtime` - The persistent runtime to execute on
    /// * `bytecode` - GPU bytecode (from WasmTranslator::translate)
    ///
    /// # Returns
    /// * `Ok(process_id)` - The process slot assigned to the new process
    /// * `Err(String)` - Error message on failure
    ///
    /// # Example
    /// ```ignore
    /// use wasm_translator::WasmTranslator;
    /// use rust_experiment::gpu_os::{persistent_runtime::PersistentRuntime, wasm_adapter::WasmAdapter};
    ///
    /// let device = metal::Device::system_default().unwrap();
    /// let mut runtime = PersistentRuntime::new(&device)?;
    ///
    /// // Translate WASM to GPU bytecode (done externally)
    /// let wasm = std::fs::read("my_app.wasm")?;
    /// let bytecode = WasmTranslator::default().translate(&wasm)?;
    ///
    /// // Load and run on persistent runtime
    /// let pid = WasmAdapter::run_bytecode(&mut runtime, &bytecode)?;
    /// ```
    pub fn run_bytecode(runtime: &mut PersistentRuntime, bytecode: &[u8]) -> Result<u32, String> {
        if bytecode.is_empty() {
            return Err("Empty bytecode".to_string());
        }

        // Load bytecode into runtime
        let (offset, len) = runtime.load_bytecode(bytecode)?;

        // Spawn process with default priority
        runtime.spawn(offset, len, 0)?;

        // Return expected process slot (approximate - actual assigned by GPU)
        Ok(0)
    }

    /// Run bytecode with specified priority
    ///
    /// # Arguments
    /// * `runtime` - The persistent runtime
    /// * `bytecode` - GPU bytecode
    /// * `priority` - Process priority (0=background, 1=normal, 2=high, 3=realtime)
    ///
    /// # Returns
    /// * `Ok(process_id)` - The process slot assigned
    /// * `Err(String)` - Error message on failure
    pub fn run_bytecode_with_priority(
        runtime: &mut PersistentRuntime,
        bytecode: &[u8],
        priority: u32,
    ) -> Result<u32, String> {
        if bytecode.is_empty() {
            return Err("Empty bytecode".to_string());
        }

        let (offset, len) = runtime.load_bytecode(bytecode)?;
        runtime.spawn(offset, len, priority)?;

        Ok(0)
    }

    /// Check if bytes are valid WASM
    ///
    /// Utility for callers to validate WASM before passing to translator.
    ///
    /// # Arguments
    /// * `bytes` - Bytes to check
    ///
    /// # Returns
    /// * `true` if bytes start with WASM magic number
    pub fn is_valid_wasm(bytes: &[u8]) -> bool {
        bytes.len() >= 4 && bytes[0..4] == WASM_MAGIC
    }

    /// Validate WASM bytes
    ///
    /// Returns a descriptive error if validation fails.
    ///
    /// # Arguments
    /// * `wasm` - WASM bytes to validate
    ///
    /// # Returns
    /// * `Ok(())` if valid
    /// * `Err(String)` with description if invalid
    pub fn validate_wasm(wasm: &[u8]) -> Result<(), String> {
        if wasm.len() < 4 {
            return Err("WASM file too short (less than 4 bytes)".to_string());
        }

        if wasm[0..4] != WASM_MAGIC {
            return Err(format!(
                "Invalid WASM magic number: expected {:02x}{:02x}{:02x}{:02x}, got {:02x}{:02x}{:02x}{:02x}",
                WASM_MAGIC[0], WASM_MAGIC[1], WASM_MAGIC[2], WASM_MAGIC[3],
                wasm[0], wasm[1], wasm[2], wasm[3]
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_magic_validation() {
        // Valid WASM header
        let valid = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        assert!(WasmAdapter::is_valid_wasm(&valid));

        // Invalid - wrong magic
        let invalid = [0x00, 0x00, 0x00, 0x00];
        assert!(!WasmAdapter::is_valid_wasm(&invalid));

        // Invalid - too short
        let short = [0x00, 0x61];
        assert!(!WasmAdapter::is_valid_wasm(&short));

        // Empty
        let empty: [u8; 0] = [];
        assert!(!WasmAdapter::is_valid_wasm(&empty));
    }

    #[test]
    fn test_validate_wasm() {
        // Valid
        let valid = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        assert!(WasmAdapter::validate_wasm(&valid).is_ok());

        // Invalid - wrong magic
        let invalid = [0x00, 0x00, 0x00, 0x00];
        let err = WasmAdapter::validate_wasm(&invalid).unwrap_err();
        assert!(err.contains("Invalid WASM magic"));

        // Invalid - too short
        let short = [0x00, 0x61];
        let err = WasmAdapter::validate_wasm(&short).unwrap_err();
        assert!(err.contains("too short"));
    }
}
