//! Test mini_mandelbrot app - tests nested inline function pattern

use metal::Device;
use wasm_translator::{WasmTranslator, TranslatorConfig};
use rust_experiment::gpu_os::gpu_app_system::{GpuAppSystem, app_type};
use std::fs;

const WASM_PATH: &str = "test_programs/apps/mini_mandelbrot/target/wasm32-unknown-unknown/release/mini_mandelbrot.wasm";

#[test]
fn test_mini_mandelbrot_translation() {
    let wasm_bytes = fs::read(WASM_PATH).expect("Failed to read WASM file");

    println!("WASM size: {} bytes", wasm_bytes.len());

    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    // Parse header
    let code_size = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    println!("Bytecode size: {} bytes", bytecode.len());
    println!("Instructions: {}", code_size);

    assert!(bytecode.len() > 16, "Bytecode should have header + instructions");
}

#[test]
fn test_mini_mandelbrot_execution() {
    let device = Device::system_default().expect("No Metal device found");

    let wasm_bytes = fs::read(WASM_PATH).expect("Failed to read WASM file");
    let translator = WasmTranslator::new(TranslatorConfig::default());
    let bytecode = translator.translate(&wasm_bytes).expect("Translation failed");

    println!("WASM size: {} bytes", wasm_bytes.len());
    println!("Bytecode size: {} bytes", bytecode.len());

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    let slot = system.launch_by_type(app_type::BYTECODE).expect("Failed to launch");
    system.write_app_state(slot, &bytecode);
    system.run_frame();

    let result = system.read_bytecode_result(slot).unwrap_or(-999);
    println!("GPU Result: {} (expected 20 = 2*2*5)", result);

    // The app should return 2 * 2 * 5 = 20
    // If inline function handling is broken, it might return 0 or wrong value
    assert_eq!(result, 20, "mini_mandelbrot should return 2*2*5=20");
}
