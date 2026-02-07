use wasm_translator::translate_wasm;
use std::fs;

fn main() {
    let wasm_path = "test_programs/simple_math/target/wasm32-unknown-unknown/release/simple_math.wasm";
    let wasm_bytes = fs::read(wasm_path).expect("Failed to read WASM file");
    
    println!("WASM size: {} bytes", wasm_bytes.len());
    
    match translate_wasm(&wasm_bytes, "factorial") {
        Ok(bytecode) => {
            println!("Translation successful!");
            println!("Bytecode size: {} bytes", bytecode.len());
            println!("Instructions: {}", (bytecode.len() - 16) / 8);
        }
        Err(e) => {
            println!("Translation failed: {:?}", e);
        }
    }
}
