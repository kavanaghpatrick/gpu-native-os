//! HashMap Test - GPU-Native Cuckoo HashMap Verification
//!
//! Tests the gpu_std::collections::HashMap to verify it works correctly
//! when compiled to WASM and executed on the GPU.
//!
//! Test scenarios:
//! 1. Create HashMap and verify it's empty
//! 2. Insert key-value pairs
//! 3. Retrieve values and verify correctness
//! 4. Check len() returns correct count
//! 5. Test contains_key()
//! 6. Test overwrite behavior (insert same key twice)
//! 7. Test remove()
//!
//! Visual output: Green = all tests pass, Red = failure

#![no_std]
#![no_main]

use gpu_std::prelude::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// Colors
const COLOR_PASS: u32 = 0x4CAF50FF;  // Green
const COLOR_FAIL: u32 = 0xF44336FF;  // Red
const COLOR_BG: u32 = 0x212121FF;    // Dark gray

/// Run all HashMap tests, return number of passed tests
fn run_tests() -> (i32, i32) {
    let mut passed = 0i32;
    let mut total = 0i32;

    // Test 1: Create empty HashMap
    total += 1;
    let map: HashMap<i32, i32> = HashMap::new();
    if map.is_empty() && map.len() == 0 {
        passed += 1;
        debug_i32(1001);  // Test 1 passed
    } else {
        debug_i32(-1001); // Test 1 failed
    }

    // Test 2: Insert and retrieve single value
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(42, 100);
    if let Some(&val) = map.get(&42) {
        if val == 100 {
            passed += 1;
            debug_i32(1002);  // Test 2 passed
        } else {
            debug_i32(-1002); // Test 2 failed - wrong value
        }
    } else {
        debug_i32(-1002); // Test 2 failed - key not found
    }

    // Test 3: Check len() after insertions
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);
    if map.len() == 3 {
        passed += 1;
        debug_i32(1003);  // Test 3 passed
    } else {
        debug_i32(-1003); // Test 3 failed
        debug_i32(map.len() as i32); // Debug: actual length
    }

    // Test 4: contains_key returns true for existing key
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(99, 999);
    if map.contains_key(&99) {
        passed += 1;
        debug_i32(1004);  // Test 4 passed
    } else {
        debug_i32(-1004); // Test 4 failed
    }

    // Test 5: contains_key returns false for non-existent key
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(1, 100);
    if !map.contains_key(&999) {
        passed += 1;
        debug_i32(1005);  // Test 5 passed
    } else {
        debug_i32(-1005); // Test 5 failed
    }

    // Test 6: Overwrite existing key
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(5, 50);
    let old = map.insert(5, 500);  // Should return Some(50)
    if old == Some(50) {
        if let Some(&val) = map.get(&5) {
            if val == 500 && map.len() == 1 {
                passed += 1;
                debug_i32(1006);  // Test 6 passed
            } else {
                debug_i32(-1006); // Test 6 failed - wrong value or len
            }
        } else {
            debug_i32(-1006); // Test 6 failed - key not found
        }
    } else {
        debug_i32(-1006); // Test 6 failed - wrong old value
    }

    // Test 7: Remove key
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(7, 70);
    map.insert(8, 80);
    let removed = map.remove(&7);
    if removed == Some(70) && map.len() == 1 && !map.contains_key(&7) {
        passed += 1;
        debug_i32(1007);  // Test 7 passed
    } else {
        debug_i32(-1007); // Test 7 failed
    }

    // Test 8: Multiple insertions and retrievals
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    let mut insert_ok = true;
    let mut i = 0i32;
    while i < 10 {
        map.insert(i, i * 10);
        i += 1;
    }

    // Verify all values
    let mut j = 0i32;
    while j < 10 {
        if let Some(&val) = map.get(&j) {
            if val != j * 10 {
                insert_ok = false;
            }
        } else {
            insert_ok = false;
        }
        j += 1;
    }

    if insert_ok && map.len() == 10 {
        passed += 1;
        debug_i32(1008);  // Test 8 passed
    } else {
        debug_i32(-1008); // Test 8 failed
    }

    // Test 9: get returns None for non-existent key
    total += 1;
    let map: HashMap<i32, i32> = HashMap::new();
    if map.get(&12345).is_none() {
        passed += 1;
        debug_i32(1009);  // Test 9 passed
    } else {
        debug_i32(-1009); // Test 9 failed
    }

    // Test 10: Clear removes all entries
    total += 1;
    let mut map: HashMap<i32, i32> = HashMap::new();
    map.insert(1, 10);
    map.insert(2, 20);
    map.clear();
    if map.is_empty() && map.len() == 0 && map.get(&1).is_none() {
        passed += 1;
        debug_i32(1010);  // Test 10 passed
    } else {
        debug_i32(-1010); // Test 10 failed
    }

    // Output summary
    debug_i32(9999);  // Marker for end of tests
    debug_i32(passed);
    debug_i32(total);

    (passed, total)
}

/// Main entry point - called every frame
#[no_mangle]
pub extern "C" fn main() -> i32 {
    // Run tests
    let (passed, total) = run_tests();

    // Determine overall status
    let all_passed = passed == total;
    let status_color = if all_passed { COLOR_PASS } else { COLOR_FAIL };

    // Draw background
    emit_quad(0.0, 0.0, 800.0, 600.0, COLOR_BG);

    // Draw large status indicator (center of screen)
    emit_quad(300.0, 200.0, 200.0, 200.0, status_color);

    // Draw test count indicator (passed/total as progress bar)
    let bar_width = 400.0;
    let bar_height = 30.0;
    let bar_x = 200.0;
    let bar_y = 500.0;

    // Background bar (gray)
    emit_quad(bar_x, bar_y, bar_width, bar_height, 0x424242FF);

    // Progress bar (green portion)
    let progress_width = bar_width * (passed as f32) / (total as f32);
    emit_quad(bar_x, bar_y, progress_width, bar_height, COLOR_PASS);

    // Return quad count
    4
}
