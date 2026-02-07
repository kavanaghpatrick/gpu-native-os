//! GPU Desktop Visual Testing
//!
//! Automated visual testing for the GPU desktop environment.
//! Captures screenshots and compares against baselines.
//!
//! Usage:
//!   cargo run --release --example desktop_visual_test
//!   cargo run --release --example desktop_visual_test -- --update  # Update baselines

use metal::*;
use std::env;

use rust_experiment::gpu_os::desktop::{
    GpuDesktop, FileBrowserApp, TerminalApp, DocumentViewerApp, TextEditorApp,
};
use rust_experiment::gpu_os::screenshot::{GpuScreenshot, VisualTestRunner, VisualTestResult};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer};

const SCREEN_WIDTH: u32 = 1280;
const SCREEN_HEIGHT: u32 = 800;

/// Run visual tests on the GPU desktop
fn main() {
    println!("GPU Desktop Visual Tests");
    println!("========================\n");

    let args: Vec<String> = env::args().collect();
    let update_baselines = args.iter().any(|a| a == "--update");

    if update_baselines {
        println!("Mode: Updating baselines\n");
    } else {
        println!("Mode: Testing against baselines\n");
    }

    // Initialize Metal
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();

    // Create visual test runner
    let runner = VisualTestRunner::new(
        "tests/visual_baselines",
        "tests/visual_output",
    ).with_threshold(10); // Allow small differences due to anti-aliasing

    let mut results: Vec<VisualTestResult> = Vec::new();

    // Test 1: Empty desktop
    println!("Test 1: Empty desktop...");
    let result = test_empty_desktop(&device, &command_queue, &runner, update_baselines);
    print_result(&result);
    results.push(result);

    // Test 2: Desktop with windows
    println!("Test 2: Desktop with windows...");
    let result = test_desktop_with_windows(&device, &command_queue, &runner, update_baselines);
    print_result(&result);
    results.push(result);

    // Test 3: Desktop with apps
    println!("Test 3: Desktop with apps...");
    let result = test_desktop_with_apps(&device, &command_queue, &runner, update_baselines);
    print_result(&result);
    results.push(result);

    // Test 4: File browser content
    println!("Test 4: File browser content...");
    let result = test_file_browser(&device, &command_queue, &runner, update_baselines);
    print_result(&result);
    results.push(result);

    // Summary
    println!("\n========================");
    println!("Results Summary");
    println!("========================");

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    for result in &results {
        let status = if result.passed { "PASS" } else { "FAIL" };
        println!("  [{}] {}", status, result.name);
    }

    println!("\n{}/{} tests passed", passed, total);

    if passed < total {
        std::process::exit(1);
    }
}

fn print_result(result: &VisualTestResult) {
    if result.passed {
        println!("  PASS: {}", result.message);
    } else {
        println!("  FAIL: {}", result.message);
    }
    println!();
}

/// Render desktop to texture and capture screenshot
fn render_desktop_to_screenshot(
    device: &Device,
    command_queue: &CommandQueue,
    desktop: &mut GpuDesktop,
) -> GpuScreenshot {
    // Create render target texture
    let tex_desc = TextureDescriptor::new();
    tex_desc.set_width(SCREEN_WIDTH as u64);
    tex_desc.set_height(SCREEN_HEIGHT as u64);
    tex_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    tex_desc.set_texture_type(MTLTextureType::D2);
    tex_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    tex_desc.set_storage_mode(MTLStorageMode::Managed);

    let texture = device.new_texture(&tex_desc);

    // Create render pass
    let render_pass_desc = RenderPassDescriptor::new();
    let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
    color_attachment.set_texture(Some(&texture));
    color_attachment.set_load_action(MTLLoadAction::Clear);
    color_attachment.set_store_action(MTLStoreAction::Store);
    color_attachment.set_clear_color(MTLClearColor::new(
        desktop.background_color[0] as f64,
        desktop.background_color[1] as f64,
        desktop.background_color[2] as f64,
        1.0,
    ));

    // Render
    let cmd = command_queue.new_command_buffer();
    let encoder = cmd.new_render_command_encoder(&render_pass_desc);

    desktop.update(0.016); // One frame at 60fps
    desktop.render(encoder);

    encoder.end_encoding();

    // Synchronize managed texture
    let blit = cmd.new_blit_command_encoder();
    blit.synchronize_resource(&texture);
    blit.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // Capture screenshot
    let mut screenshot = GpuScreenshot::new(SCREEN_WIDTH, SCREEN_HEIGHT);
    screenshot.capture(&texture);

    screenshot
}

/// Test 1: Empty desktop (just background and dock)
fn test_empty_desktop(
    device: &Device,
    command_queue: &CommandQueue,
    runner: &VisualTestRunner,
    update: bool,
) -> VisualTestResult {
    let mut desktop = GpuDesktop::new(
        device,
        SCREEN_WIDTH as f32,
        SCREEN_HEIGHT as f32,
        MTLPixelFormat::BGRA8Unorm,
    ).expect("Failed to create desktop");

    let screenshot = render_desktop_to_screenshot(device, command_queue, &mut desktop);

    if update {
        runner.update_baseline("empty_desktop", &screenshot).expect("Failed to update baseline");
        VisualTestResult {
            name: "empty_desktop".to_string(),
            passed: true,
            diff: None,
            message: "Baseline updated".to_string(),
        }
    } else {
        runner.test("empty_desktop", &screenshot)
    }
}

/// Test 2: Desktop with windows
fn test_desktop_with_windows(
    device: &Device,
    command_queue: &CommandQueue,
    runner: &VisualTestRunner,
    update: bool,
) -> VisualTestResult {
    let mut desktop = GpuDesktop::new(
        device,
        SCREEN_WIDTH as f32,
        SCREEN_HEIGHT as f32,
        MTLPixelFormat::BGRA8Unorm,
    ).expect("Failed to create desktop");

    // Create test windows
    desktop.create_window("Window 1", 100.0, 100.0, 300.0, 200.0);
    desktop.create_window("Window 2", 250.0, 180.0, 350.0, 250.0);
    desktop.create_window("Window 3", 400.0, 120.0, 280.0, 180.0);

    let screenshot = render_desktop_to_screenshot(device, command_queue, &mut desktop);

    if update {
        runner.update_baseline("windows", &screenshot).expect("Failed to update baseline");
        VisualTestResult {
            name: "windows".to_string(),
            passed: true,
            diff: None,
            message: "Baseline updated".to_string(),
        }
    } else {
        runner.test("windows", &screenshot)
    }
}

/// Test 3: Desktop with apps
fn test_desktop_with_apps(
    device: &Device,
    command_queue: &CommandQueue,
    runner: &VisualTestRunner,
    update: bool,
) -> VisualTestResult {
    let mut desktop = GpuDesktop::new(
        device,
        SCREEN_WIDTH as f32,
        SCREEN_HEIGHT as f32,
        MTLPixelFormat::BGRA8Unorm,
    ).expect("Failed to create desktop");

    // Register apps
    let files_id = desktop.apps.register("Files", 0);
    let docs_id = desktop.apps.register("Documents", 2);

    // Launch apps
    let file_browser = Box::new(FileBrowserApp::new());
    let _ = desktop.launch_app(files_id, file_browser);

    let mut doc_viewer = DocumentViewerApp::new();
    doc_viewer.set_title("Test Doc");
    doc_viewer.load_html("<h1>Test</h1><p>Content</p>");
    let _ = desktop.launch_app(docs_id, Box::new(doc_viewer));

    let screenshot = render_desktop_to_screenshot(device, command_queue, &mut desktop);

    if update {
        runner.update_baseline("apps", &screenshot).expect("Failed to update baseline");
        VisualTestResult {
            name: "apps".to_string(),
            passed: true,
            diff: None,
            message: "Baseline updated".to_string(),
        }
    } else {
        runner.test("apps", &screenshot)
    }
}

/// Test 4: File browser with content
fn test_file_browser(
    device: &Device,
    command_queue: &CommandQueue,
    runner: &VisualTestRunner,
    update: bool,
) -> VisualTestResult {
    let mut desktop = GpuDesktop::new(
        device,
        SCREEN_WIDTH as f32,
        SCREEN_HEIGHT as f32,
        MTLPixelFormat::BGRA8Unorm,
    ).expect("Failed to create desktop");

    // Launch file browser
    let files_id = desktop.apps.register("Files", 0);
    let mut file_browser = FileBrowserApp::new();
    file_browser.navigate_to("/tmp"); // Use predictable path
    let _ = desktop.launch_app(files_id, Box::new(file_browser));

    let screenshot = render_desktop_to_screenshot(device, command_queue, &mut desktop);

    if update {
        runner.update_baseline("file_browser", &screenshot).expect("Failed to update baseline");
        VisualTestResult {
            name: "file_browser".to_string(),
            passed: true,
            diff: None,
            message: "Baseline updated".to_string(),
        }
    } else {
        runner.test("file_browser", &screenshot)
    }
}
