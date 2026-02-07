//! GPU Boot Sequence - Automatic App Discovery and Loading
//!
//! Issue #170 - GPU Boot Sequence
//!
//! Architecture:
//! - CPU sets up memory and starts I/O thread
//! - GPU does ALL the work: init, discover apps, load, launch
//!
//! THE GPU IS THE COMPUTER. CPU just provides buffers.

use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::mem;
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

pub const MAX_BOOT_APPS: usize = 32;
pub const MAX_SLOTS: u32 = 64;

// Boot phases
pub const PHASE_UNINIT: u32 = 0;
pub const PHASE_SYSTEM_INIT: u32 = 1;
pub const PHASE_SYSTEM_APPS: u32 = 2;
pub const PHASE_DISCOVER: u32 = 3;
pub const PHASE_LOADING: u32 = 4;
pub const PHASE_COMPLETE: u32 = 5;

// System app types (from gpu_app_system)
pub const APP_TYPE_COMPOSITOR: u32 = 200;
pub const APP_TYPE_DOCK: u32 = 201;
pub const APP_TYPE_MENUBAR: u32 = 202;
pub const APP_TYPE_WINDOW_CHROME: u32 = 203;

// App flags
pub const APP_FLAG_ACTIVE: u32 = 1;
pub const APP_FLAG_VISIBLE: u32 = 2;

// Priority levels
pub const PRIORITY_REALTIME: u32 = 0;

pub const INVALID_SLOT: u32 = 0xFFFFFFFF;
pub const INVALID_HANDLE: u32 = 0xFFFFFFFF;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════════

/// Boot State (GPU-resident)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BootState {
    // Phase tracking (8 bytes)
    pub current_phase: u32,
    pub phase_complete: u32,

    // App discovery (136 bytes)
    pub discovered_apps: [u32; MAX_BOOT_APPS],
    pub discovered_count: u32,

    // Pending loads (136 bytes)
    pub pending_handles: [u32; MAX_BOOT_APPS],
    pub pending_count: u32,
    pub loaded_count: u32,

    // Errors (8 bytes)
    pub error_count: u32,
    pub last_error: u32,

    // Stats (8 bytes)
    pub boot_start_frame: u32,
    pub boot_end_frame: u32,

    // Padding
    pub _pad: [u32; 2],
}

impl Default for BootState {
    fn default() -> Self {
        Self {
            current_phase: PHASE_UNINIT,
            phase_complete: 0,
            discovered_apps: [INVALID_SLOT; MAX_BOOT_APPS],
            discovered_count: 0,
            pending_handles: [INVALID_HANDLE; MAX_BOOT_APPS],
            pending_count: 0,
            loaded_count: 0,
            error_count: 0,
            last_error: 0,
            boot_start_frame: 0,
            boot_end_frame: 0,
            _pad: [0; 2],
        }
    }
}

/// Boot statistics
#[derive(Clone, Debug, Default)]
pub struct BootStats {
    pub current_phase: u32,
    pub phase_complete: u32,
    pub discovered_count: u32,
    pub pending_count: u32,
    pub loaded_count: u32,
    pub error_count: u32,
    pub boot_frames: u32,
}

/// App table header (simplified for boot)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AppTableHeader {
    pub magic: u32,
    pub version: u32,
    pub max_slots: u32,
    pub active_count: u32,
    pub next_slot_hint: u32,
    pub _pad: [u32; 3],
    // Followed by: free_bitmap (8 bytes), then GpuAppDescriptor array
}

const _: () = assert!(mem::size_of::<AppTableHeader>() == 32);

// ═══════════════════════════════════════════════════════════════════════════════
// GPU BOOT SEQUENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU Boot Sequence Handler
pub struct GpuBootSequence {
    // GPU buffers
    boot_state_buffer: Buffer,
    app_table_buffer: Buffer,
    unified_state_buffer: Buffer,

    // Compute pipelines
    phase1_pipeline: ComputePipelineState,
    phase2_pipeline: ComputePipelineState,
    check_complete_pipeline: ComputePipelineState,

    // Configuration
    max_slots: u32,
}

impl GpuBootSequence {
    /// Create a new boot sequence handler
    pub fn new(device: &Device, max_slots: u32) -> Result<Self, String> {
        // Create boot state buffer
        let boot_state_buffer = device.new_buffer(
            mem::size_of::<BootState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create app table buffer
        // Header (32) + bitmap (8) + descriptors (128 * max_slots)
        let header_size = 32usize;
        let bitmap_size = 8usize;
        let descriptor_size = 128usize;
        let table_size = header_size + bitmap_size + (max_slots as usize * descriptor_size);

        let app_table_buffer = device.new_buffer(
            table_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create unified state buffer (64KB per slot)
        let state_chunk_size = 64 * 1024u64;
        let unified_state_buffer = device.new_buffer(
            max_slots as u64 * state_chunk_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Compile shaders
        let shader_source = include_str!("shaders/boot_sequence.metal");
        let library = device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| format!("Failed to compile boot_sequence shader: {}", e))?;

        let phase1_pipeline = {
            let func = library
                .get_function("boot_phase1_system_init", None)
                .map_err(|e| format!("Failed to get phase1 function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create phase1 pipeline: {}", e))?
        };

        let phase2_pipeline = {
            let func = library
                .get_function("boot_phase2_system_apps", None)
                .map_err(|e| format!("Failed to get phase2 function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create phase2 pipeline: {}", e))?
        };

        let check_complete_pipeline = {
            let func = library
                .get_function("boot_check_complete", None)
                .map_err(|e| format!("Failed to get check_complete function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create check_complete pipeline: {}", e))?
        };

        Ok(Self {
            boot_state_buffer,
            app_table_buffer,
            unified_state_buffer,
            phase1_pipeline,
            phase2_pipeline,
            check_complete_pipeline,
            max_slots,
        })
    }

    /// Run Phase 1: System Initialization (GPU-side)
    pub fn run_phase1(&self, device: &Device) {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let max_slots_buf = device.new_buffer_with_data(
            &self.max_slots as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.phase1_pipeline);
        encoder.set_buffer(0, Some(&self.app_table_buffer), 0);
        encoder.set_buffer(1, Some(&self.boot_state_buffer), 0);
        encoder.set_buffer(2, Some(&max_slots_buf), 0);

        // One thread per slot
        encoder.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Run Phase 2: System Apps (GPU-side)
    pub fn run_phase2(&self, device: &Device) {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.phase2_pipeline);
        encoder.set_buffer(0, Some(&self.app_table_buffer), 0);
        encoder.set_buffer(1, Some(&self.unified_state_buffer), 0);
        encoder.set_buffer(2, Some(&self.boot_state_buffer), 0);

        // 4 threads for 4 system apps
        encoder.dispatch_threads(MTLSize::new(4, 1, 1), MTLSize::new(4, 1, 1));

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Run the complete boot sequence
    pub fn boot(&self, device: &Device) -> BootStats {
        // Phase 1: System Init
        self.run_phase1(device);

        // Phase 2: System Apps
        self.run_phase2(device);

        // For now, skip phases 3-4 (app discovery/loading)
        // These require filesystem index integration

        // Mark boot complete
        self.mark_complete(device);

        self.read_stats()
    }

    /// Mark boot as complete (GPU-side)
    fn mark_complete(&self, device: &Device) {
        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.check_complete_pipeline);
        encoder.set_buffer(0, Some(&self.boot_state_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Check if boot is complete
    pub fn is_complete(&self) -> bool {
        let state = self.read_boot_state();
        state.phase_complete >= PHASE_COMPLETE
    }

    /// Read boot state
    pub fn read_boot_state(&self) -> BootState {
        let ptr = self.boot_state_buffer.contents() as *const BootState;
        unsafe { *ptr }
    }

    /// Read boot statistics
    pub fn read_stats(&self) -> BootStats {
        let state = self.read_boot_state();
        BootStats {
            current_phase: state.current_phase,
            phase_complete: state.phase_complete,
            discovered_count: state.discovered_count,
            pending_count: state.pending_count,
            loaded_count: state.loaded_count,
            error_count: state.error_count,
            boot_frames: state.boot_end_frame.saturating_sub(state.boot_start_frame),
        }
    }

    /// Get active app count
    pub fn active_app_count(&self) -> u32 {
        let ptr = self.app_table_buffer.contents() as *const AppTableHeader;
        unsafe { (*ptr).active_count }
    }

    /// Read app descriptor at slot
    pub fn read_app_descriptor(&self, slot: u32) -> Option<(u32, u32, u32)> {
        if slot >= self.max_slots {
            return None;
        }

        let header_size = 32usize;
        let bitmap_size = 8usize;
        let descriptor_size = 128usize;

        let offset = header_size + bitmap_size + (slot as usize * descriptor_size);
        let ptr = self.app_table_buffer.contents() as *const u8;

        unsafe {
            let desc_ptr = ptr.add(offset);
            let flags = *(desc_ptr as *const u32);
            let app_type = *(desc_ptr.add(4) as *const u32);
            let slot_id = *(desc_ptr.add(8) as *const u32);

            if flags & APP_FLAG_ACTIVE != 0 {
                Some((flags, app_type, slot_id))
            } else {
                None
            }
        }
    }

    /// Get buffer references
    pub fn boot_state_buffer(&self) -> &Buffer {
        &self.boot_state_buffer
    }

    pub fn app_table_buffer(&self) -> &Buffer {
        &self.app_table_buffer
    }

    pub fn unified_state_buffer(&self) -> &Buffer {
        &self.unified_state_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_state_size() {
        let size = mem::size_of::<BootState>();
        println!("BootState size: {} bytes", size);
        // Should be < 512 bytes
        assert!(size < 512);
    }

    #[test]
    fn test_app_table_header_size() {
        assert_eq!(mem::size_of::<AppTableHeader>(), 32);
    }
}
