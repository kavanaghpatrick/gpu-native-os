//! GPU App Loader - Load .gpuapp from Filesystem
//!
//! Issue #168 - GPU App Loader
//!
//! Architecture:
//! - GPU parses, validates, and initializes apps
//! - CPU is only an I/O coprocessor that fetches bytes
//!
//! THE GPU IS THE COMPUTER. CPU just moves bytes.

use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::fs;
use std::io;
use std::path::Path;

use crate::gpu_os::content_pipeline::{
    ContentPipeline, FileHandle, STATUS_READY, STATUS_ERROR, INVALID_HANDLE,
};
use crate::gpu_os::gpu_app_system::{BytecodeInst, BytecodeHeader};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

pub const GPUAPP_MAGIC: &[u8; 6] = b"GPUAPP";
pub const GPUAPP_VERSION: u16 = 1;
pub const GPUAPP_HEADER_SIZE: usize = 64;

pub const MAX_BYTECODE_SIZE: u32 = 65536;  // Max instructions
pub const MAX_APP_STATE: u32 = 1024 * 1024;  // 1MB
pub const MAX_APP_VERTICES: u32 = 65536;

pub const INVALID_SLOT: u32 = 0xFFFFFFFF;

// App flags (from gpu_app_system)
pub const ACTIVE_FLAG: u32 = 1;
pub const VISIBLE_FLAG: u32 = 2;
pub const DIRTY_FLAG: u32 = 4;

// App types
pub const APP_TYPE_BYTECODE: u32 = 101;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════════

/// GpuAppFileHeader (64 bytes) - Fixed header for .gpuapp files
///
/// GPU parses this directly - no string parsing, no TOML/JSON.
/// O(1) validation: check magic bytes, version, sizes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuAppFileHeader {
    pub magic: [u8; 6],           // "GPUAPP"
    pub version: u16,              // 1
    pub flags: u32,
    pub code_offset: u32,          // Offset to bytecode from file start
    pub code_size: u32,            // Number of instructions
    pub state_size: u32,           // Bytes needed for app state
    pub vertex_budget: u32,        // Max vertices
    pub thread_count: u32,         // Recommended thread count
    pub entry_point: u32,          // Starting PC
    pub name: [u8; 24],            // Null-terminated name
    pub _reserved: [u8; 4],
}

impl Default for GpuAppFileHeader {
    fn default() -> Self {
        Self {
            magic: *GPUAPP_MAGIC,
            version: GPUAPP_VERSION,
            flags: 0,
            code_offset: GPUAPP_HEADER_SIZE as u32,
            code_size: 0,
            state_size: 0,
            vertex_budget: 1024,
            thread_count: 256,
            entry_point: 0,
            name: [0; 24],
            _reserved: [0; 4],
        }
    }
}

impl GpuAppFileHeader {
    pub fn is_valid(&self) -> bool {
        self.magic == *GPUAPP_MAGIC
            && self.version == GPUAPP_VERSION
            && self.code_size <= MAX_BYTECODE_SIZE
            && self.state_size <= MAX_APP_STATE
            && self.vertex_budget <= MAX_APP_VERTICES
    }

    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&c| c == 0).unwrap_or(24);
        std::str::from_utf8(&self.name[..end]).unwrap_or("")
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>()
            )
        }
    }
}

const _: () = assert!(std::mem::size_of::<GpuAppFileHeader>() == 64);

/// Pending app load tracking
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PendingAppLoad {
    pub io_handle: u32,        // Content pipeline handle
    pub requesting_app: u32,   // App that requested (terminal, dock)
    pub target_slot: u32,      // Pre-allocated slot or INVALID_SLOT
    pub flags: u32,
}

pub const MAX_PENDING_LOADS: usize = 16;

/// Loader state (GPU-resident)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LoaderState {
    pub pending_count: u32,
    pub loaded_count: u32,
    pub error_count: u32,
    pub _pad: u32,
}

impl Default for LoaderState {
    fn default() -> Self {
        Self {
            pending_count: 0,
            loaded_count: 0,
            error_count: 0,
            _pad: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU APP BUILDER
// ═══════════════════════════════════════════════════════════════════════════════

/// Build .gpuapp files from bytecode
pub struct GpuAppBuilder {
    name: String,
    bytecode: Vec<BytecodeInst>,
    state_size: u32,
    vertex_budget: u32,
    entry_point: u32,
}

impl GpuAppBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bytecode: Vec::new(),
            state_size: 256,       // Default state size
            vertex_budget: 1024,   // Default vertex budget
            entry_point: 0,
        }
    }

    pub fn add_instruction(&mut self, inst: BytecodeInst) -> &mut Self {
        self.bytecode.push(inst);
        self
    }

    pub fn set_state_size(&mut self, size: u32) -> &mut Self {
        self.state_size = size;
        self
    }

    pub fn set_vertex_budget(&mut self, budget: u32) -> &mut Self {
        self.vertex_budget = budget;
        self
    }

    pub fn set_entry_point(&mut self, pc: u32) -> &mut Self {
        self.entry_point = pc;
        self
    }

    fn name_bytes(&self) -> [u8; 24] {
        let mut name = [0u8; 24];
        let bytes = self.name.as_bytes();
        let len = bytes.len().min(23);  // Leave room for null terminator
        name[..len].copy_from_slice(&bytes[..len]);
        name
    }

    /// Build the .gpuapp file bytes
    pub fn build(&self) -> Vec<u8> {
        let header = GpuAppFileHeader {
            magic: *GPUAPP_MAGIC,
            version: GPUAPP_VERSION,
            flags: 0,
            code_offset: GPUAPP_HEADER_SIZE as u32,
            code_size: self.bytecode.len() as u32,
            state_size: self.state_size,
            vertex_budget: self.vertex_budget,
            thread_count: 256,
            entry_point: self.entry_point,
            name: self.name_bytes(),
            _reserved: [0; 4],
        };

        let inst_size = std::mem::size_of::<BytecodeInst>();
        let total_size = GPUAPP_HEADER_SIZE + self.bytecode.len() * inst_size;
        let mut data = Vec::with_capacity(total_size);

        // Write header
        data.extend_from_slice(header.as_bytes());

        // Write bytecode
        for inst in &self.bytecode {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    inst as *const _ as *const u8,
                    inst_size
                )
            };
            data.extend_from_slice(bytes);
        }

        data
    }

    /// Write to file
    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.build())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU APP LOADER
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU App Loader
///
/// CPU's only role: provide I/O coprocessor
/// GPU does: parse header, validate, allocate slot, copy bytecode, initialize
pub struct GpuAppLoader {
    // GPU buffers
    state_buffer: Buffer,
    pending_loads_buffer: Buffer,
    results_buffer: Buffer,

    // Compute pipelines
    validate_pipeline: ComputePipelineState,
    init_pipeline: ComputePipelineState,
}

impl GpuAppLoader {
    /// Create a new GPU App Loader
    pub fn new(device: &Device) -> Result<Self, String> {
        // Create buffers (Shared for CPU readback during testing)
        let state_buffer = device.new_buffer(
            std::mem::size_of::<LoaderState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pending_loads_buffer = device.new_buffer(
            (MAX_PENDING_LOADS * std::mem::size_of::<PendingAppLoad>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let results_buffer = device.new_buffer(
            256,  // Space for test results
            MTLResourceOptions::StorageModeShared,
        );

        // Compile shader
        let shader_source = include_str!("shaders/gpu_app_loader.metal");
        let library = device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| format!("Failed to compile gpu_app_loader shader: {}", e))?;

        let validate_pipeline = {
            let func = library
                .get_function("validate_gpuapp_header", None)
                .map_err(|e| format!("Failed to get validate function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create validate pipeline: {}", e))?
        };

        let init_pipeline = {
            let func = library
                .get_function("init_app_from_gpuapp", None)
                .map_err(|e| format!("Failed to get init function: {}", e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Failed to create init pipeline: {}", e))?
        };

        // Initialize state buffer
        let ptr = state_buffer.contents() as *mut LoaderState;
        unsafe {
            *ptr = LoaderState::default();
        }

        Ok(Self {
            state_buffer,
            pending_loads_buffer,
            results_buffer,
            validate_pipeline,
            init_pipeline,
        })
    }

    /// Validate a header (GPU-side)
    ///
    /// Issue #285 fix: Correct buffer bindings to match Metal shader signature:
    ///   buffer(0) = header_data
    ///   buffer(1) = file_size
    ///   buffer(2) = result
    pub fn gpu_validate_header(&self, device: &Device, header_data: &[u8]) -> bool {
        if header_data.len() < GPUAPP_HEADER_SIZE {
            return false;
        }

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        // Create input buffer with header data
        let header_buffer = device.new_buffer_with_data(
            header_data.as_ptr() as *const _,
            header_data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Issue #285: Create file_size buffer for bounds checking (Issue #272)
        let file_size: u32 = header_data.len() as u32;
        let file_size_buffer = device.new_buffer_with_data(
            &file_size as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.validate_pipeline);
        // Issue #285: Correct buffer bindings to match Metal shader
        encoder.set_buffer(0, Some(&header_buffer), 0);      // buffer(0) = header_data
        encoder.set_buffer(1, Some(&file_size_buffer), 0);   // buffer(1) = file_size
        encoder.set_buffer(2, Some(&self.results_buffer), 0); // buffer(2) = result

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read result
        let result_ptr = self.results_buffer.contents() as *const u32;
        unsafe { *result_ptr != 0 }
    }

    /// Initialize an app from loaded .gpuapp bytes (GPU-side)
    ///
    /// This runs the GPU kernel that:
    /// 1. Validates the header
    /// 2. Allocates an app slot
    /// 3. Copies bytecode to unified state
    /// 4. Initializes the app descriptor
    pub fn gpu_init_app(
        &self,
        device: &Device,
        file_data: &[u8],
        app_table_buffer: &Buffer,
        unified_state_buffer: &Buffer,
    ) -> Option<u32> {
        if file_data.len() < GPUAPP_HEADER_SIZE {
            return None;
        }

        let queue = device.new_command_queue();
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        // Create input buffer with file data
        let file_buffer = device.new_buffer_with_data(
            file_data.as_ptr() as *const _,
            file_data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let file_size: u32 = file_data.len() as u32;
        let file_size_buffer = device.new_buffer_with_data(
            &file_size as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(&file_buffer), 0);
        encoder.set_buffer(1, Some(&file_size_buffer), 0);
        encoder.set_buffer(2, Some(app_table_buffer), 0);
        encoder.set_buffer(3, Some(unified_state_buffer), 0);
        encoder.set_buffer(4, Some(&self.results_buffer), 0);

        // Run with multiple threads for parallel bytecode copy
        encoder.dispatch_threads(MTLSize::new(256, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read result (slot ID or INVALID_SLOT)
        let result_ptr = self.results_buffer.contents() as *const u32;
        let slot = unsafe { *result_ptr };

        if slot == INVALID_SLOT {
            None
        } else {
            Some(slot)
        }
    }

    /// Read loader statistics
    pub fn read_stats(&self) -> LoaderState {
        let ptr = self.state_buffer.contents() as *const LoaderState;
        unsafe { *ptr }
    }

    /// Get buffer references
    pub fn state_buffer(&self) -> &Buffer {
        &self.state_buffer
    }

    pub fn pending_loads_buffer(&self) -> &Buffer {
        &self.pending_loads_buffer
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Read a .gpuapp header from file
pub fn read_gpuapp_header(path: &Path) -> io::Result<GpuAppFileHeader> {
    let data = fs::read(path)?;
    if data.len() < GPUAPP_HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "File too small for .gpuapp header"
        ));
    }

    let header: GpuAppFileHeader = unsafe {
        std::ptr::read(data.as_ptr() as *const GpuAppFileHeader)
    };

    if !header.is_valid() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid .gpuapp header"
        ));
    }

    Ok(header)
}

/// Parse header from bytes (for testing)
pub fn parse_header(data: &[u8]) -> Option<GpuAppFileHeader> {
    if data.len() < GPUAPP_HEADER_SIZE {
        return None;
    }

    let header: GpuAppFileHeader = unsafe {
        std::ptr::read(data.as_ptr() as *const GpuAppFileHeader)
    };

    if header.is_valid() {
        Some(header)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<GpuAppFileHeader>(), 64);
    }

    #[test]
    fn test_builder_creates_valid_header() {
        let builder = GpuAppBuilder::new("test_app");
        let data = builder.build();

        assert!(data.len() >= GPUAPP_HEADER_SIZE);

        let header = parse_header(&data).expect("Should parse header");
        assert_eq!(&header.magic, GPUAPP_MAGIC);
        assert_eq!(header.version, GPUAPP_VERSION);
        assert_eq!(header.name_str(), "test_app");
    }

    #[test]
    fn test_builder_with_bytecode() {
        let mut builder = GpuAppBuilder::new("bytecode_app");
        builder.add_instruction(BytecodeInst {
            opcode: 0,  // HALT
            dst: 0,
            src1: 0,
            src2: 0,
            imm: 0.0,
        });
        builder.set_vertex_budget(100);

        let data = builder.build();
        let header = parse_header(&data).expect("Should parse");

        assert_eq!(header.code_size, 1);
        assert_eq!(header.vertex_budget, 100);
    }
}
