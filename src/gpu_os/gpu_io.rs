// Issue #112: GPU-Direct Storage with MTLIOCommandQueue
//
// THE GPU IS THE COMPUTER. File IO should bypass CPU entirely.
//
// Traditional: App → CPU Read → Memory → GPU Copy → GPU
// GPU-Direct:  App → MTLIOCommandQueue → GPU Buffer (CPU not involved!)
//
// Metal 3's MTLIOCommandQueue provides:
// - Direct file-to-GPU-buffer transfers
// - Async IO with event synchronization
// - Priority-based scheduling
// - Zero CPU involvement during transfer

use metal::*;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::c_void;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

/// Helper to get the raw object pointer from a metal type.
/// Metal types use foreign_types which wraps raw ObjC pointers.
fn device_as_ptr(device: &Device) -> *mut Object {
    // Device derefs to DeviceRef which is #[repr(transparent)] over the raw pointer
    device.deref() as *const _ as *mut Object
}

fn buffer_as_ptr(buffer: &Buffer) -> *mut Object {
    buffer.deref() as *const _ as *mut Object
}

/// Priority levels for IO operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum IOPriority {
    High = 0,
    Normal = 1,
    Low = 2,
}

/// Queue type for IO operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum IOQueueType {
    Concurrent = 0,
    Serial = 1,
}

/// Status of an IO operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum IOStatus {
    Pending = 0,
    Cancelled = 1,
    Error = 2,
    Complete = 3,
}

impl From<i64> for IOStatus {
    fn from(val: i64) -> Self {
        match val {
            0 => IOStatus::Pending,
            1 => IOStatus::Cancelled,
            2 => IOStatus::Error,
            _ => IOStatus::Complete,
        }
    }
}

/// A GPU-direct IO command queue for file operations.
///
/// This wraps MTLIOCommandQueue to enable file reads directly into GPU buffers
/// without CPU involvement.
pub struct GpuIOQueue {
    raw: *mut Object,
    device: Device,
}

// SAFETY: MTLIOCommandQueue is thread-safe
unsafe impl Send for GpuIOQueue {}
unsafe impl Sync for GpuIOQueue {}

impl GpuIOQueue {
    /// Create a new GPU IO queue.
    ///
    /// Returns None if the device doesn't support MTLIOCommandQueue (pre-Metal 3).
    pub fn new(device: &Device, priority: IOPriority, queue_type: IOQueueType) -> Option<Self> {
        unsafe {
            // Create descriptor
            let desc_class = class!(MTLIOCommandQueueDescriptor);
            let desc: *mut Object = msg_send![desc_class, new];
            if desc.is_null() {
                return None;
            }

            // Set priority and type
            let _: () = msg_send![desc, setPriority: priority as i64];
            let _: () = msg_send![desc, setType: queue_type as i64];

            // Create IO command queue from device
            let device_ptr = device_as_ptr(device);
            let mut error: *mut Object = std::ptr::null_mut();
            let queue: *mut Object = msg_send![device_ptr,
                newIOCommandQueueWithDescriptor: desc
                error: &mut error
            ];

            // Release descriptor
            let _: () = msg_send![desc, release];

            if queue.is_null() {
                if !error.is_null() {
                    let desc: *mut Object = msg_send![error, localizedDescription];
                    if !desc.is_null() {
                        let cstr: *const i8 = msg_send![desc, UTF8String];
                        if !cstr.is_null() {
                            let err_str = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                            eprintln!("IOCommandQueue creation failed: {}", err_str);
                        }
                    }
                }
                return None;
            }

            Some(Self {
                raw: queue,
                device: device.clone(),
            })
        }
    }

    /// Create a new IO command buffer.
    pub fn command_buffer(&self) -> Option<GpuIOCommandBuffer> {
        unsafe {
            let buffer: *mut Object = msg_send![self.raw, commandBuffer];
            if buffer.is_null() {
                return None;
            }
            // Retain since commandBuffer returns autoreleased
            let _: () = msg_send![buffer, retain];
            Some(GpuIOCommandBuffer { raw: buffer })
        }
    }

    /// Insert a barrier - all commands before must complete before commands after.
    pub fn enqueue_barrier(&self) {
        unsafe {
            let _: () = msg_send![self.raw, enqueueBarrier];
        }
    }

    /// Get the underlying device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Drop for GpuIOQueue {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.raw, release];
        }
    }
}

/// A file handle for GPU-direct IO.
pub struct GpuIOFileHandle {
    raw: *mut Object,
}

// SAFETY: MTLIOFileHandle is thread-safe
unsafe impl Send for GpuIOFileHandle {}
unsafe impl Sync for GpuIOFileHandle {}

impl GpuIOFileHandle {
    /// Open a file for GPU-direct reading.
    ///
    /// Returns None if the file doesn't exist or can't be opened.
    pub fn open(device: &Device, path: impl AsRef<Path>) -> Option<Self> {
        let path_str = path.as_ref().to_str()?;
        let path_cstring = std::ffi::CString::new(path_str).ok()?;

        unsafe {
            // Create NSURL from path
            let nsstring_class = class!(NSString);
            let path_ns: *mut Object = msg_send![nsstring_class,
                stringWithUTF8String: path_cstring.as_ptr()
            ];
            if path_ns.is_null() {
                return None;
            }

            let nsurl_class = class!(NSURL);
            let url: *mut Object = msg_send![nsurl_class,
                fileURLWithPath: path_ns
            ];
            if url.is_null() {
                return None;
            }

            // Create file handle from device
            let device_ptr = device_as_ptr(device);
            let mut error: *mut Object = std::ptr::null_mut();
            let handle: *mut Object = msg_send![device_ptr,
                newIOFileHandleWithURL: url
                error: &mut error
            ];

            if handle.is_null() {
                if !error.is_null() {
                    let desc: *mut Object = msg_send![error, localizedDescription];
                    if !desc.is_null() {
                        let cstr: *const i8 = msg_send![desc, UTF8String];
                        if !cstr.is_null() {
                            let err_str = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                            eprintln!("IOFileHandle open failed: {}", err_str);
                        }
                    }
                }
                return None;
            }

            Some(Self { raw: handle })
        }
    }
}

impl Drop for GpuIOFileHandle {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.raw, release];
        }
    }
}

/// A command buffer for GPU-direct IO operations.
pub struct GpuIOCommandBuffer {
    raw: *mut Object,
}

// SAFETY: MTLIOCommandBuffer is thread-safe
unsafe impl Send for GpuIOCommandBuffer {}
unsafe impl Sync for GpuIOCommandBuffer {}

impl GpuIOCommandBuffer {
    /// Load data from file directly into a Metal buffer.
    ///
    /// # Arguments
    /// * `buffer` - Destination Metal buffer
    /// * `offset` - Offset in destination buffer
    /// * `size` - Number of bytes to read
    /// * `file` - Source file handle
    /// * `file_offset` - Offset in source file
    pub fn load_buffer(
        &self,
        buffer: &Buffer,
        offset: u64,
        size: u64,
        file: &GpuIOFileHandle,
        file_offset: u64,
    ) {
        unsafe {
            let buffer_ptr = buffer_as_ptr(buffer);
            let _: () = msg_send![self.raw,
                loadBuffer: buffer_ptr
                offset: offset as usize
                size: size as usize
                sourceHandle: file.raw
                sourceHandleOffset: file_offset as usize
            ];
        }
    }

    /// Load data from file directly into raw memory.
    ///
    /// # Safety
    /// The pointer must remain valid until the command buffer completes.
    pub unsafe fn load_bytes(
        &self,
        ptr: *mut c_void,
        size: u64,
        file: &GpuIOFileHandle,
        file_offset: u64,
    ) {
        let non_null = NonNull::new(ptr).expect("Null pointer");
        let _: () = msg_send![self.raw,
            loadBytes: non_null.as_ptr()
            size: size as usize
            sourceHandle: file.raw
            sourceHandleOffset: file_offset as usize
        ];
    }

    /// Copy the IO status to a buffer (for GPU-side status checking).
    pub fn copy_status_to_buffer(&self, buffer: &Buffer, offset: u64) {
        unsafe {
            let buffer_ptr = buffer_as_ptr(buffer);
            let _: () = msg_send![self.raw,
                copyStatusToBuffer: buffer_ptr
                offset: offset as usize
            ];
        }
    }

    /// Add a barrier - commands before must complete before commands after.
    pub fn add_barrier(&self) {
        unsafe {
            let _: () = msg_send![self.raw, addBarrier];
        }
    }

    /// Enqueue the command buffer for execution.
    pub fn enqueue(&self) {
        unsafe {
            let _: () = msg_send![self.raw, enqueue];
        }
    }

    /// Commit the command buffer for execution.
    pub fn commit(&self) {
        unsafe {
            let _: () = msg_send![self.raw, commit];
        }
    }

    /// Wait until all IO operations complete.
    pub fn wait_until_completed(&self) {
        unsafe {
            let _: () = msg_send![self.raw, waitUntilCompleted];
        }
    }

    /// Get the current status of the command buffer.
    pub fn status(&self) -> IOStatus {
        unsafe {
            let status: i64 = msg_send![self.raw, status];
            IOStatus::from(status)
        }
    }

    /// Try to cancel pending IO operations.
    pub fn try_cancel(&self) {
        unsafe {
            let _: () = msg_send![self.raw, tryCancel];
        }
    }
}

impl Drop for GpuIOCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.raw, release];
        }
    }
}

/// High-level wrapper for GPU-direct file loading.
///
/// Combines GpuIOQueue, GpuIOFileHandle, and Metal buffer management
/// into a simple API.
pub struct GpuIOBuffer {
    buffer: Buffer,
    file_size: u64,
}

impl GpuIOBuffer {
    /// Load a file directly into a GPU buffer.
    ///
    /// # Arguments
    /// * `queue` - IO queue to use
    /// * `path` - Path to file to load
    ///
    /// # Returns
    /// The loaded buffer and actual file size, or None if loading failed.
    pub fn load_file(queue: &GpuIOQueue, path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();

        // Get file size
        let metadata = std::fs::metadata(path).ok()?;
        let file_size = metadata.len();

        if file_size == 0 {
            return None;
        }

        // Open file handle
        let file_handle = GpuIOFileHandle::open(queue.device(), path)?;

        // Create destination buffer (page-aligned)
        let aligned_size = (file_size + 4095) & !4095;
        let buffer = queue.device().new_buffer(
            aligned_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Create and execute IO command
        let cmd_buffer = queue.command_buffer()?;
        cmd_buffer.load_buffer(&buffer, 0, file_size, &file_handle, 0);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Check status
        if cmd_buffer.status() != IOStatus::Complete {
            return None;
        }

        Some(Self {
            buffer,
            file_size,
        })
    }

    /// Load a file asynchronously (returns immediately).
    ///
    /// Call `wait()` or check `is_complete()` before accessing the buffer.
    pub fn load_file_async(queue: &GpuIOQueue, path: impl AsRef<Path>) -> Option<GpuIOPendingLoad> {
        let path = path.as_ref();

        let metadata = std::fs::metadata(path).ok()?;
        let file_size = metadata.len();

        if file_size == 0 {
            return None;
        }

        let file_handle = GpuIOFileHandle::open(queue.device(), path)?;

        let aligned_size = (file_size + 4095) & !4095;
        let buffer = queue.device().new_buffer(
            aligned_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Create status buffer for GPU-side checking
        let status_buffer = queue.device().new_buffer(
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buffer = queue.command_buffer()?;
        cmd_buffer.load_buffer(&buffer, 0, file_size, &file_handle, 0);
        cmd_buffer.copy_status_to_buffer(&status_buffer, 0);
        cmd_buffer.commit();

        Some(GpuIOPendingLoad {
            cmd_buffer,
            buffer,
            status_buffer,
            file_size,
        })
    }

    /// Get the Metal buffer for GPU access.
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the file size.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
}

/// A pending async IO load operation.
pub struct GpuIOPendingLoad {
    cmd_buffer: GpuIOCommandBuffer,
    buffer: Buffer,
    status_buffer: Buffer,
    file_size: u64,
}

impl GpuIOPendingLoad {
    /// Check if the load is complete.
    pub fn is_complete(&self) -> bool {
        self.cmd_buffer.status() == IOStatus::Complete
    }

    /// Wait for the load to complete and return the buffer.
    pub fn wait(self) -> Option<GpuIOBuffer> {
        self.cmd_buffer.wait_until_completed();

        if self.cmd_buffer.status() != IOStatus::Complete {
            return None;
        }

        Some(GpuIOBuffer {
            buffer: self.buffer,
            file_size: self.file_size,
        })
    }

    /// Get the status buffer (for GPU-side status checking).
    ///
    /// The GPU can read this buffer to check if IO is complete
    /// without CPU involvement.
    pub fn status_buffer(&self) -> &Buffer {
        &self.status_buffer
    }

    /// Get the destination buffer (may contain partial data until complete).
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// Check if the device supports GPU-direct IO (Metal 3+).
pub fn supports_gpu_io(device: &Device) -> bool {
    // Try to create an IO queue - if it works, we support it
    GpuIOQueue::new(device, IOPriority::Normal, IOQueueType::Concurrent).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_io_support() {
        let device = Device::system_default().expect("No Metal device");
        let supports = supports_gpu_io(&device);
        println!("GPU-direct IO supported: {}", supports);
        // Metal 3 is required (Apple Silicon with macOS 13+)
    }
}
