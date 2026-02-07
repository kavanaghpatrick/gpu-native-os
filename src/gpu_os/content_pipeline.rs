// Issue #289: Content Pipeline with Proper Bounds Checking
//
// THE GPU IS THE COMPUTER. This module provides secure handle-based
// access to GPU-resident content buffers with strict bounds validation.
//
// Security requirements:
// - All slot-based access MUST validate bounds before ptr.add()
// - Read operations return Option<T> to handle OOB gracefully
// - Write operations return Result<(), ContentPipelineError> for error handling

use metal::*;
use std::mem;

/// Error types for content pipeline operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentPipelineError {
    /// Slot index is out of bounds
    SlotOutOfBounds { slot: u32, max_handles: u32 },
    /// Buffer is not initialized
    BufferNotInitialized,
    /// Invalid handle state
    InvalidHandle,
}

impl std::fmt::Display for ContentPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentPipelineError::SlotOutOfBounds { slot, max_handles } => {
                write!(f, "Slot {} is out of bounds (max: {})", slot, max_handles)
            }
            ContentPipelineError::BufferNotInitialized => {
                write!(f, "Buffer is not initialized")
            }
            ContentPipelineError::InvalidHandle => {
                write!(f, "Invalid handle state")
            }
        }
    }
}

impl std::error::Error for ContentPipelineError {}

/// Status of a file handle in the content pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct HandleStatus {
    /// 0 = free, 1 = loading, 2 = ready, 3 = error
    pub state: u32,
    /// Error code if state == 3
    pub error_code: u32,
    /// Progress percentage (0-100) if state == 1
    pub progress: u32,
    /// Reserved for alignment
    pub _pad: u32,
}

/// Buffer information for a file handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct HandleBufferInfo {
    /// Offset into the content buffer
    pub buffer_offset: u64,
    /// Size of the content in bytes
    pub content_size: u64,
    /// Total allocated size (may be larger than content_size)
    pub allocated_size: u64,
    /// Checksum of the content
    pub checksum: u64,
}

/// A file handle in the content pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct FileHandle {
    /// Unique handle ID
    pub handle_id: u32,
    /// File type identifier
    pub file_type: u32,
    /// Flags (bit 0: valid, bit 1: dirty, bit 2: pinned)
    pub flags: u32,
    /// Reference count
    pub ref_count: u32,
    /// Status of the handle
    pub status: HandleStatus,
    /// Buffer information
    pub buffer_info: HandleBufferInfo,
}

/// GPU-resident content pipeline with bounds-checked handle access.
///
/// This structure manages file handles in a GPU buffer with strict
/// bounds validation to prevent out-of-bounds memory access.
pub struct ContentPipeline {
    device: Device,
    /// GPU buffer containing FileHandle array
    handles_buffer: Buffer,
    /// Maximum number of handles (bounds limit)
    max_handles: u32,
    /// Current number of active handles
    active_count: u32,
}

impl ContentPipeline {
    /// Create a new content pipeline with the specified capacity.
    ///
    /// # Arguments
    /// * `device` - Metal device
    /// * `max_handles` - Maximum number of file handles to support
    pub fn new(device: &Device, max_handles: u32) -> Self {
        let buffer_size = (max_handles as u64) * (mem::size_of::<FileHandle>() as u64);

        let handles_buffer = device.new_buffer(
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero-initialize the buffer
        unsafe {
            let ptr = handles_buffer.contents() as *mut u8;
            std::ptr::write_bytes(ptr, 0, buffer_size as usize);
        }

        Self {
            device: device.clone(),
            handles_buffer,
            max_handles,
            active_count: 0,
        }
    }

    /// Get the maximum number of handles this pipeline can manage.
    pub fn max_handles(&self) -> u32 {
        self.max_handles
    }

    /// Get the current number of active handles.
    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Get the underlying GPU buffer (for shader binding).
    pub fn handles_buffer(&self) -> &Buffer {
        &self.handles_buffer
    }

    /// Read a file handle from the specified slot.
    ///
    /// Returns `None` if the slot is out of bounds.
    ///
    /// # Arguments
    /// * `slot` - Index of the handle slot to read
    ///
    /// # Safety
    /// This function performs bounds checking before any pointer arithmetic.
    pub fn read_handle(&self, slot: u32) -> Option<FileHandle> {
        // SECURITY: Bounds check BEFORE any pointer arithmetic
        if slot >= self.max_handles {
            return None;
        }

        let ptr = self.handles_buffer.contents() as *const FileHandle;
        // SAFETY: We verified slot < max_handles above
        Some(unsafe { *ptr.add(slot as usize) })
    }

    /// Read a file handle, returning an error with details if out of bounds.
    ///
    /// # Arguments
    /// * `slot` - Index of the handle slot to read
    pub fn read_handle_checked(&self, slot: u32) -> Result<FileHandle, ContentPipelineError> {
        if slot >= self.max_handles {
            return Err(ContentPipelineError::SlotOutOfBounds {
                slot,
                max_handles: self.max_handles,
            });
        }

        let ptr = self.handles_buffer.contents() as *const FileHandle;
        Ok(unsafe { *ptr.add(slot as usize) })
    }

    /// Write a handle status to the specified slot.
    ///
    /// Returns an error if the slot is out of bounds.
    ///
    /// # Arguments
    /// * `slot` - Index of the handle slot to write
    /// * `status` - New status to write
    ///
    /// # Safety
    /// This function performs bounds checking before any pointer arithmetic.
    pub fn write_handle_status(
        &mut self,
        slot: u32,
        status: HandleStatus,
    ) -> Result<(), ContentPipelineError> {
        // SECURITY: Bounds check BEFORE any pointer arithmetic
        if slot >= self.max_handles {
            return Err(ContentPipelineError::SlotOutOfBounds {
                slot,
                max_handles: self.max_handles,
            });
        }

        let ptr = self.handles_buffer.contents() as *mut FileHandle;
        // SAFETY: We verified slot < max_handles above
        unsafe {
            (*ptr.add(slot as usize)).status = status;
        }
        Ok(())
    }

    /// Write buffer info to the specified slot.
    ///
    /// Returns an error if the slot is out of bounds.
    ///
    /// # Arguments
    /// * `slot` - Index of the handle slot to write
    /// * `buffer_info` - New buffer info to write
    ///
    /// # Safety
    /// This function performs bounds checking before any pointer arithmetic.
    pub fn write_handle_buffer_info(
        &mut self,
        slot: u32,
        buffer_info: HandleBufferInfo,
    ) -> Result<(), ContentPipelineError> {
        // SECURITY: Bounds check BEFORE any pointer arithmetic
        if slot >= self.max_handles {
            return Err(ContentPipelineError::SlotOutOfBounds {
                slot,
                max_handles: self.max_handles,
            });
        }

        let ptr = self.handles_buffer.contents() as *mut FileHandle;
        // SAFETY: We verified slot < max_handles above
        unsafe {
            (*ptr.add(slot as usize)).buffer_info = buffer_info;
        }
        Ok(())
    }

    /// Write a complete file handle to the specified slot.
    ///
    /// Returns an error if the slot is out of bounds.
    ///
    /// # Arguments
    /// * `slot` - Index of the handle slot to write
    /// * `handle` - Complete handle to write
    pub fn write_handle(
        &mut self,
        slot: u32,
        handle: FileHandle,
    ) -> Result<(), ContentPipelineError> {
        // SECURITY: Bounds check BEFORE any pointer arithmetic
        if slot >= self.max_handles {
            return Err(ContentPipelineError::SlotOutOfBounds {
                slot,
                max_handles: self.max_handles,
            });
        }

        let ptr = self.handles_buffer.contents() as *mut FileHandle;
        // SAFETY: We verified slot < max_handles above
        unsafe {
            *ptr.add(slot as usize) = handle;
        }
        Ok(())
    }

    /// Allocate a new handle slot.
    ///
    /// Returns the slot index if successful, or an error if no slots available.
    pub fn allocate_handle(&mut self) -> Result<u32, ContentPipelineError> {
        // Find first free slot
        for slot in 0..self.max_handles {
            if let Some(handle) = self.read_handle(slot) {
                if handle.flags == 0 {
                    // Mark as allocated
                    let new_handle = FileHandle {
                        handle_id: slot,
                        flags: 1, // valid flag
                        ..Default::default()
                    };
                    self.write_handle(slot, new_handle)?;
                    self.active_count += 1;
                    return Ok(slot);
                }
            }
        }

        Err(ContentPipelineError::SlotOutOfBounds {
            slot: self.max_handles,
            max_handles: self.max_handles,
        })
    }

    /// Free a handle slot.
    ///
    /// Returns an error if the slot is out of bounds.
    pub fn free_handle(&mut self, slot: u32) -> Result<(), ContentPipelineError> {
        // SECURITY: Bounds check BEFORE any pointer arithmetic
        if slot >= self.max_handles {
            return Err(ContentPipelineError::SlotOutOfBounds {
                slot,
                max_handles: self.max_handles,
            });
        }

        self.write_handle(slot, FileHandle::default())?;
        if self.active_count > 0 {
            self.active_count -= 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_checking_read() {
        let device = Device::system_default().expect("No Metal device");
        let pipeline = ContentPipeline::new(&device, 10);

        // Valid slot
        assert!(pipeline.read_handle(0).is_some());
        assert!(pipeline.read_handle(9).is_some());

        // Out of bounds - should return None, not crash
        assert!(pipeline.read_handle(10).is_none());
        assert!(pipeline.read_handle(100).is_none());
        assert!(pipeline.read_handle(u32::MAX).is_none());
    }

    #[test]
    fn test_bounds_checking_read_checked() {
        let device = Device::system_default().expect("No Metal device");
        let pipeline = ContentPipeline::new(&device, 10);

        // Valid slot
        assert!(pipeline.read_handle_checked(0).is_ok());
        assert!(pipeline.read_handle_checked(9).is_ok());

        // Out of bounds - should return error with details
        let err = pipeline.read_handle_checked(10).unwrap_err();
        assert_eq!(err, ContentPipelineError::SlotOutOfBounds {
            slot: 10,
            max_handles: 10,
        });
    }

    #[test]
    fn test_bounds_checking_write_status() {
        let device = Device::system_default().expect("No Metal device");
        let mut pipeline = ContentPipeline::new(&device, 10);

        let status = HandleStatus {
            state: 2,
            error_code: 0,
            progress: 100,
            _pad: 0,
        };

        // Valid slot
        assert!(pipeline.write_handle_status(0, status).is_ok());
        assert!(pipeline.write_handle_status(9, status).is_ok());

        // Out of bounds - should return error
        assert!(pipeline.write_handle_status(10, status).is_err());
        assert!(pipeline.write_handle_status(u32::MAX, status).is_err());
    }

    #[test]
    fn test_bounds_checking_write_buffer_info() {
        let device = Device::system_default().expect("No Metal device");
        let mut pipeline = ContentPipeline::new(&device, 10);

        let buffer_info = HandleBufferInfo {
            buffer_offset: 0,
            content_size: 1024,
            allocated_size: 4096,
            checksum: 0xDEADBEEF,
        };

        // Valid slot
        assert!(pipeline.write_handle_buffer_info(0, buffer_info).is_ok());
        assert!(pipeline.write_handle_buffer_info(9, buffer_info).is_ok());

        // Out of bounds - should return error
        assert!(pipeline.write_handle_buffer_info(10, buffer_info).is_err());
        assert!(pipeline.write_handle_buffer_info(u32::MAX, buffer_info).is_err());
    }

    #[test]
    fn test_allocate_and_free() {
        let device = Device::system_default().expect("No Metal device");
        let mut pipeline = ContentPipeline::new(&device, 3);

        // Allocate all slots
        let slot0 = pipeline.allocate_handle().unwrap();
        let slot1 = pipeline.allocate_handle().unwrap();
        let slot2 = pipeline.allocate_handle().unwrap();

        assert_eq!(pipeline.active_count(), 3);

        // Should fail - no more slots
        assert!(pipeline.allocate_handle().is_err());

        // Free a slot
        pipeline.free_handle(slot1).unwrap();
        assert_eq!(pipeline.active_count(), 2);

        // Now allocation should succeed
        let slot_new = pipeline.allocate_handle().unwrap();
        assert_eq!(slot_new, slot1); // Should reuse freed slot
    }

    #[test]
    fn test_read_write_roundtrip() {
        let device = Device::system_default().expect("No Metal device");
        let mut pipeline = ContentPipeline::new(&device, 10);

        let handle = FileHandle {
            handle_id: 42,
            file_type: 1,
            flags: 0x07, // valid | dirty | pinned
            ref_count: 3,
            status: HandleStatus {
                state: 2,
                error_code: 0,
                progress: 100,
                _pad: 0,
            },
            buffer_info: HandleBufferInfo {
                buffer_offset: 1024,
                content_size: 512,
                allocated_size: 1024,
                checksum: 0xCAFEBABE,
            },
        };

        // Write and read back
        pipeline.write_handle(5, handle).unwrap();
        let read_back = pipeline.read_handle(5).unwrap();

        assert_eq!(read_back.handle_id, 42);
        assert_eq!(read_back.file_type, 1);
        assert_eq!(read_back.flags, 0x07);
        assert_eq!(read_back.ref_count, 3);
        assert_eq!(read_back.status.state, 2);
        assert_eq!(read_back.buffer_info.checksum, 0xCAFEBABE);
    }
}
