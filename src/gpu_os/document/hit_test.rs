//! GPU-Native Hit Testing
//!
//! Parallel hit testing to find which element is under the mouse cursor.
//! Uses Metal compute shaders for <1ms response time on large documents.

use metal::*;
use super::parser::Element;
use super::style::ComputedStyle;
use super::layout::LayoutBox;

/// Result of a hit test
#[derive(Clone, Copy, Debug, Default)]
pub struct HitTestResult {
    /// Element index that was hit, or u32::MAX if no hit
    pub element_id: u32,
    /// Element type (ELEM_* constant)
    pub element_type: u32,
    /// Whether element is a link (<a>)
    pub is_link: bool,
    /// Whether element is a button
    pub is_button: bool,
    /// Whether element is an input/form element
    pub is_input: bool,
    /// Depth in the element tree
    pub depth: i32,
}

impl HitTestResult {
    pub fn none() -> Self {
        Self {
            element_id: u32::MAX,
            element_type: 0,
            is_link: false,
            is_button: false,
            is_input: false,
            depth: -1,
        }
    }

    pub fn hit(&self) -> bool {
        self.element_id != u32::MAX
    }
}

/// Parameters passed to GPU for hit testing
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HitTestParams {
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub scroll_x: f32,
    pub scroll_y: f32,
    pub element_count: u32,
    pub _padding: [u32; 3],
}

/// Element type constants for hit test results
pub const ELEM_A: u32 = 4;
pub const ELEM_BUTTON: u32 = 25;
pub const ELEM_INPUT: u32 = 24;
pub const ELEM_TEXTAREA: u32 = 26;
pub const ELEM_SELECT: u32 = 27;

/// GPU-Native Hit Tester
pub struct GpuHitTester {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    hit_test_pipeline: ComputePipelineState,
    hit_test_all_pipeline: ComputePipelineState,

    // Buffers for hit test results
    hit_element_id_buffer: Buffer,
    hit_depth_buffer: Buffer,
    hit_flags_buffer: Buffer,
    params_buffer: Buffer,

    max_elements: usize,
}

const THREAD_COUNT: u64 = 1024;
const HIT_TEST_SHADER: &str = include_str!("hit_test.metal");

impl GpuHitTester {
    /// Create a new GPU hit tester
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(HIT_TEST_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile hit test shader: {}", e))?;

        // Create pipelines
        let hit_test_fn = library
            .get_function("hit_test", None)
            .map_err(|e| format!("Failed to find hit_test function: {}", e))?;
        let hit_test_all_fn = library
            .get_function("hit_test_all", None)
            .map_err(|e| format!("Failed to find hit_test_all function: {}", e))?;

        let hit_test_pipeline = device
            .new_compute_pipeline_state_with_function(&hit_test_fn)
            .map_err(|e| format!("Failed to create hit test pipeline: {}", e))?;
        let hit_test_all_pipeline = device
            .new_compute_pipeline_state_with_function(&hit_test_all_fn)
            .map_err(|e| format!("Failed to create hit test all pipeline: {}", e))?;

        // Allocate result buffers
        let max_elements = 16384usize;

        let hit_element_id_buffer = device.new_buffer(
            4,  // single u32
            MTLResourceOptions::StorageModeShared,
        );
        let hit_depth_buffer = device.new_buffer(
            4,  // single i32
            MTLResourceOptions::StorageModeShared,
        );
        let hit_flags_buffer = device.new_buffer(
            (max_elements * 4) as u64,  // u32 per element
            MTLResourceOptions::StorageModeShared,
        );
        let params_buffer = device.new_buffer(
            std::mem::size_of::<HitTestParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device: device.clone(),
            command_queue,
            hit_test_pipeline,
            hit_test_all_pipeline,
            hit_element_id_buffer,
            hit_depth_buffer,
            hit_flags_buffer,
            params_buffer,
            max_elements,
        })
    }

    /// Perform hit test and return the deepest element under the cursor
    ///
    /// # Arguments
    /// * `mouse_x`, `mouse_y` - Mouse position in viewport coordinates
    /// * `scroll_x`, `scroll_y` - Current scroll offset
    /// * `boxes` - Layout boxes buffer (from GpuLayoutEngine)
    /// * `elements` - Element buffer (from GpuParser)
    /// * `styles` - Computed styles buffer (from GpuStyler)
    /// * `element_count` - Number of elements
    ///
    /// # Returns
    /// HitTestResult with element info, or HitTestResult::none() if no hit
    pub fn hit_test(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        scroll_x: f32,
        scroll_y: f32,
        boxes: &Buffer,
        elements: &[Element],
        styles: &Buffer,
        element_count: usize,
    ) -> HitTestResult {
        if element_count == 0 {
            return HitTestResult::none();
        }

        let element_count = element_count.min(self.max_elements);

        // Set up parameters
        let params = HitTestParams {
            mouse_x,
            mouse_y,
            scroll_x,
            scroll_y,
            element_count: element_count as u32,
            _padding: [0; 3],
        };

        unsafe {
            std::ptr::write(self.params_buffer.contents() as *mut HitTestParams, params);

            // Reset result buffers
            *(self.hit_element_id_buffer.contents() as *mut u32) = u32::MAX;
            *(self.hit_depth_buffer.contents() as *mut i32) = -1;
        }

        // Execute kernel
        let command_buffer = self.command_queue.new_command_buffer();
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.hit_test_pipeline);
            encoder.set_buffer(0, Some(boxes), 0);
            encoder.set_buffer(1, Some(&create_temp_buffer_from_elements(&self.device, elements)), 0);
            encoder.set_buffer(2, Some(styles), 0);
            encoder.set_buffer(3, Some(&self.params_buffer), 0);
            encoder.set_buffer(4, Some(&self.hit_element_id_buffer), 0);
            encoder.set_buffer(5, Some(&self.hit_depth_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        let element_id = unsafe { *(self.hit_element_id_buffer.contents() as *const u32) };
        let depth = unsafe { *(self.hit_depth_buffer.contents() as *const i32) };

        if element_id == u32::MAX || element_id as usize >= element_count {
            return HitTestResult::none();
        }

        let elem = &elements[element_id as usize];

        HitTestResult {
            element_id,
            element_type: elem.element_type,
            is_link: elem.element_type == ELEM_A,
            is_button: elem.element_type == ELEM_BUTTON,
            is_input: elem.element_type == ELEM_INPUT
                || elem.element_type == ELEM_TEXTAREA
                || elem.element_type == ELEM_SELECT,
            depth,
        }
    }

    /// Return all elements under the cursor, sorted by depth (deepest first)
    ///
    /// Useful for event bubbling - returns the full chain from target to root.
    pub fn hit_test_all(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        scroll_x: f32,
        scroll_y: f32,
        boxes: &Buffer,
        elements: &[Element],
        styles: &Buffer,
        element_count: usize,
    ) -> Vec<HitTestResult> {
        if element_count == 0 {
            return Vec::new();
        }

        let element_count = element_count.min(self.max_elements);

        // Set up parameters
        let params = HitTestParams {
            mouse_x,
            mouse_y,
            scroll_x,
            scroll_y,
            element_count: element_count as u32,
            _padding: [0; 3],
        };

        unsafe {
            std::ptr::write(self.params_buffer.contents() as *mut HitTestParams, params);

            // Clear hit flags
            let flags_ptr = self.hit_flags_buffer.contents() as *mut u32;
            for i in 0..element_count {
                *flags_ptr.add(i) = 0;
            }
        }

        // Execute kernel
        let command_buffer = self.command_queue.new_command_buffer();
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.hit_test_all_pipeline);
            encoder.set_buffer(0, Some(boxes), 0);
            encoder.set_buffer(1, Some(&create_temp_buffer_from_elements(&self.device, elements)), 0);
            encoder.set_buffer(2, Some(styles), 0);
            encoder.set_buffer(3, Some(&self.params_buffer), 0);
            encoder.set_buffer(4, Some(&self.hit_flags_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results and build list
        let mut results = Vec::new();
        let flags_ptr = self.hit_flags_buffer.contents() as *const u32;

        for i in 0..element_count {
            let hit = unsafe { *flags_ptr.add(i) };
            if hit != 0 {
                let elem = &elements[i];

                // Calculate depth
                let mut depth = 0i32;
                let mut parent = elem.parent;
                while parent >= 0 && depth < 100 {
                    depth += 1;
                    parent = elements[parent as usize].parent;
                }

                results.push(HitTestResult {
                    element_id: i as u32,
                    element_type: elem.element_type,
                    is_link: elem.element_type == ELEM_A,
                    is_button: elem.element_type == ELEM_BUTTON,
                    is_input: elem.element_type == ELEM_INPUT
                        || elem.element_type == ELEM_TEXTAREA
                        || elem.element_type == ELEM_SELECT,
                    depth,
                });
            }
        }

        // Sort by depth (deepest first for event bubbling)
        results.sort_by(|a, b| b.depth.cmp(&a.depth));

        results
    }
}

/// Helper to create a temporary buffer from element slice
fn create_temp_buffer_from_elements(device: &Device, elements: &[Element]) -> Buffer {
    let buffer = device.new_buffer(
        (elements.len() * std::mem::size_of::<Element>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            elements.as_ptr(),
            buffer.contents() as *mut Element,
            elements.len(),
        );
    }

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::ELEM_DIV;

    #[test]
    fn test_hit_test_result_none() {
        let result = HitTestResult::none();
        assert!(!result.hit());
        assert_eq!(result.element_id, u32::MAX);
    }

    #[test]
    fn test_hit_test_result_hit() {
        let result = HitTestResult {
            element_id: 5,
            element_type: ELEM_DIV,
            is_link: false,
            is_button: false,
            is_input: false,
            depth: 2,
        };
        assert!(result.hit());
        assert_eq!(result.element_id, 5);
    }
}
