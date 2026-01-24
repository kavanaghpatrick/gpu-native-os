// GPU-Native Text Editor
//
// A text editor running entirely on the GPU using the GpuApp framework.
// Document lives in unified memory, GPU handles layout and rendering.

use super::app::{GpuApp, AppBuilder, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

pub const MAX_CHARS: usize = 65536;
pub const MAX_LINES: usize = 4096;
pub const VISIBLE_LINES: usize = 40;
pub const CHARS_PER_LINE: usize = 80;
pub const VISIBLE_CHARS: usize = VISIBLE_LINES * CHARS_PER_LINE;
pub const VERTICES_PER_CHAR: usize = 6;
pub const MAX_VERTICES: usize = VISIBLE_CHARS * VERTICES_PER_CHAR * 2; // chars + cursor/selection

// Edit types
pub const EDIT_NONE: u32 = 0;
pub const EDIT_INSERT_CHAR: u32 = 1;
pub const EDIT_DELETE_BACK: u32 = 2;
pub const EDIT_DELETE_FWD: u32 = 3;
pub const EDIT_NEWLINE: u32 = 4;
pub const EDIT_MOVE_LEFT: u32 = 10;
pub const EDIT_MOVE_RIGHT: u32 = 11;
pub const EDIT_MOVE_UP: u32 = 12;
pub const EDIT_MOVE_DOWN: u32 = 13;
pub const EDIT_MOVE_HOME: u32 = 14;
pub const EDIT_MOVE_END: u32 = 15;
pub const EDIT_PAGE_UP: u32 = 16;
pub const EDIT_PAGE_DOWN: u32 = 17;

// ============================================================================
// Data Structures
// ============================================================================

/// Document state - lives in GPU memory
#[repr(C)]
#[derive(Clone)]
pub struct Document {
    pub char_count: u32,
    pub cursor_pos: u32,
    pub selection_start: u32,
    pub selection_end: u32,
    pub scroll_line: u32,
    pub target_column: u32,
    pub dirty: u32,
    pub _padding: u32,
}

impl Default for Document {
    fn default() -> Self {
        Self {
            char_count: 0,
            cursor_pos: 0,
            selection_start: 0,
            selection_end: 0,
            scroll_line: 0,
            target_column: 0,
            dirty: 0,
            _padding: 0,
        }
    }
}

/// Layout cache - computed by GPU
#[repr(C)]
#[derive(Clone)]
pub struct LayoutCache {
    pub line_count: u32,
    pub visible_lines: u32,
    pub cursor_line: u32,
    pub cursor_column: u32,
}

/// Editor parameters - per frame
#[repr(C)]
#[derive(Clone, Default)]
pub struct EditorParams {
    pub delta_time: f32,
    pub time: f32,
    pub pending_edit: u32,
    pub edit_char: u32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub _padding: [u32; 2],
}

/// Text vertex
#[repr(C)]
#[derive(Clone, Default)]
pub struct TextVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

// ============================================================================
// Shader Source
// ============================================================================

fn shader_source() -> String {
    format!(r#"
{header}

// ============================================================================
// Text Editor Structures
// ============================================================================

struct Document {{
    uint char_count;
    uint cursor_pos;
    uint selection_start;
    uint selection_end;
    uint scroll_line;
    uint target_column;
    uint dirty;
    uint _padding;
}};

struct LayoutCache {{
    uint line_count;
    uint visible_lines;
    uint cursor_line;
    uint cursor_column;
}};

struct EditorParams {{
    float delta_time;
    float time;
    uint pending_edit;
    uint edit_char;
    float viewport_width;
    float viewport_height;
    uint _padding[2];
}};

struct TextVertex {{
    float2 position;
    float2 uv;
    float4 color;
}};

// Edit types
constant uint EDIT_NONE = 0;
constant uint EDIT_INSERT_CHAR = 1;
constant uint EDIT_DELETE_BACK = 2;
constant uint EDIT_DELETE_FWD = 3;
constant uint EDIT_NEWLINE = 4;
constant uint EDIT_MOVE_LEFT = 10;
constant uint EDIT_MOVE_RIGHT = 11;
constant uint EDIT_MOVE_UP = 12;
constant uint EDIT_MOVE_DOWN = 13;
constant uint EDIT_MOVE_HOME = 14;
constant uint EDIT_MOVE_END = 15;
constant uint EDIT_PAGE_UP = 16;
constant uint EDIT_PAGE_DOWN = 17;

// Visual constants
constant float CHAR_WIDTH = 0.012;
constant float CHAR_HEIGHT = 0.025;
constant float LINE_HEIGHT = 0.028;
constant float MARGIN_LEFT = 0.06;
constant float MARGIN_TOP = 0.05;
constant uint VISIBLE_LINES = 40;
constant uint CHARS_PER_LINE = 80;

// Colors
constant float4 COLOR_TEXT = float4(0.804, 0.839, 0.957, 1.0);       // #cdd6f4
constant float4 COLOR_BACKGROUND = float4(0.118, 0.118, 0.180, 1.0); // #1e1e2e
constant float4 COLOR_CURSOR = float4(0.961, 0.878, 0.863, 1.0);     // #f5e0dc
constant float4 COLOR_LINE_NUM = float4(0.424, 0.443, 0.525, 1.0);   // #6c7086
constant float4 COLOR_CURRENT_LINE = float4(0.192, 0.204, 0.267, 1.0); // #313244
constant float4 COLOR_SELECTION = float4(0.271, 0.278, 0.353, 1.0);  // #45475a

// ============================================================================
// Bitmap Font Data (5x7 pixel font embedded in shader)
// ============================================================================

// Simple 5x7 bitmap font - each character is 7 rows of 5 bits
// Stored as array of 7 bytes per character (ASCII 32-126)
constant uchar FONT_DATA[95 * 7] = {{
    // Space (32)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // ! (33)
    0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04,
    // " (34)
    0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00,
    // # (35)
    0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A,
    // $ (36)
    0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04,
    // % (37)
    0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03,
    // & (38)
    0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0D,
    // ' (39)
    0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00,
    // ( (40)
    0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02,
    // ) (41)
    0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08,
    // * (42)
    0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00,
    // + (43)
    0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00,
    // , (44)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08,
    // - (45)
    0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00,
    // . (46)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
    // / (47)
    0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10,
    // 0 (48)
    0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E,
    // 1 (49)
    0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E,
    // 2 (50)
    0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F,
    // 3 (51)
    0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E,
    // 4 (52)
    0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02,
    // 5 (53)
    0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E,
    // 6 (54)
    0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E,
    // 7 (55)
    0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08,
    // 8 (56)
    0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E,
    // 9 (57)
    0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C,
    // : (58)
    0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00,
    // ; (59)
    0x00, 0x00, 0x04, 0x00, 0x04, 0x04, 0x08,
    // < (60)
    0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02,
    // = (61)
    0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00,
    // > (62)
    0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08,
    // ? (63)
    0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04,
    // @ (64)
    0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E,
    // A (65)
    0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11,
    // B (66)
    0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E,
    // C (67)
    0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E,
    // D (68)
    0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E,
    // E (69)
    0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F,
    // F (70)
    0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10,
    // G (71)
    0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E,
    // H (72)
    0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11,
    // I (73)
    0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E,
    // J (74)
    0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C,
    // K (75)
    0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11,
    // L (76)
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F,
    // M (77)
    0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11,
    // N (78)
    0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11,
    // O (79)
    0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E,
    // P (80)
    0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10,
    // Q (81)
    0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D,
    // R (82)
    0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11,
    // S (83)
    0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E,
    // T (84)
    0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    // U (85)
    0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E,
    // V (86)
    0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04,
    // W (87)
    0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11,
    // X (88)
    0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11,
    // Y (89)
    0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04,
    // Z (90)
    0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F,
    // [ (91)
    0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E,
    // \ (92)
    0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01,
    // ] (93)
    0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E,
    // ^ (94)
    0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00,
    // _ (95)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F,
    // ` (96)
    0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00,
    // a (97)
    0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F,
    // b (98)
    0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E,
    // c (99)
    0x00, 0x00, 0x0F, 0x10, 0x10, 0x10, 0x0F,
    // d (100)
    0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F,
    // e (101)
    0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E,
    // f (102)
    0x06, 0x08, 0x1C, 0x08, 0x08, 0x08, 0x08,
    // g (103)
    0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E,
    // h (104)
    0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11,
    // i (105)
    0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E,
    // j (106)
    0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C,
    // k (107)
    0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12,
    // l (108)
    0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E,
    // m (109)
    0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11,
    // n (110)
    0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11,
    // o (111)
    0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E,
    // p (112)
    0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10,
    // q (113)
    0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01,
    // r (114)
    0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10,
    // s (115)
    0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E,
    // t (116)
    0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06,
    // u (117)
    0x00, 0x00, 0x11, 0x11, 0x11, 0x11, 0x0F,
    // v (118)
    0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04,
    // w (119)
    0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A,
    // x (120)
    0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11,
    // y (121)
    0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E,
    // z (122)
    0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F,
    // {{ (123)
    0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02,
    // | (124)
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    // }} (125)
    0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08,
    // ~ (126)
    0x00, 0x08, 0x15, 0x02, 0x00, 0x00, 0x00,
}};

// Get font pixel (returns true if pixel is set)
bool get_font_pixel(uint char_code, uint px, uint py) {{
    if (char_code < 32 || char_code > 126) return false;
    if (px >= 5 || py >= 7) return false;

    uint font_index = (char_code - 32) * 7 + py;
    uchar row = FONT_DATA[font_index];
    return (row >> (4 - px)) & 1;
}}

// ============================================================================
// Main Compute Kernel
// ============================================================================

kernel void text_editor_kernel(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant EditorParams& params [[buffer(2)]],
    device Document* doc [[buffer(3)]],
    device uint* chars [[buffer(4)]],
    device LayoutCache* layout [[buffer(5)]],
    device TextVertex* vertices [[buffer(6)]],
    device atomic_uint* vertex_count [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {{
    // Shared memory for coordination
    threadgroup uint tg_line_count;
    threadgroup uint tg_cursor_line;
    threadgroup uint tg_cursor_col;
    threadgroup uint tg_char_count;
    threadgroup uint tg_vertex_idx;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: APPLY PENDING EDITS
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {{
        tg_char_count = doc->char_count;
        uint edit = params.pending_edit;
        uint cursor = doc->cursor_pos;
        uint count = tg_char_count;

        if (edit == EDIT_INSERT_CHAR && count < 65535) {{
            // Shift chars right (done by other threads below)
            doc->char_count = count + 1;
            doc->cursor_pos = cursor + 1;
            tg_char_count = count + 1;
        }}
        else if (edit == EDIT_NEWLINE && count < 65535) {{
            doc->char_count = count + 1;
            doc->cursor_pos = cursor + 1;
            tg_char_count = count + 1;
        }}
        else if (edit == EDIT_DELETE_BACK && cursor > 0) {{
            doc->char_count = count - 1;
            doc->cursor_pos = cursor - 1;
            tg_char_count = count - 1;
        }}
        else if (edit == EDIT_DELETE_FWD && cursor < count) {{
            doc->char_count = count - 1;
            tg_char_count = count - 1;
        }}
        else if (edit == EDIT_MOVE_LEFT && cursor > 0) {{
            doc->cursor_pos = cursor - 1;
        }}
        else if (edit == EDIT_MOVE_RIGHT && cursor < count) {{
            doc->cursor_pos = cursor + 1;
        }}
        else if (edit == EDIT_MOVE_HOME) {{
            // Find start of current line
            uint pos = cursor;
            while (pos > 0 && chars[pos - 1] != 10) pos--;
            doc->cursor_pos = pos;
        }}
        else if (edit == EDIT_MOVE_END) {{
            // Find end of current line
            uint pos = cursor;
            while (pos < count && chars[pos] != 10) pos++;
            doc->cursor_pos = pos;
        }}
        else if (edit == EDIT_MOVE_UP) {{
            // Find start of current line and calculate column
            uint line_start = cursor;
            while (line_start > 0 && chars[line_start - 1] != 10) line_start--;
            uint col = cursor - line_start;

            // Find start of previous line
            if (line_start > 0) {{
                uint prev_line_end = line_start - 1; // newline char
                uint prev_line_start = prev_line_end;
                while (prev_line_start > 0 && chars[prev_line_start - 1] != 10) prev_line_start--;

                // Move to same column or end of previous line
                uint prev_line_len = prev_line_end - prev_line_start;
                if (col > prev_line_len) col = prev_line_len;
                doc->cursor_pos = prev_line_start + col;
            }}
        }}
        else if (edit == EDIT_MOVE_DOWN) {{
            // Find start of current line and calculate column
            uint line_start = cursor;
            while (line_start > 0 && chars[line_start - 1] != 10) line_start--;
            uint col = cursor - line_start;

            // Find end of current line (newline char)
            uint line_end = cursor;
            while (line_end < count && chars[line_end] != 10) line_end++;

            // If there's a next line
            if (line_end < count) {{
                uint next_line_start = line_end + 1;
                uint next_line_end = next_line_start;
                while (next_line_end < count && chars[next_line_end] != 10) next_line_end++;

                // Move to same column or end of next line
                uint next_line_len = next_line_end - next_line_start;
                if (col > next_line_len) col = next_line_len;
                doc->cursor_pos = next_line_start + col;
            }}
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel character shifting for insert/delete
    uint edit = params.pending_edit;
    uint cursor = doc->cursor_pos;
    uint old_count = doc->char_count;

    if (edit == EDIT_INSERT_CHAR || edit == EDIT_NEWLINE) {{
        // Shift right: each thread handles multiple chars
        for (uint i = tid; i < old_count; i += 1024) {{
            if (i >= cursor - 1) {{
                uint src_idx = old_count - 1 - (i - (cursor - 1));
                if (src_idx >= cursor - 1 && src_idx < old_count - 1) {{
                    chars[src_idx + 1] = chars[src_idx];
                }}
            }}
        }}
    }}

    threadgroup_barrier(mem_flags::mem_device);

    // Insert the new character
    if (tid == 0) {{
        if (edit == EDIT_INSERT_CHAR) {{
            chars[cursor - 1] = params.edit_char;
        }}
        else if (edit == EDIT_NEWLINE) {{
            chars[cursor - 1] = 10; // newline
        }}
    }}

    if (edit == EDIT_DELETE_BACK) {{
        // Shift left from cursor-1
        uint del_pos = cursor; // cursor already decremented
        for (uint i = tid; i < old_count; i += 1024) {{
            if (i >= del_pos && i < old_count - 1) {{
                chars[i] = chars[i + 1];
            }}
        }}
    }}
    else if (edit == EDIT_DELETE_FWD) {{
        // Shift left from cursor
        for (uint i = tid; i < old_count; i += 1024) {{
            if (i >= cursor && i < old_count - 1) {{
                chars[i] = chars[i + 1];
            }}
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: COMPUTE LINE LAYOUT
    // ═══════════════════════════════════════════════════════════════════

    // Count newlines to find cursor line/column
    if (tid == 0) {{
        uint line = 0;
        uint col = 0;
        uint count = doc->char_count;
        uint cur = doc->cursor_pos;

        for (uint i = 0; i < count && i < cur; i++) {{
            if (chars[i] == 10) {{
                line++;
                col = 0;
            }} else {{
                col++;
            }}
        }}

        tg_cursor_line = line;
        tg_cursor_col = col;

        // Count total lines
        uint total_lines = 1;
        for (uint i = 0; i < count; i++) {{
            if (chars[i] == 10) total_lines++;
        }}
        tg_line_count = total_lines;

        layout->line_count = total_lines;
        layout->cursor_line = line;
        layout->cursor_column = col;

        // Auto-scroll to keep cursor visible
        if (line < doc->scroll_line) {{
            doc->scroll_line = line;
        }}
        if (line >= doc->scroll_line + VISIBLE_LINES) {{
            doc->scroll_line = line - VISIBLE_LINES + 1;
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: GENERATE GEOMETRY
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {{
        atomic_store_explicit(vertex_count, 0, memory_order_relaxed);
        tg_vertex_idx = 0;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 generates all visible character geometry
    // (In a more optimized version, we'd parallelize this)
    if (tid == 0) {{
        uint count = doc->char_count;
        uint scroll = doc->scroll_line;
        uint cursor_pos = doc->cursor_pos;
        float time = params.time;

        uint current_line = 0;
        uint current_col = 0;
        uint vidx = 0;

        // Generate line number background and current line highlight
        for (uint line = 0; line < VISIBLE_LINES && (scroll + line) < tg_line_count; line++) {{
            float y = MARGIN_TOP + float(line) * LINE_HEIGHT;

            // Current line highlight
            if (scroll + line == tg_cursor_line) {{
                float x0 = 0.0;
                float x1 = 1.0;
                float y0 = y - 0.002;
                float y1 = y + LINE_HEIGHT - 0.002;

                vertices[vidx++] = TextVertex{{float2(x0, y0), float2(0, 0), COLOR_CURRENT_LINE}};
                vertices[vidx++] = TextVertex{{float2(x1, y0), float2(1, 0), COLOR_CURRENT_LINE}};
                vertices[vidx++] = TextVertex{{float2(x0, y1), float2(0, 1), COLOR_CURRENT_LINE}};
                vertices[vidx++] = TextVertex{{float2(x1, y0), float2(1, 0), COLOR_CURRENT_LINE}};
                vertices[vidx++] = TextVertex{{float2(x1, y1), float2(1, 1), COLOR_CURRENT_LINE}};
                vertices[vidx++] = TextVertex{{float2(x0, y1), float2(0, 1), COLOR_CURRENT_LINE}};
            }}
        }}

        // Generate text characters
        for (uint i = 0; i < count && vidx < 65000; i++) {{
            uint ch = chars[i];

            if (ch == 10) {{ // newline
                current_line++;
                current_col = 0;
                continue;
            }}

            // Skip lines before scroll
            if (current_line < scroll) {{
                current_col++;
                continue;
            }}

            // Stop after visible lines
            if (current_line >= scroll + VISIBLE_LINES) {{
                break;
            }}

            uint visible_line = current_line - scroll;
            float x = MARGIN_LEFT + float(current_col) * CHAR_WIDTH;
            float y = MARGIN_TOP + float(visible_line) * LINE_HEIGHT;

            // Skip if off screen horizontally
            if (current_col >= CHARS_PER_LINE) {{
                current_col++;
                continue;
            }}

            // Generate character quad with embedded font rendering
            float x0 = x;
            float y0 = y;
            float x1 = x + CHAR_WIDTH;
            float y1 = y + CHAR_HEIGHT;

            // Encode: uv.x = char_code + local_u (0-0.99), uv.y = local_v (0-1)
            float char_base = float(ch);

            vertices[vidx++] = TextVertex{{float2(x0, y0), float2(char_base, 0.0), COLOR_TEXT}};
            vertices[vidx++] = TextVertex{{float2(x1, y0), float2(char_base + 0.99, 0.0), COLOR_TEXT}};
            vertices[vidx++] = TextVertex{{float2(x0, y1), float2(char_base, 0.99), COLOR_TEXT}};
            vertices[vidx++] = TextVertex{{float2(x1, y0), float2(char_base + 0.99, 0.0), COLOR_TEXT}};
            vertices[vidx++] = TextVertex{{float2(x1, y1), float2(char_base + 0.99, 0.99), COLOR_TEXT}};
            vertices[vidx++] = TextVertex{{float2(x0, y1), float2(char_base, 0.99), COLOR_TEXT}};

            current_col++;
        }}

        // Generate cursor
        float cursor_blink = fmod(time * 2.0, 2.0) < 1.0 ? 1.0 : 0.0;
        if (cursor_blink > 0.5 && tg_cursor_line >= scroll && tg_cursor_line < scroll + VISIBLE_LINES) {{
            uint visible_line = tg_cursor_line - scroll;
            float cx = MARGIN_LEFT + float(tg_cursor_col) * CHAR_WIDTH;
            float cy = MARGIN_TOP + float(visible_line) * LINE_HEIGHT;

            float cx0 = cx - 0.001;
            float cx1 = cx + 0.002;
            float cy0 = cy;
            float cy1 = cy + CHAR_HEIGHT;

            vertices[vidx++] = TextVertex{{float2(cx0, cy0), float2(0, 0), COLOR_CURSOR}};
            vertices[vidx++] = TextVertex{{float2(cx1, cy0), float2(0, 0), COLOR_CURSOR}};
            vertices[vidx++] = TextVertex{{float2(cx0, cy1), float2(0, 0), COLOR_CURSOR}};
            vertices[vidx++] = TextVertex{{float2(cx1, cy0), float2(0, 0), COLOR_CURSOR}};
            vertices[vidx++] = TextVertex{{float2(cx1, cy1), float2(0, 0), COLOR_CURSOR}};
            vertices[vidx++] = TextVertex{{float2(cx0, cy1), float2(0, 0), COLOR_CURSOR}};
        }}

        // Generate line numbers
        for (uint line = 0; line < VISIBLE_LINES && (scroll + line) < tg_line_count; line++) {{
            uint line_num = scroll + line + 1;
            float y = MARGIN_TOP + float(line) * LINE_HEIGHT;

            // Up to 4 digits
            uint digits[4];
            uint num_digits = 0;
            uint n = line_num;
            while (n > 0 && num_digits < 4) {{
                digits[num_digits++] = n % 10;
                n /= 10;
            }}
            if (num_digits == 0) {{
                digits[0] = 0;
                num_digits = 1;
            }}

            // Render digits right-aligned
            for (uint d = 0; d < num_digits; d++) {{
                uint digit = digits[num_digits - 1 - d];
                float dx = 0.01 + float(d) * CHAR_WIDTH;
                float dx0 = dx;
                float dx1 = dx + CHAR_WIDTH;
                float dy0 = y;
                float dy1 = y + CHAR_HEIGHT;

                // Encode: uv.x = char_code + local_u, uv.y = local_v
                float char_base = float(48 + digit); // '0' = 48

                vertices[vidx++] = TextVertex{{float2(dx0, dy0), float2(char_base, 0.0), COLOR_LINE_NUM}};
                vertices[vidx++] = TextVertex{{float2(dx1, dy0), float2(char_base + 0.99, 0.0), COLOR_LINE_NUM}};
                vertices[vidx++] = TextVertex{{float2(dx0, dy1), float2(char_base, 0.99), COLOR_LINE_NUM}};
                vertices[vidx++] = TextVertex{{float2(dx1, dy0), float2(char_base + 0.99, 0.0), COLOR_LINE_NUM}};
                vertices[vidx++] = TextVertex{{float2(dx1, dy1), float2(char_base + 0.99, 0.99), COLOR_LINE_NUM}};
                vertices[vidx++] = TextVertex{{float2(dx0, dy1), float2(char_base, 0.99), COLOR_LINE_NUM}};
            }}
        }}

        atomic_store_explicit(vertex_count, vidx, memory_order_relaxed);
    }}
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 uv;
    float4 color;
}};

vertex VertexOut text_vertex(
    const device TextVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    TextVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position * 2.0 - 1.0, 0.0, 1.0);
    out.position.y = -out.position.y;
    out.uv = v.uv;
    out.color = v.color;
    return out;
}}

// ============================================================================
// Fragment Shader
// ============================================================================

fragment float4 text_fragment(VertexOut in [[stage_in]]) {{
    // Check if this is a character quad (uv.x encodes character code + local_u)
    // uv.x = char_code + local_u (0-0.99)
    // uv.y = local_v (0-0.99)
    float char_code_f = floor(in.uv.x);
    uint char_code = uint(char_code_f);

    if (char_code >= 32 && char_code <= 126) {{
        // Extract local position within character cell
        float local_u = in.uv.x - char_code_f;  // 0 to 0.99
        float local_v = in.uv.y;                 // 0 to 0.99

        // Map to 5x7 pixel grid
        uint px = uint(local_u * 5.0);
        uint py = uint(local_v * 7.0);

        // Clamp to valid range
        px = min(px, 4u);
        py = min(py, 6u);

        // Sample the bitmap font
        if (get_font_pixel(char_code, px, py)) {{
            return in.color;
        }} else {{
            discard_fragment();
        }}
    }}

    // Solid color (cursor, highlights, backgrounds)
    return in.color;
}}
"#, header = APP_SHADER_HEADER)
}

// ============================================================================
// TextEditor App
// ============================================================================

pub struct TextEditor {
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,
    params_buffer: Buffer,
    doc_buffer: Buffer,
    chars_buffer: Buffer,
    layout_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,
    current_params: EditorParams,
    pending_edit: u32,
    edit_char: u32,
    time: f32,
}

impl TextEditor {
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "TextEditor");

        let source = shader_source();
        let library = builder.compile_library(&source)?;

        let compute_pipeline = builder.create_compute_pipeline(&library, "text_editor_kernel")?;
        let render_pipeline = builder.create_render_pipeline(&library, "text_vertex", "text_fragment")?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<EditorParams>());
        let doc_buffer = builder.create_buffer(mem::size_of::<Document>());
        let chars_buffer = builder.create_buffer(MAX_CHARS * 4); // u32 per char
        let layout_buffer = builder.create_buffer(mem::size_of::<LayoutCache>());
        let vertices_buffer = builder.create_buffer(MAX_VERTICES * mem::size_of::<TextVertex>());
        let vertex_count_buffer = builder.create_buffer(4); // atomic u32

        // Initialize document with sample text
        let sample_text = "// GPU-Native Text Editor\n// All text processing runs on the GPU!\n\nfn main() {\n    println!(\"Hello, GPU!\");\n}\n\nTry typing here...\nUse arrow keys to move.\nBackspace to delete.\n";

        unsafe {
            let doc_ptr = doc_buffer.contents() as *mut Document;
            *doc_ptr = Document {
                char_count: sample_text.len() as u32,
                cursor_pos: sample_text.len() as u32,
                ..Default::default()
            };

            let chars_ptr = chars_buffer.contents() as *mut u32;
            for (i, c) in sample_text.chars().enumerate() {
                *chars_ptr.add(i) = c as u32;
            }
        }

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            doc_buffer,
            chars_buffer,
            layout_buffer,
            vertices_buffer,
            vertex_count_buffer,
            current_params: EditorParams::default(),
            pending_edit: EDIT_NONE,
            edit_char: 0,
            time: 0.0,
        })
    }

    pub fn document(&self) -> Document {
        unsafe { (*(self.doc_buffer.contents() as *const Document)).clone() }
    }

    pub fn layout(&self) -> LayoutCache {
        unsafe { (*(self.layout_buffer.contents() as *const LayoutCache)).clone() }
    }

    fn vertex_count_val(&self) -> usize {
        unsafe { *(self.vertex_count_buffer.contents() as *const u32) as usize }
    }
}

impl GpuApp for TextEditor {
    fn name(&self) -> &str {
        "Text Editor"
    }

    fn compute_pipeline(&self) -> &ComputePipelineState {
        &self.compute_pipeline
    }

    fn render_pipeline(&self) -> &RenderPipelineState {
        &self.render_pipeline
    }

    fn vertices_buffer(&self) -> &Buffer {
        &self.vertices_buffer
    }

    fn vertex_count(&self) -> usize {
        self.vertex_count_val()
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.doc_buffer,          // slot 3
            &self.chars_buffer,        // slot 4
            &self.layout_buffer,       // slot 5
            &self.vertices_buffer,     // slot 6
            &self.vertex_count_buffer, // slot 7
        ]
    }

    fn update_params(&mut self, _frame_state: &FrameState, delta_time: f32) {
        self.time += delta_time;

        self.current_params = EditorParams {
            delta_time,
            time: self.time,
            pending_edit: self.pending_edit,
            edit_char: self.edit_char,
            viewport_width: 1.0,
            viewport_height: 1.0,
            _padding: [0; 2],
        };

        unsafe {
            let ptr = self.params_buffer.contents() as *mut EditorParams;
            *ptr = self.current_params.clone();
        }

        // Clear pending edit after sending
        self.pending_edit = EDIT_NONE;
        self.edit_char = 0;
    }

    fn handle_input(&mut self, event: &InputEvent) {
        if event.event_type != InputEventType::KeyDown as u16 {
            return;
        }

        // Map keycodes to edit operations
        // Using common HID keycodes
        match event.keycode {
            0x50 => self.pending_edit = EDIT_MOVE_LEFT,  // Left arrow
            0x4F => self.pending_edit = EDIT_MOVE_RIGHT, // Right arrow
            0x52 => self.pending_edit = EDIT_MOVE_UP,    // Up arrow
            0x51 => self.pending_edit = EDIT_MOVE_DOWN,  // Down arrow
            0x2A => self.pending_edit = EDIT_DELETE_BACK, // Backspace
            0x4C => self.pending_edit = EDIT_DELETE_FWD,  // Delete
            0x28 => self.pending_edit = EDIT_NEWLINE,     // Enter
            0x4A => self.pending_edit = EDIT_MOVE_HOME,   // Home
            0x4D => self.pending_edit = EDIT_MOVE_END,    // End
            _ => {}
        }
    }

    fn clear_color(&self) -> MTLClearColor {
        // Dark purple-gray background
        MTLClearColor::new(0.118, 0.118, 0.180, 1.0)
    }
}

// Helper to queue a character insertion from the example
impl TextEditor {
    pub fn insert_char(&mut self, c: char) {
        self.pending_edit = EDIT_INSERT_CHAR;
        self.edit_char = c as u32;
    }

    pub fn delete_back(&mut self) {
        self.pending_edit = EDIT_DELETE_BACK;
    }

    pub fn delete_forward(&mut self) {
        self.pending_edit = EDIT_DELETE_FWD;
    }

    pub fn move_cursor(&mut self, direction: u32) {
        self.pending_edit = direction;
    }

    pub fn newline(&mut self) {
        self.pending_edit = EDIT_NEWLINE;
    }

    /// Load file contents into the editor
    pub fn load_file(&mut self, path: &str) -> Result<(), String> {
        use std::fs;

        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Truncate to MAX_CHARS if needed
        let content: String = content.chars().take(MAX_CHARS - 1).collect();
        let char_count = content.len();

        unsafe {
            // Update document state
            let doc_ptr = self.doc_buffer.contents() as *mut Document;
            (*doc_ptr).char_count = char_count as u32;
            (*doc_ptr).cursor_pos = 0;
            (*doc_ptr).selection_start = 0;
            (*doc_ptr).selection_end = 0;
            (*doc_ptr).scroll_line = 0;
            (*doc_ptr).target_column = 0;
            (*doc_ptr).dirty = 0;

            // Copy characters to GPU buffer
            let chars_ptr = self.chars_buffer.contents() as *mut u32;
            for (i, c) in content.chars().enumerate() {
                *chars_ptr.add(i) = c as u32;
            }
        }

        Ok(())
    }

    /// Clear the editor content
    pub fn clear(&mut self) {
        unsafe {
            let doc_ptr = self.doc_buffer.contents() as *mut Document;
            (*doc_ptr).char_count = 0;
            (*doc_ptr).cursor_pos = 0;
            (*doc_ptr).selection_start = 0;
            (*doc_ptr).selection_end = 0;
            (*doc_ptr).scroll_line = 0;
            (*doc_ptr).target_column = 0;
            (*doc_ptr).dirty = 0;
        }
    }

    /// Load file using GPU-Direct Storage (Metal 3 MTLIOCommandQueue)
    ///
    /// This bypasses CPU for the file read - data goes directly from disk to GPU buffer.
    /// UTF-8 decoding still happens on CPU (future: GPU UTF-8 decoder kernel).
    pub fn load_file_gpu_direct(
        &mut self,
        device: &metal::Device,
        path: &str,
    ) -> Result<(), String> {
        use crate::gpu_os::gpu_io::{GpuIOQueue, GpuIOBuffer, IOPriority, IOQueueType};

        // Create GPU I/O queue
        let io_queue = GpuIOQueue::new(device, IOPriority::Normal, IOQueueType::Concurrent)
            .ok_or_else(|| "GPU-Direct I/O not supported (requires Metal 3)".to_string())?;

        // Load file directly to GPU buffer
        let gpu_buffer = GpuIOBuffer::load_file(&io_queue, path)
            .ok_or_else(|| format!("Failed to load file via GPU-Direct: {}", path))?;

        let file_size = gpu_buffer.file_size() as usize;
        if file_size == 0 {
            self.clear();
            return Ok(());
        }

        // Read bytes from GPU buffer (shared memory - zero-copy access)
        let bytes: &[u8] = unsafe {
            let ptr = gpu_buffer.metal_buffer().contents() as *const u8;
            std::slice::from_raw_parts(ptr, file_size)
        };

        // Convert UTF-8 to string (CPU - future: GPU UTF-8 decoder)
        let content = std::str::from_utf8(bytes)
            .map_err(|e| format!("Invalid UTF-8: {}", e))?;

        // Truncate to MAX_CHARS if needed
        let content: String = content.chars().take(MAX_CHARS - 1).collect();
        let char_count = content.len();

        unsafe {
            // Update document state
            let doc_ptr = self.doc_buffer.contents() as *mut Document;
            (*doc_ptr).char_count = char_count as u32;
            (*doc_ptr).cursor_pos = 0;
            (*doc_ptr).selection_start = 0;
            (*doc_ptr).selection_end = 0;
            (*doc_ptr).scroll_line = 0;
            (*doc_ptr).target_column = 0;
            (*doc_ptr).dirty = 0;

            // Copy characters to GPU buffer
            let chars_ptr = self.chars_buffer.contents() as *mut u32;
            for (i, c) in content.chars().enumerate() {
                *chars_ptr.add(i) = c as u32;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<Document>(), 32);
        assert_eq!(mem::size_of::<EditorParams>(), 32);
        assert_eq!(mem::size_of::<TextVertex>(), 32);
    }
}
