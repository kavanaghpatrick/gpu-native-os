# GPU-Native Text Editor PRD

## Overview

A text editor running entirely on the GPU using the GpuApp framework. The document lives in unified memory, and the GPU handles layout, rendering, cursor positioning, and selection—all in a single compute dispatch per frame.

**Goal:** Prove the GPU-Native OS can build real productivity applications, not just visualizations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED MEMORY (Shared)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DOCUMENT BUFFER (64KB - ~64K characters max)                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ chars: [u32; 65536]      │ Unicode codepoints                          │ │
│  │ char_count: u32          │ Current document length                     │ │
│  │ cursor_pos: u32          │ Insertion point                             │ │
│  │ selection_start: u32     │ Selection anchor (or == cursor if none)     │ │
│  │ selection_end: u32       │ Selection end                               │ │
│  │ scroll_line: u32         │ First visible line                          │ │
│  │ dirty: u32               │ Document modified flag                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  LAYOUT CACHE (computed by GPU each frame)                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ line_starts: [u32; 4096] │ Character index where each line begins      │ │
│  │ line_count: u32          │ Total lines in document                     │ │
│  │ visible_lines: u32       │ Lines that fit in viewport                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  GLYPH ATLAS (preloaded, read-only)                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 16x16 bitmap per ASCII character (32-126)                              │ │
│  │ Simple monospace font for MVP                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  PENDING EDITS (CPU writes, GPU reads & applies)                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ edit_type: u32           │ INSERT_CHAR, DELETE_CHAR, etc.              │ │
│  │ edit_char: u32           │ Character to insert                         │ │
│  │ edit_count: u32          │ Number of chars (for delete)                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## GPU Kernel Phases

All 1024 threads participate in every phase:

### Phase 1: Apply Pending Edits
```
Thread 0 reads pending edit from CPU
All threads shift characters in parallel for insert/delete
- INSERT: threads >= cursor shift right by 1
- DELETE: threads > cursor shift left by 1
Thread 0 clears pending edit flag
```

### Phase 2: Line Layout
```
Thread N checks if chars[N] == '\n'
SIMD prefix sum to compute line numbers
Store line_starts array for visible lines
```

### Phase 3: Cursor & Selection
```
Compute cursor (line, column) from cursor_pos
Compute selection range (line, col) for start/end
Determine if cursor is visible; auto-scroll if needed
```

### Phase 4: Geometry Generation
```
Thread N generates vertices for visible character N:
- Character glyph quad (textured)
- Selection highlight quad (if in selection)
- Cursor quad (if at cursor position, with blink)
```

## Data Structures

### Document (Rust)
```rust
#[repr(C)]
pub struct Document {
    pub chars: [u32; 65536],      // Unicode codepoints
    pub char_count: u32,
    pub cursor_pos: u32,
    pub selection_start: u32,
    pub selection_end: u32,
    pub scroll_line: u32,
    pub target_column: u32,       // For up/down navigation
    pub dirty: u32,
    pub _padding: [u32; 8],
}

#[repr(C)]
pub struct LayoutCache {
    pub line_starts: [u32; 4096],
    pub line_count: u32,
    pub visible_lines: u32,
    pub chars_per_line: u32,      // For wrapping (0 = no wrap)
    pub _padding: u32,
}

#[repr(C)]
pub struct EditorParams {
    pub delta_time: f32,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub pending_edit: u32,        // Edit type
    pub edit_char: u32,           // Character for insert
    pub edit_count: u32,          // Count for delete
    pub blink_time: f32,          // For cursor blink
    pub _padding: u32,
}
```

### Edit Types
```rust
pub const EDIT_NONE: u32 = 0;
pub const EDIT_INSERT_CHAR: u32 = 1;
pub const EDIT_DELETE_BACK: u32 = 2;    // Backspace
pub const EDIT_DELETE_FWD: u32 = 3;     // Delete key
pub const EDIT_NEWLINE: u32 = 4;
pub const EDIT_MOVE_LEFT: u32 = 10;
pub const EDIT_MOVE_RIGHT: u32 = 11;
pub const EDIT_MOVE_UP: u32 = 12;
pub const EDIT_MOVE_DOWN: u32 = 13;
pub const EDIT_MOVE_HOME: u32 = 14;
pub const EDIT_MOVE_END: u32 = 15;
pub const EDIT_SELECT_ALL: u32 = 20;
pub const EDIT_DELETE_SELECTION: u32 = 21;
```

## Text Rendering Approach

### Option A: Simple Bitmap Font (MVP)
- Precomputed 16x16 grayscale bitmap per ASCII character
- Single texture atlas (16 chars × 6 rows = 96 chars)
- GPU samples texture for each character quad
- Fast, simple, sufficient for code/text editing

### Option B: MSDF (Future)
- Multi-channel signed distance field
- Resolution-independent sharp text
- More complex shader, better quality at all sizes

**MVP uses Option A** for simplicity.

## Keyboard Handling

```rust
impl GpuApp for TextEditor {
    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            INPUT_KEY_DOWN => {
                match event.keycode {
                    KEY_LEFT => self.pending_edit = EDIT_MOVE_LEFT,
                    KEY_RIGHT => self.pending_edit = EDIT_MOVE_RIGHT,
                    KEY_UP => self.pending_edit = EDIT_MOVE_UP,
                    KEY_DOWN => self.pending_edit = EDIT_MOVE_DOWN,
                    KEY_BACKSPACE => self.pending_edit = EDIT_DELETE_BACK,
                    KEY_DELETE => self.pending_edit = EDIT_DELETE_FWD,
                    KEY_ENTER => self.pending_edit = EDIT_NEWLINE,
                    KEY_HOME => self.pending_edit = EDIT_MOVE_HOME,
                    KEY_END => self.pending_edit = EDIT_MOVE_END,
                    _ => {
                        // Printable character
                        if let Some(c) = keycode_to_char(event.keycode, event.modifiers) {
                            self.pending_edit = EDIT_INSERT_CHAR;
                            self.edit_char = c as u32;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
```

## Visual Design

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU Text Editor                                          [—][×] │
├─────────────────────────────────────────────────────────────────┤
│  1 │ // GPU-Native Text Editor                                  │
│  2 │ // All rendering happens on the GPU                        │
│  3 │ fn main() {                                                │
│  4 │     let editor = TextEditor::new();█                       │
│  5 │     editor.run();                                          │
│  6 │ }                                                          │
│  7 │                                                            │
│    │                                                            │
├─────────────────────────────────────────────────────────────────┤
│ Ln 4, Col 35 │ UTF-8 │ 6 lines │ GPU: 0.2ms                     │
└─────────────────────────────────────────────────────────────────┘

Colors:
- Background: #1e1e2e (dark purple-gray)
- Text: #cdd6f4 (light gray)
- Line numbers: #6c7086 (muted)
- Cursor: #f5e0dc (peach, blinking)
- Selection: #45475a (subtle highlight)
- Current line: #313244 (slightly lighter bg)
```

## MVP Features

### Must Have
- [x] Document buffer in shared memory
- [x] Insert characters at cursor
- [x] Delete characters (backspace, delete)
- [x] Cursor movement (arrows, home, end)
- [x] Line wrapping display
- [x] Scrolling (mouse wheel, page up/down)
- [x] Cursor blinking
- [x] Line numbers

### Nice to Have (v2)
- [ ] Text selection (shift+arrows, click+drag)
- [ ] Copy/paste (Cmd+C, Cmd+V)
- [ ] Undo/redo
- [ ] Find (Cmd+F)
- [ ] Syntax highlighting (keywords, strings, comments)
- [ ] Multiple cursors

### Future
- [ ] File open/save
- [ ] Tab support
- [ ] Large file support (virtual scrolling)
- [ ] LSP integration

## Implementation Plan

### Step 1: Basic Document & Rendering
1. Create `TextEditor` struct implementing `GpuApp`
2. Create document buffer with test content
3. Create simple bitmap font atlas (hardcoded ASCII)
4. GPU kernel: generate quads for visible characters
5. Render text on screen (read-only)

### Step 2: Cursor & Navigation
1. Add cursor position to document
2. GPU kernel: render cursor (blinking)
3. Handle arrow key input for movement
4. Implement home/end

### Step 3: Basic Editing
1. Handle character input
2. Implement insert at cursor (parallel shift)
3. Implement backspace/delete (parallel shift)
4. Implement newline

### Step 4: Scrolling & Line Numbers
1. Compute line breaks in GPU
2. Implement scroll_line tracking
3. Render line numbers
4. Mouse wheel scrolling
5. Auto-scroll when cursor moves off-screen

### Step 5: Polish
1. Current line highlighting
2. Cursor column memory for up/down
3. Status bar (line, column, char count)

## Buffer Slot Assignment

Following GpuApp convention:
```
Slot 0: FrameState (OS)
Slot 1: InputQueue (OS)
Slot 2: EditorParams (app per-frame params)
Slot 3: Document (chars, cursor, selection)
Slot 4: LayoutCache (line starts, computed)
Slot 5: GlyphAtlas (font texture data)
Slot 6: Vertices (output geometry)
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Frame time | < 2ms |
| Keystroke latency | < 8ms (one frame at 120Hz) |
| Max document size | 64K characters |
| Visible lines | 50+ |
| GPU memory | < 1MB total |

## Success Criteria

1. **Functional:** Can type, navigate, and edit text
2. **Responsive:** No perceptible input lag
3. **Correct:** Cursor position accurate, text renders correctly
4. **Demonstrates:** Real app on GPU-Native OS framework

## Files to Create

```
src/gpu_os/text_editor.rs     # TextEditor app (implements GpuApp)
examples/text_editor.rs       # Demo application
assets/font_atlas.rs          # Embedded bitmap font data
```

## Appendix: Parallel Character Shifting

The key GPU algorithm for insert/delete:

```metal
// INSERT: Shift characters right to make room
kernel void insert_char(device uint* chars, uint cursor_pos, uint new_char, uint tid) {
    if (tid >= cursor_pos && tid < char_count) {
        // Read my current char before anyone writes
        uint my_char = chars[tid];
        threadgroup_barrier(mem_flags::mem_device);

        // Write to position + 1
        if (tid < char_count) {
            chars[tid + 1] = my_char;
        }

        // Thread at cursor writes the new char
        if (tid == cursor_pos) {
            chars[tid] = new_char;
        }
    }
}

// DELETE: Shift characters left to close gap
kernel void delete_char(device uint* chars, uint delete_pos, uint tid) {
    if (tid > delete_pos && tid < char_count) {
        uint my_char = chars[tid];
        threadgroup_barrier(mem_flags::mem_device);
        chars[tid - 1] = my_char;
    }
}
```

This achieves O(1) parallel insert/delete regardless of document size (up to 1024 threads).
