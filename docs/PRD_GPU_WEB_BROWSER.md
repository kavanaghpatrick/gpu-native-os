# PRD: GPU-Native Web Browser Engine

## Executive Summary

Build a production-quality GPU-native web browser rendering engine capable of loading and displaying real-world webpages with visual fidelity matching Chrome. All parsing, styling, layout, and rendering operations execute on the GPU via Metal compute shaders.

**Target:** Render any standard HTML5/CSS3 webpage identically to Chrome's rendering.

---

## 1. Vision & Goals

### 1.1 Primary Goal
A GPU-accelerated browser engine that:
- Loads URLs and renders HTML/CSS with Chrome-equivalent output
- Executes 100% of rendering pipeline on GPU (no CPU layout/paint)
- Achieves 60fps scrolling on documents with 10,000+ elements
- Supports standard web content (news sites, documentation, blogs)

### 1.2 Success Criteria
| Metric | Target |
|--------|--------|
| Visual accuracy vs Chrome | 95%+ pixel similarity |
| First contentful paint | <100ms for 100KB HTML |
| Scroll performance | 60fps, <16ms frame time |
| Memory efficiency | <200MB for typical webpage |
| Element capacity | 100,000+ elements |

### 1.3 Non-Goals (V1)
- JavaScript execution
- Web APIs (localStorage, fetch, WebSocket)
- Audio/Video playback
- WebGL/Canvas 2D
- Print layout
- Accessibility (screen readers)

---

## 2. Current State Analysis

### 2.1 Implemented Features
| Component | Status | Quality |
|-----------|--------|---------|
| HTML Tokenizer | Complete | Production |
| HTML Parser | Complete | Production |
| CSS Selector Matching | Partial | Basic selectors only |
| Style Resolution | Partial | No inheritance |
| Flexbox Layout | Complete | Production |
| Block Layout | Complete | Production |
| Text Rendering | Partial | Single font, no wrap |
| Background/Border | Complete | Production |
| Vertical Scrolling | Complete | Production |
| Image Loading | Complete | Not integrated |

### 2.2 Critical Gaps
1. **Text wrapping** - Text overflows containers
2. **CSS inheritance** - Properties don't cascade
3. **Hit testing** - No mouse interaction
4. **Link navigation** - Links not clickable
5. **Image rendering** - Images not displayed
6. **Inline styles** - `style=""` attribute ignored
7. **Font system** - Single hardcoded font
8. **Advanced selectors** - No descendant, child, attribute selectors

---

## 3. Architecture Overview

### 3.1 Pipeline Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU RENDERING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │  FETCH   │──▶│  PARSE   │──▶│  STYLE   │──▶│  LAYOUT  │         │
│  │  (CPU)   │   │  (GPU)   │   │  (GPU)   │   │  (GPU)   │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│       │              │              │              │                 │
│       ▼              ▼              ▼              ▼                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │  HTML    │   │  DOM     │   │ Computed │   │  Layout  │         │
│  │  Bytes   │   │  Tree    │   │  Styles  │   │  Boxes   │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│                                                      │               │
│                                                      ▼               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │ DISPLAY  │◀──│ RASTERIZE│◀──│  PAINT   │◀──│  TEXT    │         │
│  │          │   │  (GPU)   │   │  (GPU)   │   │  (GPU)   │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow
```
1. FETCH:    URL → HTTP Request → HTML bytes (CPU, async)
2. TOKENIZE: HTML bytes → Token stream (GPU, parallel)
3. PARSE:    Tokens → DOM elements + text buffer (GPU, parallel)
4. STYLE:    Elements × CSS → Computed styles (GPU, parallel)
5. LAYOUT:   Elements + Styles → Layout boxes (GPU, multi-pass)
6. TEXT:     Text + Styles → Glyph positions (GPU, parallel)
7. PAINT:    Layout + Text → Vertices (GPU, parallel)
8. RASTER:   Vertices + Textures → Pixels (GPU, fragment shader)
```

### 3.3 Memory Architecture
```
GPU Buffers (Shared Memory):
├── html_buffer:        4 MB   (raw HTML bytes)
├── token_buffer:       4 MB   (256K tokens × 16 bytes)
├── element_buffer:    12 MB   (256K elements × 48 bytes)
├── style_buffer:      32 MB   (256K styles × 128 bytes)
├── layout_buffer:     16 MB   (256K boxes × 64 bytes)
├── text_buffer:        8 MB   (text content)
├── glyph_buffer:      16 MB   (400K glyphs × 40 bytes)
├── vertex_buffer:     64 MB   (1M vertices × 64 bytes)
├── image_atlas:       64 MB   (4096×4096 RGBA)
├── font_atlas:        16 MB   (4096×4096 R8)
└── TOTAL:            ~236 MB
```

---

## 4. Feature Specifications

### 4.1 HTML Support

#### 4.1.1 Document Structure
```
DOCTYPE, html, head, body, title, meta, link, style, script
```

#### 4.1.2 Semantic Elements
```
header, footer, nav, main, section, article, aside, figure, figcaption
```

#### 4.1.3 Text Elements
```
h1-h6, p, span, div, a, strong, em, b, i, u, s, small, mark, code, pre,
blockquote, q, cite, abbr, time, sub, sup, br, hr, wbr
```

#### 4.1.4 List Elements
```
ul, ol, li, dl, dt, dd
```

#### 4.1.5 Table Elements
```
table, thead, tbody, tfoot, tr, th, td, caption, colgroup, col
```

#### 4.1.6 Form Elements (Display Only V1)
```
form, input, button, textarea, select, option, optgroup, label, fieldset, legend
```

#### 4.1.7 Media Elements
```
img, picture, source, video (poster only), audio (hidden)
```

#### 4.1.8 Interactive Elements
```
a (with href), details, summary
```

### 4.2 CSS Support

#### 4.2.1 Selectors (Priority Order)

**Phase 1 - Basic:**
```css
*           /* Universal */
E           /* Type */
.class      /* Class */
#id         /* ID */
E, F        /* Grouping */
```

**Phase 2 - Combinators:**
```css
E F         /* Descendant */
E > F       /* Child */
E + F       /* Adjacent sibling */
E ~ F       /* General sibling */
```

**Phase 3 - Attribute:**
```css
[attr]           /* Has attribute */
[attr=val]       /* Exact match */
[attr~=val]      /* Word match */
[attr^=val]      /* Starts with */
[attr$=val]      /* Ends with */
[attr*=val]      /* Contains */
```

**Phase 4 - Pseudo-classes:**
```css
:hover, :active, :focus, :visited, :link
:first-child, :last-child, :nth-child(n)
:first-of-type, :last-of-type, :nth-of-type(n)
:not(selector), :empty, :root
```

**Phase 5 - Pseudo-elements:**
```css
::before, ::after, ::first-line, ::first-letter, ::selection
```

#### 4.2.2 Properties

**Box Model:**
```css
width, height, min-width, max-width, min-height, max-height
margin (all sides), padding (all sides)
box-sizing: content-box | border-box
```

**Display & Position:**
```css
display: none | block | inline | inline-block | flex | inline-flex | grid | table | table-row | table-cell
position: static | relative | absolute | fixed | sticky
top, right, bottom, left
z-index
float: none | left | right
clear: none | left | right | both
overflow: visible | hidden | scroll | auto
overflow-x, overflow-y
visibility: visible | hidden | collapse
```

**Flexbox:**
```css
flex-direction: row | row-reverse | column | column-reverse
flex-wrap: nowrap | wrap | wrap-reverse
justify-content: flex-start | flex-end | center | space-between | space-around | space-evenly
align-items: flex-start | flex-end | center | baseline | stretch
align-content: flex-start | flex-end | center | space-between | space-around | stretch
flex-grow, flex-shrink, flex-basis
order
align-self
gap, row-gap, column-gap
```

**Grid (Phase 2):**
```css
grid-template-columns, grid-template-rows
grid-column, grid-row
grid-gap
```

**Typography:**
```css
font-family (system fonts + web fonts)
font-size (px, em, rem, %, vw, vh)
font-weight: 100-900 | normal | bold
font-style: normal | italic | oblique
font-variant: normal | small-caps
line-height (number, px, em, %)
text-align: left | right | center | justify
text-decoration: none | underline | overline | line-through
text-transform: none | uppercase | lowercase | capitalize
text-indent
letter-spacing
word-spacing
white-space: normal | nowrap | pre | pre-wrap | pre-line
word-break: normal | break-all | keep-all
overflow-wrap: normal | break-word
vertical-align
```

**Colors & Backgrounds:**
```css
color (hex, rgb, rgba, hsl, hsla, named)
background-color
background-image: url() | linear-gradient() | radial-gradient()
background-position
background-size: auto | cover | contain | length
background-repeat: repeat | no-repeat | repeat-x | repeat-y
background-attachment: scroll | fixed
opacity
```

**Borders:**
```css
border (shorthand)
border-width, border-style, border-color (all sides)
border-radius (all corners)
outline, outline-width, outline-style, outline-color, outline-offset
```

**Shadows:**
```css
box-shadow
text-shadow
```

**Transforms (Phase 2):**
```css
transform: translate | rotate | scale | skew | matrix
transform-origin
```

**Transitions (Phase 2):**
```css
transition-property
transition-duration
transition-timing-function
transition-delay
```

**Lists:**
```css
list-style-type: disc | circle | square | decimal | none
list-style-position: inside | outside
list-style-image
```

**Tables:**
```css
border-collapse: collapse | separate
border-spacing
table-layout: auto | fixed
caption-side: top | bottom
empty-cells: show | hide
```

**Images:**
```css
object-fit: fill | contain | cover | none | scale-down
object-position
```

**Miscellaneous:**
```css
cursor: auto | default | pointer | text | move | etc.
user-select: auto | none | text | all
pointer-events: auto | none
```

#### 4.2.3 Values & Units

**Length:**
```
px, em, rem, %, vw, vh, vmin, vmax, ch, ex
```

**Color:**
```
#RGB, #RRGGBB, #RGBA, #RRGGBBAA
rgb(r,g,b), rgba(r,g,b,a)
hsl(h,s,l), hsla(h,s,l,a)
Named colors (140 CSS color names)
transparent, currentColor
```

**Calc:**
```css
calc(100% - 20px)
```

### 4.3 Text Rendering

#### 4.3.1 Font System
```
Font Loading:
├── System fonts (San Francisco, Helvetica, Arial, etc.)
├── Generic families (serif, sans-serif, monospace, cursive, fantasy)
├── Web fonts (@font-face with woff2/woff/ttf)
└── Font fallback chain

Font Atlas:
├── SDF (Signed Distance Field) rendering
├── Dynamic glyph caching
├── Unicode support (BMP + common SMP)
├── Font metrics (ascender, descender, line-gap, x-height)
└── Kerning tables
```

#### 4.3.2 Text Layout
```
Text Shaping:
├── Bidirectional text (LTR/RTL)
├── Word breaking (Unicode UAX #14)
├── Line breaking algorithm
├── Hyphenation (optional)
├── Tab stops
└── White space handling

Text Measurement:
├── Glyph advance widths
├── Kerning pairs
├── Line height calculation
├── Baseline alignment
└── Vertical metrics
```

#### 4.3.3 Text Rendering Pipeline
```
GPU Text Pipeline:
1. Text Segmentation: Split by font/style/direction
2. Font Resolution: Match font-family to available fonts
3. Glyph Mapping: Unicode → Glyph ID
4. Glyph Positioning: Apply kerning, advance
5. Line Breaking: Fit glyphs to container width
6. Vertex Generation: Create glyph quads with UV coords
7. Fragment Shading: Sample SDF atlas, apply AA
```

### 4.4 Image Rendering

#### 4.4.1 Image Loading
```
Formats: PNG, JPEG, GIF (static), WebP, SVG (rasterized)
Loading: Async fetch → GPU decode (MTKTextureLoader)
Caching: LRU cache with configurable size
Placeholder: Gray box during load, alt text on error
```

#### 4.4.2 Image Layout
```
Intrinsic sizing: Use natural dimensions
Explicit sizing: width/height attributes or CSS
Aspect ratio: Preserve unless both dimensions specified
Object-fit: fill, contain, cover, none, scale-down
Object-position: Alignment within container
```

#### 4.4.3 Image Atlas
```
Atlas Management:
├── Dynamic packing (shelf algorithm)
├── Multiple atlases if needed
├── Texture eviction (LRU)
├── Mipmap generation
└── Format: RGBA8 or BGRA8
```

### 4.5 Interaction System

#### 4.5.1 Hit Testing
```metal
// GPU Hit Test Kernel
kernel void hit_test(
    device const LayoutBox* boxes,
    device const Element* elements,
    constant float2& mouse_pos,
    device atomic_uint* hit_element,
    device atomic_float* hit_z_index
) {
    // For each element, check if mouse_pos is inside box
    // Track highest z-index element that contains point
    // Return element ID via atomic compare-exchange
}
```

#### 4.5.2 Event Handling
```
Mouse Events:
├── click, dblclick
├── mousedown, mouseup
├── mousemove, mouseenter, mouseleave
├── mouseover, mouseout
└── contextmenu

Keyboard Events:
├── keydown, keyup, keypress
├── Tab navigation
└── Arrow key scrolling

Scroll Events:
├── wheel (vertical/horizontal)
├── Touch scroll (trackpad)
└── Scroll position tracking
```

#### 4.5.3 Link Navigation
```
Link Detection:
├── Parse href attribute
├── Resolve relative URLs
├── Handle anchor fragments (#)
└── Target attribute (_blank, _self)

Navigation:
├── History stack (back/forward)
├── URL bar update
├── Page transition
└── Scroll restoration
```

### 4.6 Layout Engine

#### 4.6.1 Box Model
```
┌─────────────────────────────────────┐
│              margin                 │
│  ┌───────────────────────────────┐  │
│  │          border               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │       padding           │  │  │
│  │  │  ┌───────────────────┐  │  │  │
│  │  │  │    content        │  │  │  │
│  │  │  │                   │  │  │  │
│  │  │  └───────────────────┘  │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

#### 4.6.2 Layout Modes

**Block Layout:**
```
- Vertical stacking
- Full width by default
- Margin collapsing
- Clear floats
```

**Inline Layout:**
```
- Horizontal flow
- Line boxes
- Vertical alignment
- Word wrapping
```

**Flexbox Layout:**
```
- Main axis / cross axis
- Flex item sizing
- Alignment & distribution
- Wrapping
```

**Positioned Layout:**
```
- Containing block resolution
- Offset calculation
- Z-index stacking
- Fixed viewport attachment
```

**Table Layout:**
```
- Column width distribution
- Row height calculation
- Border collapsing
- Caption positioning
```

#### 4.6.3 Layout Algorithm (GPU Multi-Pass)

```
Pass 1: Style Resolution
├── Selector matching
├── Cascade & specificity
├── Value computation
└── Inheritance propagation

Pass 2: Box Generation
├── Display type determination
├── Anonymous box creation
├── Containing block assignment
└── Formatting context establishment

Pass 3: Intrinsic Sizing
├── Min-content width
├── Max-content width
├── Intrinsic height
└── Aspect ratio handling

Pass 4: Width Resolution
├── Containing block width
├── Percentage resolution
├── min/max constraints
└── Flex basis calculation

Pass 5: Height Resolution
├── Content height
├── Percentage resolution
├── min/max constraints
└── Aspect ratio adjustment

Pass 6: Position Resolution
├── Block positioning
├── Inline positioning
├── Flex item positioning
└── Absolute positioning

Pass 7: Overflow Handling
├── Scroll dimensions
├── Clipping regions
├── Overflow indicators
└── Sticky positioning
```

---

## 5. GPU Shader Specifications

### 5.1 Compute Shaders

#### 5.1.1 Tokenizer (tokenizer.metal)
```metal
// Pass 1A: Boundary Detection
kernel void detect_boundaries(
    device const uchar* html,
    device atomic_uint* boundaries,
    constant uint& html_length
);

// Pass 1B: Token Extraction
kernel void extract_tokens(
    device const uchar* html,
    device const uint* boundaries,
    device Token* tokens,
    device atomic_uint* token_count
);
```

#### 5.1.2 Parser (parser.metal)
```metal
// Pass 2A: Element Allocation
kernel void allocate_elements(
    device const Token* tokens,
    device atomic_uint* element_count
);

// Pass 2B: Tree Construction
kernel void build_tree(
    device const Token* tokens,
    device const uchar* html,
    device Element* elements
);

// Pass 2C: Attribute Extraction
kernel void extract_attributes(
    device const Token* tokens,
    device const uchar* html,
    device Element* elements,
    device Attribute* attributes
);
```

#### 5.1.3 Styler (style.metal)
```metal
// Pass 3A: Selector Matching
kernel void match_selectors(
    device const Element* elements,
    device const Selector* selectors,
    device MatchResult* matches
);

// Pass 3B: Cascade Resolution
kernel void resolve_cascade(
    device const MatchResult* matches,
    device ComputedStyle* styles
);

// Pass 3C: Inheritance Propagation
kernel void propagate_inheritance(
    device const Element* elements,
    device ComputedStyle* styles
);

// Pass 3D: Value Computation
kernel void compute_values(
    device ComputedStyle* styles,
    constant Viewport& viewport
);
```

#### 5.1.4 Layout (layout.metal)
```metal
// Pass 4A: Intrinsic Sizing
kernel void compute_intrinsic_sizes(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes
);

// Pass 4B: Width Resolution
kernel void resolve_widths(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes
);

// Pass 4C: Height Resolution
kernel void resolve_heights(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes
);

// Pass 4D: Position Resolution
kernel void resolve_positions(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes
);

// Pass 4E: Flex Layout
kernel void layout_flex_containers(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes
);
```

#### 5.1.5 Text Layout (text.metal)
```metal
// Pass 5A: Text Segmentation
kernel void segment_text(
    device const uchar* text_buffer,
    device const Element* elements,
    device TextSegment* segments
);

// Pass 5B: Glyph Mapping
kernel void map_glyphs(
    device const TextSegment* segments,
    device const FontMetrics* fonts,
    device Glyph* glyphs
);

// Pass 5C: Line Breaking
kernel void break_lines(
    device const Glyph* glyphs,
    device const LayoutBox* boxes,
    device LineBox* lines
);

// Pass 5D: Glyph Positioning
kernel void position_glyphs(
    device const LineBox* lines,
    device Glyph* glyphs
);
```

#### 5.1.6 Paint (paint.metal)
```metal
// Pass 6A: Vertex Counting
kernel void count_vertices(
    device const Element* elements,
    device const ComputedStyle* styles,
    device uint* vertex_counts
);

// Pass 6B: Background Generation
kernel void generate_backgrounds(
    device const LayoutBox* boxes,
    device const ComputedStyle* styles,
    device PaintVertex* vertices
);

// Pass 6C: Border Generation
kernel void generate_borders(
    device const LayoutBox* boxes,
    device const ComputedStyle* styles,
    device PaintVertex* vertices
);

// Pass 6D: Text Generation
kernel void generate_text_vertices(
    device const Glyph* glyphs,
    device const ComputedStyle* styles,
    device PaintVertex* vertices
);

// Pass 6E: Image Generation
kernel void generate_image_vertices(
    device const LayoutBox* boxes,
    device const ImageInfo* images,
    device PaintVertex* vertices
);

// Pass 6F: Shadow Generation
kernel void generate_shadows(
    device const LayoutBox* boxes,
    device const ComputedStyle* styles,
    device PaintVertex* vertices
);
```

### 5.2 Render Shaders

#### 5.2.1 Vertex Shader
```metal
vertex VertexOut document_vertex(
    device const PaintVertex* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.tex_coord = vertices[vid].tex_coord;
    out.color = vertices[vid].color;
    out.flags = vertices[vid].flags;
    return out;
}
```

#### 5.2.2 Fragment Shader
```metal
fragment float4 document_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> font_atlas [[texture(0)]],
    texture2d<float> image_atlas [[texture(1)]],
    constant Uniforms& uniforms [[buffer(0)]]
) {
    if (in.flags & FLAG_TEXT) {
        // SDF text rendering with anti-aliasing
        float d = font_atlas.sample(samp, in.tex_coord).r;
        float aa = fwidth(d) * 0.75;
        float alpha = smoothstep(0.5 - aa, 0.5 + aa, d);
        return float4(in.color.rgb, in.color.a * alpha);
    }
    else if (in.flags & FLAG_IMAGE) {
        // Image texture sampling
        return image_atlas.sample(samp, in.tex_coord) * in.color;
    }
    else if (in.flags & FLAG_GRADIENT) {
        // Gradient interpolation
        return sample_gradient(in.tex_coord, in.gradient_id);
    }
    else {
        // Solid color
        return in.color;
    }
}
```

---

## 6. Data Structures

### 6.1 Core Structures

```rust
// Token from HTML tokenizer
#[repr(C)]
pub struct Token {
    pub token_type: u32,    // TAG_OPEN, TAG_CLOSE, TEXT, etc.
    pub start: u32,         // Byte offset in HTML
    pub end: u32,           // End byte offset
    pub flags: u32,         // Self-closing, has attributes, etc.
}

// DOM Element
#[repr(C)]
pub struct Element {
    pub element_type: u32,      // DIV, P, A, IMG, etc.
    pub parent: i32,            // Parent element index (-1 for root)
    pub first_child: i32,       // First child index (-1 if none)
    pub next_sibling: i32,      // Next sibling index (-1 if none)
    pub prev_sibling: i32,      // Previous sibling (-1 if none)
    pub last_child: i32,        // Last child for append
    pub text_start: u32,        // Offset in text buffer
    pub text_length: u32,       // Length of text content
    pub attr_start: u32,        // Offset in attribute buffer
    pub attr_count: u32,        // Number of attributes
    pub token_index: u32,       // Original token reference
    pub flags: u32,             // Void element, has ID, has class, etc.
}

// Computed Style (all CSS properties)
#[repr(C)]
pub struct ComputedStyle {
    // Display & Position (16 bytes)
    pub display: u32,
    pub position: u32,
    pub float_: u32,
    pub clear: u32,

    // Box Model (64 bytes)
    pub width: f32,
    pub height: f32,
    pub min_width: f32,
    pub max_width: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub margin: [f32; 4],       // top, right, bottom, left
    pub padding: [f32; 4],
    pub border_width: [f32; 4],

    // Position Offsets (16 bytes)
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,

    // Flexbox (32 bytes)
    pub flex_direction: u32,
    pub flex_wrap: u32,
    pub justify_content: u32,
    pub align_items: u32,
    pub align_content: u32,
    pub align_self: u32,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: f32,
    pub order: i32,
    pub gap: [f32; 2],

    // Typography (32 bytes)
    pub font_size: f32,
    pub font_weight: u32,
    pub font_style: u32,
    pub line_height: f32,
    pub letter_spacing: f32,
    pub word_spacing: f32,
    pub text_align: u32,
    pub text_decoration: u32,
    pub text_transform: u32,
    pub white_space: u32,
    pub vertical_align: u32,

    // Colors (32 bytes)
    pub color: [f32; 4],            // RGBA
    pub background_color: [f32; 4],
    pub border_color: [f32; 4],
    pub outline_color: [f32; 4],

    // Visual (32 bytes)
    pub opacity: f32,
    pub visibility: u32,
    pub overflow_x: u32,
    pub overflow_y: u32,
    pub z_index: i32,
    pub border_radius: [f32; 4],    // corners
    pub box_shadow: u32,            // index into shadow buffer
    pub text_shadow: u32,

    // Background (16 bytes)
    pub background_image: u32,      // index into gradient/image buffer
    pub background_position: [f32; 2],
    pub background_size: u32,
    pub background_repeat: u32,

    // List & Table (8 bytes)
    pub list_style_type: u32,
    pub border_collapse: u32,

    // Misc (8 bytes)
    pub cursor: u32,
    pub pointer_events: u32,

    // Total: 256 bytes (power of 2 for GPU alignment)
    pub _padding: [f32; 6],
}

// Layout Box
#[repr(C)]
pub struct LayoutBox {
    // Border box (position relative to parent)
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,

    // Content box (absolute position)
    pub content_x: f32,
    pub content_y: f32,
    pub content_width: f32,
    pub content_height: f32,

    // Padding box
    pub padding_x: f32,
    pub padding_y: f32,
    pub padding_width: f32,
    pub padding_height: f32,

    // Scroll dimensions
    pub scroll_width: f32,
    pub scroll_height: f32,
    pub scroll_x: f32,
    pub scroll_y: f32,

    // Clipping
    pub clip_x: f32,
    pub clip_y: f32,
    pub clip_width: f32,
    pub clip_height: f32,

    // Stacking
    pub z_order: i32,
    pub stacking_context: u32,

    // Flags
    pub flags: u32,             // is_clipping, is_scrollable, etc.
    pub _padding: f32,

    // Total: 96 bytes
}

// Text Glyph
#[repr(C)]
pub struct Glyph {
    pub codepoint: u32,         // Unicode code point
    pub font_id: u32,           // Font index
    pub glyph_id: u32,          // Glyph ID in font
    pub element_id: u32,        // Owning element
    pub x: f32,                 // Position
    pub y: f32,
    pub advance: f32,           // Horizontal advance
    pub atlas_x: f32,           // UV in atlas
    pub atlas_y: f32,
    pub atlas_w: f32,
    pub atlas_h: f32,
    pub flags: u32,             // whitespace, line_start, etc.
    // Total: 48 bytes
}

// Paint Vertex
#[repr(C)]
pub struct PaintVertex {
    pub position: [f32; 2],     // NDC coordinates
    pub tex_coord: [f32; 2],    // UV for texture sampling
    pub color: [f32; 4],        // RGBA
    pub flags: u32,             // BACKGROUND, BORDER, TEXT, IMAGE, etc.
    pub extra: [u32; 3],        // gradient_id, texture_id, etc.
    // Total: 48 bytes
}
```

### 6.2 Attribute System

```rust
#[repr(C)]
pub struct Attribute {
    pub name_start: u32,        // Offset in HTML buffer
    pub name_length: u32,
    pub value_start: u32,
    pub value_length: u32,
}

// Common attribute indices (pre-parsed for fast access)
#[repr(C)]
pub struct ElementAttributes {
    pub id: u32,                // Index into string table (0 = none)
    pub class_list: u32,        // Index into class list buffer
    pub href: u32,              // For links
    pub src: u32,               // For images
    pub alt: u32,               // For images
    pub style: u32,             // Inline style string
    pub width: f32,             // For images/tables
    pub height: f32,
}
```

---

## 7. Implementation Phases

### Phase 1: Text Rendering Improvements (Week 1-2)

**Goal:** Proper text wrapping and font support

**Tasks:**
1. Implement text measurement kernel
2. Add line breaking algorithm (UAX #14 simplified)
3. Implement text wrapping in layout pass
4. Add font-size scaling (currently hardcoded)
5. Support font-weight (bold vs regular)
6. Add text-decoration (underline)

**Deliverable:** Text wraps correctly within containers

### Phase 2: CSS Inheritance (Week 2-3)

**Goal:** Properties cascade from parent to child

**Tasks:**
1. Identify inheritable properties (color, font-*, line-height, etc.)
2. Implement inheritance propagation kernel
3. Add computed value resolution
4. Support `inherit` keyword
5. Add `initial` and `unset` keywords

**Deliverable:** Font colors and sizes inherit correctly

### Phase 3: Advanced Selectors (Week 3-4)

**Goal:** Support descendant, child, and attribute selectors

**Tasks:**
1. Implement descendant selector matching
2. Implement child selector (>)
3. Implement adjacent sibling (+)
4. Add attribute selectors ([attr], [attr=val])
5. Implement :first-child, :last-child

**Deliverable:** Real-world CSS selectors work

### Phase 4: Hit Testing & Links (Week 4-5)

**Goal:** Clickable links with navigation

**Tasks:**
1. Implement GPU hit testing kernel
2. Add mouse position tracking
3. Detect clicks on link elements
4. Parse and resolve href URLs
5. Implement navigation (fetch new page)
6. Add cursor feedback (pointer on links)

**Deliverable:** Links are clickable and navigate

### Phase 5: Image Integration (Week 5-6)

**Goal:** Display images in documents

**Tasks:**
1. Connect image loader to parser
2. Extract img src attributes
3. Implement async image loading
4. Add images to paint pipeline
5. Support object-fit CSS
6. Add loading placeholders

**Deliverable:** Images display correctly

### Phase 6: Positioned Layout (Week 6-7)

**Goal:** Support position: relative/absolute/fixed

**Tasks:**
1. Implement containing block resolution
2. Add offset calculation (top/right/bottom/left)
3. Implement z-index stacking
4. Add position: fixed (viewport-relative)
5. Implement position: sticky

**Deliverable:** Positioned elements work

### Phase 7: Overflow & Scrolling (Week 7-8)

**Goal:** Proper overflow handling with nested scrolling

**Tasks:**
1. Implement overflow: hidden clipping
2. Add overflow: scroll containers
3. Implement horizontal scrolling
4. Add nested scroll containers
5. Implement scroll-to-anchor (#fragments)

**Deliverable:** Overflow containers scroll independently

### Phase 8: Visual Effects (Week 8-9)

**Goal:** Shadows, gradients, and opacity

**Tasks:**
1. Implement box-shadow rendering
2. Add text-shadow
3. Implement linear-gradient backgrounds
4. Add radial-gradient
5. Implement opacity compositing

**Deliverable:** Visual effects render correctly

### Phase 9: Tables (Week 9-10)

**Goal:** Table layout support

**Tasks:**
1. Implement table layout algorithm
2. Add column width distribution
3. Implement border-collapse
4. Add caption positioning
5. Support nested tables

**Deliverable:** Tables render correctly

### Phase 10: Performance & Polish (Week 10-12)

**Goal:** Production-quality performance

**Tasks:**
1. Implement viewport culling
2. Add incremental layout updates
3. Optimize text rendering
4. Add resource caching
5. Profile and optimize bottlenecks
6. Fix visual bugs vs Chrome

**Deliverable:** 60fps on complex pages

---

## 8. Testing Strategy

### 8.1 Unit Tests
```
- Tokenizer: HTML snippets → expected tokens
- Parser: Tokens → expected DOM structure
- Styler: Selectors → expected matches
- Layout: Elements → expected box dimensions
- Paint: Layout → expected vertices
```

### 8.2 Visual Regression Tests
```
- Render reference HTML files
- Compare screenshots vs Chrome
- Track pixel differences
- Automated CI testing
```

### 8.3 Performance Benchmarks
```
- Tokenization: tokens/second
- Full pipeline: elements/second
- Render time: ms per frame
- Memory usage: bytes per element
```

### 8.4 Test Pages
```
- css-tricks.com (complex layouts)
- news.ycombinator.com (simple, text-heavy)
- wikipedia.org (tables, images)
- github.com (modern CSS)
- docs.rs (documentation)
```

---

## 9. Success Metrics

### 9.1 Visual Accuracy
| Page Type | Target Similarity |
|-----------|-------------------|
| Text-only pages | 99% |
| Simple layouts | 97% |
| Complex CSS | 95% |
| Tables | 95% |
| With images | 95% |

### 9.2 Performance
| Metric | Target |
|--------|--------|
| 10K element page load | <200ms |
| Scroll frame time | <16ms |
| Memory per element | <500 bytes |
| GPU utilization | >80% |

### 9.3 Compatibility
| Standard | Support Level |
|----------|---------------|
| HTML5 | 95% |
| CSS3 | 85% |
| CSS Selectors L3 | 90% |
| Flexbox | 100% |
| Grid | 50% |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Text wrapping complexity | High | Start with simple break algorithm |
| CSS inheritance edge cases | Medium | Follow CSS spec exactly |
| Hit testing performance | Medium | Spatial partitioning (quad-tree) |
| Memory limits on large pages | High | Streaming/paging architecture |
| Font rendering quality | High | Use SDF with proper metrics |
| Table layout complexity | Medium | Defer to Phase 9 |

---

## 11. Dependencies

### 11.1 External Crates
```toml
metal = "0.33"          # Metal bindings
winit = "0.30"          # Windowing
cocoa = "0.26"          # macOS integration
objc = "0.2"            # Objective-C FFI
ureq = "2.9"            # HTTP client
ttf-parser = "0.25"     # Font parsing
```

### 11.2 System Requirements
```
- macOS 12.0+ (Metal 3)
- Apple Silicon or Intel with Metal
- 8GB RAM minimum
- GPU with 4GB VRAM recommended
```

---

## 12. File Structure

```
src/gpu_os/document/
├── mod.rs              # Module exports
├── tokenizer.rs        # HTML tokenizer
├── tokenizer.metal     # GPU tokenization shaders
├── parser.rs           # DOM parser
├── parser.metal        # GPU parsing shaders
├── style.rs            # CSS style resolution
├── style.metal         # GPU styling shaders
├── selector.rs         # CSS selector system
├── layout.rs           # Layout engine
├── layout.metal        # GPU layout shaders
├── text.rs             # Text shaping & layout
├── text.metal          # GPU text shaders
├── paint.rs            # Vertex generation
├── paint.metal         # GPU paint shaders
├── image.rs            # Image loading & atlas
├── font.rs             # Font loading & atlas
├── hit_test.rs         # Interaction system
└── render.rs           # Final rendering

examples/
├── document_viewer.rs  # Main browser application
├── document_images.rs  # Image rendering demo
└── document_test.rs    # Visual regression tests
```

---

## Appendix A: CSS Property Reference

[Full list of supported CSS properties with syntax and default values - see implementation phases for priority]

## Appendix B: HTML Element Reference

[Full list of supported HTML elements with rendering behavior - see Section 4.1]

## Appendix C: GPU Memory Layout

[Detailed buffer layouts for GPU data structures - see Section 3.3]
