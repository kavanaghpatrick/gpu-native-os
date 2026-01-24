# PRD: GPU CSS Layout Engine Rewrite

## Executive Summary

Complete rewrite of the document layout engine to implement CSS 2.1 Visual Formatting Model correctly. The current implementation has fundamental algorithmic issues that cannot be fixed incrementally.

## Problem Statement

Current layout issues:
1. **No external CSS loading** - Wikipedia's display:none rules come from stylesheets we don't fetch
2. **No CSS selector matching** - Cannot match `.class`, `#id`, or complex selectors to elements
3. **No inline formatting context** - All elements treated as blocks, causing text stacking
4. **Incorrect height calculation** - Parents have massive heights (150,000px instead of ~5,000px)
5. **No proper margin collapsing** - Adjacent margins summed instead of collapsed
6. **Out-of-flow elements included** - position:absolute/fixed elements contribute to parent height

## Goals

1. Render Wikipedia article pages with correct layout
2. Support CSS 2.1 Visual Formatting Model (block + inline)
3. Maintain GPU-native architecture (all computation on GPU)
4. Achieve <100ms layout time for 15,000 element documents

## Non-Goals

1. Full CSS3 support (flexbox, grid) - future work
2. JavaScript execution
3. Form interactivity
4. Animations/transitions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPU (I/O Only)                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Fetch HTML                                                  │
│  2. Fetch external CSS (parallel)                               │
│  3. Upload to GPU buffers                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GPU Pipeline                                 │
├─────────────────────────────────────────────────────────────────┤
│  Pass 1: Tokenize HTML (parallel)                               │
│  Pass 2: Build DOM tree (parallel prefix sum)                   │
│  Pass 3: Parse CSS rules (parallel)                             │
│  Pass 4: Match selectors to elements (parallel)                 │
│  Pass 5: Cascade & compute styles (parallel)                    │
│  Pass 6: Build formatting contexts (parallel)                   │
│  Pass 7: Layout - widths top-down (level-parallel)              │
│  Pass 8: Layout - heights bottom-up (level-parallel)            │
│  Pass 9: Position elements (level-parallel)                     │
│  Pass 10: Paint (parallel)                                      │
└─────────────────────────────────────────────────────────────────┘
```

## CSS 2.1 Visual Formatting Model

### Block Formatting Context (BFC)

In a BFC, boxes are laid out vertically starting at the top of the containing block:

```
┌─────────────────────────────────┐
│ Containing Block                │
│ ┌─────────────────────────────┐ │
│ │ Block Box 1                 │ │
│ │ margin-bottom: 20px         │ │
│ └─────────────────────────────┘ │
│         ↕ collapsed margin      │
│ ┌─────────────────────────────┐ │
│ │ Block Box 2                 │ │
│ │ margin-top: 30px            │ │
│ └─────────────────────────────┘ │
│                                 │
│ Collapsed margin = max(20,30)   │
│                  = 30px         │
└─────────────────────────────────┘
```

### Inline Formatting Context (IFC)

In an IFC, boxes are laid out horizontally in line boxes:

```
┌─────────────────────────────────────────┐
│ Line Box 1                              │
│ ┌────┐ ┌──────────┐ ┌────┐              │
│ │word│ │  word    │ │word│              │
│ └────┘ └──────────┘ └────┘              │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ Line Box 2                              │
│ ┌──────────────┐ ┌────┐                 │
│ │ longer word  │ │wd  │                 │
│ └──────────────┘ └────┘                 │
└─────────────────────────────────────────┘
```

### Height Calculation Algorithm (CSS 2.1 Section 10.6.3)

For block elements with `height: auto`:

```
height = 0
prev_margin_bottom = 0
is_first = true

for each in-flow child:
    if child is out-of-flow (absolute/fixed):
        continue  // Does not contribute to height

    if is_first:
        // First child's top margin may collapse with parent
        height += child.margin_top (if not collapsing with parent)
        is_first = false
    else:
        // Collapse adjacent margins
        collapsed = max(prev_margin_bottom, child.margin_top)
        height += collapsed

    height += child.border_top + child.padding_top
    height += child.content_height
    height += child.padding_bottom + child.border_bottom

    prev_margin_bottom = child.margin_bottom

// Last child's bottom margin may collapse with parent
height += prev_margin_bottom (if not collapsing with parent)
```

### Margin Collapsing Rules (CSS 2.1 Section 8.3.1)

Margins collapse when:
1. Both belong to in-flow block-level boxes in the same BFC
2. No line boxes, clearance, padding, or border separate them
3. Both belong to vertically-adjacent box edges

Margins do NOT collapse when:
1. Boxes are not in the same BFC
2. Either box establishes a new BFC (overflow != visible, float, etc.)
3. Either box has clearance
4. Parent has padding or border separating it from child

## Data Structures

### CSSRule (GPU buffer)
```rust
#[repr(C)]
struct CSSRule {
    selector_start: u32,      // Offset into selector buffer
    selector_length: u16,     // Length of selector string
    specificity: u16,         // Packed: (id << 10) | (class << 5) | element
    property_start: u32,      // Offset into property buffer
    property_count: u16,      // Number of properties
    source_order: u16,        // For cascade tie-breaking
}
```

### Selector (GPU buffer)
```rust
#[repr(C)]
struct SelectorPart {
    selector_type: u8,        // ELEMENT, CLASS, ID, ATTRIBUTE, PSEUDO
    combinator: u8,           // NONE, DESCENDANT, CHILD, SIBLING, ADJACENT
    name_hash: u32,           // Hash of element/class/id name
    name_start: u32,          // Offset for full name comparison
    name_length: u16,
    _padding: u16,
}
```

### FormattingContext (GPU buffer)
```rust
#[repr(C)]
struct FormattingContext {
    context_type: u32,        // BFC, IFC
    root_element: u32,        // Element that establishes this context
    first_child_context: i32, // Nested contexts
    next_sibling_context: i32,
}
```

### LineBox (GPU buffer for IFC)
```rust
#[repr(C)]
struct LineBox {
    y: f32,                   // Y position in containing block
    height: f32,              // Line height (max of inline heights)
    baseline: f32,            // Baseline position
    width: f32,               // Actual width used
    first_item: u32,          // First inline item index
    item_count: u32,          // Number of items on this line
    _padding: [f32; 2],
}
```

## Implementation Phases

### Phase 1: External CSS Loading (CPU)
- Parse `<link rel="stylesheet">` and `<style>` tags
- Fetch external stylesheets in parallel
- Concatenate into single CSS buffer for GPU

### Phase 2: CSS Parsing (GPU)
- Tokenize CSS (parallel per-character)
- Build rule list with selectors and declarations
- Compute specificity for each selector

### Phase 3: Selector Matching (GPU)
- For each element, test against all selectors (parallel)
- Use hash-based filtering for fast rejection
- Build matched-rules list per element

### Phase 4: Style Cascade (GPU)
- Sort matched rules by specificity + source order
- Apply properties in cascade order
- Handle inheritance for unset properties

### Phase 5: Formatting Context Construction (GPU)
- Identify BFC-establishing elements
- Identify IFC regions (sequences of inline content)
- Build context tree

### Phase 6: Block Layout (GPU, level-parallel)
- Pass A: Widths top-down (containing block → children)
- Pass B: Heights bottom-up (leaves → root)
- Pass C: Positions top-down

### Phase 7: Inline Layout (GPU)
- Break text into words
- Measure word widths
- Pack words into line boxes
- Position inline boxes on lines

## Success Metrics

1. **Correctness**: Wikipedia renders with proper spacing (no overlapping text)
2. **Performance**: <100ms layout for 15,000 elements
3. **Coverage**: 90% of Wikipedia article pages render correctly

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CSS parsing complexity | Start with subset (class, id, element selectors only) |
| Selector matching performance | Hash-based filtering, limit selector depth |
| Inline layout complexity | Implement word-based breaking first, then character-level |
| Memory usage | Stream large documents in chunks |

## Testing Strategy

1. **Unit tests**: Each kernel tested in isolation with known inputs
2. **Reference tests**: Compare layout output against browser (headless Chrome)
3. **Regression tests**: Wikipedia snapshots to detect regressions
4. **Performance tests**: Benchmark layout time on large documents

## Timeline

- Phase 1-2 (CSS Loading + Parsing): Issue #94, #95
- Phase 3-4 (Selector Matching + Cascade): Issue #96, #97
- Phase 5 (Formatting Contexts): Issue #98
- Phase 6 (Block Layout Rewrite): Issue #99
- Phase 7 (Inline Layout): Issue #100
- Integration & Testing: Issue #101
