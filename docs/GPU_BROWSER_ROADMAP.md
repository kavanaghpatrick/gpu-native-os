# GPU-Native Browser Implementation Roadmap

## Key Insight from Servo Analysis

Servo already has the architecture we need! The key discovery is that Servo uses a **conditional parallelism pattern** (`use_rayon: bool`) that we can extend to GPU compute.

## Three Integration Strategies

### Strategy A: Fork Servo + GPU Kernels (Recommended)
**Effort: 4-8 weeks**

1. Fork Servo repository
2. Add `use_gpu: bool` flag alongside `use_rayon`
3. Implement Metal compute kernels for hot paths
4. Keep Stylo/html5ever for parsing (mature, spec-compliant)

**Pros:**
- Leverages battle-tested HTML/CSS parsing
- Incremental migration (GPU for hot paths, CPU fallback)
- Full CSS specification compliance

**Cons:**
- Large codebase (~500K lines)
- Complex build system
- SpiderMonkey dependency

### Strategy B: Minimal GPU Browser (Current Path)
**Effort: 8-16 weeks**

1. Keep our existing GPU document pipeline
2. Add external CSS loading (Issue #94)
3. Implement proper CSS cascade
4. Focus on common CSS subset (80/20 rule)

**Pros:**
- Full control over GPU architecture
- No external dependencies
- Optimized for our single-threadgroup model

**Cons:**
- Months of CSS edge cases
- Limited specification compliance
- Maintenance burden

### Strategy C: Servo Components as Libraries
**Effort: 6-12 weeks**

Use individual Servo components without full browser:
1. **stylo** - CSS parsing and cascade (GPU-ready data)
2. **html5ever** - HTML parsing
3. **taffy** - Flexbox/Grid algorithms (reference)

Replace layout/rendering with our GPU pipeline.

**Pros:**
- Best of both worlds
- Spec-compliant parsing
- GPU-native layout/rendering

**Cons:**
- Component integration complexity
- Version management

## Recommended Approach: Strategy C

### Phase 1: Component Integration (Week 1-2)

Add dependencies to Cargo.toml:
```toml
[dependencies]
# HTML parsing (mature, no GPU needed)
html5ever = "0.38"
markup5ever = "0.38"

# CSS parsing (produces data for GPU)
cssparser = "0.36"

# Layout algorithms reference (CPU reference, GPU reimplementation)
taffy = { version = "0.9", default-features = false }
```

### Phase 2: CSS Data Pipeline (Week 3-4)

1. Parse CSS with cssparser
2. Convert to GPU-friendly format
3. Upload to Metal buffers
4. Run GPU selector matching

### Phase 3: GPU Layout Kernels (Week 5-8)

1. Implement widths kernel (top-down)
2. Implement heights kernel (bottom-up)
3. Implement positions kernel
4. Handle percentage units
5. Handle min/max constraints

### Phase 4: Text Integration (Week 9-10)

1. Connect SDF text engine (already built!)
2. GPU text measurement
3. Line breaking algorithm
4. Text vertex generation

### Phase 5: Rendering Polish (Week 11-12)

1. Background gradients
2. Border rendering (including radius)
3. Box shadows
4. Image support (from Issue #37)

## Quick Win: Test Current Architecture on Complex Pages

Before investing in new components, benchmark current system:

```bash
# Test current document viewer on local HTML
cargo run --example document_viewer -- examples/test_pages/wikipedia.html

# Profile GPU kernel times
cargo run --example document_viewer -- --profile
```

## GPU Kernel Priority Order

Based on Servo analysis, implement in this order for maximum impact:

1. **Selector Matching** (50-100x speedup) - Biggest win
2. **Text Measurement** (10-100x speedup) - Major bottleneck
3. **Block Layout** (5-20x speedup) - Common case
4. **Cascade Resolution** (10-50x speedup) - CPU-heavy

## Integration with Existing GPU-OS

Our existing infrastructure maps perfectly:

| GPU-OS Component | Browser Use |
|------------------|-------------|
| `text_render.rs` | Glyph atlas, text vertices |
| `layout.metal` | Layout computation |
| `paint.metal` | Vertex generation |
| `document/parser.rs` | Extend for full HTML |
| `document/style.rs` | Extend for full CSS |

## Concrete Next Steps

### This Week
1. Add html5ever for HTML parsing
2. Add cssparser for CSS parsing
3. Create benchmark page (local Wikipedia snapshot)
4. Measure current performance baseline

### Next Week
1. Implement GPU selector matching kernel
2. Benchmark vs CPU matching
3. Begin cascade resolution kernel

### Week 3-4
1. Integrate with existing layout.metal
2. Add percentage unit resolution
3. Test on real web pages
