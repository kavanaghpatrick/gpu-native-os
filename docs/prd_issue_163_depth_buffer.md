# PRD: Issue #163 - Fix Depth Buffer Configuration

## Problem
The visual_megakernel demo shows a black screen despite generating 126+ vertices. The render pipeline lacks depth buffer configuration, causing incorrect z-ordering and potential overdraw issues.

## Root Cause
The render pipeline descriptor doesn't configure:
1. Depth attachment pixel format
2. Depth compare function
3. Depth write mask

## Solution

### Metal Render Pipeline Changes

```metal
// In render pipeline descriptor setup:
renderPipelineDescriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
```

### Depth Stencil State

```rust
// Create depth stencil descriptor
let depth_desc = DepthStencilDescriptor::new();
depth_desc.set_depth_compare_function(MTLCompareFunction::LessEqual);
depth_desc.set_depth_write_enabled(true);

let depth_state = device.new_depth_stencil_state(&depth_desc);
```

### Render Pass Depth Attachment

```rust
// In render pass descriptor:
let depth_attachment = render_pass_desc.depth_attachment().unwrap();
depth_attachment.set_texture(&depth_texture);
depth_attachment.set_load_action(MTLLoadAction::Clear);
depth_attachment.set_store_action(MTLStoreAction::DontCare);
depth_attachment.set_clear_depth(1.0);
```

## Depth Values by Layer

| Component | Depth (z) | Purpose |
|-----------|-----------|---------|
| Background | 0.0 | Compositor fills screen |
| Windows | 0.1 - 0.9 | User app content |
| Dock | 0.95 | Always above windows |
| MenuBar | 0.98 | Always at top |
| Popups | 0.99 | Above everything |

## Test Cases

1. Depth texture created with correct format
2. Depth stencil state configured
3. Render pass has depth attachment
4. Vertices with different z values render in correct order

## Files to Modify

- `examples/visual_megakernel.rs` - Add depth buffer setup
- `src/gpu_os/gpu_app_system.rs` - Ensure vertices have correct z values
