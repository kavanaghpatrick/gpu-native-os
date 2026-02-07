# PRD: Issue #165 - Fix GpuOs Boot Initialization

## Problem
`GpuOs::boot()` doesn't call `initialize_system_app_states()`, so system apps (compositor, dock, menubar) have uninitialized state and don't generate vertices.

## Root Cause

```rust
// boot() creates GpuOs but doesn't initialize states:
pub fn boot(device: &Device) -> Result<Self, String> {
    let mut system = GpuAppSystem::new(device)?;
    // ... launches system apps
    // MISSING: initialize_system_app_states()
    Ok(os)
}

// boot_with_size() DOES call it:
pub fn boot_with_size(device: &Device, width: f32, height: f32) -> Result<Self, String> {
    let mut os = Self::boot(device)?;
    os.initialize_system_app_states(width, height);  // <-- Called here
    Ok(os)
}
```

## Solution

### Option 1: Always Initialize in boot()

```rust
pub fn boot(device: &Device) -> Result<Self, String> {
    let mut system = GpuAppSystem::new(device)?;
    // ... existing code ...

    let mut os = GpuOs { /* ... */ };
    os.launch_system_apps()?;

    // ADD: Initialize with default screen size
    os.initialize_system_app_states(os.screen_width, os.screen_height);

    Ok(os)
}
```

### Option 2: Make boot_with_size the Primary API

Deprecate `boot()` and always require screen dimensions:

```rust
pub fn boot(device: &Device, width: f32, height: f32) -> Result<Self, String> {
    // Combine boot and boot_with_size
}
```

## Recommended: Option 1
Keep backward compatibility, ensure initialization always happens.

## Test Cases

1. `GpuOs::boot()` initializes system app states
2. Compositor has valid screen dimensions after boot
3. Dock has items after boot
4. System apps generate vertices on first frame

## Files to Modify

- `src/gpu_os/gpu_os.rs` - Add initialization call to boot()
