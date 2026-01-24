# PRD: GPU Filesystem Browser GUI

## Overview

A native macOS GUI application that lets users browse and search their filesystem using GPU-accelerated lookups. The app scans a directory tree into GPU memory and provides instant search results.

## User Flow

1. **Launch** - App opens with `/Users/patrickkavanagh` pre-loaded
2. **Scan Progress** - Shows loading indicator while scanning directory tree
3. **Search** - Type in search box, see results update in real-time
4. **Results** - Click a result to reveal in Finder or copy path

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  GPU Filesystem Browser                              [─][□][×]│
├─────────────────────────────────────────────────────────────┤
│  Base: /Users/patrickkavanagh  [Change...]  [Rescan]        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 🔍 Search: [_________________________]                │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Results (247 matches)                          23µs GPU    │
│  ─────────────────────────────────────────────────────────  │
│  📁 /Users/patrickkavanagh/Documents/project               │
│  📁 /Users/patrickkavanagh/Downloads/project-backup        │
│  📄 /Users/patrickkavanagh/rust-experiment/Cargo.toml      │
│  📄 /Users/patrickkavanagh/notes/project-ideas.md          │
│  ...                                                        │
│                                                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Status: 45,231 files indexed (12MB GPU) │ Ready            │
└─────────────────────────────────────────────────────────────┘
```

## Features

### P0 (MVP)
- [x] Scan directory tree into GPU memory
- [ ] Native macOS window with search box
- [ ] Real-time search as you type
- [ ] Display file/folder results with icons
- [ ] Show search timing (GPU microseconds)
- [ ] Click result to copy path

### P1 (Nice to Have)
- [ ] Change base directory button
- [ ] Rescan button
- [ ] Reveal in Finder on double-click
- [ ] Filter by file type (files only, folders only)
- [ ] Search history

## Technical Approach

### Option A: egui (Recommended)
- Pure Rust, cross-platform
- Simple integration with existing Metal code
- `eframe` provides window management
- Fast to implement

```toml
[dependencies]
eframe = "0.27"
egui = "0.27"
```

### Option B: Native Cocoa
- Uses existing cocoa/objc crates already in project
- More complex but matches existing demo code
- Better macOS native feel

### Option C: Tauri + Web UI
- HTML/CSS/JS frontend
- Rust backend with existing GPU code
- Most flexible UI but heaviest

## Implementation Plan

1. Add `eframe`/`egui` dependencies to Cargo.toml
2. Create `examples/filesystem_browser.rs`
3. Build scan logic (reuse from filesystem_search.rs)
4. Create egui window with search box
5. Wire up GPU search on text change
6. Display results in scrollable list
7. Add click-to-copy functionality

## Success Criteria

- App launches and shows window
- Scans /Users/patrickkavanagh in <5 seconds
- Search results appear as you type (<50ms latency)
- Can copy file paths from results

## Files to Create

```
examples/filesystem_browser.rs  # Main GUI application
```

## Dependencies to Add

```toml
eframe = "0.27"
egui = "0.27"
```

## Launch Command

```bash
cargo run --release --example filesystem_browser
```
