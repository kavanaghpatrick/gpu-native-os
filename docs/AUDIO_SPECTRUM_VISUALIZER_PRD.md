# PRD: 1024-Band Audio Spectrum Visualizer

**Version**: 1.0
**Date**: 2026-01-23
**Status**: Ready for Implementation
**Target Platform**: Apple Silicon / Metal

---

## 1. Executive Summary

### 1.1 Overview

A real-time audio spectrum visualizer that performs a **1024-point FFT entirely on the GPU** within a single threadgroup of 1024 threads. This demo showcases the GPU-Native OS architecture by achieving perfect thread-to-data alignment: **Thread N owns Frequency Bin N**.

### 1.2 Why This Is Impressive

| Aspect | Traditional Approach | GPU-Native OS |
|--------|---------------------|---------------|
| FFT Location | CPU (vDSP, FFTW) | GPU Compute |
| Thread Model | Multi-threaded CPU | 1024 lockstep threads |
| Data Transfer | CPU FFT → GPU texture | Zero-copy unified memory |
| Thread Mapping | N/A | Thread N = Bin N (perfect 1:1) |
| Latency | 5-15ms | <2ms audio-to-visual |
| Synchronization | Mutexes, locks | `threadgroup_barrier()` |

**Key Insight**: A 1024-point FFT requires exactly 1024 complex values, and we have exactly 1024 threads. Each thread owns one frequency bin from sample input through FFT computation to final pixel rendering.

### 1.3 Goals

1. **Primary**: Demonstrate parallel Cooley-Tukey FFT in a single threadgroup
2. **Secondary**: Achieve <16ms latency from audio sample to displayed pixel
3. **Tertiary**: Showcase multiple visualization modes with GPU-computed aesthetics

### 1.4 Non-Goals

- Audio playback/synthesis (CPU handles audio I/O)
- Professional audio analysis accuracy (demo-quality is sufficient)
- Multiple simultaneous audio streams

---

## 2. User Experience

### 2.1 What the User Sees

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1024-Band Audio Spectrum Visualizer                    🎵 Playing: Song.mp3│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ████                                              │
│                          ██████                                             │
│                         ████████                                            │
│                        ██████████                                           │
│                       ████████████      ████                                │
│                      ██████████████    ██████                               │
│         ████        ████████████████  ████████        ████                  │
│        ██████      ██████████████████████████████    ██████                 │
│  ████████████████████████████████████████████████████████████████████       │
│  ████████████████████████████████████████████████████████████████████       │
│                                                                             │
│  20Hz                    1kHz                    10kHz              20kHz   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Bars] [Waveform] [Spectrogram]    Zoom: [−][+]    Colors: [Plasma]  ▼     │
│                                                                             │
│  Peak: 1.2kHz @ -12dB    FFT: 0.3ms    Frame: 0.8ms    120 FPS              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 User Interactions

| Input | Action |
|-------|--------|
| Click mode button | Switch visualization (Bars/Waveform/Spectrogram) |
| Zoom +/- | Adjust frequency range display |
| Color dropdown | Change color mapping |
| Drag on spectrum | Zoom to frequency region |
| Spacebar | Pause/resume visualization |
| Number keys 1-3 | Quick mode switch |

### 2.3 Visual Feedback

- **Beat Detection**: Screen flash/pulse on bass peaks
- **Smooth Animation**: 60-120fps interpolated bar heights
- **Hover Info**: Frequency/amplitude tooltip on mouse hover
- **Peak Hold**: Falling peak indicators above bars

---

## 3. Technical Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CPU DOMAIN                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Audio Capture Thread                            │    │
│  │  CoreAudio HAL → Ring Buffer (2048 samples) → Unified Memory        │    │
│  └──────────────────────────────────┬──────────────────────────────────┘    │
│                                     │ Write samples                          │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    UNIFIED MEMORY                                     │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │   │
│  │  │ Audio Ring   │ │ FFT Scratch  │ │ Spectrum     │ │ Widget      │  │   │
│  │  │ Buffer       │ │ (Complex)    │ │ Magnitude    │ │ State       │  │   │
│  │  │ 4KB          │ │ 16KB         │ │ 4KB          │ │ 2KB         │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │ Read by GPU                            │
│                                     ▼                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GPU DOMAIN                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              SINGLE THREADGROUP (1024 Threads)                       │    │
│  │                                                                      │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │  │ PHASE 1     │    │ PHASE 2     │    │ PHASE 3     │              │    │
│  │  │ Window &    │ ─▶ │ FFT         │ ─▶ │ Magnitude & │              │    │
│  │  │ Bit-Reverse │    │ Butterfly   │    │ dB Convert  │              │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │         │                  │                  │                       │    │
│  │         ▼                  ▼                  ▼                       │    │
│  │  Thread N loads    Thread N computes   Thread N owns                 │    │
│  │  Sample N          its butterfly ops   Bin N magnitude               │    │
│  │                                                                      │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │  │ PHASE 4     │    │ PHASE 5     │    │ PHASE 6     │              │    │
│  │  │ Smoothing & │ ─▶ │ Vertex      │ ─▶ │ State &     │              │    │
│  │  │ Peak Hold   │    │ Generation  │    │ UI Update   │              │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              RENDER PASS (Hardware Rasterization)                    │    │
│  │  Vertex Buffer → Vertex Shader → Fragment Shader → Framebuffer      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Thread Assignment (Unified Worker Model)

**All 1024 threads participate in ALL phases** - no fixed SIMD roles.

| Phase | All 1024 Threads Do | Barrier After |
|-------|---------------------|---------------|
| 1. Window & Load | Thread N: Load sample[bit_reverse(N)], apply Hann window | Yes |
| 2. FFT Butterfly | Thread N: Participate in log2(1024)=10 butterfly stages | Yes per stage |
| 3. Magnitude | Thread N: Compute `sqrt(re[N]² + im[N]²)` | Yes |
| 4. Smoothing | Thread N: Blend with previous frame, update peak | Yes |
| 5. Vertex Gen | Thread N: Generate bar/waveform vertex for bin N | Yes |
| 6. State Update | Thread N: Update widget state if N < widget_count | No |

### 3.3 Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPU-ACCESSIBLE MEMORY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AUDIO RING BUFFER (4KB) - Written by CPU, Read by GPU                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ head: u32 │ tail: u32 │ samples: [f32; 1024] │ overflow_samples: [f32; N]│ │
│  │ 4 bytes   │ 4 bytes   │ 4096 bytes           │                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  FFT SCRATCH (16KB) - GPU Threadgroup Memory (on-chip, fast)                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ complex: [Complex; 1024]  │ 8KB (re + im as f32 each)                   │ │
│  │ twiddle_re: [f32; 512]    │ 2KB (precomputed cos values)                │ │
│  │ twiddle_im: [f32; 512]    │ 2KB (precomputed -sin values)               │ │
│  │ scratch: [f32; 1024]      │ 4KB (temp storage)                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SPECTRUM STATE (8KB) - GPU Read/Write                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ magnitude: [f32; 1024]     │ 4KB (current frame, linear scale)         │ │
│  │ magnitude_db: [f32; 1024]  │ 4KB (current frame, dB scale)             │ │
│  │ smoothed: [f32; 1024]      │ 4KB (exponential moving average)          │ │
│  │ peak_hold: [f32; 1024]     │ 4KB (falling peak values)                 │ │
│  │ peak_age: [u16; 1024]      │ 2KB (frames since peak)                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  VISUALIZER STATE (256B) - GPU Read/Write                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ mode: u32                  │ 0=Bars, 1=Waveform, 2=Spectrogram         │ │
│  │ color_scheme: u32          │ 0=Plasma, 1=Viridis, 2=Fire, 3=Ice       │ │
│  │ zoom_start: f32            │ Start frequency (normalized 0-1)          │ │
│  │ zoom_end: f32              │ End frequency (normalized 0-1)            │ │
│  │ smoothing: f32             │ EMA factor (0.0-1.0)                       │ │
│  │ peak_decay: f32            │ Peak fall rate per frame                  │ │
│  │ beat_threshold: f32        │ dB threshold for beat detection           │ │
│  │ beat_detected: u32         │ 1 if beat this frame                      │ │
│  │ bass_energy: f32           │ Sum of bins 0-32 (sub-bass region)        │ │
│  │ frame_count: u32           │ Total frames rendered                     │ │
│  │ last_fft_time_us: u32      │ FFT computation time                      │ │
│  │ last_frame_time_us: u32    │ Total frame time                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  VERTEX BUFFER (96KB) - GPU Write, Hardware Rasterizer Read                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Bar mode: 1024 bars × 6 vertices × 16 bytes = 96KB                     │ │
│  │ Waveform: 1024 line segments × 2 vertices × 16 bytes = 32KB           │ │
│  │ Spectrogram: 1024 × 1 row × 4 bytes (color) = 4KB per row             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SPECTROGRAM HISTORY (512KB) - GPU Read/Write                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ history: [u32; 1024 × 128] │ 128 rows of 1024 color values            │ │
│  │ write_row: u32             │ Current row index (circular)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Total: ~640KB GPU memory                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Per-Frame Pipeline

```
Time (ms)    Operation                                    Threads Active
─────────────────────────────────────────────────────────────────────────
0.00         CPU: Audio callback fills ring buffer        0 (CPU only)

0.10         GPU Kernel Dispatch                          1024
             ├─ Phase 1: Window & Bit-Reverse Load        1024
             │   └─ threadgroup_barrier()
0.20         ├─ Phase 2: FFT Stage 1 (512 butterflies)    1024
             │   └─ threadgroup_barrier()
0.22         ├─ Phase 2: FFT Stage 2 (256 butterflies)    1024
             │   └─ threadgroup_barrier()
             │   ... (10 stages total) ...
0.40         ├─ Phase 2: FFT Stage 10 (1 butterfly)       1024
             │   └─ threadgroup_barrier()
0.45         ├─ Phase 3: Magnitude & dB Conversion        1024
             │   └─ threadgroup_barrier()
0.50         ├─ Phase 4: Smoothing & Peak Detection       1024
             │   └─ threadgroup_barrier()
0.60         ├─ Phase 5: Vertex Generation                1024
             │   └─ threadgroup_barrier()
0.65         └─ Phase 6: State Update                     1024

0.70         Render Pass: Hardware Rasterization          (GPU fixed function)
             ├─ Vertex Shader: Transform bars
             └─ Fragment Shader: Color mapping

1.50         Frame Complete, VSync                        0 (idle)

Total GPU active time: ~1.5ms (well under 8.33ms @ 120Hz)
```

### 3.5 Audio Input Strategy

```rust
// CPU-side audio capture (runs in dedicated audio thread)
struct AudioCapture {
    ring_buffer: Arc<AtomicRingBuffer>,  // Lock-free, shared with GPU
    sample_rate: u32,                      // 44100 or 48000 Hz
    hop_size: usize,                       // 512 samples (50% overlap)
}

impl AudioCapture {
    fn audio_callback(&mut self, samples: &[f32]) {
        // CoreAudio callback - runs at audio interrupt priority
        // Write samples to ring buffer (lock-free)
        let head = self.ring_buffer.head.load(Ordering::Relaxed);
        for (i, &sample) in samples.iter().enumerate() {
            let slot = (head + i as u32) % 2048;
            self.ring_buffer.samples[slot as usize] = sample;
        }
        self.ring_buffer.head.store(head + samples.len() as u32, Ordering::Release);
    }
}
```

---

## 4. Data Structures

### 4.1 Rust/Metal Shared Structures

```rust
// Must match Metal shader definitions exactly

/// Audio ring buffer for CPU→GPU transfer
#[repr(C)]
pub struct AudioRingBuffer {
    pub head: u32,              // Written by CPU (atomic)
    pub tail: u32,              // Written by GPU (atomic)
    pub _padding: [u32; 2],     // Alignment
    pub samples: [f32; 2048],   // Ring buffer (8KB)
}  // 8208 bytes total

/// Complex number for FFT
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}  // 8 bytes

/// Per-bin spectrum data
#[repr(C)]
pub struct SpectrumBin {
    pub magnitude: f32,         // Linear magnitude
    pub magnitude_db: f32,      // dB scale (-96 to 0)
    pub smoothed: f32,          // EMA smoothed value
    pub peak_hold: f32,         // Falling peak
    pub peak_age: u16,          // Frames since peak set
    pub _padding: u16,
}  // 20 bytes

/// Visualizer configuration and state
#[repr(C)]
pub struct VisualizerState {
    pub mode: u32,              // VisualizationMode enum
    pub color_scheme: u32,      // ColorScheme enum
    pub zoom_start: f32,        // 0.0 = 0 Hz
    pub zoom_end: f32,          // 1.0 = Nyquist
    pub smoothing: f32,         // EMA alpha (0.1 = smooth, 0.9 = responsive)
    pub peak_decay: f32,        // dB per frame decay
    pub beat_threshold: f32,    // dB above average for beat
    pub beat_detected: u32,     // Boolean (0/1)
    pub bass_energy: f32,       // Sum of low bins
    pub frame_count: u32,
    pub fft_time_us: u32,
    pub frame_time_us: u32,
    pub sample_rate: u32,       // 44100 or 48000
    pub _padding: [u32; 3],
}  // 64 bytes

/// Vertex for spectrum bars
#[repr(C)]
pub struct SpectrumVertex {
    pub position: [f32; 2],     // x, y in clip space
    pub color: [f32; 4],        // RGBA
    pub uv: [f32; 2],           // For gradient/texture
}  // 32 bytes

/// Parameters passed to kernel each frame
#[repr(C)]
pub struct AudioKernelParams {
    pub sample_count: u32,      // New samples available
    pub delta_time: f32,        // Frame delta in seconds
    pub time: f32,              // Total elapsed time
    pub width: f32,             // Viewport width
    pub height: f32,            // Viewport height
}  // 20 bytes

/// Visualization mode enum
#[repr(u32)]
pub enum VisualizationMode {
    Bars = 0,
    Waveform = 1,
    Spectrogram = 2,
}

/// Color scheme enum
#[repr(u32)]
pub enum ColorScheme {
    Plasma = 0,      // Purple → Pink → Yellow
    Viridis = 1,     // Purple → Green → Yellow
    Fire = 2,        // Black → Red → Orange → Yellow
    Ice = 3,         // Black → Blue → Cyan → White
    Monochrome = 4,  // Black → Green (classic VU style)
}
```

### 4.2 Size Verification

```rust
#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use super::*;

    #[test]
    fn verify_struct_sizes() {
        assert_eq!(size_of::<AudioRingBuffer>(), 8208);
        assert_eq!(size_of::<Complex>(), 8);
        assert_eq!(size_of::<SpectrumBin>(), 20);
        assert_eq!(size_of::<VisualizerState>(), 64);
        assert_eq!(size_of::<SpectrumVertex>(), 32);
        assert_eq!(size_of::<AudioKernelParams>(), 20);
    }
}
```

---

## 5. Shader Pseudocode

### 5.1 Parallel Cooley-Tukey FFT

```metal
// ============================================================================
// 1024-Point Radix-2 Decimation-in-Time FFT
// All 1024 threads participate in every stage
// ============================================================================

kernel void audio_spectrum_kernel(
    device AudioRingBuffer* audio [[buffer(0)]],
    device SpectrumBin* spectrum [[buffer(1)]],
    device VisualizerState* state [[buffer(2)]],
    device SpectrumVertex* vertices [[buffer(3)]],
    constant AudioKernelParams& params [[buffer(4)]],
    device float* spectrogram_history [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // =========================================================================
    // Threadgroup Shared Memory
    // =========================================================================
    threadgroup float re[1024];          // Real parts
    threadgroup float im[1024];          // Imaginary parts
    threadgroup float scratch[1024];     // For reductions

    // Precomputed twiddle factors (could also be constant buffer)
    threadgroup float twiddle_re[512];
    threadgroup float twiddle_im[512];

    // =========================================================================
    // PHASE 1: Bit-Reverse Load + Hann Window
    // =========================================================================

    // Precompute twiddle factors (thread 0-511 each computes one)
    if (tid < 512) {
        float angle = -2.0 * M_PI_F * float(tid) / 1024.0;
        twiddle_re[tid] = cos(angle);
        twiddle_im[tid] = sin(angle);
    }

    // Bit-reverse permutation lookup
    uint reversed = bit_reverse_10bit(tid);

    // Read sample from ring buffer with bit-reversed index
    uint tail = atomic_load_explicit(&audio->tail, memory_order_relaxed);
    uint sample_idx = (tail + reversed) % 2048;
    float sample = audio->samples[sample_idx];

    // Apply Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/N))
    float window = 0.5 * (1.0 - cos(2.0 * M_PI_F * float(tid) / 1024.0));

    // Store windowed sample (bit-reversed order → natural order after FFT)
    re[tid] = sample * window;
    im[tid] = 0.0;  // Input is real

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 2: FFT Butterfly Operations (10 stages for N=1024)
    // =========================================================================

    // Cooley-Tukey radix-2 DIT FFT
    // Stage s: butterfly size = 2^(s+1), twiddle step = N / 2^(s+1)

    for (uint stage = 0; stage < 10; stage++) {
        uint butterfly_size = 1u << (stage + 1);  // 2, 4, 8, ..., 1024
        uint half_size = butterfly_size >> 1;      // 1, 2, 4, ..., 512
        uint twiddle_step = 1024 >> (stage + 1);   // 512, 256, ..., 1

        // Determine which butterfly this thread participates in
        uint butterfly_idx = tid / butterfly_size;
        uint pos_in_butterfly = tid % butterfly_size;

        // Only bottom half of each butterfly does work
        // (top half reads the result)
        if (pos_in_butterfly < half_size) {
            uint bottom_idx = butterfly_idx * butterfly_size + pos_in_butterfly;
            uint top_idx = bottom_idx + half_size;
            uint twiddle_idx = pos_in_butterfly * twiddle_step;

            // Load butterfly inputs
            float a_re = re[bottom_idx];
            float a_im = im[bottom_idx];
            float b_re = re[top_idx];
            float b_im = im[top_idx];

            // Twiddle factor W_N^k = e^(-2*pi*i*k/N)
            float w_re = twiddle_re[twiddle_idx];
            float w_im = twiddle_im[twiddle_idx];

            // Complex multiply: b * W
            float bw_re = b_re * w_re - b_im * w_im;
            float bw_im = b_re * w_im + b_im * w_re;

            // Butterfly output
            re[bottom_idx] = a_re + bw_re;
            im[bottom_idx] = a_im + bw_im;
            re[top_idx] = a_re - bw_re;
            im[top_idx] = a_im - bw_im;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // PHASE 3: Magnitude & dB Conversion
    // =========================================================================

    // Compute magnitude: |X[k]| = sqrt(re[k]^2 + im[k]^2)
    float mag = sqrt(re[tid] * re[tid] + im[tid] * im[tid]);

    // Normalize by N/2 for proper amplitude scaling
    mag = mag / 512.0;

    // Convert to dB: 20 * log10(mag), clamped to [-96, 0]
    float mag_db = (mag > 1e-10) ? 20.0 * log10(mag) : -96.0;
    mag_db = clamp(mag_db, -96.0f, 0.0f);

    // Store to spectrum buffer
    spectrum[tid].magnitude = mag;
    spectrum[tid].magnitude_db = mag_db;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 4: Smoothing & Peak Detection
    // =========================================================================

    float prev_smoothed = spectrum[tid].smoothed;
    float alpha = state->smoothing;

    // Exponential moving average
    float smoothed = alpha * mag_db + (1.0 - alpha) * prev_smoothed;
    spectrum[tid].smoothed = smoothed;

    // Peak hold with decay
    float prev_peak = spectrum[tid].peak_hold;
    uint prev_age = spectrum[tid].peak_age;

    if (mag_db > prev_peak) {
        // New peak
        spectrum[tid].peak_hold = mag_db;
        spectrum[tid].peak_age = 0;
    } else {
        // Decay existing peak
        float decayed = prev_peak - state->peak_decay;
        spectrum[tid].peak_hold = max(decayed, mag_db);
        spectrum[tid].peak_age = min(prev_age + 1, 65535u);
    }

    // Beat detection (bass bins 0-32)
    scratch[tid] = (tid < 32) ? mag : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for bass energy
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float bass_energy = scratch[0];
        state->bass_energy = bass_energy;
        state->beat_detected = (bass_energy > state->beat_threshold) ? 1 : 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 5: Vertex Generation (mode-dependent)
    // =========================================================================

    uint mode = state->mode;
    float zoom_start = state->zoom_start;
    float zoom_end = state->zoom_end;
    float width = params.width;
    float height = params.height;

    // Map bin to screen X position (with zoom)
    float bin_normalized = float(tid) / 1024.0;
    bool in_range = (bin_normalized >= zoom_start) && (bin_normalized <= zoom_end);
    float x_norm = (bin_normalized - zoom_start) / (zoom_end - zoom_start);
    float x = x_norm * 2.0 - 1.0;  // Convert to clip space [-1, 1]

    // Map dB to screen Y position
    float y_norm = (smoothed + 96.0) / 96.0;  // -96dB→0, 0dB→1
    float y = y_norm * 2.0 - 1.0;  // Convert to clip space

    // Get color from palette
    float4 color = get_color(state->color_scheme, y_norm, state->beat_detected);

    if (mode == 0) {
        // ─────────────────────────────────────────────────────────────────────
        // BAR MODE: Each thread generates 6 vertices (2 triangles) for one bar
        // ─────────────────────────────────────────────────────────────────────
        float bar_width = 2.0 / (1024.0 / (zoom_end - zoom_start)) * 0.8;

        uint base = tid * 6;

        if (in_range) {
            // Bottom-left, Bottom-right, Top-right (Triangle 1)
            vertices[base + 0] = make_vertex(x - bar_width/2, -1.0, color);
            vertices[base + 1] = make_vertex(x + bar_width/2, -1.0, color);
            vertices[base + 2] = make_vertex(x + bar_width/2, y, color);

            // Bottom-left, Top-right, Top-left (Triangle 2)
            vertices[base + 3] = make_vertex(x - bar_width/2, -1.0, color);
            vertices[base + 4] = make_vertex(x + bar_width/2, y, color);
            vertices[base + 5] = make_vertex(x - bar_width/2, y, color);
        } else {
            // Zero out vertices outside zoom range
            for (uint i = 0; i < 6; i++) {
                vertices[base + i] = make_vertex(0, -2, float4(0));  // Off-screen
            }
        }
    }
    else if (mode == 1) {
        // ─────────────────────────────────────────────────────────────────────
        // WAVEFORM MODE: Each thread generates 2 vertices for a line segment
        // ─────────────────────────────────────────────────────────────────────
        uint base = tid * 2;

        // Current point
        vertices[base + 0] = make_vertex(x, y, color);

        // Next point (thread tid+1's position) - use simd_shuffle for efficiency
        float next_x = simd_shuffle_down(x, 1);
        float next_y = simd_shuffle_down(y, 1);

        // Handle SIMD group boundary
        if (simd_lane == 31 && tid < 1023) {
            next_x = (float(tid + 1) / 1024.0 - zoom_start) / (zoom_end - zoom_start) * 2.0 - 1.0;
            next_y = (spectrum[tid + 1].smoothed + 96.0) / 96.0 * 2.0 - 1.0;
        }

        vertices[base + 1] = make_vertex(next_x, next_y, color);
    }
    else if (mode == 2) {
        // ─────────────────────────────────────────────────────────────────────
        // SPECTROGRAM MODE: Each thread writes one pixel to current row
        // ─────────────────────────────────────────────────────────────────────
        uint row = state->frame_count % 128;  // Circular buffer of 128 rows
        uint pixel_idx = row * 1024 + tid;

        // Pack color to RGBA8
        uint packed = pack_color_rgba8(color);
        spectrogram_history[pixel_idx] = packed;

        // Generate vertices for full-screen quad (only thread 0)
        if (tid == 0) {
            // Two triangles covering the screen
            vertices[0] = make_vertex(-1, -1, float4(1));  // BL
            vertices[1] = make_vertex(1, -1, float4(1));   // BR
            vertices[2] = make_vertex(1, 1, float4(1));    // TR
            vertices[3] = make_vertex(-1, -1, float4(1));  // BL
            vertices[4] = make_vertex(1, 1, float4(1));    // TR
            vertices[5] = make_vertex(-1, 1, float4(1));   // TL
        }
    }

    // =========================================================================
    // PHASE 6: State Update
    // =========================================================================

    if (tid == 0) {
        // Update audio ring buffer tail (mark samples as consumed)
        uint old_tail = atomic_load_explicit(&audio->tail, memory_order_relaxed);
        atomic_store_explicit(&audio->tail, old_tail + 1024, memory_order_relaxed);

        state->frame_count++;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

// 10-bit bit reversal for N=1024
inline uint bit_reverse_10bit(uint n) {
    n = ((n & 0x3E0) >> 5) | ((n & 0x01F) << 5);  // Swap 5-bit halves
    n = ((n & 0x318) >> 3) | ((n & 0x063) << 3) | (n & 0x084);
    n = ((n & 0x252) >> 1) | ((n & 0x129) << 1) | (n & 0x084);
    return n;
}

// Color palette functions
inline float4 get_color(uint scheme, float value, uint beat) {
    float v = clamp(value, 0.0f, 1.0f);
    float4 color;

    if (scheme == 0) {
        // Plasma: Purple → Pink → Yellow
        color.r = v * v * 0.8 + 0.2;
        color.g = v * (1.0 - v) * 2.0;
        color.b = (1.0 - v) * 0.8 + 0.2;
        color.a = 1.0;
    }
    else if (scheme == 1) {
        // Viridis: Purple → Teal → Yellow
        color.r = v * v;
        color.g = v * 0.8 + 0.1;
        color.b = (1.0 - v * v) * 0.8;
        color.a = 1.0;
    }
    else if (scheme == 2) {
        // Fire: Black → Red → Orange → Yellow
        color.r = min(v * 2.0, 1.0);
        color.g = max(v * 2.0 - 0.5, 0.0);
        color.b = max(v * 2.0 - 1.0, 0.0);
        color.a = 1.0;
    }
    else if (scheme == 3) {
        // Ice: Black → Blue → Cyan → White
        color.r = max(v - 0.5, 0.0) * 2.0;
        color.g = max(v - 0.25, 0.0) * 1.5;
        color.b = min(v * 2.0, 1.0);
        color.a = 1.0;
    }
    else {
        // Monochrome: Green VU style
        color.r = 0.0;
        color.g = v;
        color.b = 0.0;
        color.a = 1.0;
    }

    // Beat flash effect
    if (beat != 0) {
        color.rgb = mix(color.rgb, float3(1.0), 0.3);
    }

    return color;
}

inline SpectrumVertex make_vertex(float x, float y, float4 color) {
    SpectrumVertex v;
    v.position = float2(x, y);
    v.color = color;
    v.uv = float2(0, 0);
    return v;
}
```

### 5.2 Bit-Reverse Lookup Table Alternative

For maximum performance, precompute bit-reverse indices:

```metal
constant uint bit_reverse_lut[1024] = {
    0, 512, 256, 768, 128, 640, 384, 896, 64, 576, 320, 832, 192, 704, 448, 960,
    // ... 1024 entries total (generated at compile time)
};

// Usage: much faster than computing
uint reversed = bit_reverse_lut[tid];
```

---

## 6. Visualization Modes

### 6.1 Spectrum Bars (Mode 0)

```
Visual:
████████                                  Amplitude
██████████████                            shown as
████████████████████                      vertical
██████████████████████████                bar height
████████████████████████████████████████
────────────────────────────────────────
20Hz            1kHz           10kHz  20kHz

Features:
- 1024 individual bars (or grouped for display)
- Logarithmic frequency scale option
- Peak hold markers (falling white lines above peaks)
- Configurable bar width and gap
- Beat-reactive color flash
```

### 6.2 Waveform (Mode 1)

```
Visual:
                    ▄▄▄▄
        ▄▄▄▄▄▄▄▄▄▄██████▄▄▄▄▄▄▄▄
    ▄▄██████████████████████████████▄▄
────────────────────────────────────────── 0dB
    ▀▀██████████████████████████████▀▀
        ▀▀▀▀▀▀▀▀▀▀██████▀▀▀▀▀▀▀▀
                    ▀▀▀▀

Features:
- Continuous line connecting frequency bins
- Mirrored display (symmetry around 0dB line)
- Glow effect using additive blending
- Thickness varies with amplitude
```

### 6.3 Spectrogram (Mode 2)

```
Visual:
Time ↓
████████████████████████████████ ← Current
████░░████████████░░████████████
████░░░░██████████░░░░██████████
████░░░░░░████████░░░░░░████████
████░░░░████████████░░░░████████
████░░████████████████░░████████
────────────────────────────────
20Hz            1kHz        20kHz

Features:
- Scrolling waterfall display (128 rows history)
- Color intensity = amplitude (dB)
- Time flows downward (newest at top)
- Optional frequency scale markers
```

---

## 7. Widget Integration

### 7.1 UI Layout (GPU-Native Widgets)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ROOT (Container, full screen)                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ HEADER (Container, height: 40px)                                         │ │
│ │ ┌────────────────────────────────────┬──────────────────────────────────┐ │
│ │ │ TITLE (Text: "1024-Band Spectrum") │ NOW_PLAYING (Text, right-align)  │ │
│ │ └────────────────────────────────────┴──────────────────────────────────┘ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ SPECTRUM_VIEW (Custom widget, flex: 1)                                   │ │
│ │                                                                          │ │
│ │              [Visualization rendered here]                               │ │
│ │                                                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ FREQ_LABELS (Container, height: 20px)                                    │ │
│ │ │ 20Hz │ 100Hz │ 1kHz │ 5kHz │ 10kHz │ 20kHz │                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ CONTROLS (Container, height: 60px)                                       │ │
│ │ ┌──────────────────┬───────────────┬────────────────┬─────────────────┐ │ │
│ │ │ MODE_BUTTONS     │ ZOOM_CONTROLS │ COLOR_DROPDOWN │ STATS_DISPLAY   │ │ │
│ │ │ [Bars][Wave][Sp] │ [−] [100%] [+]│ [Plasma    ▼]  │ FFT: 0.3ms      │ │ │
│ │ └──────────────────┴───────────────┴────────────────┴─────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Widget State Structures

```rust
/// Button group for mode selection
#[repr(C)]
pub struct ModeButtonGroup {
    pub buttons: [WidgetCompact; 3],  // Bars, Waveform, Spectrogram
    pub selected_index: u32,
}

/// Slider for zoom/smoothing controls
#[repr(C)]
pub struct SliderWidget {
    pub base: WidgetCompact,
    pub value: f32,           // 0.0 - 1.0
    pub min_value: f32,
    pub max_value: f32,
    pub step: f32,
}

/// Dropdown for color scheme selection
#[repr(C)]
pub struct DropdownWidget {
    pub base: WidgetCompact,
    pub selected_index: u32,
    pub option_count: u32,
    pub expanded: u32,        // Boolean
}
```

### 7.3 Widget Actions (GPU-Side)

```metal
// In the unified kernel, widget updates happen in Phase 6
if (tid < widget_count) {
    WidgetCompact w = widgets[tid];
    uint widget_type = get_widget_type(w.packed_style);

    if (widget_type == WIDGET_BUTTON && tid == hovered_widget && mouse_clicked) {
        // Mode button clicked - update visualizer state
        if (tid == MODE_BUTTON_BARS) state->mode = 0;
        else if (tid == MODE_BUTTON_WAVE) state->mode = 1;
        else if (tid == MODE_BUTTON_SPECTRO) state->mode = 2;
    }

    if (widget_type == WIDGET_SLIDER && tid == dragging_widget) {
        // Slider being dragged
        float normalized = (cursor_x - w.bounds.x) / w.bounds.width;
        if (tid == SLIDER_ZOOM) {
            state->zoom_end = clamp(normalized, state->zoom_start + 0.1, 1.0);
        }
        else if (tid == SLIDER_SMOOTHING) {
            state->smoothing = clamp(normalized, 0.05, 0.95);
        }
    }
}
```

---

## 8. Visual Design

### 8.1 Color Mapping Palettes

```metal
// Perceptually uniform color maps for audio visualization

// PLASMA: High contrast, visually striking
float3 plasma(float t) {
    return float3(
        t * t * t * 0.15 + t * t * 0.35 + t * 0.15 + 0.05,
        t * (1.0 - t) * 1.5,
        sqrt(t) * 0.4 + (1.0 - t) * 0.5
    );
}

// VIRIDIS: Perceptually uniform, colorblind-friendly
float3 viridis(float t) {
    const float3 c0 = float3(0.267, 0.004, 0.329);
    const float3 c1 = float3(0.282, 0.140, 0.458);
    const float3 c2 = float3(0.254, 0.265, 0.530);
    const float3 c3 = float3(0.163, 0.471, 0.558);
    const float3 c4 = float3(0.134, 0.658, 0.517);
    const float3 c5 = float3(0.477, 0.821, 0.318);
    const float3 c6 = float3(0.993, 0.906, 0.144);

    t = clamp(t, 0.0, 1.0);
    float t6 = t * 6.0;
    int i = int(t6);
    float f = fract(t6);

    float3 colors[7] = {c0, c1, c2, c3, c4, c5, c6};
    return mix(colors[min(i, 5)], colors[min(i + 1, 6)], f);
}

// FIRE: Classic audio visualization
float3 fire(float t) {
    return float3(
        min(t * 3.0, 1.0),
        clamp((t - 0.25) * 2.0, 0.0, 1.0),
        clamp((t - 0.5) * 2.0, 0.0, 1.0)
    );
}
```

### 8.2 Beat Detection Effects

```metal
// Beat-reactive visual effects

struct BeatEffects {
    float pulse_intensity;    // 0-1, decays after beat
    float flash_alpha;        // Screen flash overlay
    float zoom_factor;        // Momentary zoom on beat
};

BeatEffects update_beat_effects(BeatEffects prev, bool beat_detected, float dt) {
    BeatEffects fx;

    if (beat_detected) {
        fx.pulse_intensity = 1.0;
        fx.flash_alpha = 0.3;
        fx.zoom_factor = 1.05;
    } else {
        // Exponential decay
        fx.pulse_intensity = prev.pulse_intensity * exp(-dt * 10.0);
        fx.flash_alpha = prev.flash_alpha * exp(-dt * 15.0);
        fx.zoom_factor = 1.0 + (prev.zoom_factor - 1.0) * exp(-dt * 8.0);
    }

    return fx;
}

// Apply effects in fragment shader
float4 apply_beat_effects(float4 color, BeatEffects fx) {
    // Pulse: brighten on beat
    color.rgb += fx.pulse_intensity * 0.2;

    // Glow: increase saturation
    float luma = dot(color.rgb, float3(0.299, 0.587, 0.114));
    color.rgb = mix(float3(luma), color.rgb, 1.0 + fx.pulse_intensity * 0.5);

    return color;
}
```

### 8.3 Gradient and Glow

```metal
// Bar gradient: darker at bottom, brighter at top
float4 bar_gradient(float y_normalized, float4 base_color) {
    float gradient = 0.3 + y_normalized * 0.7;
    return float4(base_color.rgb * gradient, base_color.a);
}

// Glow effect using gaussian blur approximation
float4 apply_glow(texture2d<float> source, float2 uv, float radius) {
    float4 color = source.sample(sampler, uv);
    float4 glow = float4(0);

    // 5-tap gaussian approximation
    const float weights[5] = {0.227, 0.194, 0.122, 0.054, 0.016};
    const float offsets[5] = {0.0, 1.0, 2.0, 3.0, 4.0};

    for (int i = 0; i < 5; i++) {
        float2 offset = float2(offsets[i] * radius / source.get_width(), 0);
        glow += source.sample(sampler, uv + offset) * weights[i];
        glow += source.sample(sampler, uv - offset) * weights[i];
    }

    return color + glow * 0.5;
}
```

---

## 9. Performance Targets

### 9.1 Timing Budget

| Metric | Target | Maximum | Notes |
|--------|--------|---------|-------|
| FFT Computation | 0.5ms | 1.0ms | 10 butterfly stages |
| Total Kernel | 1.5ms | 3.0ms | All 6 phases |
| Render Pass | 1.0ms | 2.0ms | Hardware rasterization |
| Frame Total | 3.0ms | 5.0ms | Kernel + Render |
| Audio-to-Visual Latency | 8ms | 16ms | One frame + buffer |
| Frame Rate | 120fps | 60fps | VSync locked |

### 9.2 Memory Budget

| Buffer | Size | Usage |
|--------|------|-------|
| Audio Ring | 8KB | CPU writes, GPU reads |
| FFT Scratch | 16KB | Threadgroup memory |
| Spectrum State | 18KB | Magnitudes + peaks |
| Visualizer State | 256B | Configuration |
| Vertex Buffer | 96KB | Bar/line vertices |
| Spectrogram | 512KB | 128-row history |
| **Total** | **~650KB** | Well under GPU limits |

### 9.3 Threadgroup Occupancy

```
Target: 1024 threads with 16KB threadgroup memory

Apple M4 GPU:
- Max threads per threadgroup: 1024
- Max threadgroup memory: 32KB
- Our usage: 16KB → 100% occupancy achievable

Register Usage (per thread):
- FFT: ~20 registers (re, im, twiddle, indices)
- Spectrum: ~10 registers (magnitude, smoothed, peak)
- Vertex: ~8 registers (position, color)
- Total: ~38 registers per thread → well within limits
```

---

## 10. Implementation Milestones

### Milestone 1: Audio Pipeline Foundation (Week 1)

- [ ] Set up CoreAudio capture callback
- [ ] Implement lock-free ring buffer (CPU→GPU)
- [ ] Create Metal buffers with shared storage mode
- [ ] Verify audio samples reach GPU memory
- [ ] Add basic FPS/latency instrumentation

**Deliverable**: Audio samples visible in GPU debugger

### Milestone 2: GPU FFT Implementation (Week 2)

- [ ] Implement bit-reverse permutation
- [ ] Implement Hann window application
- [ ] Implement single butterfly stage (verify correctness)
- [ ] Implement full 10-stage FFT
- [ ] Add threadgroup barriers between stages
- [ ] Verify FFT output matches CPU reference (vDSP)

**Deliverable**: Correct FFT magnitudes in spectrum buffer

### Milestone 3: Basic Spectrum Bars (Week 3)

- [ ] Implement magnitude → dB conversion
- [ ] Implement vertex generation for bars
- [ ] Create vertex/fragment shaders for bars
- [ ] Add basic color mapping (single color)
- [ ] Verify visual output matches audio

**Deliverable**: Working bar visualizer with live audio

### Milestone 4: Smoothing and Peak Hold (Week 4)

- [ ] Implement EMA smoothing
- [ ] Implement peak hold with decay
- [ ] Add peak hold markers (thin lines above bars)
- [ ] Tune smoothing/decay parameters
- [ ] Add beat detection (bass energy threshold)

**Deliverable**: Smooth, professional-looking visualization

### Milestone 5: Multiple Visualization Modes (Week 5)

- [ ] Implement waveform mode (line rendering)
- [ ] Implement spectrogram mode (texture scrolling)
- [ ] Add mode switching logic
- [ ] Optimize vertex generation for each mode

**Deliverable**: All three modes working

### Milestone 6: Color Palettes and Effects (Week 6)

- [ ] Implement plasma, viridis, fire, ice palettes
- [ ] Add beat-reactive flash effect
- [ ] Add glow/bloom post-processing
- [ ] Add bar gradient shading

**Deliverable**: Visually polished output

### Milestone 7: UI Integration (Week 7)

- [ ] Integrate with GPU-Native OS widget system
- [ ] Add mode selection buttons
- [ ] Add zoom slider
- [ ] Add color scheme dropdown
- [ ] Add stats display (FPS, FFT time)

**Deliverable**: Full interactive application

### Milestone 8: Polish and Optimization (Week 8)

- [ ] Profile with Metal System Trace
- [ ] Optimize hot paths (FFT butterflies, vertex gen)
- [ ] Add frequency scale labels
- [ ] Add logarithmic frequency option
- [ ] Final performance verification

**Deliverable**: Production-ready demo

---

## 11. Future Enhancements

### 11.1 Audio Features

- **Stereo Visualization**: Separate left/right channels (2048 threads, 512 per FFT)
- **Multi-band EQ Display**: Grouped frequency bands (bass, mid, treble)
- **Note Detection**: Pitch tracking using autocorrelation
- **Audio Source Selection**: Microphone input, system audio capture

### 11.2 Visual Features

- **3D Spectrum**: Rotating 3D bars using depth buffer
- **Particle System**: Beat-reactive particles (GPU instancing)
- **Circular Layout**: Radial spectrum display
- **Custom Backgrounds**: Reactive gradient backgrounds

### 11.3 Technical Improvements

- **2048-Point FFT**: Higher frequency resolution (needs 2 threadgroups or occupancy trade-off)
- **Overlap-Add**: 50% overlap for smoother updates
- **SIMD Optimization**: Use `simd_shuffle` for twiddle factor distribution
- **Persistent Threads**: Keep kernel running across frames

### 11.4 Integration

- **Music Player**: Integrate with audio playback (AVFoundation)
- **Spotify Integration**: Visualize currently playing track
- **MIDI Visualization**: Show MIDI events as color-coded hits
- **OSC Control**: External control of visualization parameters

---

## Appendix A: FFT Mathematics Reference

### Cooley-Tukey Radix-2 DIT FFT

For N = 1024 = 2^10:

```
X[k] = Σ(n=0 to N-1) x[n] * W_N^(nk)

where W_N = e^(-2πi/N) (primitive Nth root of unity)

Butterfly operation:
    A' = A + W * B
    B' = A - W * B

where W = W_N^k for appropriate k in each stage
```

### Stage-by-Stage Breakdown

| Stage | Butterfly Size | Butterflies | Twiddle Step |
|-------|---------------|-------------|--------------|
| 0 | 2 | 512 | 512 |
| 1 | 4 | 256 | 256 |
| 2 | 8 | 128 | 128 |
| 3 | 16 | 64 | 64 |
| 4 | 32 | 32 | 32 |
| 5 | 64 | 16 | 16 |
| 6 | 128 | 8 | 8 |
| 7 | 256 | 4 | 4 |
| 8 | 512 | 2 | 2 |
| 9 | 1024 | 1 | 1 |

### Frequency Bin Mapping

```
bin_frequency[k] = k * sample_rate / N

For sample_rate = 44100 Hz, N = 1024:
- Bin 0: 0 Hz (DC)
- Bin 1: 43 Hz
- Bin 10: 430 Hz
- Bin 100: 4.3 kHz
- Bin 512: 22.05 kHz (Nyquist)
- Bins 513-1023: Mirror of 0-511 (conjugate symmetric for real input)
```

---

## Appendix B: CoreAudio Setup (Rust)

```rust
use coreaudio::audio_unit::{AudioUnit, IOType, SampleFormat};
use coreaudio::audio_unit::render_callback::{self, data};
use std::sync::Arc;

pub struct AudioCaptureConfig {
    pub sample_rate: f64,
    pub buffer_size: u32,
    pub channels: u32,
}

impl Default for AudioCaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            buffer_size: 1024,
            channels: 1,  // Mono for FFT
        }
    }
}

pub fn setup_audio_capture(
    ring_buffer: Arc<AudioRingBuffer>,
    config: AudioCaptureConfig,
) -> Result<AudioUnit, coreaudio::Error> {
    let mut audio_unit = AudioUnit::new(IOType::DefaultOutput)?;

    audio_unit.set_property(
        kAudioUnitProperty_StreamFormat,
        Scope::Input,
        Element::Output,
        Some(&AudioStreamBasicDescription {
            mSampleRate: config.sample_rate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: config.channels,
            mBitsPerChannel: 32,
            mReserved: 0,
        }),
    )?;

    audio_unit.set_render_callback(move |args: render_callback::Args<data::Float32>| {
        let buffer = args.data;
        let ring = ring_buffer.clone();

        // Write samples to ring buffer
        let head = ring.head.load(Ordering::Relaxed);
        for (i, &sample) in buffer.iter().enumerate() {
            let slot = (head as usize + i) % 2048;
            ring.samples[slot] = sample;
        }
        ring.head.store(head + buffer.len() as u32, Ordering::Release);

        Ok(())
    })?;

    audio_unit.start()?;
    Ok(audio_unit)
}
```

---

## Appendix C: Verification Test Plan

### C.1 FFT Correctness

```rust
#[test]
fn test_fft_against_reference() {
    // Generate test signal: 440 Hz sine wave
    let sample_rate = 44100.0;
    let freq = 440.0;
    let samples: Vec<f32> = (0..1024)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    // Run GPU FFT
    let gpu_magnitudes = run_gpu_fft(&samples);

    // Run CPU reference (vDSP)
    let cpu_magnitudes = run_vdsp_fft(&samples);

    // Expected peak at bin k = 440 * 1024 / 44100 ≈ 10
    let expected_peak_bin = (freq * 1024.0 / sample_rate).round() as usize;

    // Verify peak location
    let gpu_peak = gpu_magnitudes.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(gpu_peak, expected_peak_bin, "Peak bin mismatch");

    // Verify magnitude accuracy (within 1%)
    for i in 0..512 {
        let error = (gpu_magnitudes[i] - cpu_magnitudes[i]).abs() / cpu_magnitudes[i].max(1e-10);
        assert!(error < 0.01, "Magnitude error at bin {}: {:.2}%", i, error * 100.0);
    }
}
```

### C.2 Latency Measurement

```rust
#[test]
fn test_audio_to_visual_latency() {
    // Inject impulse at known time
    let impulse_time = Instant::now();
    ring_buffer.samples[0] = 1.0;  // Impulse
    ring_buffer.head.store(1024, Ordering::Release);

    // Wait for visualization to respond
    loop {
        run_frame();
        if spectrum_state.magnitude[0] > 0.5 {
            let latency = impulse_time.elapsed();
            println!("Audio-to-visual latency: {:?}", latency);
            assert!(latency < Duration::from_millis(16), "Latency too high");
            break;
        }
    }
}
```

---

*This PRD defines a complete GPU-native audio visualization system that showcases the power of the single-threadgroup architecture. The 1024-point FFT perfectly matches the 1024 threads, enabling thread N to own frequency bin N from audio input through FFT computation to final pixel rendering.*
