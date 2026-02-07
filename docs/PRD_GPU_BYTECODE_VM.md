# PRD: GPU Bytecode Virtual Machine

## Goal

Enable dynamic app loading from filesystem without CPU involvement at load time. Apps are bytecode programs that the GPU interprets directly.

## Requirements

The bytecode must be expressive enough to run real programs:

| Demo | Key Operations |
|------|----------------|
| Game of Life | Nested loops, neighbor counting, modulo, array read/write |
| Particles | Vector math, physics integration, bounds checking |
| Mandelbrot | Complex number math, iteration loops, color mapping |
| Terminal | Text buffer manipulation, cursor movement, input handling |

## Instruction Set Architecture (ISA)

### Design Principles

1. **Register-based** - More efficient than stack-based on GPU (less memory traffic)
2. **SIMD-friendly** - Native float4 operations
3. **Parallel-aware** - Thread ID as built-in register
4. **Simple encoding** - Fixed 8-byte instructions (easy to decode)

### Register File

Each thread has 32 registers:

| Register | Name | Purpose |
|----------|------|---------|
| r0 | ZERO | Always 0.0 |
| r1 | TID | Thread ID (read-only) |
| r2 | TGSIZE | Threadgroup size (read-only) |
| r3 | FRAME | Frame number (read-only) |
| r4-r7 | A0-A3 | Arguments / return values |
| r8-r23 | T0-T15 | Temporaries |
| r24-r31 | S0-S7 | Saved across calls |

Each register is a `float4` (16 bytes). Use `.x`, `.y`, `.z`, `.w` for components.

### Instruction Encoding

Fixed 8 bytes per instruction:

```
┌────────────────────────────────────────────────────────────────┐
│ opcode(8) │ dst(5) │ src1(5) │ src2(5) │ flags(5) │ imm(32)    │
└────────────────────────────────────────────────────────────────┘
```

- `opcode`: Operation (256 possible)
- `dst`: Destination register (0-31)
- `src1`, `src2`: Source registers (0-31)
- `flags`: Swizzle, component select, etc.
- `imm`: Immediate value (float or int, context-dependent)

### Instruction Categories

#### 1. Arithmetic (0x00-0x1F)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0x00 | NOP | No operation |
| 0x01 | MOV | dst = src1 |
| 0x02 | ADD | dst = src1 + src2 |
| 0x03 | SUB | dst = src1 - src2 |
| 0x04 | MUL | dst = src1 * src2 |
| 0x05 | DIV | dst = src1 / src2 |
| 0x06 | MOD | dst = src1 % src2 (int) |
| 0x07 | NEG | dst = -src1 |
| 0x08 | ABS | dst = abs(src1) |
| 0x09 | MIN | dst = min(src1, src2) |
| 0x0A | MAX | dst = max(src1, src2) |
| 0x0B | FLOOR | dst = floor(src1) |
| 0x0C | CEIL | dst = ceil(src1) |
| 0x0D | FRACT | dst = fract(src1) |
| 0x0E | SQRT | dst = sqrt(src1) |
| 0x0F | SIN | dst = sin(src1) |
| 0x10 | COS | dst = cos(src1) |
| 0x11 | ADDI | dst = src1 + imm |
| 0x12 | MULI | dst = src1 * imm |
| 0x13 | LOADI | dst = imm (broadcast to all components) |
| 0x14 | LOAD4 | dst = float4(imm[0], imm[1], imm[2], imm[3]) |

#### 2. Vector Operations (0x20-0x3F)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0x20 | DOT | dst.x = dot(src1, src2) |
| 0x21 | CROSS | dst.xyz = cross(src1.xyz, src2.xyz) |
| 0x22 | LEN | dst.x = length(src1) |
| 0x23 | NORM | dst = normalize(src1) |
| 0x24 | LERP | dst = mix(src1, src2, imm) |
| 0x25 | SWIZZLE | dst = src1.swizzle (flags encode swizzle) |
| 0x26 | SPLAT | dst = src1.component (broadcast one component) |

#### 3. Comparison & Logic (0x40-0x5F)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0x40 | EQ | dst = (src1 == src2) ? 1.0 : 0.0 |
| 0x41 | NE | dst = (src1 != src2) ? 1.0 : 0.0 |
| 0x42 | LT | dst = (src1 < src2) ? 1.0 : 0.0 |
| 0x43 | LE | dst = (src1 <= src2) ? 1.0 : 0.0 |
| 0x44 | GT | dst = (src1 > src2) ? 1.0 : 0.0 |
| 0x45 | GE | dst = (src1 >= src2) ? 1.0 : 0.0 |
| 0x46 | AND | dst = src1 & src2 (bitwise) |
| 0x47 | OR | dst = src1 | src2 (bitwise) |
| 0x48 | XOR | dst = src1 ^ src2 (bitwise) |
| 0x49 | NOT | dst = ~src1 (bitwise) |
| 0x4A | SEL | dst = (src1.x != 0) ? src2 : imm |

#### 4. Control Flow (0x60-0x7F)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0x60 | JMP | pc = imm |
| 0x61 | JZ | if (src1.x == 0) pc = imm |
| 0x62 | JNZ | if (src1.x != 0) pc = imm |
| 0x63 | CALL | push pc, pc = imm |
| 0x64 | RET | pc = pop |
| 0x65 | LOOP | Loop header: init counter |
| 0x66 | ENDLOOP | Loop footer: decrement, branch back |
| 0x67 | BREAK | Exit current loop |
| 0x68 | CONTINUE | Skip to next iteration |

#### 5. Memory Access (0x80-0x9F)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0x80 | LD | dst = state[src1.x + imm] (load float4) |
| 0x81 | ST | state[src1.x + imm] = src2 (store float4) |
| 0x82 | LD1 | dst.x = state_bytes[src1.x + imm] (load byte) |
| 0x83 | ST1 | state_bytes[src1.x + imm] = src2.x (store byte) |
| 0x84 | LD4 | dst = state_f32[src1.x + imm] (load single float) |
| 0x85 | ST4 | state_f32[src1.x + imm] = src2.x (store single float) |
| 0x86 | LDI | dst = state_i32[src1.x + imm] as float |
| 0x87 | STI | state_i32[src1.x + imm] = int(src2.x) |

#### 6. Graphics Output (0xA0-0xBF)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0xA0 | QUAD | Emit quad: pos=src1.xy, size=src1.zw, color=src2, depth=imm |
| 0xA1 | TRI | Emit triangle: v0=src1.xyz, v1=src2.xyz, v2=state |
| 0xA2 | VERTEX | Emit single vertex: pos=src1.xyz, color=src2, uv=imm |
| 0xA3 | SETDEPTH | Set depth for subsequent primitives |
| 0xA4 | CIRCLE | Emit circle: center=src1.xy, radius=src1.z, color=src2 |

#### 7. Special (0xE0-0xFF)

| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0xE0 | BARRIER | threadgroup_barrier() |
| 0xE1 | ATOMIC_ADD | dst = atomic_add(state[src1.x], src2.x) |
| 0xE2 | ATOMIC_MAX | dst = atomic_max(state[src1.x], src2.x) |
| 0xE3 | ATOMIC_MIN | dst = atomic_min(state[src1.x], src2.x) |
| 0xE4 | RAND | dst = random(src1.x as seed) |
| 0xE5 | TIME | dst.x = frame_time |
| 0xE6 | SCREEN | dst = float4(screen_w, screen_h, 0, 0) |
| 0xFF | HALT | Stop execution |

## Example Programs

### Game of Life (Bytecode)

```asm
; Header: grid_width at state[0], grid_height at state[1]

; Calculate cells per thread
LOADI   r8, 0           ; r8 = 0 (offset to grid_width)
LDI     r9, r8, 0       ; r9.x = grid_width
LDI     r10, r8, 1      ; r10.x = grid_height
MUL     r11, r9, r10    ; r11.x = grid_size

; Calculate my range: start = tid * cells_per_thread
MOV     r12, r1         ; r12 = tid
DIV     r13, r11, r2    ; r13 = grid_size / tg_size
ADDI    r13, r13, 1     ; r13 = cells_per_thread (rounded up)
MUL     r14, r12, r13   ; r14 = start
ADD     r15, r14, r13   ; r15 = end
MIN     r15, r15, r11   ; r15 = min(end, grid_size)

; Loop over my cells
MOV     r16, r14        ; r16 = i (loop counter)
loop_start:
GE      r17, r16, r15   ; r17 = (i >= end)
JNZ     r17, loop_end   ; if done, exit

; x = i % width, y = i / width
MOD     r18, r16, r9    ; r18.x = x
DIV     r19, r16, r9    ; r19.x = y

; Count neighbors (unrolled inner loop for simplicity)
LOADI   r20, 0          ; r20 = neighbor count
; ... neighbor counting code (8 neighbor checks) ...
; Each check: compute nx, ny, load cell, add to count

; Apply Conway's rules
LOADI   r21, 8          ; offset to grid data
ADD     r22, r21, r16   ; r22 = grid[i] offset
LD1     r23, r22, 0     ; r23.x = current cell (0 or 1)

; (alive && (neighbors == 2 || neighbors == 3)) || (!alive && neighbors == 3)
EQ      r24, r20, 2     ; neighbors == 2
EQ      r25, r20, 3     ; neighbors == 3
OR      r26, r24, r25   ; 2 or 3 neighbors
MUL     r27, r23, r26   ; alive && (2 or 3)
EQ      r28, r23, 0     ; !alive
MUL     r29, r28, r25   ; !alive && 3
OR      r30, r27, r29   ; next state

ST1     r22, r30, 0     ; grid[i] = next

; Emit quad if alive
JZ      r30, skip_quad
; Calculate screen position
MUL     r4, r18, 8      ; r4.x = x * cell_size
MUL     r5, r19, 8      ; r5.x = y * cell_size
; Pack into pos/size
; r4.xy = pos, r4.zw = size(8,8)
LOADI   r5, 0           ; green color
ADDI    r5.y, r5, 1     ; r5 = (0, 1, 0, 1) = green
QUAD    r4, r5, 0.5     ; emit quad

skip_quad:
ADDI    r16, r16, 1     ; i++
JMP     loop_start

loop_end:
HALT
```

### Particles (Bytecode)

```asm
; State layout:
;   [0]: float4(count, gravity_x, gravity_y, dt)
;   [1..N]: Particle { pos: float2, vel: float2, lifetime: float, ... }

; Load constants
LOADI   r8, 0
LD      r9, r8, 0           ; r9 = (count, gravity_x, gravity_y, dt)
SPLAT   r10, r9, 0          ; r10 = count
SPLAT   r11, r9, 1          ; r11 = gravity_x
SPLAT   r12, r9, 2          ; r12 = gravity_y
SPLAT   r13, r9, 3          ; r13 = dt

; Calculate my particle range
DIV     r14, r10, r2        ; particles_per_thread
ADDI    r14, r14, 1
MUL     r15, r1, r14        ; start = tid * per_thread
ADD     r16, r15, r14       ; end
MIN     r16, r16, r10       ; clamp to count

; Loop over particles
MOV     r17, r15            ; i = start
particle_loop:
GE      r18, r17, r16
JNZ     r18, particle_done

; Load particle (each particle is 2 float4s: pos/vel, color/lifetime)
ADDI    r19, r17, 1         ; particle offset (skip header)
MULI    r19, r19, 2         ; 2 float4s per particle
LD      r20, r19, 0         ; r20 = (pos_x, pos_y, vel_x, vel_y)
LD      r21, r19, 1         ; r21 = (color_r, color_g, color_b, lifetime)

; Apply gravity: vel += gravity * dt
MUL     r22, r13, r11       ; dt * gravity_x
MUL     r23, r13, r12       ; dt * gravity_y
ADDI    r20.z, r20.z, r22   ; vel_x +=
ADDI    r20.w, r20.w, r23   ; vel_y +=

; Update position: pos += vel * dt
MUL     r24, r20.z, r13     ; vel_x * dt
MUL     r25, r20.w, r13     ; vel_y * dt
ADD     r20.x, r20.x, r24   ; pos_x +=
ADD     r20.y, r20.y, r25   ; pos_y +=

; Decrease lifetime
SUB     r21.w, r21.w, r13   ; lifetime -= dt

; Store back
ST      r19, r20, 0
ST      r19, r21, 1

; Emit quad
; r4.xy = pos, r4.zw = size(4,4)
MOV     r4.x, r20.x
MOV     r4.y, r20.y
LOADI   r4.z, 4
LOADI   r4.w, 4
MOV     r5, r21             ; color from particle
QUAD    r4, r5, 0.5

ADDI    r17, r17, 1         ; i++
JMP     particle_loop

particle_done:
HALT
```

## File Format

### Header (64 bytes)

```
┌─────────────────────────────────────────────────────────────────┐
│ magic: "GPUAPP" (6 bytes)                                       │
│ version: u16                                                     │
│ flags: u32                                                       │
│ code_offset: u32                                                 │
│ code_size: u32                                                   │
│ state_size: u32 (bytes needed for app state)                    │
│ vertex_budget: u32 (max vertices app can emit)                  │
│ thread_count: u32 (recommended threadgroup size)                │
│ entry_point: u32 (instruction offset for main)                  │
│ name: [u8; 24] (null-terminated app name)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Code Section

Array of 8-byte instructions following header.

## GPU Interpreter

```metal
inline void interpret_bytecode(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    uint window_count,
    constant uchar* bytecode,  // Loaded from filesystem
    uint code_size,
    uint tid,
    uint tg_size
) {
    // Register file (per-thread)
    float4 regs[32];
    regs[0] = float4(0);        // ZERO
    regs[1] = float4(tid);      // TID
    regs[2] = float4(tg_size);  // TGSIZE
    regs[3] = float4(frame);    // FRAME

    // Get state and vertex pointers
    device float4* state = (device float4*)(unified_state + app->state_offset);
    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
    uint vert_idx = 0;

    // Program counter
    uint pc = 0;
    uint max_instructions = 100000;  // Safety limit

    while (pc < code_size && max_instructions-- > 0) {
        // Fetch instruction
        device Instruction* inst = (device Instruction*)(bytecode + pc * 8);
        uint op = inst->opcode;
        uint dst = inst->dst;
        uint src1 = inst->src1;
        uint src2 = inst->src2;
        float imm = inst->imm_f;

        switch (op) {
            case OP_NOP: break;
            case OP_MOV: regs[dst] = regs[src1]; break;
            case OP_ADD: regs[dst] = regs[src1] + regs[src2]; break;
            case OP_MUL: regs[dst] = regs[src1] * regs[src2]; break;
            case OP_ADDI: regs[dst] = regs[src1] + imm; break;
            case OP_LOADI: regs[dst] = float4(imm); break;

            case OP_LD: {
                uint idx = uint(regs[src1].x) + uint(imm);
                regs[dst] = state[idx];
                break;
            }
            case OP_ST: {
                uint idx = uint(regs[src1].x) + uint(imm);
                state[idx] = regs[src2];
                break;
            }

            case OP_JMP: pc = uint(imm) - 1; break;  // -1 because pc++ at end
            case OP_JZ: if (regs[src1].x == 0) pc = uint(imm) - 1; break;
            case OP_JNZ: if (regs[src1].x != 0) pc = uint(imm) - 1; break;

            case OP_QUAD: {
                float2 pos = regs[src1].xy;
                float2 size = regs[src1].zw;
                float4 color = regs[src2];
                float depth = imm;
                write_quad(verts + vert_idx, pos, size, depth, color);
                vert_idx += 6;
                break;
            }

            case OP_HALT: goto done;

            // ... other opcodes ...
        }
        pc++;
    }

done:
    // Thread 0 commits vertex count
    if (tid == 0) {
        app->vertex_count = vert_idx;
    }
}
```

## Integration with Megakernel

```metal
inline void dispatch_app_update(...) {
    switch (app->app_type) {
        // ... existing cases ...

        case APP_TYPE_DYNAMIC:
            // Bytecode pointer is stored at start of state
            device DynamicAppHeader* header = (device DynamicAppHeader*)(unified_state + app->state_offset);
            interpret_bytecode(
                app, unified_state, unified_vertices, windows, window_count,
                header->bytecode_ptr, header->bytecode_size,
                tid, tg_size
            );
            break;
    }
}
```

## Terminal Integration

When user types `launch <app_name>`:

```metal
// In terminal_update or command handler:

// 1. Search filesystem index
uint file_idx = gpu_search_index(index, app_name, ".gpuapp");
if (file_idx == INVALID) {
    // Show error
    return;
}

// 2. Queue file load (via work queue to I/O system)
queue_load_request(index[file_idx].path, index[file_idx].size);

// 3. When load completes, allocate app slot
uint slot = allocate_app_slot();  // atomic bitmap

// 4. Initialize app with bytecode
apps[slot].app_type = APP_TYPE_DYNAMIC;
apps[slot].state_offset = allocate_state(header->state_size);
// Copy bytecode pointer into state
```

## Performance Considerations

1. **Instruction Dispatch**: Switch statement is ~10-20 cycles on GPU
2. **Register Access**: Fast (thread-local memory)
3. **State Access**: Medium (device memory, but cached)
4. **Vertex Emission**: Same cost as native apps

Expected overhead: ~10-50x slower than native Metal for pure computation.
For UI apps with mostly I/O and simple logic: acceptable.
For compute-heavy apps (particles, fractals): may need native fallback.

## Hybrid Approach

For performance-critical apps, support both:

```
my_app.gpuapp:
  - manifest.toml (metadata)
  - main.gpubc (bytecode for logic)
  - compute.metallib (optional: pre-compiled Metal for hot paths)
```

The bytecode can call into pre-compiled Metal functions via a syscall-like mechanism.

## Success Criteria

1. Game of Life runs from bytecode at 60fps
2. Particles simulation runs from bytecode at 30fps minimum
3. Terminal can discover and launch apps from ~/apps/
4. Apps load without CPU involvement (after initial I/O queue setup)
