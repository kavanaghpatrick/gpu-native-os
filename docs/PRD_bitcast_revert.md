# PRD: Revert Integer Representation to Bit-Cast Convention

**Issue**: #298 (#213 Revert)
**Priority**: P0 - Critical
**Status**: Ready for Implementation (Rev 2 - post-review)
**Date**: 2026-02-08
**Reviews**: Grok-4 (standard + adversarial), Codex/GPT-5.2 - 3 critical fixes applied

## Problem Statement

Issue #213 changed the GPU bytecode VM's integer representation from bit-cast (`as_type<float>`) to float-value (`float()`) convention. This broke:

1. **Colors**: Packed RGBA values > 2^24 lose precision (alpha channel corrupted)
2. **Positions**: Stack-allocated float arrays stored via `i32.store`/`i64.store` and read via `f32.load` produce garbage values (e.g. Y=10,569,646,080 instead of Y=240)
3. **Bitwise ops**: Large values lose low bits
4. **Branch correctness**: JZ/JNZ using float comparison fails for `-0.0f` (0x80000000) and NaN patterns
5. **Memory round-trips**: `i32.store` → `f32.load` path corrupts IEEE754-encoded floats

**Result**: WASM apps render 1 visible square instead of 5 bouncing balls. 2048 tile colors have alpha=0 (invisible).

## Decision Record

**4-round multi-AI debate** (Grok-4, Gemini, Codex/GPT-5.2, Claude) reached **unanimous consensus** on all points:

| Decision | Vote | Rationale |
|----------|------|-----------|
| Revert to bit-cast (`as_type<>`) | 3/3 | Float-value loses precision above 2^24 |
| Keep `float4` register file (not `uint4`) | 3/3 | Minimal refactor; float ops work directly |
| JZ/JNZ: use `as_type<uint>(reg.x) == 0u` | 3/3 | IEEE754: `-0.0f == +0.0f` is true; NaN != anything |
| Comparisons return `as_type<float>(1u/0u)` | 3/3 | `1.0f` = bits `0x3F800000`, wrong for integer ops |
| FTZ is moot | 3/3 | `as_type<>` bypasses float ALU entirely |
| Conversion ops bridge the two worlds | 3/3 | F32_CONVERT_I32 etc. are intentional type changes |

### Key Insight: The -0.0f Bug

```
WASM i32 value: -2147483648 (i32::MIN, common value)
Bit pattern:    0x80000000
As float:       -0.0f
IEEE754 rule:   -0.0f == +0.0f  →  TRUE

Result: JZ incorrectly branches on a NON-ZERO integer.
Fix:    as_type<uint>(reg.x) == 0u  →  0x80000000 != 0  →  FALSE (correct)
```

## Architecture

### The Two Worlds

```
┌─────────────────────────────────────────────────────┐
│              REGISTER FILE (float4)                 │
│                                                     │
│  Integer values: stored as bit-cast floats          │
│    as_type<float>(42u) = 5.885e-44 (denormal)       │
│    Bits preserved exactly: 0x0000002A               │
│                                                     │
│  Float values: stored as actual floats              │
│    3.14f stored as 3.14f                            │
│    Bits: 0x4048F5C3 (IEEE754 of 3.14)              │
└─────────────────────────────────────────────────────┘
         │                              │
    ┌────▼────┐                    ┌────▼────┐
    │ INTEGER │                    │  FLOAT  │
    │   OPS   │                    │   OPS   │
    │         │                    │         │
    │ Extract:│                    │ Direct: │
    │ as_type │                    │ reg.x + │
    │ <uint>  │                    │ reg.y   │
    │         │                    │         │
    │ Store:  │                    │ Store:  │
    │ as_type │                    │ Direct  │
    │ <float> │                    │ assign  │
    └────┬────┘                    └────┬────┘
         │                              │
    ┌────▼──────────────────────────────▼────┐
    │        CONVERSION OPS (bridge)         │
    │                                        │
    │  INT_TO_F: as_type<int>(reg.x) → float │
    │  F_TO_INT: int(reg.x) → as_type<float> │
    └────────────────────────────────────────┘
```

### Convention Rules

| Operation Type | Extract From Register | Store To Register |
|---------------|----------------------|-------------------|
| Integer arithmetic | `as_type<int>(reg.x)` | `as_type<float>(as_type<uint>(result))` |
| Unsigned arithmetic | `as_type<uint>(reg.x)` | `as_type<float>(result)` |
| Bitwise ops | `as_type<uint>(reg.x)` | `as_type<float>(result)` |
| Comparisons (int) | `as_type<int>(reg.x)` | `as_type<float>(cond ? 1u : 0u)` |
| Float arithmetic | `reg.x` (direct) | `reg.x = result` (direct) |
| Comparisons (float) | `reg.x` (direct) | `as_type<float>(cond ? 1u : 0u)` |
| JZ/JNZ | `as_type<uint>(reg.x) == 0u` | N/A |
| Load immediate int | N/A | `as_type<float>(uint_val)` |
| Load immediate float | N/A | `reg.x = float_val` (direct) |
| INT_TO_F conversion | `as_type<int>(reg.x)` | `float(int_val)` (direct) |
| F_TO_INT conversion | `int(reg.x)` (direct) | `as_type<float>(as_type<uint>(int_val))` |
| Memory (LD4/ST4) | Address: `as_type<uint>(reg.x)` | Value: `as_type<float>(word)` |

## Scope of Changes

### File: `src/gpu_os/gpu_app_system.rs` (Metal Shader)

**Total: ~45 opcode handlers need fixing**

#### Category 1: Integer Immediate Loads (2 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_LOADI_INT` | ~5989 | `float(int_val)` | `as_type<float>(as_type<uint>(int_val))` |
| `OP_LOADI_UINT` | ~5994 | `float(uint_val)` | `as_type<float>(uint_val)` |

#### Category 2: Integer Arithmetic (8 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_INT_ADD` | ~5688 | `int a = int(reg.x)` / `float(a+b)` | `as_type<int>` / `as_type<float>(as_type<uint>(a+b))` |
| `OP_INT_SUB` | ~5695 | Same pattern | Same fix |
| `OP_INT_MUL` | ~5702 | Same pattern | Same fix |
| `OP_INT_DIV_S` | ~5709 | Same pattern | Same fix |
| `OP_INT_DIV_U` | ~5721 | Same pattern | Same fix |
| `OP_INT_REM_S` | ~5733 | Same pattern | Same fix |
| `OP_INT_REM_U` | ~5747 | Same pattern | Same fix |
| `OP_INT_NEG` | ~5759 | Same pattern | Same fix |

#### Category 3: Bitwise Operations (13 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_BIT_AND` | ~5767 | `uint a = uint(reg.x)` / `float(a&b)` | `as_type<uint>` / `as_type<float>(a&b)` |
| `OP_BIT_OR` | ~5774 | Same pattern | Same fix |
| `OP_BIT_XOR` | ~5781 | Same pattern | Same fix |
| `OP_BIT_NOT` | ~5788 | Same pattern | Same fix |
| `OP_SHL` | ~5794 | Same pattern | Same fix |
| `OP_SHR_U` | ~5801 | Same pattern | Same fix |
| `OP_SHR_S` | ~5808 | Same pattern (signed) | `as_type<float>(as_type<uint>(a>>b))` |
| `OP_ROTL` | ~5815 | Same pattern | Same fix |
| `OP_ROTR` | ~5822 | Same pattern | Same fix |
| `OP_CLZ` | ~5829 | `float(clz(a))` | `as_type<float>(clz(a))` |
| `OP_CTZ` | ~5836 | `float(ctz(a))` | `as_type<float>(ctz(a))` |
| `OP_POPCNT` | ~5843 | `float(popcount(a))` | `as_type<float>(popcount(a))` |
| `OP_I32_EXTEND8_S` | check | May need fix | Check |

#### Category 4: Comparison Operations (12 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_INT_EQ` | ~5856 | `(a==b) ? 1.0f : 0.0f` | `as_type<float>(uint(a==b))` |
| `OP_INT_NE` | ~5863 | Same | Same |
| `OP_INT_LT_S` | ~5870 | Same | Same |
| `OP_INT_LT_U` | ~5877 | Same | Same |
| `OP_INT_LE_S` | ~5884 | Same | Same |
| `OP_INT_LE_U` | ~5891 | Same | Same |
| `OP_F64_EQ` | ~5035 | Same | Same |
| `OP_F64_NE` | ~5036 | Same | Same |
| `OP_F64_LT` | ~5037 | Same | Same |
| `OP_F64_GT` | ~5038 | Same | Same |
| `OP_F64_LE` | ~5039 | Same | Same |
| `OP_F64_GE` | ~5040 | Same | Same |

**Note**: `OP_INT_GT_S`, `OP_INT_GT_U`, `OP_INT_GE_S`, `OP_INT_GE_U` - check if they exist and have the same pattern.

#### Category 5: Branch Operations (2 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_JZ` | ~5279 | `regs[s1].x == 0.0f` | `as_type<uint>(regs[s1].x) == 0u` |
| `OP_JNZ` | ~5280 | `regs[s1].x != 0.0f` | `as_type<uint>(regs[s1].x) != 0u` |

#### Category 6: Load/Store Value Handling (2-3 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_LD4` | ~5384 | `float(as_type<int>(words[idx]))` | `as_type<float>(words[idx])` |
| `OP_LD` + `OP_ST` | ~5289 | Address: `uint(regs.x)` | Address: `as_type<uint>(regs.x)` |

**Note**: LD/ST address calculation currently uses `uint(regs.x)` which works in the current float-value convention. After switching to bit-cast, addresses stored via LOADI_UINT will be bit-cast floats, so the address extraction MUST also change to `as_type<uint>(regs.x)`.

#### Category 7: Misc Boolean Returns (3 handlers)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_ATOMIC_CAS` | ~6112 | `? 1.0f : 0.0f` | `as_type<float>(cond ? 1u : 0u)` |
| `OP_V4_EQ` | ~6704 | `? 1.0f : 0.0f` (x4) | `as_type<float>(uint(cond))` (x4) |
| `OP_V4_LT` | ~6717 | `? 1.0f : 0.0f` (x4) | `as_type<float>(uint(cond))` (x4) |
| `OP_SPINLOCK` | ~6828 | `? 1.0f : 0.0f` | `as_type<float>(cond ? 1u : 0u)` |

#### Category 8: F64 Conversion Ops (REVIEW FIX - previously missing)

| Opcode | Line | Current (WRONG) | Fix |
|--------|------|-----------------|-----|
| `OP_F64_FROM_I32_S` | ~5058 | `ds_from_i32(int(regs[s1].x))` | `ds_from_i32(as_type<int>(regs[s1].x))` |
| `OP_F64_FROM_I32_U` | ~5061 | `ds_from_u32(uint(regs[s1].x))` | `ds_from_u32(as_type<uint>(regs[s1].x))` |

#### Category 9: i64 Operations (REVIEW FIX - previously missing)

All i64 arithmetic/bitwise/comparison handlers that extract i32 values from registers also need updating. These include any handler that reads `int(regs[s].x)` or `uint(regs[s].x)` for 64-bit operations. Audit all `OP_I64_*` and `OP_INT64_*` handlers.

### File: `wasm_translator/src/translate.rs`

| Location | Current (WRONG) | Fix |
|----------|-----------------|-----|
| `I64Const` ~line 1289 | `loadi_uint(dst, low)` | `loadi(dst, 0.0); setx(dst, f32::from_bits(low)); sety(dst, f32::from_bits(high))` |

**Already fixed in this session** - verify it's committed.

### File: `wasm_translator/src/emit.rs`

No changes needed - emitter methods pass through to assembler.

## Post-Review Fixes (Rev 2)

Three critical issues found during review by Grok-4 and Codex/GPT-5.2:

### REVIEW FIX 1: Immediate Encoding Mismatch (Codex - CRITICAL)

**The assembler encodes branch targets and memory offsets as float VALUES** (e.g., `42 as f32`). The shader reads them via `imm` (float) or `imm_bits` (raw uint bits of the float).

**RULE**:
- **Register values**: Use `as_type<uint>(regs[s].x)` to extract integer bits
- **Immediate values**: Keep `uint(imm)` for float-encoded immediates (branch targets, LD/ST offsets)
- **Immediate raw bits**: Use `imm_bits` only where the assembler stores raw uint bits (LOADI_UINT, SETX, etc.)

Violating this rule would cause branch target `42` (encoded as `42.0f`, bits `0x42280000`) to be read as uint `1,110,441,984` - catastrophic control flow corruption.

### REVIEW FIX 2: Missing ST1/ST2 Store Values + LD_RGBA Address (Codex - CRITICAL)

Added to Category 6. All store ops must extract values via `as_type<uint>()`:
- `OP_ST1`: `uchar(regs[s2].x)` → `uchar(as_type<uint>(regs[s2].x))`
- `OP_ST2`: `ushort(uint(regs[s2].x))` → `ushort(as_type<uint>(regs[s2].x))`
- `OP_LD_RGBA` address: `uint(regs[s1].x)` → `as_type<uint>(regs[s1].x)`

### REVIEW FIX 3: Missing F64 Conversion Ops (Codex - CRITICAL)

`OP_F64_FROM_I32_S` and `OP_F64_FROM_I32_U` extract integers from registers but were not in scope. Added as Category 8.

### REVIEW NOTE: i64 Opcodes (Grok)

i64 arithmetic/bitwise ops were not individually audited. Added as Category 9 for completeness. Most i64 ops work on the `.xy` components via `as_type<ulong>()` which already does bit-cast, so they may be correct. Needs verification during implementation.

## What NOT to Change

These are **CORRECT** and must stay as-is:

| Operation | Why Correct |
|-----------|-------------|
| `OP_INT_TO_F` / `OP_UINT_TO_F` | Intentional int→float conversion |
| `OP_F_TO_INT` / `OP_F_TO_UINT` | Intentional float→int conversion |
| Float arithmetic (FADD, FMUL, etc.) | Operate on actual float values |
| Float comparisons (FEQ, FLT, etc.) | Compare actual float values (but result still needs bit-cast 1u/0u) |
| `OP_LOADI` | Loads actual float immediate |
| `OP_SETX/SETY/SETZ/SETW` | Already uses `as_type<float>(imm_bits)` |
| `OP_LOADI_RGBA` | New opcode, already correct |
| `OP_LD_RGBA` | New opcode, already correct |
| ~~`OP_ST1`~~ | ~~Byte store~~ **MOVED TO SCOPE** - needs `uchar(as_type<uint>(reg.x))` |
| ~~`OP_LD1`~~ | ~~Byte load~~ **MOVED TO SCOPE** - needs `as_type<float>(uint(byte))` |

**Important note on OP_LD1**: After the bit-cast revert, `OP_LD1` loading a byte and storing via `float(byte_val)` is still correct IF the byte is used as an integer input to other ops. But since we're switching to bit-cast convention everywhere, it should arguably use `as_type<float>(uint(byte_val))` for consistency. However, for values 0-255, `float(x)` and `as_type<float>(uint(x))` produce different results - `float(5)` = 5.0f (bits 0x40A00000) vs `as_type<float>(5u)` = denormal (bits 0x00000005). **LD1 must use `as_type<float>(uint(byte_val))` in the new convention** since subsequent integer ops will use `as_type<uint>()` to extract.

**Correction**: OP_LD1 and OP_LD2 MUST also be fixed:
- `OP_LD1`: `regs[d].x = as_type<float>(uint(bytes[idx]));`
- `OP_LD2`: `regs[d].x = as_type<float>(uint(halfwords[idx]));`

## Pre-Implementation: NaN Payload Test

Before starting, run this test to confirm Apple Silicon preserves NaN payloads through float4 storage:

```metal
kernel void test_nan_payload(device uint* out [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    // Store NaN with custom payload
    float4 reg;
    reg.x = as_type<float>(0x7FC00001u);  // quiet NaN, payload=1
    reg.y = as_type<float>(0x7FBFFFFFu);  // signaling NaN, max payload
    reg.z = as_type<float>(0xFFFFFFFFu);  // negative NaN, all bits
    reg.w = as_type<float>(0x80000000u);  // -0.0f

    // Round-trip through device memory
    device float4* mem = (device float4*)(out + 16);
    mem[0] = reg;
    float4 loaded = mem[0];

    // Check bit-exact preservation
    out[0] = as_type<uint>(loaded.x);  // expect 0x7FC00001
    out[1] = as_type<uint>(loaded.y);  // expect 0x7FBFFFFF
    out[2] = as_type<uint>(loaded.z);  // expect 0xFFFFFFFF
    out[3] = as_type<uint>(loaded.w);  // expect 0x80000000
}
```

**Expected**: All 4 values preserved exactly. If any differ, we need the `uint4` register file (Option B from the debate).

## Implementation Order

Execute all changes in a single pass (per debate consensus). The mechanical pattern is:

### Step 1: Fix LOADI_UINT and LOADI_INT
```metal
// BEFORE:
case OP_LOADI_UINT: { uint v = imm_bits; regs[d] = float4(float(v), 0,0,0); break; }
case OP_LOADI_INT:  { int v = as_type<int>(imm_bits); regs[d] = float4(float(v), 0,0,0); break; }

// AFTER:
case OP_LOADI_UINT: { regs[d] = float4(as_type<float>(imm_bits), 0,0,0); break; }
case OP_LOADI_INT:  { regs[d] = float4(as_type<float>(imm_bits), 0,0,0); break; }
```

### Step 2: Fix all integer arithmetic (INT_ADD through INT_NEG)
```metal
// BEFORE:
case OP_INT_ADD: { int a = int(regs[s1].x); int b = int(regs[s2].x); regs[d].x = float(a + b); break; }

// AFTER:
case OP_INT_ADD: { int a = as_type<int>(regs[s1].x); int b = as_type<int>(regs[s2].x); regs[d].x = as_type<float>(as_type<uint>(a + b)); break; }
```

### Step 3: Fix all bitwise ops (BIT_AND through POPCNT)
```metal
// BEFORE:
case OP_BIT_AND: { uint a = uint(regs[s1].x); uint b = uint(regs[s2].x); regs[d].x = float(a & b); break; }

// AFTER:
case OP_BIT_AND: { uint a = as_type<uint>(regs[s1].x); uint b = as_type<uint>(regs[s2].x); regs[d].x = as_type<float>(a & b); break; }
```

### Step 4: Fix all comparisons (INT_EQ through F64_GE)
```metal
// BEFORE:
case OP_INT_EQ: { int a = int(regs[s1].x); int b = int(regs[s2].x); regs[d].x = (a==b) ? 1.0f : 0.0f; break; }

// AFTER:
case OP_INT_EQ: { int a = as_type<int>(regs[s1].x); int b = as_type<int>(regs[s2].x); regs[d].x = as_type<float>(uint(a == b)); break; }
```

### Step 5: Fix JZ/JNZ
```metal
// BEFORE:
case OP_JZ:  if (regs[s1].x == 0.0f) pc = uint(imm) - 1; break;
case OP_JNZ: if (regs[s1].x != 0.0f) pc = uint(imm) - 1; break;

// AFTER:
// CRITICAL: Keep uint(imm) for branch targets! The assembler encodes targets as float VALUES.
// Only the CONDITION register needs as_type<uint>().
case OP_JZ:  if (as_type<uint>(regs[s1].x) == 0u) pc = uint(imm) - 1; break;
case OP_JNZ: if (as_type<uint>(regs[s1].x) != 0u) pc = uint(imm) - 1; break;
```

### Step 6: Fix LD/ST address extraction and LD4/LD1/LD2 values
```metal
// LD/ST addresses from REGISTERS need as_type<uint>():
// BEFORE: uint idx = uint(regs[s1].x) + uint(imm);
// AFTER:  uint idx = as_type<uint>(regs[s1].x) + uint(imm);
// CRITICAL: Keep uint(imm) for immediate offsets! Assembler encodes offsets as float VALUES.
// Only REGISTER-sourced addresses change to as_type<uint>().

// LD4 value:
// BEFORE: regs[d] = float4(float(as_type<int>(words[idx])), 0,0,0);
// AFTER:  regs[d] = float4(as_type<float>(words[idx]), 0,0,0);

// LD1/LD2 values:
// BEFORE: regs[d] = float4(float(bytes[idx]), 0,0,0);
// AFTER:  regs[d] = float4(as_type<float>(uint(bytes[idx])), 0,0,0);

// ST1/ST2 values (REVIEW FIX - previously missing):
// BEFORE: bytes[idx] = uchar(regs[s2].x);
// AFTER:  bytes[idx] = uchar(as_type<uint>(regs[s2].x));
// BEFORE: halfwords[idx] = ushort(uint(regs[s2].x));
// AFTER:  halfwords[idx] = ushort(as_type<uint>(regs[s2].x));
```

### Step 7: Fix misc boolean returns (ATOMIC_CAS, V4_EQ, V4_LT, SPINLOCK)

### Step 8: Remove workarounds
- Remove `OP_LOADI_RGBA` (no longer needed - LOADI_UINT preserves bits)
- Remove heuristic alpha fix in OP_QUAD
- Remove dual-path color handling in OP_QUAD
- Remove `last_const` tracking in translator
- Simplify OP_QUAD back to single path: `uint packed = as_type<uint>(regs[d].x);`

## Testing Plan

### Test 1: NaN Payload Preservation (pre-implementation)
Run the Metal shader test above. Must pass before proceeding.

### Test 2: Existing Unit Tests
```bash
cargo test
```
All existing tests must still pass.

### Test 3: Visual WASM Apps
```bash
cargo run --example visual_wasm_apps
```
Verify each app (keys 1-8):
- **Bouncing Balls**: 5 colored rectangles bouncing, correct positions AND colors
- **2048**: Board with colored tiles, all visible (alpha=1)
- **Snake**: Grid with colored segments
- **Mandelbrot**: Colored fractal pattern
- **Particles**: Moving colored particles
- **Clock**: Animated display
- **Pong**: Two paddles + ball
- **Game of Life**: Grid cells

### Test 4: Edge Case Values
Create a test WASM app that exercises:
- `i32::MIN` (-2147483648 = 0x80000000) through JZ → must NOT branch
- `0x7FC00000` (NaN pattern as integer) through JNZ → MUST branch
- `0xFFFFFFFF` through BIT_AND → must preserve all bits
- `0xFF0000FF` through emit_quad → must render red with alpha=1
- `i32.store(IEEE754_of_620.0)` then `f32.load` → must get 620.0

### Test 5: Denormal Address Test
Verify address 8192 (stack pointer) works through bit-cast:
- `as_type<float>(8192u)` = denormal
- `as_type<uint>(denormal)` = 8192 (round-trip)

## Rollback Plan

If the bit-cast revert causes unexpected failures:
1. The NaN payload test catches hardware-level issues before implementation
2. Git revert to pre-change commit
3. Fall back to Option B (uint4 register file) which eliminates all float-domain concerns

## Success Criteria

1. All 11 WASM apps render with correct colors AND correct positions
2. `cargo test` passes
3. NaN payload test passes
4. No regression in frame rate (as_type is zero-cost)
5. Zero workarounds/heuristics needed (LOADI_RGBA, alpha fix removed)
