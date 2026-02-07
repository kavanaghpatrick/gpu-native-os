// Plasma - Classic demoscene plasma effect
//
// Creates swirling, colorful plasma patterns using sine wave interference.
// Move the mouse to shift the color palette.

struct PlasmaVertex {
    float4 position;
    float2 uv;
    float2 cursor;
    float time;
    float _pad[3];
};

kernel void plasma_compute(
    constant FrameState& frame [[buffer(0)]],
    device PlasmaVertex* vertices [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < 6) {
        float2 positions[6] = {
            float2(-1, -1), float2(1, -1), float2(-1, 1),
            float2(-1, 1), float2(1, -1), float2(1, 1)
        };
        float2 uvs[6] = {
            float2(0, 0), float2(1, 0), float2(0, 1),
            float2(0, 1), float2(1, 0), float2(1, 1)
        };

        vertices[tid].position = float4(positions[tid], 0, 1);
        vertices[tid].uv = uvs[tid];
        vertices[tid].cursor = float2(frame.cursor_x, frame.cursor_y);
        vertices[tid].time = frame.time;
    }
}

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float2 cursor;
    float time;
};

vertex VertexOut plasma_vertex(
    const device PlasmaVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    PlasmaVertex v = vertices[vid];
    VertexOut out;
    out.position = v.position;
    out.uv = v.uv;
    out.cursor = v.cursor;
    out.time = v.time;
    return out;
}

fragment float4 plasma_fragment(VertexOut in [[stage_in]]) {
    float2 uv = in.uv;
    float2 cursor = in.cursor;
    float time = in.time;

    // Scale UV for more detail
    float2 p = uv * 8.0;

    // Classic plasma formula with multiple sine waves
    float v1 = sin(p.x + time);
    float v2 = sin(p.y + time * 0.7);
    float v3 = sin(p.x + p.y + time * 0.5);
    float v4 = sin(length(p - float2(4.0, 4.0)) + time);

    // Cursor influence - creates a swirl effect
    float2 toCenter = uv - cursor;
    float dist = length(toCenter);
    float v5 = sin(dist * 10.0 - time * 2.0) * (1.0 - smoothstep(0.0, 0.5, dist));

    float plasma = (v1 + v2 + v3 + v4 + v5) * 0.2 + 0.5;

    // Create vibrant colors from plasma value
    float hue = plasma * 360.0 + time * 30.0 + cursor.x * 180.0;
    float saturation = 0.8 + 0.2 * sin(time + plasma * 3.14159);
    float brightness = 0.6 + 0.4 * plasma;

    float3 rgb = hsv_to_rgb(hue, saturation, brightness);
    return float4(rgb, 1.0);
}
