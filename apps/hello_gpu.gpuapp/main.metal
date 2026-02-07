// Hello GPU - A simple dynamic app demo
//
// Shows a fullscreen gradient that responds to mouse position and time.
// Uses FrameState from APP_SHADER_HEADER.

kernel void hello_compute(
    device FrameState& frame [[buffer(0)]],
    device float4* vertices [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < 6) {
        // Fullscreen quad positions
        float2 positions[6] = {
            float2(-1, -1), float2(1, -1), float2(-1, 1),
            float2(-1, 1), float2(1, -1), float2(1, 1)
        };

        // UV coordinates for color interpolation
        float2 uvs[6] = {
            float2(0, 0), float2(1, 0), float2(0, 1),
            float2(0, 1), float2(1, 0), float2(1, 1)
        };

        // Store position in first 6 vertices
        vertices[tid] = float4(positions[tid], 0, 1);

        // Store UV in next 6 vertices (used for color)
        vertices[tid + 6] = float4(uvs[tid], frame.cursor_x, frame.cursor_y);

        // Store time-based color modifier
        float t = frame.time;
        float r = sin(t * 2.0) * 0.5 + 0.5;
        float g = sin(t * 2.0 + 2.094) * 0.5 + 0.5;  // 2.094 = 2pi/3
        float b = sin(t * 2.0 + 4.189) * 0.5 + 0.5;  // 4.189 = 4pi/3
        vertices[tid + 12] = float4(r, g, b, 1.0);
    }
}

struct VertexOut {
    float4 position [[position]];
    float4 uv_cursor;
    float4 time_color;
};

vertex VertexOut hello_vertex(
    device float4* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = vertices[vid];
    out.uv_cursor = vertices[vid + 6];
    out.time_color = vertices[vid + 12];
    return out;
}

fragment float4 hello_fragment(VertexOut in [[stage_in]]) {
    float2 uv = in.uv_cursor.xy;
    float2 cursor = in.uv_cursor.zw;
    float4 time_color = in.time_color;

    // Base gradient from UV
    float3 color = float3(uv.x, uv.y, 0.5);

    // Add cursor influence - brighter near cursor
    float dist = length(uv - cursor);
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);
    color = mix(color, float3(1.0, 1.0, 1.0), glow * 0.5);

    // Mix in time-based color
    color = mix(color, time_color.rgb, 0.3);

    return float4(color, 1.0);
}
