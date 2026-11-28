// Front to back blending of depth-sorted splat lists over 16 x 16 pixel tiles, and its gradient.
//
// Per splat: mean2d (x, y), conic (a, b, c), opacity, shade (r, g, b), depth. Per tile: the start
// and length of its run in `owner`, which lists splat indices tile by tile, nearest first.
// Images are row-major: colour (H, W, 3), cover and depth (H, W).

static const uint TILE = 16;
static const uint BLOCK = TILE * TILE;
static const float MOST_ALPHA = 0.99;
static const float LEAST_ALPHA = 1.0 / 255.0;
static const float LEAST_THROUGH = 1e-4;

groupshared float2 s_mean[BLOCK];
groupshared float3 s_conic[BLOCK];
groupshared float s_opacity[BLOCK];
groupshared float3 s_shade[BLOCK];
groupshared float s_depth[BLOCK];
groupshared int s_index[BLOCK];
groupshared uint s_count;

// Opacity a splat lends a pixel, with its Gaussian falloff and the offset from its centre.
float splat_alpha(float2 mean, float3 conic, float opacity, float2 pixel, out float falloff, out float2 offset)
{
    offset = pixel - mean;
    float power = -0.5 * (conic.x * offset.x * offset.x + conic.z * offset.y * offset.y) - conic.y * offset.x * offset.y;
    falloff = exp(min(power, 0.0));
    float alpha = min(opacity * falloff, MOST_ALPHA);
    return alpha < LEAST_ALPHA ? 0.0 : alpha;
}

void load(uint slot, int index, float* mean2d, float* conic, float* opacity, float* shade, float* depth)
{
    s_index[slot] = index;
    s_mean[slot] = float2(mean2d[2 * index], mean2d[2 * index + 1]);
    s_conic[slot] = float3(conic[3 * index], conic[3 * index + 1], conic[3 * index + 2]);
    s_opacity[slot] = opacity[index];
    s_shade[slot] = float3(shade[3 * index], shade[3 * index + 1], shade[3 * index + 2]);
    s_depth[slot] = depth[index];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blend_forward(
    uint3 group: SV_GroupID,
    uint3 local: SV_GroupThreadID,
    uniform float* mean2d,
    uniform float* conic,
    uniform float* opacity,
    uniform float* shade,
    uniform float* depth,
    uniform int* owner,
    uniform int* tile_start,
    uniform int* tile_count,
    uniform float* out_colour,
    uniform float* out_cover,
    uniform float* out_depth,
    uniform float* out_through,
    uniform int* out_last,
    uniform uint width,
    uniform uint height,
    uniform uint tiles_x)
{
    uint tile = group.y * tiles_x + group.x;
    uint thread = local.y * TILE + local.x;
    uint px = group.x * TILE + local.x;
    uint py = group.y * TILE + local.y;
    bool inside = px < width && py < height;
    float2 pixel = float2(px + 0.5, py + 0.5);
    int start = tile_start[tile];
    int count = tile_count[tile];

    float through = 1.0;
    float3 colour = float3(0.0);
    float depth_sum = 0.0;
    int last = 0;
    bool done = !inside;

    for (int base = 0; base < count; base += int(BLOCK))
    {
        if (thread == 0)
            s_count = 0;
        GroupMemoryBarrierWithGroupSync();
        if (done)
            InterlockedAdd(s_count, 1u);
        int fetch = base + int(thread);
        if (fetch < count)
            load(thread, owner[start + fetch], mean2d, conic, opacity, shade, depth);
        GroupMemoryBarrierWithGroupSync();
        if (s_count == BLOCK)
            break;
        int batch = min(int(BLOCK), count - base);
        for (int k = 0; k < batch && !done; k++)
        {
            float falloff;
            float2 offset;
            float alpha = splat_alpha(s_mean[k], s_conic[k], s_opacity[k], pixel, falloff, offset);
            if (alpha == 0.0)
                continue;
            float next = through * (1.0 - alpha);
            if (next < LEAST_THROUGH)
            {
                done = true;
                break;
            }
            float weight = alpha * through;
            colour += weight * s_shade[k];
            depth_sum += weight * s_depth[k];
            through = next;
            last = base + k + 1;
        }
    }
    if (inside)
    {
        uint at = py * width + px;
        out_colour[3 * at] = colour.x;
        out_colour[3 * at + 1] = colour.y;
        out_colour[3 * at + 2] = colour.z;
        out_cover[at] = 1.0 - through;
        out_depth[at] = depth_sum;
        out_through[at] = through;
        out_last[at] = last;
    }
}

groupshared int s_deepest;

// Sums a gradient across the wave and adds it once to `target`.
void spread(float value, float* target, bool lead)
{
    float total = WaveActiveSum(value);
    if (lead && total != 0.0)
        InterlockedAdd(*target, total);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blend_backward(
    uint3 group: SV_GroupID,
    uint3 local: SV_GroupThreadID,
    uniform float* mean2d,
    uniform float* conic,
    uniform float* opacity,
    uniform float* shade,
    uniform float* depth,
    uniform int* owner,
    uniform int* tile_start,
    uniform int* tile_count,
    uniform float* in_through,
    uniform int* in_last,
    uniform float* grad_colour,
    uniform float* grad_cover,
    uniform float* grad_depth,
    uniform float* d_mean2d,
    uniform float* d_conic,
    uniform float* d_opacity,
    uniform float* d_shade,
    uniform float* d_depth,
    uniform uint width,
    uniform uint height,
    uniform uint tiles_x)
{
    uint tile = group.y * tiles_x + group.x;
    uint thread = local.y * TILE + local.x;
    uint px = group.x * TILE + local.x;
    uint py = group.y * TILE + local.y;
    bool inside = px < width && py < height;
    float2 pixel = float2(px + 0.5, py + 0.5);
    int start = tile_start[tile];
    uint at = inside ? py * width + px : 0;

    float through = inside ? in_through[at] : 1.0;
    int last = inside ? in_last[at] : 0;
    float3 g_colour = inside ? float3(grad_colour[3 * at], grad_colour[3 * at + 1], grad_colour[3 * at + 2]) : float3(0.0);
    float g_cover = inside ? grad_cover[at] : 0.0;
    float g_depth = inside ? grad_depth[at] : 0.0;
    float behind = 0.0;
    bool lead = WaveIsFirstLane();

    if (thread == 0)
        s_deepest = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(s_deepest, last);
    GroupMemoryBarrierWithGroupSync();
    int deepest = s_deepest;

    for (int top = deepest; top > 0; top -= int(BLOCK))
    {
        int base = max(0, top - int(BLOCK));
        GroupMemoryBarrierWithGroupSync();
        int fetch = base + int(thread);
        if (fetch < top)
            load(thread, owner[start + fetch], mean2d, conic, opacity, shade, depth);
        GroupMemoryBarrierWithGroupSync();
        for (int k = top - base - 1; k >= 0; k--)
        {
            int position = base + k;
            float g_mx = 0.0, g_my = 0.0, g_ca = 0.0, g_cb = 0.0, g_cc = 0.0, g_op = 0.0;
            float3 g_sh = float3(0.0);
            float g_dp = 0.0;
            if (position < last)
            {
                float falloff;
                float2 offset;
                float alpha = splat_alpha(s_mean[k], s_conic[k], s_opacity[k], pixel, falloff, offset);
                if (alpha > 0.0)
                {
                    through = through / (1.0 - alpha);
                    float weight = alpha * through;
                    g_sh = weight * g_colour;
                    g_dp = weight * g_depth;
                    float value = dot(s_shade[k], g_colour) + s_depth[k] * g_depth + g_cover;
                    if (alpha < MOST_ALPHA)
                    {
                        float d_alpha = through * value - behind / (1.0 - alpha);
                        g_op = d_alpha * falloff;
                        float d_power = d_alpha * s_opacity[k] * falloff;
                        float3 co = s_conic[k];
                        g_ca = -0.5 * d_power * offset.x * offset.x;
                        g_cb = -d_power * offset.x * offset.y;
                        g_cc = -0.5 * d_power * offset.y * offset.y;
                        g_mx = d_power * (co.x * offset.x + co.y * offset.y);
                        g_my = d_power * (co.z * offset.y + co.y * offset.x);
                    }
                    behind += weight * value;
                }
            }
            if (WaveActiveAnyTrue(g_op != 0.0 || g_sh.x != 0.0 || g_sh.y != 0.0 || g_sh.z != 0.0 || g_dp != 0.0))
            {
                int index = s_index[k];
                spread(g_mx, &d_mean2d[2 * index], lead);
                spread(g_my, &d_mean2d[2 * index + 1], lead);
                spread(g_ca, &d_conic[3 * index], lead);
                spread(g_cb, &d_conic[3 * index + 1], lead);
                spread(g_cc, &d_conic[3 * index + 2], lead);
                spread(g_op, &d_opacity[index], lead);
                spread(g_sh.x, &d_shade[3 * index], lead);
                spread(g_sh.y, &d_shade[3 * index + 1], lead);
                spread(g_sh.z, &d_shade[3 * index + 2], lead);
                spread(g_dp, &d_depth[index], lead);
            }
        }
    }
}
