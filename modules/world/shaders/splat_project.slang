// Projection of splats into a camera, their view-dependent colour, and the gradient of both.
//
// Per splat: mean (3), scale (3, linear), quaternion (w, x, y, z), opacity, and `bands` spherical
// harmonic colours (3 each). The camera is an eye (3) and rows (3 x 3, row-major): the picture's
// right, down and forward directions. Outputs follow the inputs of splat_blend.slang.

static const float DILATION = 0.3;
static const float LEAST_ALPHA = 1.0 / 255.0;
static const float SH_C0 = 0.28209479177387814;
static const float SH_C1 = 0.4886025119029199;
static const float SH_C2[5] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
                                0.5462742152960396 };
static const float SH_C3[7] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                                -0.4570457994644658, 1.445305721320277, -0.5900435899266435 };

struct Projected : IDifferentiable
{
    float2 mean2d;
    float3 conic;
    float depth;
    float3 shade;
    float a;
    float b;
    float c;
};

struct Camera
{
    float focal;
    float2 centre;
    float2 limit;
    int bands;
};

[Differentiable]
float3 sh_colour(Array<float3, 16> sh, float3 d, no_diff int bands)
{
    float3 result = SH_C0 * sh[0];
    if (bands >= 4)
    {
        float x = d.x, y = d.y, z = d.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        if (bands >= 9)
        {
            float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
            result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                     + SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];
            if (bands >= 16)
            {
                result = result + SH_C3[0] * y * (3.0 * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10]
                         + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                         + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                         + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14]
                         + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15];
            }
        }
    }
    return max(result + 0.5, float3(0.0));
}

[Differentiable]
Projected project(float3 mean, float3 scale, float4 quat, Array<float3, 16> sh, float3 eye, float3x3 rows,
                  no_diff Camera camera)
{
    float3 offset = mean - eye;
    float3 local = mul(rows, offset);
    float z = local.z;
    float tx = clamp(local.x / z, -camera.limit.x, camera.limit.x) * z;
    float ty = clamp(local.y / z, -camera.limit.y, camera.limit.y) * z;

    float4 q = quat / max(length(quat), 1e-12);
    float w = q.x, x = q.y, y = q.z, qz = q.w;
    float3x3 turn = float3x3(
        1.0 - 2.0 * (y * y + qz * qz), 2.0 * (x * y - w * qz), 2.0 * (x * qz + w * y),
        2.0 * (x * y + w * qz), 1.0 - 2.0 * (x * x + qz * qz), 2.0 * (y * qz - w * x),
        2.0 * (x * qz - w * y), 2.0 * (y * qz + w * x), 1.0 - 2.0 * (x * x + y * y));
    float3x3 shape = float3x3(turn[0] * scale, turn[1] * scale, turn[2] * scale);
    float3x3 world = mul(shape, transpose(shape));
    float3x3 s = mul(mul(rows, world), transpose(rows));

    float f = camera.focal;
    float j0 = f / z;
    float j2 = -f * tx / (z * z);
    float j5 = -f * ty / (z * z);
    float a = j0 * j0 * s[0][0] + 2.0 * j0 * j2 * s[0][2] + j2 * j2 * s[2][2] + DILATION;
    float b = j0 * j0 * s[0][1] + j0 * j5 * s[0][2] + j2 * j0 * s[1][2] + j2 * j5 * s[2][2];
    float c = j0 * j0 * s[1][1] + 2.0 * j0 * j5 * s[1][2] + j5 * j5 * s[2][2] + DILATION;
    float det = max(a * c - b * b, 1e-12);

    Projected result;
    result.conic = float3(c / det, -b / det, a / det);
    result.mean2d = float2(f * local.x / z + camera.centre.x, f * local.y / z + camera.centre.y);
    result.depth = z;
    result.shade = sh_colour(sh, offset / max(length(offset), 1e-12), camera.bands);
    result.a = a;
    result.b = b;
    result.c = c;
    return result;
}

Array<float3, 16> load_sh(float* sh, uint index, int bands)
{
    Array<float3, 16> result;
    for (int k = 0; k < 16; k++)
    {
        uint at = (index * uint(bands) + uint(k)) * 3;
        result[k] = k < bands ? float3(sh[at], sh[at + 1], sh[at + 2]) : float3(0.0);
    }
    return result;
}

float3x3 load_rows(float* rows)
{
    return float3x3(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], rows[8]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void project_forward(
    uint3 tid: SV_DispatchThreadID,
    uniform float* means,
    uniform float* scales,
    uniform float* quats,
    uniform float* opacities,
    uniform float* sh,
    uniform float* eye,
    uniform float* rows,
    uniform float* out_mean2d,
    uniform float* out_conic,
    uniform float* out_depth,
    uniform float* out_shade,
    uniform int* out_rect,
    uniform uint count,
    uniform float focal,
    uniform float width,
    uniform float height,
    uniform float near,
    uniform float widest,
    uniform int bands,
    uniform int tile,
    uniform int tiles_x,
    uniform int tiles_y)
{
    uint i = tid.x;
    if (i >= count)
        return;
    float3 mean = float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    float3 e = float3(eye[0], eye[1], eye[2]);
    float3x3 r = load_rows(rows);
    float z = dot(r[2], mean - e);
    out_rect[4 * i + 2] = 0;
    out_rect[4 * i + 3] = 0;
    out_depth[i] = z;
    if (!(z > near))
    {
        out_mean2d[2 * i] = 0.0; out_mean2d[2 * i + 1] = 0.0;
        out_conic[3 * i] = 0.0; out_conic[3 * i + 1] = 0.0; out_conic[3 * i + 2] = 0.0;
        out_shade[3 * i] = 0.0; out_shade[3 * i + 1] = 0.0; out_shade[3 * i + 2] = 0.0;
        return;
    }
    Camera camera;
    camera.focal = focal;
    camera.centre = float2(width * 0.5, height * 0.5);
    camera.limit = float2(1.3 * camera.centre.x / focal, 1.3 * camera.centre.y / focal);
    camera.bands = bands;
    float3 scale = float3(scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]);
    float4 quat = float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
    Projected p = project(mean, scale, quat, load_sh(sh, i, bands), e, r, camera);
    out_mean2d[2 * i] = p.mean2d.x; out_mean2d[2 * i + 1] = p.mean2d.y;
    out_conic[3 * i] = p.conic.x; out_conic[3 * i + 1] = p.conic.y; out_conic[3 * i + 2] = p.conic.z;
    out_shade[3 * i] = p.shade.x; out_shade[3 * i + 1] = p.shade.y; out_shade[3 * i + 2] = p.shade.z;

    // Tiles the splat reaches, each axis out to where it lends a pixel the least counted opacity.
    float det = max(p.a * p.c - p.b * p.b, 1e-12);
    float middle = 0.5 * (p.a + p.c);
    float largest = middle + sqrt(max(middle * middle - det, 0.1));
    float radius = ceil(3.0 * sqrt(largest));
    float reach = 2.0 * log(max(opacities[i] / LEAST_ALPHA, 1.0));
    float rx = sqrt(reach * p.a);
    float ry = sqrt(reach * p.c);
    float u = p.mean2d.x, v = p.mean2d.y;
    bool seen = u + rx >= 0.0 && u - rx < width && v + ry >= 0.0 && v - ry < height && reach > 0.0
                && isfinite(u) && isfinite(v) && (widest <= 0.0 || radius <= widest);
    if (!seen)
        return;
    int x0 = clamp(int(floor((u - rx) / tile)), 0, tiles_x - 1);
    int x1 = clamp(int(floor((u + rx) / tile)), 0, tiles_x - 1);
    int y0 = clamp(int(floor((v - ry) / tile)), 0, tiles_y - 1);
    int y1 = clamp(int(floor((v + ry) / tile)), 0, tiles_y - 1);
    out_rect[4 * i] = x0;
    out_rect[4 * i + 1] = y0;
    out_rect[4 * i + 2] = x1 - x0 + 1;
    out_rect[4 * i + 3] = y1 - y0 + 1;
}

void spread_camera(float value, float* target)
{
    float total = WaveActiveSum(value);
    if (WaveIsFirstLane() && total != 0.0)
        InterlockedAdd(*target, total);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void project_backward(
    uint3 tid: SV_DispatchThreadID,
    uniform float* means,
    uniform float* scales,
    uniform float* quats,
    uniform float* sh,
    uniform float* eye,
    uniform float* rows,
    uniform float* in_depth,
    uniform float* grad_mean2d,
    uniform float* grad_conic,
    uniform float* grad_depth,
    uniform float* grad_shade,
    uniform float* d_means,
    uniform float* d_scales,
    uniform float* d_quats,
    uniform float* d_sh,
    uniform float* d_eye,
    uniform float* d_rows,
    uniform uint count,
    uniform float focal,
    uniform float width,
    uniform float height,
    uniform float near,
    uniform int bands,
    uniform int camera_grad)
{
    uint i = tid.x;
    bool live = i < count && in_depth[min(i, count - 1)] > near;
    float3 g_eye = float3(0.0);
    float3x3 g_rows = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (live)
    {
        Camera camera;
        camera.focal = focal;
        camera.centre = float2(width * 0.5, height * 0.5);
        camera.limit = float2(1.3 * camera.centre.x / focal, 1.3 * camera.centre.y / focal);
        camera.bands = bands;
        var mean = diffPair(float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]));
        var scale = diffPair(float3(scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]));
        var quat = diffPair(float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]));
        var colours = diffPair(load_sh(sh, i, bands));
        var e = diffPair(float3(eye[0], eye[1], eye[2]));
        var r = diffPair(load_rows(rows));
        Projected.Differential d;
        d.mean2d = float2(grad_mean2d[2 * i], grad_mean2d[2 * i + 1]);
        d.conic = float3(grad_conic[3 * i], grad_conic[3 * i + 1], grad_conic[3 * i + 2]);
        d.depth = grad_depth[i];
        d.shade = float3(grad_shade[3 * i], grad_shade[3 * i + 1], grad_shade[3 * i + 2]);
        d.a = 0.0;
        d.b = 0.0;
        d.c = 0.0;
        bwd_diff(project)(mean, scale, quat, colours, e, r, camera, d);
        d_means[3 * i] = mean.d.x; d_means[3 * i + 1] = mean.d.y; d_means[3 * i + 2] = mean.d.z;
        d_scales[3 * i] = scale.d.x; d_scales[3 * i + 1] = scale.d.y; d_scales[3 * i + 2] = scale.d.z;
        d_quats[4 * i] = quat.d.x; d_quats[4 * i + 1] = quat.d.y; d_quats[4 * i + 2] = quat.d.z;
        d_quats[4 * i + 3] = quat.d.w;
        for (int k = 0; k < bands; k++)
        {
            uint at = (i * uint(bands) + uint(k)) * 3;
            d_sh[at] = colours.d[k].x; d_sh[at + 1] = colours.d[k].y; d_sh[at + 2] = colours.d[k].z;
        }
        g_eye = e.d;
        g_rows = r.d;
    }
    if (camera_grad != 0)
    {
        spread_camera(g_eye.x, &d_eye[0]);
        spread_camera(g_eye.y, &d_eye[1]);
        spread_camera(g_eye.z, &d_eye[2]);
        for (int k = 0; k < 9; k++)
            spread_camera(g_rows[k / 3][k % 3], &d_rows[k]);
    }
}
