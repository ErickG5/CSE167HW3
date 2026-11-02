import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);
    float aspectRatio = float(width) / float(height);

    float pxPlacementX = float(tid.x + 0.5) / float(width);
    float pxPlacementY = float(tid.y + 0.5) / float(height);

    float xPlanePos = -aspectRatio + pxPlacementX * (aspectRatio - -aspectRatio);
    float yPlanePos = 1 + pxPlacementY * (-1 - 1);

    float3 origin = float3(0, 0, 0);
    float3 direction = float3(xPlanePos, yPlanePos, -1) - origin;

    float closeT = 1e30;
    int closestIndex = -1;

    for (int i = 0; i < num_spheres; i++) {
        float3 sphereCenter = spheres[i].center;
        float sphereRadius = spheres[i].radius;    
        float3 oc = origin - sphereCenter;
        float A = dot(direction, direction);
        float B = 2 * dot(oc, direction);
        float C = dot(oc, oc) - (sphereRadius * sphereRadius);
        float2 t;
        bool tF = solve_quadratic(A, B, C, t);
        if (tF == false) {
            continue;
        }
        float tNear = min(t.x, t.y);
        float tFar = max(t.x, t.y);
        float tHit;
        if (tNear > 0) {
            tHit = tNear;
        } else if (tFar > 0) {
            tHit = tFar;
        } else {
            continue;
        }
        if (tHit < closeT) {
            closeT = tHit;
            closestIndex = i;
        }
    }

    if (closestIndex == -1) {
        return;
    }
    float3 hitPoint = origin + (closeT * direction);
    float3 normal = normalize(hitPoint - spheres[closestIndex].center);
    image[tid] = float4((normal + 1) / 2, 1.0);
}
