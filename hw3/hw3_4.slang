import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
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




    float3 color = 0.0;
    float3 throughput = 1.0;
    for (int b = 0; b < max_depth; b++) {
        // compute contribution from point lights    
        float closeT = 1e30;
        int closestIndex = -1;
        for (int i = 0; i < num_shapes; i++) {
            float3 sphereCenter = spheres[shapes[i].index].center;
            float sphereRadius = spheres[shapes[i].index].radius;
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
            if (tNear > EPSILON) {
                tHit = tNear;
            } else if (tFar > EPSILON) {
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
            color += background * throughput;
            break;
        }

        float3 hitPoint = origin + (closeT * direction);
        float3 normal = normalize(hitPoint - spheres[shapes[closestIndex].index].center);
        float3 contribution = float3(0, 0, 0);
        for (int j = 0; j < num_lights; j++) {
            float3 toLight = lights[j].position - hitPoint;
            float3 lightDir = normalize(toLight);
            float cosine = max(dot(normal, lightDir), 0.0);
            float sqrDist = dot(toLight, toLight);
            float3 material = shapes[closestIndex].Kd;
            float3 lightColor = lights[j].intensity;
            float3 combination = material * lightColor * cosine / sqrDist;
            bool blocked = false;
            for (int k = 0; k < num_shapes; k++) {
                float3 sphereCenter = spheres[shapes[k].index].center;
                float sphereRadius = spheres[shapes[k].index].radius;
                float3 oc = hitPoint - sphereCenter;
                float A = dot(toLight, toLight);
                float B = 2 * dot(oc, toLight);
                float C = dot(oc, oc) - (sphereRadius * sphereRadius);
                float2 t;
                bool tF = solve_quadratic(A, B, C, t);
                if (tF == false) {
                    continue;
                }
                float tNear = min(t.x, t.y);
                float tFar = max(t.x, t.y);
                float tHit;
                if (tNear > EPSILON) {
                    tHit = tNear;
                } else if (tFar > EPSILON) {
                    tHit = tFar;
                } else {
                    continue;
                }
                if (tHit < 1.0) {
                    blocked = true;
                    break;
                }
            }
            if (blocked == false) {
                contribution = contribution + combination;
            }
        }
        color += throughput * contribution;
        throughput *= shapes[closestIndex].Kr;
        origin = hitPoint;
        direction = reflect(direction, normal);
    }
    image[tid] = float4(color, 1.0);
}
