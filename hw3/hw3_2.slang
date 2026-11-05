import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
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
    float3 normal = normalize(hitPoint - spheres[shapes[closestIndex].index].center);
    //I figure this is probably to get get complete color calculation after all lights
    float3 color = float3(0, 0, 0);
    for (int j = 0; j < num_lights; j++) {
        //this finds the ray going from the hit point to the light
        float3 toLight = lights[j].position - hitPoint;
        //this normalizes the light direction
        float3 lightDir = normalize(toLight);
        //basically the dot product can't be negative because that would mean its behind it and this is just an easier way of dealing with it
        float cosine = max(dot(normal, lightDir), 0.0);
        //gets distance of light ray
        float sqrDist = dot(toLight, toLight);
        //gets color that the shape is supposed to be on closest circle
        float3 material = shapes[closestIndex].Kd;
        //gets light intensity
        float3 lightColor = lights[j].intensity;
        //full diffuse color equation calculated
        float3 combination = material * lightColor * cosine / sqrDist;
        bool blocked = false;
        //this basically checks I'm pretty sure if the light ray gets blocked by a shape
        for (int k = 0; k < num_shapes; k++) {
            //does the same as the previous part
            //except instead of checking for camera we check for other spheres
            float3 sphereCenter = spheres[shapes[k].index].center;
            float sphereRadius = spheres[shapes[k].index].radius;
            //vector between hypothetical spheres
            float3 oc = hitPoint - sphereCenter;
            //this is calculating the shadow ray
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
            //ignores hit points on actual current sphere
            if (tNear > EPSILON) {
                tHit = tNear;
            } else if (tFar > EPSILON) {
                tHit = tFar;
            } else {
                continue;
            }
            //this checks if we were able to reach the light, because if we havn't that means that something blocked
            //this comes down to what value the t we got
            if (tHit < 1.0) {
                blocked = true;
                break;
            }
        }
        if (blocked == false) {
            color = color + combination;            
        }
    }
    image[tid] = float4(color,1.0); 

}
