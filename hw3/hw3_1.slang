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

    //turns the tid into what percentage of the screen it should be at
    float pxPlacementX = float(tid.x + 0.5) / float(width);
    float pxPlacementY = float(tid.y + 0.5) / float(height);

    //turns that percentage into an actual position
    float xPlanePos = -aspectRatio + pxPlacementX * (aspectRatio - -aspectRatio);
    float yPlanePos = 1 + pxPlacementY * (-1 - 1);

    //I assume this is just so in the future you can move the origin/camera
    //You also set it to -1 because that is technically where the image plane is at in reference to the camera
    float3 origin = float3(0, 0, 0);
    float3 direction = float3(xPlanePos, yPlanePos, -1) - origin;

    //I assume this is just a shortcut for a super long distance and a short distance
    float closeT = 1e30;
    int closestIndex = -1;

    //iterate through all of the spheres
    for (int i = 0; i < num_spheres; i++) {
        //gets sphere center
        float3 sphereCenter = spheres[i].center;
        //gets sphere raidus
        float sphereRadius = spheres[i].radius;    
        //probably just gets center point in reference to origin
        //this also gets rid of xc later down in the equation
        float3 oc = origin - sphereCenter;

        //just to solve the expanded equation
        float A = dot(direction, direction);
        float B = 2 * dot(oc, direction);
        float C = dot(oc, oc) - (sphereRadius * sphereRadius);
        float2 t;
        bool tF = solve_quadratic(A, B, C, t);
        if (tF == false) {
            continue;
        }
        //the t values gets our intersection points in reference to time
        //we can have multiple so t.x and t.y is possible
        float tNear = min(t.x, t.y);
        float tFar = max(t.x, t.y);
        float tHit;
        //we check that they are greater then 0 because there is a chance that maybe the camera is in the sphere and we don't want to take that into account
        if (tNear > 0) {
            tHit = tNear;
        } else if (tFar > 0) {
            tHit = tFar;
        } else {
            continue;
        }
        //this is basically just to keep track of closest point to the screen so we render that
        //this is set to always work the first time
        if (tHit < closeT) {
            closeT = tHit;
            //so I guess its set to -1 if its an error
            closestIndex = i;
        }
    }

    if (closestIndex == -1) {
        return;
    }
    //finds the point itself using the ray equation
    float3 hitPoint = origin + (closeT * direction);
    //finds the normal
    float3 normal = normalize(hitPoint - spheres[closestIndex].center);
    //uses the normal to set the color
    image[tid] = float4((normal + 1) / 2, 1.0);
}
