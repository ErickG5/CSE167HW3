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

    //tldr is that basically before we assumed the fov was 90 degrees and now it isn't
    float fovLength = radians(camera.fov_x);
    //so the fov basically determines like how much of the screen we can see from the stationary 1 position
    float focalWidth = 2 * tan(fovLength/2);
    float focalHeight = focalWidth/aspectRatio;

    //since it affects the length of the viewable screen then we need to take it into account when tranlating that percentage into screen space
    float xPlanePos = -focalWidth/2 + (pxPlacementX * focalWidth);
    float yPlanePos = focalHeight/2 + (pxPlacementY * -focalHeight);

    float3 origin = float3(0, 0, 0);
    //both the origin and direction need to be put into world space
    float3 originToWorld = xform_pt(camera.cam_to_world, origin);
    float3 direction = float3(xPlanePos, yPlanePos, -1) - origin;
    //changes the camera to point ray from camera space(camera to screen) to world space, a place where the camera and transformed object coexist
    float3 directionToWorld = xform_vec(camera.cam_to_world, direction);



    float3 color = 0.0;
    float3 throughput = 1.0;
    float3 originToObjectFinal = float3(0, 0, 0);
    float3 directionToObjectFinal = float3(0, 0, 0);
    for (int b = 0; b < max_depth; b++) {
        // compute contribution from point lights    
        float closeT = 1e30;
        int closestIndex = -1;
        for (int i = 0; i < num_shapes; i++) {
            //now this is pretty much to change the ray so it hits the theoretical untransformed object
            float3 directionToObject = xform_vec(shapes[i].world_to_obj, directionToWorld);
            float3 originToObject = xform_pt(shapes[i].world_to_obj, originToWorld);
            
            float3 sphereCenter = spheres[shapes[i].index].center;
            float sphereRadius = spheres[shapes[i].index].radius;
            float3 oc = originToObject - sphereCenter;
            float A = dot(directionToObject, directionToObject);
            float B = 2 * dot(oc, directionToObject);
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
                originToObjectFinal = originToObject;
                directionToObjectFinal = directionToObject;
            }
        }
        if (closestIndex == -1) {
            color += background * throughput;
            break;
        }

        float3 hitPoint = originToObjectFinal + (closeT * directionToObjectFinal);
        //so once it hits the untransformed object, that hitpoint doesn't tell us where to render it on the screen so we need to change it back
        float3 hitPointTransformed = xform_pt(shapes[closestIndex].obj_to_world, hitPoint);
        float3 normal = normalize(hitPoint - spheres[shapes[closestIndex].index].center);
        //you also need to normalize it again because the transformation can make it longer then 1
        //world_to_obj is already the inverse of obj_to_world, so we just need to take the transpose of it
        float3 normalTransformed = normalize(xform_vec(transpose(shapes[closestIndex].world_to_obj), normal));
        float3 contribution = float3(0, 0, 0);
        for (int j = 0; j < num_lights; j++) {
            float3 toLight = lights[j].position - hitPointTransformed;
            float3 lightDir = normalize(toLight);
            float cosine = max(dot(normalTransformed, lightDir), 0.0);
            float sqrDist = dot(toLight, toLight);
            float3 material = shapes[closestIndex].Kd;
            float3 lightColor = lights[j].intensity;
            float3 combination = material * lightColor * cosine / sqrDist;
            bool blocked = false;
            for (int k = 0; k < num_shapes; k++) {
                float3 sphereCenter = spheres[shapes[k].index].center;
                float sphereRadius = spheres[shapes[k].index].radius;
                //hitPointTransformed is still in world space and for this to work it needs to be in objectSpace
                float3 oc = xform_pt(shapes[k].world_to_obj, hitPointTransformed) - sphereCenter;
                //changes the light to be in object space so its compatible
                float3 toLightObject = xform_vec(shapes[k].world_to_obj, toLight);
                float A = dot(toLightObject, toLightObject);
                float B = 2 * dot(oc, toLightObject);
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
        originToWorld = hitPointTransformed;
        directionToWorld = reflect(directionToWorld, normalTransformed);
    }
    image[tid] = float4(color, 1.0);
}
