typedef struct {
    float3 orig, dir;
} ray;

__device__ float3 ray_at(ray r, float lambda) {
    return add(r.orig, mul(r.dir, lambda));
}

typedef struct {
    float aspect_ratio;
    unsigned int image_width, image_height, samples_per_pixel;
    float3 center, pixel00_loc, pixel_delta;
} camera;


typedef struct {
    float3 center;
    float radius;
} sphere;

typedef struct {
    bool is_none;
    float3 point, normal;
    float t;
    bool front_face;
} hit_record;



__device__ hit_record hit_sphere(sphere s, ray r, float2 t) {
    hit_record record;
    record.is_none = true;
    float3 oc = sub(s.center, r.orig);
    float a = dot(r.dir, r.dir);
    float h = dot(r.dir, oc);
    float c = dot(oc, oc) - s.radius*s.radius;
    float discriminant = h*h - a*c;
    if (discriminant < 0.0) return record;

    float sqrtd = sqrtf(discriminant);
    float root_a = (h - sqrtd) / a;
    float root_b = (h + sqrtd) / a;

    float root;
    if (contains(t, root_a)) {
        root = root_a;
    } else if (contains(t, root_b)){
        root = root_b;
    } else {
        return record;
    }

    record.is_none = false;
    record.t = root;
    record.point = ray_at(r,root);
    record.normal = div(sub(record.point, s.center), s.radius);
    record.normal = div(sub(record.point, s.center), s.radius);
    record.normal = invert_if_dot(record.normal, r.dir, false);
    return record;
}