// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
using uint64_t = unsigned long long;
using uint32_t = unsigned int;
typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;
__device__ uint64_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}
// ldexpf(pcg32_random_r(&rng), -32);

__device__ float3 random_norm_float3(pcg32_random_t* rng) {
    for (int i = 0; i < 10; i++) {
        float3 p = make_float3(
            ldexpf(pcg32_random_r(rng), -32),
            ldexpf(pcg32_random_r(rng), -32),
            ldexpf(pcg32_random_r(rng), -32)
        );
        if (dot(p, p) <= 1.0) return normalize(p);
    }
    return make_float3(1.0, 0.0, 0.0);
}

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
    bool is_none;
    float3 point, normal;
    float t;
    bool front_face;
} hit_record;

class hittable {
    public:
        __device__ virtual hit_record hit(ray r, float2 t) = 0;
};

class material{
    public:
        __device__ virtual ray scatter(const ray in_r, const hit_record record, float3 color) = 0;
};

class diffuse : public material {
    float3 fcolor;
    __device__ diffuse(float3 fcolor) : fcolor(fcolor) {}
    __device__ ray scatter(const ray in_r, const hit_record record, float3 color) override {
        int a = 1;
         //
        // random diffuse
        ray r = {0.1, 0.1};
        return r;
    }
};

struct sphere : hittable {
    float3 center;
    float radius;
    __device__ sphere(float3 c, float r) : center(c), radius(r) {}
    __device__ hit_record hit(ray r, float2 t) override {
        hit_record record;
        record.is_none = true;
        float3 oc = sub(center, r.orig);
        float a = dot(r.dir, r.dir);
        float h = dot(r.dir, oc);
        float c = dot(oc, oc) - radius*radius;
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
        record.normal = div(sub(record.point, center), radius);
        // record.normal = div(sub(record.point, center), radius);
        record.normal = invert_if_dot(record.normal, r.dir, false);
        return record;
    }
};