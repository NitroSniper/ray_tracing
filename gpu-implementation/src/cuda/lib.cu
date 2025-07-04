// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
using uint64_t = unsigned long long;
using uint32_t = unsigned int;
typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;
__device__ static pcg32_random_t pcg32_global;
__device__ uint64_t pcg32_random_r()
{
    uint64_t oldstate = pcg32_global.state;
    // Advance internal state
    pcg32_global.state = oldstate * 6364136223846793005ULL + (pcg32_global.inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ float3 random_norm_float3() {
    for (int i = 0; i < 10; i++) {
        float3 p = make_float3(
            ldexpf(pcg32_random_r(), -32),
            ldexpf(pcg32_random_r(), -32),
            ldexpf(pcg32_random_r(), -32)
        );
        if (dot(p, p) <= 1.0) return normalize(p);
    }
    return make_float3(1.0, 0.0, 0.0);
}


struct ray {
    float3 orig, dir;
    __device__ float3 at(float lambda) {
        return add(orig, mul(dir, lambda));
    }
};

typedef struct {
    float aspect_ratio;
    unsigned int image_width, image_height, samples_per_pixel;
    float3 center, pixel00_loc, pixel_delta;
} camera;

struct light {
    float3 color;
    ray ray;
};

struct hit_record {
    bool is_none;
    float3 point, normal;
    float t;
    bool front_face;
};

struct diffuse {
     float3 fcolor;
     __device__ diffuse(float3 fcolor) : fcolor(fcolor) {}
     __device__ light scatter(const ray in_r, const hit_record record) {
         float3 scatter_direction = add(record.normal, random_norm_float3());
         scatter_direction = float3_near_zero_mag(scatter_direction) ? record.normal : scatter_direction;
         light l = {fcolor, {record.point, scatter_direction}};
         return l;
     }
};

struct reflect {
     float3 fcolor;
     __device__ reflect(float3 fcolor) : fcolor(fcolor) {}
     __device__ light scatter(const ray in_r, const hit_record record) {
         float3 reflected = float3_reflect(in_r.dir, record.normal);
         light l = {fcolor, {record.point, reflected}};
         return l;
     }
};


enum class MaterialType {
    DIFFUSE=1,
    REFLECT,
};

struct material {
    MaterialType type;

    union {
        diffuse diff;
        reflect refl;
    };

    __device__ material(const diffuse d) : type(MaterialType::DIFFUSE), diff(d) {}
    __device__ material(const reflect r) : type(MaterialType::REFLECT), refl(r) {}
    __device__ light scatter(const ray in_r, const hit_record record) {
        switch (type) {
            case MaterialType::DIFFUSE:
                return diff.scatter(in_r, record);
            case MaterialType::REFLECT:
                return refl.scatter(in_r, record);
        }
    }
};

struct sphere {
    float3 center;
    float radius;
    material mat;
    __device__ sphere(float3 c, float r, material m) : center(c), radius(r), mat(m) {}
    __device__ hit_record hit(ray r, float2 t) {
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
        record.point = r.at(root);
        record.normal = div(sub(record.point, center), radius);
        // record.normal = div(sub(record.point, center), radius);
        record.normal = invert_if_dot(record.normal, r.dir, false);
        return record;
    }
};

enum class HittableType {
    SPHERE=1,
};
struct hittable {
    HittableType type;
    union {
        sphere s;
    };
    __device__ hittable(const sphere& s) : type(HittableType::SPHERE), s(s) {}
    __device__ hit_record hit(ray r, float2 t) {
        switch (type) {
            case HittableType::SPHERE: return s.hit(r, t);
        }
    }
};