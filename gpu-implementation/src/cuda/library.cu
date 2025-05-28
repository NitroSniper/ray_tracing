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

__device__ float3 random_float3() {
    return make_float3(
        ldexpf(pcg32_random_r(), -32),
        ldexpf(pcg32_random_r(), -32),
        ldexpf(pcg32_random_r(), -32)
    );
}
__device__ float3 random_norm_float3() {
    for (int i = 0; i < 20; i++) {
        float3 p = random_float3();
        if (dot(p, p) <= 1.0) return normalize(p);
    }
    return make_float3(1.0, 0.0, 0.0);
}

template <typename Derived>
class Ownership {
public:
    Ownership() = default;

    // Delete copy constructor and copy assignment
    Ownership(const Ownership&) = delete;
    Ownership& operator=(const Ownership&) = delete;

    // Default move constructor and move assignment
    Ownership(Ownership&&) = default;
    Ownership& operator=(Ownership&&) = default;

    // Provide clone method (must be implemented in Derived)
    __device__ Derived clone() const {
        Derived copy;
        memcpy(&copy, static_cast<const Derived*>(this), sizeof(Derived));
        return copy;
    }
};

struct Camera : public Ownership<Camera> {
    float aspect_ratio;
    unsigned int image_width, image_height;
    float3 center, pixel00_loc, pixel_delta;

    Camera() = default;
};


struct GuiState : public Ownership<GuiState> {
    unsigned int sample2_per_pixel, block_dim, max_depth;
    bool show_random, random_norm;

    GuiState() = default;
};

struct Ray : public Ownership<Ray> {
    float3 orig, dir;

    Ray() = default;

    __device__ Ray(float3 o, float3 d) : orig(o), dir(d) {}
    __device__ float3 at(float lambda) {
        return add(orig, mul(dir, lambda));
    }
};

struct HitRecord : public Ownership<HitRecord> {
    bool is_none = true;
    float3 point, normal;
    float t;
    bool front_face;

    HitRecord() = default;
};

struct Light : public Ownership<Light> {
    float3 color;
    Ray ray;

    Light() = default;

    __device__ Light(float3 c, Ray r)
    : color(c)
    , ray(static_cast<Ray&&>(r)) {}
};

struct Diffuse : public Ownership<Diffuse> {
    float3 color;

    Diffuse() = default;

    __device__ Diffuse(float3 c) : color(c) {}
    __device__ Light scatter(Ray& in_r, HitRecord& record) {
        float3 scatter_direction = add(record.normal, random_norm_float3());
        scatter_direction = float3_near_zero_mag(scatter_direction) ? record.normal : scatter_direction;
        return Light(color, Ray(record.point, scatter_direction));
    }
};

struct Reflect : public Ownership<Reflect> {
    float3 color;

    Reflect() = default;

    __device__ Reflect(float3 c) : color(c) {}
    __device__ Light scatter(Ray& in_r, HitRecord& record) {
        float3 reflected = float3_reflect(in_r.dir, record.normal);
        return Light(color, Ray(record.point, reflected));
    }
};

enum class MaterialType {
    DIFFUSE=1,
    REFLECT,
};

struct Material : public Ownership<Material> {
    MaterialType type;

    union {
        Diffuse diff;
        Reflect refl;
    };

    Material() = default;

    __device__ Material(Diffuse d)
        : type(MaterialType::DIFFUSE)
        , diff(static_cast<Diffuse&&>(d)) {}

    __device__ Material(Reflect r)
        : type(MaterialType::REFLECT)
        , refl(static_cast<Reflect&&>(r)) {}

    __device__ Light scatter(Ray& in_r, HitRecord& record) {
        switch (type) {
            case MaterialType::DIFFUSE:
                return diff.scatter(in_r, record);
            case MaterialType::REFLECT:
                return refl.scatter(in_r, record);
        }
        return Light();
    }
};

struct Sphere : public Ownership<Sphere> {
    float3 center;
    float radius;
    Diffuse mat;

    Sphere() = default;

    __device__ Sphere(float3 c, float r, Diffuse m)
        : center(c)
        , radius(r)
        , mat(static_cast<Diffuse&&>(m)) {}

    __device__ HitRecord hit(Ray& r, float2 t) {
        HitRecord record;
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

struct LightRecord : public Ownership<LightRecord> {
    HitRecord record;
    Light light;

    LightRecord() = default;

    __device__ LightRecord(HitRecord r, Light l)
        : record(static_cast<HitRecord&&>(r))
        , light(static_cast<Light&&>(l)) {}
};
