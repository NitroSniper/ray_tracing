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
    unsigned int image_width, image_height, samples_per_pixel;
    float3 center, pixel00_loc, pixel_delta;

    Camera() = default;
};

struct GuiState : public Ownership<GuiState> {
    bool show_random;
    bool random_norm;

    GuiState() = default;
};

struct Ray {
    float3 orig, dir;
    __device__ Ray(float3 o, float3 d) : orig(o), dir(d) {}
    Ray() = default;
    __device__ float3 at(float lambda) {
        return add(orig, mul(dir, lambda));
    }
};
