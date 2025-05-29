// Float 4


__forceinline__ __device__ float4 mul(const float4 a, const float s) {
    return make_float4(a.x*s, a.y*s, a.z*s, a.w*s);
}

__forceinline__ __device__ float4 add(const float4 a, const float s) {
    return make_float4(a.x+s, a.y+s, a.z+s, a.w+s);
}

__forceinline__ __device__ float4 div(const float4 a, const float s) {
    return make_float4(a.x/s, a.y/s, a.z/s, a.w/s);
}

__forceinline__ __device__ float4 sub(const float4 a, const float s) {
    return make_float4(a.x-s, a.y-s, a.z-s, a.w-s);
}

__forceinline__ __device__ float4 div(const float s, const float4 a) {
    return make_float4(s / a.x, s / a.y, s / a.z, s / a.w);
}

__forceinline__ __device__ float4 sub(const float s, const float4 a) {
    return make_float4(s - a.x, s - a.y, s - a.z, s - a.w);
}

__forceinline__ __device__ float4 mul(const float4 a, const float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__forceinline__ __device__ float4 add(const float4 a, const float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ float4 div(const float4 a, const float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__forceinline__ __device__ float4 sub(const float4 a, const float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Float 3
__forceinline__ __device__ float3 mul(const float3 a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 add(const float3 a, const float s) {
    return make_float3(a.x + s, a.y + s, a.z + s);
}

__forceinline__ __device__ float3 div(const float3 a, const float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__forceinline__ __device__ float3 sub(const float3 a, const float s) {
    return make_float3(a.x - s, a.y - s, a.z - s);
}

__forceinline__ __device__ float3 div(const float s, const float3 a) {
    return make_float3(s / a.x, s / a.y, s / a.z);
}

__forceinline__ __device__ float3 sub(const float s, const float3 a) {
    return make_float3(s - a.x, s - a.y, s - a.z);
}

__forceinline__ __device__ float3 mul(const float3 a, const float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3 add(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 div(const float3 a, const float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__forceinline__ __device__ float3 sub(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 sqrt(const float3 a) {
    return make_float3(sqrtf(a.x),sqrtf(a.y),sqrtf(a.z));
}

__forceinline__ __device__ float4 __ffffmaf_rz(const float4 a, const float4 b, const float4 c) {
    float x = __fmaf_rz(a.x, b.x, c.x);
    float y = __fmaf_rz(a.y, b.y, c.y);
    float z = __fmaf_rz(a.z, b.z, c.z);
    float w = __fmaf_rz(a.w, b.w, c.w);
    return make_float4(x,y,z,w);
}

__forceinline__ __device__ float3 __fffmaf_rz(const float3 a, const float3 b, const float3 c) {
    float x = __fmaf_rz(a.x, b.x, c.x);
    float y = __fmaf_rz(a.y, b.y, c.y);
    float z = __fmaf_rz(a.z, b.z, c.z);
    return make_float3(x,y,z);
}

__forceinline__ __device__ float4 make_float4_f3(const float3 a, const float w) {
    return make_float4(a.x, a.y, a.z, w);
}


__forceinline__ __device__ float dot(const float3 a, const float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__forceinline__ __device__ bool contains(const float2 a, const float b) {
    return a.x < b && a.y > b;
}
__forceinline__ __device__ bool float3_near_zero_mag(const float3 a) {
    float s =  1e-8;
    return fabsf(a.x) < s && fabsf(a.y) < s && fabsf(a.z) < s;
}
__forceinline__ __device__ float3 normalize(const float3 a) {
    return mul(a, rnorm3df(a.x, a.y, a.z));
}

__forceinline__ __device__ float3 float3_reflect(const float3 a, const float3 b) {
    // a - b * 2 * dot(a, b)
    return sub(a, mul(b, 2.0f * dot(a, b)));
}

__device__ float3 invert_if_dot(float3 a, const float3 b, bool is_negative) {
    float multiplier = is_negative ? -1.0 : 1.0;
    float val = dot(a, b);
    if (val*multiplier > 0) {
        a = mul(a, -1.0);
    }
    return a;
}


