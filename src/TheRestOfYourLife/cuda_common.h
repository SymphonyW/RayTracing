#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 数学常量
#define CUDART_PI_F 3.141592654f
#define CUDART_INF_F __int_as_float(0x7f800000)

// CUDA随机数生成
__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ inline float random_float(curandState* state, float min, float max) {
    return min + (max - min) * random_float(state);
}

// Vec3类 - GPU版本
class vec3 {
public:
    float e[3];

    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(float t) {
        return *this *= 1/t;
    }

    __host__ __device__ float length() const {
        return sqrtf(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __device__ bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }
};

// Vec3工具函数
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_sphere(curandState* state) {
    vec3 p;
    do {
        p = 2.0f * vec3(random_float(state), random_float(state), random_float(state)) - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ inline vec3 random_unit_vector(curandState* state) {
    return unit_vector(random_in_unit_sphere(state));
}

__device__ inline vec3 random_in_unit_disk(curandState* state) {
    vec3 p;
    do {
        p = 2.0f * vec3(random_float(state), random_float(state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// ONB (正交基底) - GPU版本
struct ONB {
    vec3 axis[3];

    __device__ ONB() {}
    __device__ ONB(const vec3& n) {
        axis[2] = unit_vector(n);
        vec3 a = (fabsf(axis[2].x()) > 0.9f) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        axis[1] = unit_vector(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }

    __device__ const vec3& u() const { return axis[0]; }
    __device__ const vec3& v() const { return axis[1]; }
    __device__ const vec3& w() const { return axis[2]; }

    __device__ vec3 transform(const vec3& v) const {
        return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
    }
};

// 余弦加权半球随机方向（局部坐标）
__device__ inline vec3 random_cosine_direction(curandState* state) {
    float r1 = random_float(state);
    float r2 = random_float(state);

    float phi = 2.0f * CUDART_PI_F * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    return vec3(x, y, z);
}

// 朝球体方向的重要性采样
__device__ inline vec3 random_to_sphere(float radius, float distance_squared, curandState* state) {
    float r1 = random_float(state);
    float r2 = random_float(state);
    float z = 1.0f + r2 * (sqrtf(1.0f - radius * radius / distance_squared) - 1.0f);

    float phi = 2.0f * CUDART_PI_F * r1;
    float x = cosf(phi) * sqrtf(1.0f - z * z);
    float y = sinf(phi) * sqrtf(1.0f - z * z);

    return vec3(x, y, z);
}

// 类型别名
using point3 = vec3;
using color = vec3;

// Ray类
class ray {
public:
    point3 orig;
    vec3 dir;
    float tm;

    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& direction, float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    __host__ __device__ float time() const { return tm; }

    __host__ __device__ point3 at(float t) const {
        return orig + t * dir;
    }
};

#endif
