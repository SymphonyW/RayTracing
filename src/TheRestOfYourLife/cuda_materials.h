#ifndef CUDA_MATERIALS_H
#define CUDA_MATERIALS_H

#include "cuda_common.h"

// 材质类型枚举
enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT
};

// 散射记录
struct ScatterRecord {
    vec3 attenuation;
    ray scattered;      // skip_pdf时用的散射光线
    bool skip_pdf;      // true表示镜面材质，跳过PDF采样
};

// 材质结构
struct Material {
    MaterialType type;
    vec3 albedo;        // 漫反射颜色或反照率
    float fuzz;         // 金属模糊度
    float ir;           // 折射率
    vec3 emit;          // 发光颜色

    __device__ vec3 emitted(bool front_face) const {
        if (type == DIFFUSE_LIGHT && front_face) {
            return emit;
        }
        return vec3(0, 0, 0);
    }

    __device__ bool scatter(const ray& r_in, const vec3& p, const vec3& normal, bool front_face,
                           ScatterRecord& srec, curandState* state) const {
        if (type == LAMBERTIAN) {
            srec.attenuation = albedo;
            srec.skip_pdf = false;
            // 散射方向由PDF采样决定，不在此处设置
            return true;
        }
        else if (type == METAL) {
            vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
            srec.scattered = ray(p, reflected + fuzz * random_in_unit_sphere(state), r_in.time());
            srec.attenuation = albedo;
            srec.skip_pdf = true;
            return (dot(srec.scattered.direction(), normal) > 0);
        }
        else if (type == DIELECTRIC) {
            srec.attenuation = vec3(1.0f, 1.0f, 1.0f);
            srec.skip_pdf = true;
            float refraction_ratio = front_face ? (1.0f / ir) : ir;
            vec3 unit_direction = unit_vector(r_in.direction());

            float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            vec3 direction;

            // Schlick近似
            auto reflectance = [](float cosine, float ref_idx) -> float {
                float r0 = (1 - ref_idx) / (1 + ref_idx);
                r0 = r0 * r0;
                return r0 + (1 - r0) * powf((1 - cosine), 5);
            };

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(state))
                direction = reflect(unit_direction, normal);
            else
                direction = refract(unit_direction, normal, refraction_ratio);

            srec.scattered = ray(p, direction, r_in.time());
            return true;
        }
        else if (type == DIFFUSE_LIGHT) {
            return false;
        }
        return false;
    }

    // 材质的散射PDF值（仅用于非skip_pdf材质）
    __device__ float scattering_pdf(const vec3& normal, const ray& scattered) const {
        if (type == LAMBERTIAN) {
            float cos_theta = dot(normal, unit_vector(scattered.direction()));
            return cos_theta < 0.0f ? 0.0f : cos_theta / CUDART_PI_F;
        }
        return 0.0f;
    }
};

// 创建材质的工具函数
__host__ __device__ Material make_lambertian(const vec3& albedo) {
    Material m;
    m.type = LAMBERTIAN;
    m.albedo = albedo;
    m.fuzz = 0.0f;
    m.ir = 0.0f;
    m.emit = vec3(0, 0, 0);
    return m;
}

__host__ __device__ Material make_metal(const vec3& albedo, float fuzz) {
    Material m;
    m.type = METAL;
    m.albedo = albedo;
    m.fuzz = fuzz;
    m.ir = 0.0f;
    m.emit = vec3(0, 0, 0);
    return m;
}

__host__ __device__ Material make_dielectric(float ir) {
    Material m;
    m.type = DIELECTRIC;
    m.albedo = vec3(1, 1, 1);
    m.fuzz = 0.0f;
    m.ir = ir;
    m.emit = vec3(0, 0, 0);
    return m;
}

__host__ __device__ Material make_diffuse_light(const vec3& emit) {
    Material m;
    m.type = DIFFUSE_LIGHT;
    m.albedo = vec3(0, 0, 0);
    m.fuzz = 0.0f;
    m.ir = 0.0f;
    m.emit = emit;
    return m;
}

#endif
