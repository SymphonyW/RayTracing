#ifndef CUDA_GEOMETRY_H
#define CUDA_GEOMETRY_H

#include "cuda_common.h"
#include "cuda_materials.h"

// 几何体类型枚举
enum GeometryType {
    SPHERE,
    QUAD
};

// 碰撞记录
struct HitRecord {
    vec3 p;
    vec3 normal;
    float t;
    bool front_face;
    int mat_idx;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// 球体
struct Sphere {
    vec3 center;
    float radius;
    int mat_idx;

    __device__ bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float half_b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = half_b * half_b - a * c;

        if (discriminant < 0) return false;

        float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_idx = mat_idx;

        return true;
    }

    // 球体重要性采样：PDF值
    __device__ float pdf_value(const vec3& origin, const vec3& direction) const {
        HitRecord rec;
        if (!hit(ray(origin, direction), 0.001f, CUDART_INF_F, rec))
            return 0.0f;

        float dist_squared = (center - origin).length_squared();
        float cos_theta_max = sqrtf(1.0f - radius * radius / dist_squared);
        float solid_angle = 2.0f * CUDART_PI_F * (1.0f - cos_theta_max);

        return 1.0f / solid_angle;
    }

    // 球体重要性采样：生成随机方向
    __device__ vec3 random_direction(const vec3& origin, curandState* state) const {
        vec3 direction = center - origin;
        float distance_squared = direction.length_squared();
        ONB uvw(direction);
        return uvw.transform(random_to_sphere(radius, distance_squared, state));
    }
};

// 四边形
struct Quad {
    vec3 Q;      // 起始顶点
    vec3 u, v;   // 两条边向量
    vec3 normal;
    float D;
    vec3 w;
    float area;
    int mat_idx;

    __host__ __device__ void initialize() {
        vec3 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
        area = n.length();
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        float denom = dot(normal, r.direction());

        // 射线平行于平面
        if (fabsf(denom) < 1e-8f)
            return false;

        float t = (D - dot(normal, r.origin())) / denom;
        if (t < t_min || t > t_max)
            return false;

        vec3 intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        float alpha = dot(w, cross(planar_hitpt_vector, v));
        float beta = dot(w, cross(u, planar_hitpt_vector));

        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
            return false;

        rec.t = t;
        rec.p = intersection;
        rec.set_face_normal(r, normal);
        rec.mat_idx = mat_idx;

        return true;
    }

    // 光源重要性采样：PDF值
    __device__ float pdf_value(const vec3& origin, const vec3& direction) const {
        HitRecord rec;
        if (!hit(ray(origin, direction), 0.001f, CUDART_INF_F, rec))
            return 0.0f;

        float distance_squared = rec.t * rec.t * dot(direction, direction);
        float cosine = fabsf(dot(direction, rec.normal) / direction.length());

        return distance_squared / (cosine * area);
    }

    // 光源重要性采样：生成随机方向
    __device__ vec3 random_direction(const vec3& origin, curandState* state) const {
        vec3 p = Q + random_float(state) * u + random_float(state) * v;
        return p - origin;
    }
};

// 场景结构
struct Scene {
    Sphere* spheres;
    int num_spheres;
    Quad* quads;
    int num_quads;
    Material* materials;
    int num_materials;

    // 光源几何体（用于重要性采样，直接存储在结构体中）
    static const int MAX_LIGHTS = 4;
    Quad light_quads[MAX_LIGHTS];
    int num_light_quads;
    Sphere light_spheres[MAX_LIGHTS];
    int num_light_spheres;

    __device__ bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (int i = 0; i < num_spheres; i++) {
            if (spheres[i].hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        for (int i = 0; i < num_quads; i++) {
            if (quads[i].hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    // 光源混合PDF值
    __device__ float light_pdf_value(const vec3& origin, const vec3& direction) const {
        int total_lights = num_light_quads + num_light_spheres;
        if (total_lights == 0) return 0.0f;

        float weight = 1.0f / total_lights;
        float sum = 0.0f;

        for (int i = 0; i < num_light_quads; i++)
            sum += weight * light_quads[i].pdf_value(origin, direction);
        for (int i = 0; i < num_light_spheres; i++)
            sum += weight * light_spheres[i].pdf_value(origin, direction);

        return sum;
    }

    // 随机选择一个光源并生成采样方向
    __device__ vec3 light_random(const vec3& origin, curandState* state) const {
        int total_lights = num_light_quads + num_light_spheres;
        if (total_lights == 0) return vec3(0, 1, 0);

        int choice = int(random_float(state) * total_lights);
        if (choice >= total_lights) choice = total_lights - 1;

        if (choice < num_light_quads)
            return light_quads[choice].random_direction(origin, state);
        else
            return light_spheres[choice - num_light_quads].random_direction(origin, state);
    }
};

#endif
