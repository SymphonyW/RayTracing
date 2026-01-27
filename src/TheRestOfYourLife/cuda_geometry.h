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
};

// 四边形
struct Quad {
    vec3 Q;      // 起始顶点
    vec3 u, v;   // 两条边向量
    vec3 normal;
    float D;
    vec3 w;
    int mat_idx;

    __host__ __device__ void initialize() {
        vec3 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
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
};

// 场景结构
struct Scene {
    Sphere* spheres;
    int num_spheres;
    Quad* quads;
    int num_quads;
    Material* materials;
    int num_materials;

    __device__ bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        // 检查所有球体
        for (int i = 0; i < num_spheres; i++) {
            if (spheres[i].hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        // 检查所有四边形
        for (int i = 0; i < num_quads; i++) {
            if (quads[i].hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif
