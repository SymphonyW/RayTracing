//==============================================================================================
// GPU加速版本的光线追踪渲染器
// 基于CUDA并行计算，使用混合PDF重要性采样（光源采样 + 余弦加权半球采样）
// 算法与CPU多线程版本对齐，通过GPU大规模并行加速实现高质量快速渲染
//==============================================================================================

#include "cuda_common.h"
#include "cuda_materials.h"
#include "cuda_geometry.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>
#include <algorithm>

// 相机参数结构
struct Camera {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

// 初始化随机数生成器
__global__ void init_rand_state(curandState* rand_state, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
    int pixel_index = j * max_x + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

// 光线颜色计算（使用重要性采样的完整版本）
__device__ vec3 ray_color(const ray& r, const Scene& scene, curandState* local_rand_state, int max_depth) {
    ray cur_ray = r;
    vec3 cur_attenuation(1.0f, 1.0f, 1.0f);
    vec3 accumulated_color(0.0f, 0.0f, 0.0f);
    
    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        
        if (!scene.hit(cur_ray, 0.001f, CUDART_INF_F, rec)) {
            // 未击中任何物体，返回背景色（黑色）
            break;
        }
        
        // 累加自发光
        vec3 emission = scene.materials[rec.mat_idx].emitted(rec.front_face);
        accumulated_color = accumulated_color + cur_attenuation * emission;
        
        // 尝试散射
        ScatterRecord srec;
        if (!scene.materials[rec.mat_idx].scatter(cur_ray, rec.p, rec.normal, rec.front_face,
                                                  srec, local_rand_state)) {
            // 不散射（光源），返回累积颜色
            break;
        }
        
        if (srec.skip_pdf) {
            // 镜面材质（金属、电介质），直接使用散射方向，无需PDF修正
            cur_attenuation = cur_attenuation * srec.attenuation;
            cur_ray = srec.scattered;
            continue;
        }
        
        // === 重要性采样：混合PDF（光源采样 + 余弦采样） ===
        ONB uvw(rec.normal);
        ray scattered;
        
        if (random_float(local_rand_state) < 0.5f) {
            // 50% 概率：朝光源方向采样
            vec3 light_dir = scene.light_random(rec.p, local_rand_state);
            scattered = ray(rec.p, light_dir, cur_ray.time());
        } else {
            // 50% 概率：余弦加权半球采样
            vec3 cosine_dir = uvw.transform(random_cosine_direction(local_rand_state));
            scattered = ray(rec.p, cosine_dir, cur_ray.time());
        }
        
        // 计算混合PDF值
        float light_pdf = scene.light_pdf_value(rec.p, scattered.direction());
        float cosine_theta = dot(unit_vector(scattered.direction()), uvw.w());
        float cosine_pdf = cosine_theta < 0.0f ? 0.0f : cosine_theta / CUDART_PI_F;
        float pdf_value = 0.5f * light_pdf + 0.5f * cosine_pdf;
        
        if (pdf_value < 1e-10f) {
            break;
        }
        
        // 材质散射PDF
        float s_pdf = scene.materials[rec.mat_idx].scattering_pdf(rec.normal, scattered);
        
        // 重要性采样修正：attenuation * scattering_pdf / mixture_pdf
        cur_attenuation = cur_attenuation * srec.attenuation * (s_pdf / pdf_value);
        cur_ray = scattered;
    }
    
    return accumulated_color;
}

// 渲染内核
__global__ void render_kernel(vec3* fb, int max_x, int max_y, int samples_per_pixel, int max_depth,
                              Camera cam, Scene scene, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
    
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    
    vec3 col(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++) {
        float u = float(i + random_float(&local_rand_state)) / float(max_x);
        float v = float(j + random_float(&local_rand_state)) / float(max_y);
        
        ray r(cam.origin, cam.lower_left_corner + u * cam.horizontal + v * cam.vertical - cam.origin);
        col += ray_color(r, scene, &local_rand_state, max_depth);
    }
    
    col /= float(samples_per_pixel);
    
    // Gamma校正
    col = vec3(sqrtf(col.x()), sqrtf(col.y()), sqrtf(col.z()));
    
    fb[pixel_index] = col;
}

// 创建Cornell Box场景
void create_cornell_box_scene(Scene& h_scene, Sphere*& d_spheres, Quad*& d_quads, Material*& d_materials) {
    // CPU端场景数据
    const int num_materials = 7;
    const int num_quads = 18;  // 6个房间墙面 + 2个立方体(每个6面)
    const int num_spheres = 2;
    
    Material* h_materials = new Material[num_materials];
    h_materials[0] = make_lambertian(vec3(0.65f, 0.05f, 0.05f));  // 红色墙面
    h_materials[1] = make_lambertian(vec3(0.12f, 0.45f, 0.15f));  // 绿色墙面
    h_materials[2] = make_lambertian(vec3(0.73f, 0.73f, 0.73f));  // 白色墙面
    h_materials[3] = make_diffuse_light(vec3(15.0f, 15.0f, 15.0f)); // 天花板光源
    h_materials[4] = make_dielectric(1.5f);  // 玻璃材质
    h_materials[5] = make_metal(vec3(0.8f, 0.85f, 0.88f), 0.05f);  // 银色金属
    h_materials[6] = make_metal(vec3(0.8f, 0.6f, 0.2f), 0.0f);  // 金色金属
    
    Quad* h_quads = new Quad[num_quads];
    // 右墙（绿色）
    h_quads[0].Q = vec3(555, 0, 0);
    h_quads[0].u = vec3(0, 0, 555);
    h_quads[0].v = vec3(0, 555, 0);
    h_quads[0].mat_idx = 1;
    h_quads[0].initialize();
    
    // 左墙（红色）
    h_quads[1].Q = vec3(0, 0, 555);
    h_quads[1].u = vec3(0, 0, -555);
    h_quads[1].v = vec3(0, 555, 0);
    h_quads[1].mat_idx = 0;
    h_quads[1].initialize();
    
    // 天花板（白色）
    h_quads[2].Q = vec3(0, 555, 0);
    h_quads[2].u = vec3(555, 0, 0);
    h_quads[2].v = vec3(0, 0, 555);
    h_quads[2].mat_idx = 2;
    h_quads[2].initialize();
    
    // 地板（白色）
    h_quads[3].Q = vec3(0, 0, 555);
    h_quads[3].u = vec3(555, 0, 0);
    h_quads[3].v = vec3(0, 0, -555);
    h_quads[3].mat_idx = 2;
    h_quads[3].initialize();
    
    // 后墙（白色）
    h_quads[4].Q = vec3(555, 0, 555);
    h_quads[4].u = vec3(-555, 0, 0);
    h_quads[4].v = vec3(0, 555, 0);
    h_quads[4].mat_idx = 2;
    h_quads[4].initialize();
    
    // 光源
    h_quads[5].Q = vec3(213, 554, 227);
    h_quads[5].u = vec3(130, 0, 0);
    h_quads[5].v = vec3(0, 0, 105);
    h_quads[5].mat_idx = 3;
    h_quads[5].initialize();
    
    // 立方体1 - 白色大盒子（右后方，带旋转）
    // 旋转角15度（sin(15°) ≈ 0.2588, cos(15°) ≈ 0.9659）
    float cos15 = 0.9659f;
    float sin15 = 0.2588f;
    // 旋转后的u和v向量
    vec3 rot_u = vec3(165 * cos15, 0, -165 * sin15);  // 绕Y轴旋转的u
    vec3 rot_v_base = vec3(165 * sin15, 0, 165 * cos15);  // 绕Y轴旋转的v
    
    // 底面
    h_quads[6].Q = vec3(265, 0, 295);
    h_quads[6].u = rot_u;
    h_quads[6].v = rot_v_base;
    h_quads[6].mat_idx = 2;
    h_quads[6].initialize();
    // 顶面
    h_quads[7].Q = vec3(265, 330, 295);
    h_quads[7].u = rot_u;
    h_quads[7].v = rot_v_base;
    h_quads[7].mat_idx = 2;
    h_quads[7].initialize();
    // 前面
    h_quads[8].Q = vec3(265, 0, 295);
    h_quads[8].u = rot_u;
    h_quads[8].v = vec3(0, 330, 0);
    h_quads[8].mat_idx = 2;
    h_quads[8].initialize();
    // 后面
    h_quads[9].Q = vec3(265 + rot_v_base.x(), 0, 295 + rot_v_base.z());
    h_quads[9].u = rot_u;
    h_quads[9].v = vec3(0, 330, 0);
    h_quads[9].mat_idx = 2;
    h_quads[9].initialize();
    // 左面
    h_quads[10].Q = vec3(265, 0, 295);
    h_quads[10].u = rot_v_base;
    h_quads[10].v = vec3(0, 330, 0);
    h_quads[10].mat_idx = 2;
    h_quads[10].initialize();
    // 右面
    h_quads[11].Q = vec3(265 + rot_u.x(), 0, 295 + rot_u.z());
    h_quads[11].u = rot_v_base;
    h_quads[11].v = vec3(0, 330, 0);
    h_quads[11].mat_idx = 2;
    h_quads[11].initialize();
    
    // 立方体2 - 白色小盒子（左前方，用于展示环境光）
    // 底面
    h_quads[12].Q = vec3(80, 0, 50);
    h_quads[12].u = vec3(90, 0, 0);
    h_quads[12].v = vec3(0, 0, 90);
    h_quads[12].mat_idx = 2;
    h_quads[12].initialize();
    // 顶面
    h_quads[13].Q = vec3(80, 150, 50);
    h_quads[13].u = vec3(90, 0, 0);
    h_quads[13].v = vec3(0, 0, 90);
    h_quads[13].mat_idx = 2;
    h_quads[13].initialize();
    // 前面（靠近红墙，会接收红色环境光）
    h_quads[14].Q = vec3(80, 0, 50);
    h_quads[14].u = vec3(90, 0, 0);
    h_quads[14].v = vec3(0, 150, 0);
    h_quads[14].mat_idx = 2;
    h_quads[14].initialize();
    // 后面
    h_quads[15].Q = vec3(80, 0, 140);
    h_quads[15].u = vec3(90, 0, 0);
    h_quads[15].v = vec3(0, 150, 0);
    h_quads[15].mat_idx = 2;
    h_quads[15].initialize();
    // 左面（靠近红墙，会接收红色环境光）
    h_quads[16].Q = vec3(80, 0, 50);
    h_quads[16].u = vec3(0, 0, 90);
    h_quads[16].v = vec3(0, 150, 0);
    h_quads[16].mat_idx = 2;
    h_quads[16].initialize();
    // 右面（会接收来自中间的光线）
    h_quads[17].Q = vec3(170, 0, 50);
    h_quads[17].u = vec3(0, 0, 90);
    h_quads[17].v = vec3(0, 150, 0);
    h_quads[17].mat_idx = 2;
    h_quads[17].initialize();
    
    Sphere* h_spheres = new Sphere[num_spheres];
    // 玻璃球（大立方体左侧）
    h_spheres[0].center = vec3(160, 90, 300);
    h_spheres[0].radius = 90;
    h_spheres[0].mat_idx = 4;
    
    // 银色金属球（右前方）
    h_spheres[1].center = vec3(420, 70, 150);
    h_spheres[1].radius = 70;
    h_spheres[1].mat_idx = 5;

    // 设置光源几何体（用于重要性采样）
    // 光源四边形：与CPU版本一致，反转方向以确保法线朝下
    Quad light_q;
    light_q.Q = vec3(343, 554, 332);
    light_q.u = vec3(-130, 0, 0);
    light_q.v = vec3(0, 0, -105);
    light_q.mat_idx = -1;
    light_q.initialize();
    h_scene.light_quads[0] = light_q;
    h_scene.num_light_quads = 1;

    // 玻璃球也作为重要性采样光源（与CPU版本一致）
    Sphere light_s;
    light_s.center = vec3(160, 90, 300);
    light_s.radius = 90;
    light_s.mat_idx = -1;
    h_scene.light_spheres[0] = light_s;
    h_scene.num_light_spheres = 1;

    CUDA_CHECK(cudaMalloc(&d_materials, num_materials * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_quads, num_quads * sizeof(Quad)));
    CUDA_CHECK(cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)));
    
    CUDA_CHECK(cudaMemcpy(d_materials, h_materials, num_materials * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quads, h_quads, num_quads * sizeof(Quad), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    
    h_scene.materials = d_materials;
    h_scene.num_materials = num_materials;
    h_scene.quads = d_quads;
    h_scene.num_quads = num_quads;
    h_scene.spheres = d_spheres;
    h_scene.num_spheres = num_spheres;
    
    delete[] h_materials;
    delete[] h_quads;
    delete[] h_spheres;
}

// 写入PPM图像
void write_ppm(const char* filename, vec3* fb, int nx, int ny) {
    std::ofstream out(filename);
    out << "P3\n" << nx << " " << ny << "\n255\n";
    
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            int pixel_index = j * nx + i;
            vec3 pixel = fb[pixel_index];
            
            int ir = int(255.99f * pixel.x());
            int ig = int(255.99f * pixel.y());
            int ib = int(255.99f * pixel.z());
            
            // 限制到0-255范围
            ir = ir < 0 ? 0 : (ir > 255 ? 255 : ir);
            ig = ig < 0 ? 0 : (ig > 255 ? 255 : ig);
            ib = ib < 0 ? 0 : (ib > 255 ? 255 : ib);
            
            out << ir << " " << ig << " " << ib << "\n";
        }
    }
    out.close();
}

int main() {

    // ========================================================================================
    // 图像质量参数 - 调整这些参数来控制渲染质量和速度
    // ========================================================================================
    
    // 图像分辨率（像素）
    // 数值越大 = 质量越好但渲染越慢
    // 推荐值：800x800（快速）, 1920x1080（平衡）, 3840x2160（高质量）
    const int image_width = 3840;
    const int image_height = 2160;
    
    // 每像素采样数（SPP）
    // 数值越大 = 噪点越少、图像越平滑但渲染越慢
    // 推荐值：10（预览）, 100（快速）, 1000（平衡）, 5000+（高质量）
    // 已使用重要性采样，收敛速度大幅提升
    const int samples_per_pixel = 5000;
    
    // 光线最大反弹深度
    // 数值越大 = 光照越真实但渲染越慢
    // 推荐值：10（快速）, 50（平衡）, 100（高质量）
    const int max_depth = 100;
    
    // ========================================================================================
    
    std::cout << "=================================================\n";
    std::cout << "============= current version:CUDA ==============\n";
    std::cout << "  GPU-Accelerated Ray Tracer (CUDA)\n";
    std::cout << "=================================================\n";
    std::cout << "Image Size:        " << image_width << " x " << image_height << "\n";
    std::cout << "Samples/Pixel:     " << samples_per_pixel << "\n";
    std::cout << "Max Ray Depth:     " << max_depth << "\n";
    std::cout << "Total Pixels:      " << (image_width * image_height) << "\n";
    std::cout << "=================================================\n\n";
    
    // 分配帧缓冲
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3* d_fb;
    CUDA_CHECK(cudaMalloc(&d_fb, fb_size));
    
    // 分配随机数状态
    curandState* d_rand_state;
    CUDA_CHECK(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));
    
    // 初始化随机数生成器
    dim3 blocks(image_width / 8 + 1, image_height / 8 + 1);
    dim3 threads(8, 8);
    init_rand_state<<<blocks, threads>>>(d_rand_state, image_width, image_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 创建场景
    Scene h_scene;
    Sphere* d_spheres;
    Quad* d_quads;
    Material* d_materials;
    create_cornell_box_scene(h_scene, d_spheres, d_quads, d_materials);
    
    // 设置相机
    Camera cam;
    vec3 lookfrom(278, 278, -800);
    vec3 lookat(278, 278, 0);
    vec3 vup(0, 1, 0);
    float vfov = 40.0f;
    float aspect = float(image_width) / float(image_height);
    
    float theta = vfov * CUDART_PI_F / 180.0f;
    float half_height = tanf(theta / 2.0f);
    float half_width = aspect * half_height;
    
    cam.origin = lookfrom;
    cam.w = unit_vector(lookfrom - lookat);
    cam.u = unit_vector(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);
    
    float focus_dist = (lookfrom - lookat).length();
    cam.lower_left_corner = cam.origin - half_width * focus_dist * cam.u 
                          - half_height * focus_dist * cam.v - focus_dist * cam.w;
    cam.horizontal = 2.0f * half_width * focus_dist * cam.u;
    cam.vertical = 2.0f * half_height * focus_dist * cam.v;
    cam.lens_radius = 0.0f;
    
    // 开始渲染
    std::cout << "Starting GPU rendering...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    // 启动渲染kernel
    render_kernel<<<blocks, threads>>>(d_fb, image_width, image_height, samples_per_pixel, 
                                       max_depth, cam, h_scene, d_rand_state);
    CUDA_CHECK(cudaGetLastError());
    
    // 显示进度条（轮询GPU状态）
    // std::cout << "\r[" << std::string(50, ' ') << "] 0%" << std::flush;
    
    cudaError_t status = cudaSuccess;
    int dots = 0;
    while ((status = cudaStreamQuery(0)) == cudaErrorNotReady) {
        // GPU还在处理中
        std::cout << "\rProcessing";
        for (int i = 0; i < (dots % 4); i++) std::cout << ".";
        for (int i = (dots % 4); i < 3; i++) std::cout << " ";
        std::cout << "   " << std::flush;
        dots++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // 确保完全同步
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "\r[" << std::string(50, '=') << "] 100% - Complete!       \n";
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Rendering completed in " << elapsed.count() << " seconds\n";
    
    // 复制帧缓冲到主机
    vec3* h_fb = new vec3[num_pixels];
    CUDA_CHECK(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    
    // 生成带时间戳的文件名
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm = *std::localtime(&now);
    std::stringstream ss;
    ss << "output/image_cuda-" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S") << ".ppm";
    std::string output_filename = ss.str();
    
    // 写入图像文件
    std::cout << "Writing image to " << output_filename << "...\n";
    write_ppm(output_filename.c_str(), h_fb, image_width, image_height);
    std::cout << "\nDone! Image saved successfully.\n";
    std::cout << "=================================================\n";
    
    // 清理
    delete[] h_fb;
    CUDA_CHECK(cudaFree(d_fb));
    CUDA_CHECK(cudaFree(d_rand_state));
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_quads));
    CUDA_CHECK(cudaFree(d_materials));
    
    return 0;
}
