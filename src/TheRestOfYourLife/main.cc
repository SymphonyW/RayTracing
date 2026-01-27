//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"

#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"

#include <ctime>    // 用于获取时间
#include <chrono>   // 用于 std::chrono::system_clock
#include <iomanip>  // 用于 std::put_time 格式化时间
#include <sstream>  // 用于 std::stringstream 拼接字符串


int main() {
    hittable_list world;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    // Cornell box sides
    world.add(make_shared<quad>(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), green));
    world.add(make_shared<quad>(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), red));
    world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

    // Light
    world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

    // Box
    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    world.add(box1);

    // Glass Sphere
    auto glass = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(190,90,190), 90, glass));

    // Light Sources
    auto empty_material = shared_ptr<material>();
    hittable_list lights;
    lights.add(
        make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));
    lights.add(make_shared<sphere>(point3(190, 90, 190), 90, empty_material));

    camera cam;

    // ========================
    // 图像输出参数
    // ========================
    cam.aspect_ratio      = 1.0;    // 宽高比 (宽度/高度)
    cam.image_width       = 1920;    // 图像宽度（像素），越大质量越高但渲染越慢
    cam.samples_per_pixel = 1000;    // 每像素采样数，越多降噪效果越好
    cam.max_depth         = 100;     // 光线最大反弹深度，越大细节越丰富
    cam.background        = color(0,0,0);  // 背景色

    // ========================
    // 相机视角参数
    // ========================
    cam.vfov     = 40;                           // 垂直视角范围 (度)
    cam.lookfrom = point3(278, 278, -800);      // 相机位置
    cam.lookat   = point3(278, 278, 0);         // 相机看向的位置
    cam.vup      = vec3(0, 1, 0);               // 相机"向上"方向

    // ========================
    // 景深参数
    // ========================
    cam.defocus_angle = 0;  // 散焦角度（0为无景深效果）

    // ========================
    // 多线程参数
    // ========================
    cam.num_threads = 0;  // 设置为 0 自动使用系统硬件线程数，或指定具体数值（如 4, 8 等）

    // ========================
    // 自动生成带时间戳的文件名
    // ========================
    
    // 1. 获取当前时间
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    
    // 2. 将时间转换为本地时间结构
    std::tm tm = *std::localtime(&now);

    // 3. 使用 stringstream 格式化文件名
    std::stringstream ss;
    
    // 格式说明: %Y-年, %m-月, %d-日, %H-时, %M-分, %S-秒
    // 也就是生成类似: output/image_2023-10-27_15-30-00.ppm
    // 注意：文件名中不能包含冒号 (:)，所以时间分隔符用了横杠 (-)
    ss << "output/image-" << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S") << ".ppm";

    cam.output_file = ss.str();
    std::cout << "Output file will be: " << cam.output_file << std::endl;
    

    cam.render(world, lights);
}
