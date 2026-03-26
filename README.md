# 光线追踪渲染器

基于 Peter Shirley 的《Ray Tracing in One Weekend》系列书籍实现的光线追踪渲染器，包含从基础到高级的光线追踪技术，并添加了 GPU 加速功能。

## 项目结构

项目分为三个主要阶段，每个阶段对应书籍的一个部分：

1. **InOneWeekend** - 基础光线追踪实现
2. **TheNextWeek** - 扩展功能实现
3. **TheRestOfYourLife** - 高级功能实现，包括 CUDA GPU 加速

## 核心功能

### 1. 基础光线追踪（InOneWeekend）
- 实现了基本的光线追踪算法
- 支持多种材质：漫反射、金属、玻璃
- 实现了相机系统和抗锯齿
- 场景由球体组成，包括地面和随机分布的球体

### 2. 扩展功能（TheNextWeek）
- 添加了更多几何形状（如四边形）
- 实现了 BVH（边界体积层次结构）加速
- 添加了纹理和噪声功能
- 实现了体积雾效果

### 3. 高级功能与 GPU 加速（TheRestOfYourLife）
- 实现了重要性采样（混合 PDF 采样）
- 添加了 CUDA GPU 加速版本
- 实现了 Cornell Box 场景
- 支持更高质量的渲染效果

## 环境要求

### 编译环境
- CMake 3.1.0 或更高版本
- C++11 兼容的编译器
- CUDA 工具包（仅 GPU 版本需要）

### 支持平台
- Windows
- Linux
- macOS

## 构建与运行

### CPU 版本

```bash
# 构建并运行 InOneWeekend
cmake --build build --target InOneWeekend
./build/Debug/InOneWeekend.exe

# 构建并运行 TheNextWeek
cmake --build build --target TheNextWeek
./build/Debug/TheNextWeek.exe

# 构建并运行 TheRestOfYourLife
cmake --build build --target TheRestOfYourLife
./build/Debug/TheRestOfYourLife.exe
```

### GPU 版本

```bash
# 构建并运行 CUDA 版本
./build_cuda.bat
./build/TheRestOfYourLife_CUDA.exe
```

## 渲染参数调整

### GPU 版本参数（main_cuda.cu）
- `image_width` 和 `image_height` - 图像分辨率
- `samples_per_pixel` - 每像素采样数（影响噪声和渲染速度）
- `max_depth` - 光线最大反弹深度（影响光照真实感）

### CPU 版本参数
可在各版本的 main.cc 文件中调整相机参数和场景设置。

## 输出

渲染结果将保存为 PPM 格式的图像文件，位于 `output` 目录中（GPU 版本）。

## 技术文档

详细的技术文档位于 `doc` 目录中，包括：
- 光线追踪原理
- 代码结构分析
- 性能优化技术
- GPU 加速实现

## 参考资料

- Peter Shirley, "Ray Tracing in One Weekend"
- Peter Shirley, "Ray Tracing: The Next Week"
- Peter Shirley, "Ray Tracing: The Rest of Your Life"

## 许可证

本项目基于 CC0 公共领域贡献协议，详见 COPYING.txt 文件。