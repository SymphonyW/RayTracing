先点击vscode下方的生成

CPU多线程版本：
源码位于main.cc中
编译运行指令：D:\raytracing> cmake --build build --target TheRestOfYourLife;.\build\Debug\TheRestOfYourLife.exe
D:\raytracing> cmake --build build --target TheNextWeek;.\build\Debug\TheNextWeek.exe
D:\raytracing> cmake --build build --target InOneWeekend;.\build\Debug\InOneWeekend.exe

GPU版本
源码位于main_cuda.cu中
编译运行指令：GPU版本：.\build_cuda.bat;.\build\TheRestOfYourLife_CUDA.exe