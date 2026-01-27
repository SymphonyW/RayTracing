@echo off
call "D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
nvcc src\TheRestOfYourLife\main_cuda.cu -o build\TheRestOfYourLife_CUDA.exe -I src\TheRestOfYourLife -O3 --std=c++11
if %errorlevel% equ 0 (
    echo Build successful!
    echo Executable created at: build\TheRestOfYourLife_CUDA.exe
) else (
    echo Build failed with error code %errorlevel%
)
@REM pause
