cmake_minimum_required(VERSION 3.16)

project(RollGimbalDynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gimbal_dynamics STATIC
    src/dynamics_model.cpp
    src/simulation.cpp
    src/stage_data.cpp
    src/stage_derivatives.cpp
    src/stage_jacobians.cpp
    src/control_gradient.cpp
    src/parameter_gradient.cpp
)
target_include_directories(gimbal_dynamics PUBLIC include PRIVATE src)
target_compile_options(gimbal_dynamics PRIVATE -Wall -Wextra)

# 热路径的向量化与展开选项：-mavx2 只影响本库的编译单元；
# -ffp-contract=off 保证不引入 FMA，浮点运算顺序与结果保持不变。
include(CheckCXXCompilerFlag)
option(GIMBAL_VECTORIZE "为热路径开启 AVX2 向量化与展开选项" ON)
set(GIMBAL_HOT_FLAGS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3;-funroll-loops;-fno-math-errno>")
check_cxx_compiler_flag("-mavx2" GIMBAL_HAS_AVX2)
if(GIMBAL_VECTORIZE AND GIMBAL_HAS_AVX2)
    list(APPEND GIMBAL_HOT_FLAGS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2;-ffp-contract=off>")
endif()
target_compile_options(gimbal_dynamics PRIVATE ${GIMBAL_HOT_FLAGS})

option(GIMBAL_DYNAMICS_BUILD_BENCHMARKS "构建基准程序" ON)
if(GIMBAL_DYNAMICS_BUILD_BENCHMARKS)
    add_executable(sequential_benchmark benchmarks/sequential_benchmark.cpp)
    target_link_libraries(sequential_benchmark PRIVATE gimbal_dynamics)
    target_compile_options(sequential_benchmark PRIVATE -Wall -Wextra)
endif()

option(GIMBAL_DYNAMICS_BUILD_TESTS "构建校验程序" ON)
if(GIMBAL_DYNAMICS_BUILD_TESTS)
    enable_testing()

    add_executable(model_consistency tests/model_consistency.cpp)
    target_link_libraries(model_consistency PRIVATE gimbal_dynamics)
    target_compile_options(model_consistency PRIVATE -Wall -Wextra)
    add_test(NAME model_consistency COMMAND model_consistency)

    add_executable(gradient_check tests/gradient_check.cpp)
    target_link_libraries(gradient_check PRIVATE gimbal_dynamics)
    target_compile_options(gradient_check PRIVATE -Wall -Wextra)
    add_test(NAME gradient_check COMMAND gradient_check)
endif()
