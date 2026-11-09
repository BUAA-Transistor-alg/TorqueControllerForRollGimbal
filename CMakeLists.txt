cmake_minimum_required(VERSION 3.16)

project(RollGimbalDynamics LANGUAGES CXX)

find_package(Threads REQUIRED)

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
    src/thread_pool.cpp
)
target_include_directories(gimbal_dynamics PUBLIC include PRIVATE src)
target_compile_options(gimbal_dynamics PRIVATE -Wall -Wextra)
target_link_libraries(gimbal_dynamics PUBLIC Threads::Threads)

# 精度开关：默认 double；打开后内部计算改用 float，仅用于精度对照测量
option(GIMBAL_REAL_FLOAT "内部计算精度切换为 float（仅用于精度对照测量）" OFF)
if(GIMBAL_REAL_FLOAT)
    target_compile_definitions(gimbal_dynamics PUBLIC GIMBAL_REAL_FLOAT)
endif()

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

    add_executable(concurrency_check tests/concurrency_check.cpp)
    target_link_libraries(concurrency_check PRIVATE gimbal_dynamics)
    target_compile_options(concurrency_check PRIVATE -Wall -Wextra)
    add_test(NAME concurrency_check COMMAND concurrency_check)
    set_tests_properties(concurrency_check PROPERTIES TIMEOUT 120)

    add_executable(exception_check tests/exception_check.cpp)
    target_link_libraries(exception_check PRIVATE gimbal_dynamics)
    target_compile_options(exception_check PRIVATE -Wall -Wextra)
    add_test(NAME exception_check COMMAND exception_check)
    set_tests_properties(exception_check PROPERTIES TIMEOUT 120)

    add_executable(thread_config_check tests/thread_config_check.cpp)
    target_link_libraries(thread_config_check PRIVATE gimbal_dynamics)
    target_compile_options(thread_config_check PRIVATE -Wall -Wextra)
    add_test(NAME thread_config_check COMMAND thread_config_check)
    set_tests_properties(thread_config_check PROPERTIES TIMEOUT 120)
endif()
