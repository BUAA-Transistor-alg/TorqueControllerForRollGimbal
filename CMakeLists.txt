cmake_minimum_required(VERSION 3.16)

project(RollGimbalDynamics LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GIMBAL_BACKEND_SOURCES
    src/gimbal_dynamics/dynamics_model.cpp
    src/gimbal_dynamics/simulation.cpp
    src/gimbal_dynamics/stage_data.cpp
    src/gimbal_dynamics/stage_derivatives.cpp
    src/gimbal_dynamics/stage_jacobians.cpp
    src/gimbal_dynamics/control_gradient.cpp
    src/gimbal_dynamics/parameter_gradient.cpp
    src/gimbal_dynamics/thread_pool.cpp
    src/gimbal_dynamics/backend_runner.cpp
)

# 热路径的向量化与展开选项：只影响本库编译单元，不改变公开接口
include(CheckCXXCompilerFlag)
option(GIMBAL_VECTORIZE "为热路径开启 AVX2 向量化与展开选项" ON)
set(GIMBAL_HOT_FLAGS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3;-funroll-loops;-fno-math-errno>")
check_cxx_compiler_flag("-mavx2" GIMBAL_HAS_AVX2)
if(GIMBAL_VECTORIZE AND GIMBAL_HAS_AVX2)
    list(APPEND GIMBAL_HOT_FLAGS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2;-ffp-contract=off>")
endif()

# 每个精度后端独立编译一份：双精度在 tcrg::gimbal_dynamics，单精度在 tcrg_f::gimbal_dynamics
function(gimbal_add_backend target prefix)
    add_library(${target} STATIC ${GIMBAL_BACKEND_SOURCES})
    target_include_directories(${target} PUBLIC include)
    target_compile_options(${target} PRIVATE -Wall -Wextra ${GIMBAL_HOT_FLAGS})
    target_link_libraries(${target} PUBLIC Threads::Threads)
    target_compile_definitions(${target} PRIVATE GIMBAL_BACKEND_PREFIX=${prefix})
endfunction()

gimbal_add_backend(gimbal_backend_double tcrg_backend_double_)
gimbal_add_backend(gimbal_backend_float tcrg_backend_float_)
target_compile_definitions(gimbal_backend_float PRIVATE tcrg=tcrg_f GIMBAL_REAL_FLOAT)

add_library(gimbal_dynamics INTERFACE)
target_include_directories(gimbal_dynamics INTERFACE include)
target_link_libraries(gimbal_dynamics INTERFACE gimbal_backend_double gimbal_backend_float)

option(GIMBAL_DYNAMICS_BUILD_BENCHMARKS "构建基准程序" ON)
if(GIMBAL_DYNAMICS_BUILD_BENCHMARKS)
    add_executable(sequential_benchmark benchmarks/sequential_benchmark.cpp)
    target_link_libraries(sequential_benchmark PRIVATE gimbal_dynamics)
    target_include_directories(sequential_benchmark PRIVATE tests)
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
