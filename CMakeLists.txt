cmake_minimum_required(VERSION 3.16)

project(RollGimbalDynamics LANGUAGES C CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GIMBAL_BACKEND_SOURCES
    src/gimbal_dynamics/dynamics_model.cpp
    src/gimbal_dynamics/simulation.cpp
    src/gimbal_dynamics/stage_data.cpp
    src/gimbal_dynamics/stage_derivatives.cpp
    src/gimbal_dynamics/stage_jacobians.cpp
    src/gimbal_dynamics/control_gradient.cpp
    src/gimbal_dynamics/parameter_gradient.cpp
    src/gimbal_dynamics/roll_locked_model.cpp
    src/gimbal_dynamics/roll_locked_simulation.cpp
    src/gimbal_dynamics/roll_locked_gradient.cpp
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

# 每个精度后端独立编译一份：双精度在 tcrg::gimbal_dynamics，单精度后缀为 tcrg::gimbal_dynamics_f
function(gimbal_add_backend target prefix)
    add_library(${target} STATIC ${GIMBAL_BACKEND_SOURCES})
    target_include_directories(${target} PUBLIC include)
    target_compile_options(${target} PRIVATE -Wall -Wextra ${GIMBAL_HOT_FLAGS})
    target_link_libraries(${target} PUBLIC Threads::Threads)
    target_compile_definitions(${target} PRIVATE GIMBAL_BACKEND_PREFIX=${prefix})
endfunction()

gimbal_add_backend(gimbal_backend_double tcrg_backend_double_)
gimbal_add_backend(gimbal_backend_float tcrg_backend_float_)
target_compile_definitions(gimbal_backend_float PRIVATE gimbal_dynamics=gimbal_dynamics_f GIMBAL_REAL_FLOAT)

add_library(gimbal_dynamics INTERFACE)
target_include_directories(gimbal_dynamics INTERFACE include)
target_link_libraries(gimbal_dynamics INTERFACE gimbal_backend_double gimbal_backend_float)

# C 接口共享库，供 Python ctypes 或其他语言调用；内部使用双精度后端
add_library(gimbal_dynamics_c SHARED src/gimbal_dynamics/c_api.cpp)
target_include_directories(gimbal_dynamics_c PUBLIC include)
target_link_libraries(gimbal_dynamics_c PRIVATE gimbal_backend_double Threads::Threads)
target_compile_options(gimbal_dynamics_c PRIVATE -Wall -Wextra)
set_target_properties(gimbal_dynamics_c PROPERTIES OUTPUT_NAME gimbal_dynamics)

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

    add_executable(roll_locked_check tests/roll_locked_check.cpp)
    target_link_libraries(roll_locked_check PRIVATE gimbal_dynamics)
    target_compile_options(roll_locked_check PRIVATE -Wall -Wextra)
    add_test(NAME roll_locked_check COMMAND roll_locked_check)

    add_executable(mpc_check tests/mpc_check.cpp)
    target_link_libraries(mpc_check PRIVATE gimbal_dynamics)
    target_compile_options(mpc_check PRIVATE -Wall -Wextra)
    add_test(NAME mpc_check COMMAND mpc_check)

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

    add_executable(c_api_check tests/c_api_check.c)
    target_link_libraries(c_api_check PRIVATE gimbal_dynamics_c)
    target_compile_options(c_api_check PRIVATE -Wall -Wextra)
    add_test(NAME c_api_check COMMAND c_api_check)
    set_tests_properties(c_api_check PROPERTIES TIMEOUT 120)

    add_executable(roll_locked_c_api_check tests/roll_locked_c_api_check.c)
    target_link_libraries(roll_locked_c_api_check PRIVATE gimbal_dynamics_c)
    target_compile_options(roll_locked_c_api_check PRIVATE -Wall -Wextra)
    add_test(NAME roll_locked_c_api_check COMMAND roll_locked_c_api_check)
    set_tests_properties(roll_locked_c_api_check PROPERTIES TIMEOUT 120)
endif()

# 实车环境：由子模组 Sentry2Communicator 的 ipc_client 读写实车控制器
option(GIMBAL_DYNAMICS_BUILD_ROBOT_IPC "构建实车 IPC 桥接库" ON)
if(GIMBAL_DYNAMICS_BUILD_ROBOT_IPC)
    set(SENTRY2_IPC_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sub_module/Sentry2Communicator/ipc_client")
    if(NOT EXISTS "${SENTRY2_IPC_CLIENT_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "子模组 Sentry2Communicator 未初始化，请执行 git submodule update --init --recursive")
    endif()
    add_subdirectory("${SENTRY2_IPC_CLIENT_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sentry2communicator_ipc_client")

    add_library(robot_ipc_c SHARED
        src/robot_ipc/c_api.cpp
        src/robot_ipc/chassis_pose.cpp
        src/robot_ipc/robot_client.cpp)
    target_include_directories(robot_ipc_c PUBLIC include)
    target_link_libraries(robot_ipc_c PRIVATE ipc_client Threads::Threads)
    target_compile_options(robot_ipc_c PRIVATE -Wall -Wextra)
    set_target_properties(robot_ipc_c PROPERTIES OUTPUT_NAME robot_ipc)

    if(GIMBAL_DYNAMICS_BUILD_TESTS)
        add_executable(robot_pose_check tests/robot_pose_check.cpp src/robot_ipc/chassis_pose.cpp)
        target_include_directories(robot_pose_check PRIVATE include)
        target_compile_options(robot_pose_check PRIVATE -Wall -Wextra)
        add_test(NAME robot_pose_check COMMAND robot_pose_check)

        add_executable(robot_ipc_check tests/robot_ipc_check.cpp)
        target_link_libraries(robot_ipc_check PRIVATE robot_ipc_c)
        target_compile_options(robot_ipc_check PRIVATE -Wall -Wextra)
        add_test(NAME robot_ipc_check COMMAND robot_ipc_check)
    endif()
endif()
