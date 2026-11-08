cmake_minimum_required(VERSION 3.16)

project(RollGimbalDynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
target_link_libraries(gimbal_dynamics PUBLIC Eigen3::Eigen)
target_compile_options(gimbal_dynamics PRIVATE -Wall -Wextra)

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
