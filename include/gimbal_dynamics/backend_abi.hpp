#pragma once

// 后端 ABI：处理器模板与按精度编译的后端之间的边界。
// 这里只出现 void* 与基本类型，且命名空间不随精度重命名。

#include <cstddef>

namespace gimbal_abi {

struct ContextConfig {
    int thread_count = 1;      // 含调用线程的总线程数
    std::size_t max_batch = 1; // 批量模式允许的最大批量
};

struct SingleStepInput {
    const void* model = nullptr;
    const void* terrain = nullptr;
    const void* base = nullptr;
    const void* angle = nullptr;
    const void* velocity = nullptr;
    const void* torque = nullptr;
};

struct SingleStepOutput {
    void* acceleration = nullptr;
};

struct ForwardInput {
    const void* model = nullptr;
    const void* terrain = nullptr;
    const void* initial = nullptr;
    const void* base = nullptr;
    const void* torque = nullptr;
    std::size_t num_steps = 0;
    double step = 0.0;
    int refinement = 1;
    const void* loss = nullptr;
};

struct ForwardOutput {
    void* loss = nullptr;
    void* step_end = nullptr;
    void* stage_state = nullptr;
    void* stage_base = nullptr;
};

struct GradientInput {
    const void* model = nullptr;
    const void* terrain = nullptr;
    const void* initial = nullptr;
    const void* base = nullptr;
    const void* torque = nullptr;
    std::size_t num_steps = 0;
    double step = 0.0;
    int refinement = 1;
    const void* loss = nullptr;
};

struct GradientOutput {
    void* loss = nullptr;
    void* final_state = nullptr;
    void* gradient = nullptr;
};

struct BatchGradientInput {
    const void* model = nullptr;
    const void* terrain = nullptr;
    const void* initial = nullptr;
    const void* base = nullptr;
    const void* torque = nullptr;
    std::size_t num_steps = 0;
    double step = 0.0;
    int refinement = 1;
    const void* losses = nullptr;
};

struct BatchGradientOutput {
    void* mean_loss = nullptr;
    void* gradient = nullptr;
    void* final_state = nullptr;
};

struct BatchForwardOutput {
    void* mean_loss = nullptr;
    void* step_end = nullptr;
    void* final_state = nullptr;
};

}  // namespace gimbal_abi

extern "C" {

struct GimbalAbiContext;

GimbalAbiContext* gimbal_backend_double_create(const gimbal_abi::ContextConfig* config);
void gimbal_backend_double_destroy(GimbalAbiContext* context);
void gimbal_backend_double_single_step(GimbalAbiContext* context,
                                       const gimbal_abi::SingleStepInput* input,
                                       gimbal_abi::SingleStepOutput* output);
void gimbal_backend_double_forward(GimbalAbiContext* context,
                                   const gimbal_abi::ForwardInput* input,
                                   gimbal_abi::ForwardOutput* output);
void gimbal_backend_double_forward_batch(GimbalAbiContext* context, std::size_t batch,
                                         const gimbal_abi::BatchGradientInput* input,
                                         gimbal_abi::BatchForwardOutput* output);
void gimbal_backend_double_control_gradient(GimbalAbiContext* context,
                                            const gimbal_abi::GradientInput* input,
                                            gimbal_abi::GradientOutput* output);
void gimbal_backend_double_control_gradient_batch(GimbalAbiContext* context, std::size_t batch,
                                                  const gimbal_abi::BatchGradientInput* input,
                                                  gimbal_abi::BatchGradientOutput* output);
void gimbal_backend_double_parameter_sensitivity(GimbalAbiContext* context,
                                                 const gimbal_abi::GradientInput* input,
                                                 gimbal_abi::GradientOutput* output);
void gimbal_backend_double_parameter_sensitivity_batch(
    GimbalAbiContext* context, std::size_t batch, const gimbal_abi::BatchGradientInput* input,
    gimbal_abi::BatchGradientOutput* output);

GimbalAbiContext* gimbal_backend_float_create(const gimbal_abi::ContextConfig* config);
void gimbal_backend_float_destroy(GimbalAbiContext* context);
void gimbal_backend_float_single_step(GimbalAbiContext* context,
                                      const gimbal_abi::SingleStepInput* input,
                                      gimbal_abi::SingleStepOutput* output);
void gimbal_backend_float_forward(GimbalAbiContext* context,
                                  const gimbal_abi::ForwardInput* input,
                                  gimbal_abi::ForwardOutput* output);
void gimbal_backend_float_forward_batch(GimbalAbiContext* context, std::size_t batch,
                                        const gimbal_abi::BatchGradientInput* input,
                                        gimbal_abi::BatchForwardOutput* output);
void gimbal_backend_float_control_gradient(GimbalAbiContext* context,
                                           const gimbal_abi::GradientInput* input,
                                           gimbal_abi::GradientOutput* output);
void gimbal_backend_float_control_gradient_batch(GimbalAbiContext* context, std::size_t batch,
                                                 const gimbal_abi::BatchGradientInput* input,
                                                 gimbal_abi::BatchGradientOutput* output);
void gimbal_backend_float_parameter_sensitivity(GimbalAbiContext* context,
                                                const gimbal_abi::GradientInput* input,
                                                gimbal_abi::GradientOutput* output);
void gimbal_backend_float_parameter_sensitivity_batch(
    GimbalAbiContext* context, std::size_t batch, const gimbal_abi::BatchGradientInput* input,
    gimbal_abi::BatchGradientOutput* output);

}  // extern "C"
