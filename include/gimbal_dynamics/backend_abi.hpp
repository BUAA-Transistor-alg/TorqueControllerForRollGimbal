#pragma once

// 后端 ABI：处理器模板与按精度编译的后端之间的边界。
// 这里只出现 void* 与基本类型，且命名空间不随精度重命名。

#include <cstddef>

namespace tcrg_abi {

struct ContextConfig {
    int thread_count = 1;  // 含调用线程的总线程数
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

}  // namespace tcrg_abi

extern "C" {

struct GimbalAbiContext;

GimbalAbiContext* tcrg_backend_double_create(const tcrg_abi::ContextConfig* config);
void tcrg_backend_double_destroy(GimbalAbiContext* context);
void tcrg_backend_double_single_step(GimbalAbiContext* context,
                                       const tcrg_abi::SingleStepInput* input,
                                       tcrg_abi::SingleStepOutput* output);
void tcrg_backend_double_forward(GimbalAbiContext* context,
                                   const tcrg_abi::ForwardInput* input,
                                   tcrg_abi::ForwardOutput* output);
void tcrg_backend_double_forward_batch(GimbalAbiContext* context, std::size_t batch,
                                         const tcrg_abi::BatchGradientInput* input,
                                         tcrg_abi::BatchForwardOutput* output);
void tcrg_backend_double_control_gradient(GimbalAbiContext* context,
                                            const tcrg_abi::GradientInput* input,
                                            tcrg_abi::GradientOutput* output);
void tcrg_backend_double_control_gradient_batch(GimbalAbiContext* context, std::size_t batch,
                                                  const tcrg_abi::BatchGradientInput* input,
                                                  tcrg_abi::BatchGradientOutput* output);
void tcrg_backend_double_parameter_sensitivity(GimbalAbiContext* context,
                                                 const tcrg_abi::GradientInput* input,
                                                 tcrg_abi::GradientOutput* output);
void tcrg_backend_double_parameter_sensitivity_batch(
    GimbalAbiContext* context, std::size_t batch, const tcrg_abi::BatchGradientInput* input,
    tcrg_abi::BatchGradientOutput* output);

GimbalAbiContext* tcrg_backend_float_create(const tcrg_abi::ContextConfig* config);
void tcrg_backend_float_destroy(GimbalAbiContext* context);
void tcrg_backend_float_single_step(GimbalAbiContext* context,
                                      const tcrg_abi::SingleStepInput* input,
                                      tcrg_abi::SingleStepOutput* output);
void tcrg_backend_float_forward(GimbalAbiContext* context,
                                  const tcrg_abi::ForwardInput* input,
                                  tcrg_abi::ForwardOutput* output);
void tcrg_backend_float_forward_batch(GimbalAbiContext* context, std::size_t batch,
                                        const tcrg_abi::BatchGradientInput* input,
                                        tcrg_abi::BatchForwardOutput* output);
void tcrg_backend_float_control_gradient(GimbalAbiContext* context,
                                           const tcrg_abi::GradientInput* input,
                                           tcrg_abi::GradientOutput* output);
void tcrg_backend_float_control_gradient_batch(GimbalAbiContext* context, std::size_t batch,
                                                 const tcrg_abi::BatchGradientInput* input,
                                                 tcrg_abi::BatchGradientOutput* output);
void tcrg_backend_float_parameter_sensitivity(GimbalAbiContext* context,
                                                const tcrg_abi::GradientInput* input,
                                                tcrg_abi::GradientOutput* output);
void tcrg_backend_float_parameter_sensitivity_batch(
    GimbalAbiContext* context, std::size_t batch, const tcrg_abi::BatchGradientInput* input,
    tcrg_abi::BatchGradientOutput* output);

}  // extern "C"
