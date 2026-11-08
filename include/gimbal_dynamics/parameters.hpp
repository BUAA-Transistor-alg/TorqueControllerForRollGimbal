#pragma once

// 动力学模型参数的扁平编号布局。
//
// 参与求导的模型参数共 56 个，顺序固定：
//   * 4 个刚体 B1..B4，每个 10 个：质量 1、质心 3、惯量对称分量 6；
//   * 4 个关节安装偏置 o1..o4，每个 3 个；
//   * 4 个关节粘性阻尼 d1..d4。
//
// 惯量按对称张量处理：每刚体的第 5..10 个分量依次对应 xx、yy、zz、xy、xz、yz，
// 加减该分量时同时作用到 (r,c) 与 (c,r)。
//
// parameter_value 与 add_parameter 同时适用于 ModelParameters（Real）与内部的
// 标量泛化模型，参数布局只在本文件维护一份。

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "gimbal_dynamics/dynamics_model.hpp"

namespace gimbal {
namespace parameters {

inline constexpr int kBodyCount = 4;
inline constexpr int kBodyParameterCount = 10;
inline constexpr int kOffsetParameterCount = 12;
inline constexpr int kDampingParameterCount = 4;
inline constexpr int kParameterCount =
    kBodyCount * kBodyParameterCount + kOffsetParameterCount + kDampingParameterCount;

// 模型标量类型：ModelParameters 为 Real，标量泛化模型为对应的 Scalar。
template <typename Model>
using ModelScalar = std::remove_reference_t<decltype(std::declval<Model&>().gravity)>;

namespace layout {

inline void check_index(int index) {
    if (index < 0 || index >= kParameterCount) {
        throw std::out_of_range("参数编号超出范围");
    }
}

// 惯量的第 slot（4..9）个对称分量所在的行与列
inline void inertia_position(int slot, int& row, int& column) {
    switch (slot) {
        case 4: row = 0; column = 0; return;
        case 5: row = 1; column = 1; return;
        case 6: row = 2; column = 2; return;
        case 7: row = 0; column = 1; return;
        case 8: row = 0; column = 2; return;
        default: row = 1; column = 2; return;
    }
}

}  // namespace layout

template <typename Model>
ModelScalar<Model> parameter_value(const Model& model, int index) {
    layout::check_index(index);
    if (index < kBodyCount * kBodyParameterCount) {
        const int body = index / kBodyParameterCount;
        const int slot = index % kBodyParameterCount;
        const auto& target = model.bodies[body];
        switch (slot) {
            case 0: return target.mass;
            case 1: return target.com[0];
            case 2: return target.com[1];
            case 3: return target.com[2];
            default: break;
        }
        int row = 0;
        int column = 0;
        layout::inertia_position(slot, row, column);
        return target.inertia(row, column);
    }
    if (index < kBodyCount * kBodyParameterCount + kOffsetParameterCount) {
        const int local = index - kBodyCount * kBodyParameterCount;
        return model.joint_offsets[local / 3][local % 3];
    }
    return model.damping[index - kBodyCount * kBodyParameterCount - kOffsetParameterCount];
}

template <typename Model>
void add_parameter(Model& model, int index, const ModelScalar<Model>& delta) {
    layout::check_index(index);
    if (index < kBodyCount * kBodyParameterCount) {
        const int body = index / kBodyParameterCount;
        const int slot = index % kBodyParameterCount;
        auto& target = model.bodies[body];
        switch (slot) {
            case 0: target.mass += delta; return;
            case 1: target.com[0] += delta; return;
            case 2: target.com[1] += delta; return;
            case 3: target.com[2] += delta; return;
            default: break;
        }
        int row = 0;
        int column = 0;
        layout::inertia_position(slot, row, column);
        target.inertia(row, column) += delta;
        if (row != column) {
            target.inertia(column, row) += delta;
        }
        return;
    }
    if (index < kBodyCount * kBodyParameterCount + kOffsetParameterCount) {
        const int local = index - kBodyCount * kBodyParameterCount;
        model.joint_offsets[local / 3][local % 3] += delta;
        return;
    }
    model.damping[index - kBodyCount * kBodyParameterCount - kOffsetParameterCount] += delta;
}

// ModelParameters 上的便捷接口
inline Real value(const ModelParameters& model, int index) {
    return parameter_value(model, index);
}

inline void set_value(ModelParameters& model, int index, Real new_value) {
    add_parameter(model, index, new_value - parameter_value(model, index));
}

// 参数名，长度 kParameterCount
inline const char* const* names() {
    static const char* const kNames[kParameterCount] = {
        "B1.mass", "B1.com_x", "B1.com_y", "B1.com_z",
        "B1.inertia_xx", "B1.inertia_yy", "B1.inertia_zz",
        "B1.inertia_xy", "B1.inertia_xz", "B1.inertia_yz",
        "B2.mass", "B2.com_x", "B2.com_y", "B2.com_z",
        "B2.inertia_xx", "B2.inertia_yy", "B2.inertia_zz",
        "B2.inertia_xy", "B2.inertia_xz", "B2.inertia_yz",
        "B3.mass", "B3.com_x", "B3.com_y", "B3.com_z",
        "B3.inertia_xx", "B3.inertia_yy", "B3.inertia_zz",
        "B3.inertia_xy", "B3.inertia_xz", "B3.inertia_yz",
        "B4.mass", "B4.com_x", "B4.com_y", "B4.com_z",
        "B4.inertia_xx", "B4.inertia_yy", "B4.inertia_zz",
        "B4.inertia_xy", "B4.inertia_xz", "B4.inertia_yz",
        "o1.x", "o1.y", "o1.z", "o2.x", "o2.y", "o2.z",
        "o3.x", "o3.y", "o3.z", "o4.x", "o4.y", "o4.z",
        "d1", "d2", "d3", "d4",
    };
    return kNames;
}

}  // namespace parameters
}  // namespace gimbal
