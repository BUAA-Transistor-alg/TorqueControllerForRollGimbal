#pragma once

// 固定尺寸向量/矩阵与基础运算。
// 行主序存储，模板标量为 double。

#include <cmath>
#include <cstddef>

namespace gimbal {
namespace math {

template <typename Scalar, int N>
struct Vector {
    Scalar data[N];

    Vector() = default;

    static Vector zero() {
        Vector result;
        for (int i = 0; i < N; ++i) {
            result.data[i] = Scalar(0);
        }
        return result;
    }

    static Vector unit(int index) {
        Vector result = zero();
        result.data[index] = Scalar(1);
        return result;
    }

    // 由 N 个分量构造：Vector4d::from(a, b, c, d)
    template <typename... Values>
    static Vector from(Values... values) {
        static_assert(sizeof...(Values) == N, "分量个数必须与维数一致");
        const Scalar items[N] = {static_cast<Scalar>(values)...};
        Vector result;
        for (int i = 0; i < N; ++i) {
            result.data[i] = items[i];
        }
        return result;
    }

    void setZero() {
        for (int i = 0; i < N; ++i) {
            data[i] = Scalar(0);
        }
    }

    Scalar& operator[](int index) { return data[index]; }
    const Scalar& operator[](int index) const { return data[index]; }
};

template <typename Scalar, int Rows, int Columns>
struct Matrix {
    Scalar data[Rows][Columns];

    Matrix() = default;

    static Matrix zero() {
        Matrix result;
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) {
                result.data[r][c] = Scalar(0);
            }
        }
        return result;
    }

    static Matrix identity() {
        static_assert(Rows == Columns, "单位阵要求方阵");
        Matrix result = zero();
        for (int i = 0; i < Rows; ++i) {
            result.data[i][i] = Scalar(1);
        }
        return result;
    }

    void setZero() {
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Columns; ++c) {
                data[r][c] = Scalar(0);
            }
        }
    }

    void setIdentity() {
        static_assert(Rows == Columns, "单位阵要求方阵");
        setZero();
        for (int i = 0; i < Rows; ++i) {
            data[i][i] = Scalar(1);
        }
    }

    Scalar& operator()(int row, int column) { return data[row][column]; }
    const Scalar& operator()(int row, int column) const { return data[row][column]; }
};

// ---- 标量别名 ----
template <typename Scalar>
using Vector3 = Vector<Scalar, 3>;
template <typename Scalar>
using Matrix3 = Matrix<Scalar, 3, 3>;
template <typename Scalar>
using Vector4 = Vector<Scalar, 4>;
template <typename Scalar>
using Matrix4 = Matrix<Scalar, 4, 4>;
template <typename Scalar>
using Matrix34 = Matrix<Scalar, 3, 4>;
template <typename Scalar>
using Vector8 = Vector<Scalar, 8>;
template <typename Scalar>
using Matrix8 = Matrix<Scalar, 8, 8>;
template <typename Scalar>
using Matrix84 = Matrix<Scalar, 8, 4>;
template <typename Scalar>
using Matrix856 = Matrix<Scalar, 8, 56>;

using Vector3d = Vector3<double>;
using Matrix3d = Matrix3<double>;
using Vector4d = Vector4<double>;
using Matrix4d = Matrix4<double>;
using Matrix34d = Matrix34<double>;
using Vector8d = Vector8<double>;
using Matrix8d = Matrix8<double>;
using Matrix84d = Matrix84<double>;

// ---- 向量运算 ----
template <typename Scalar, int N>
Vector<Scalar, N> operator+(const Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    Vector<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
        result.data[i] = lhs.data[i] + rhs.data[i];
    }
    return result;
}

template <typename Scalar, int N>
Vector<Scalar, N> operator-(const Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    Vector<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
        result.data[i] = lhs.data[i] - rhs.data[i];
    }
    return result;
}

template <typename Scalar, int N>
Vector<Scalar, N> operator-(const Vector<Scalar, N>& value) {
    Vector<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
        result.data[i] = -value.data[i];
    }
    return result;
}

template <typename Scalar, int N>
Vector<Scalar, N> operator*(const Vector<Scalar, N>& value, const Scalar& factor) {
    Vector<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
        result.data[i] = value.data[i] * factor;
    }
    return result;
}

template <typename Scalar, int N>
Vector<Scalar, N> operator*(const Scalar& factor, const Vector<Scalar, N>& value) {
    return value * factor;
}

template <typename Scalar, int N>
Vector<Scalar, N> operator/(const Vector<Scalar, N>& value, const Scalar& divisor) {
    Vector<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
        result.data[i] = value.data[i] / divisor;
    }
    return result;
}

template <typename Scalar, int N>
Vector<Scalar, N>& operator+=(Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    for (int i = 0; i < N; ++i) {
        lhs.data[i] += rhs.data[i];
    }
    return lhs;
}

template <typename Scalar, int N>
Vector<Scalar, N>& operator-=(Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    for (int i = 0; i < N; ++i) {
        lhs.data[i] -= rhs.data[i];
    }
    return lhs;
}

template <typename Scalar, int N>
Vector<Scalar, N>& operator*=(Vector<Scalar, N>& lhs, const Scalar& factor) {
    for (int i = 0; i < N; ++i) {
        lhs.data[i] *= factor;
    }
    return lhs;
}

template <typename Scalar, int N>
bool operator==(const Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    for (int i = 0; i < N; ++i) {
        if (!(lhs.data[i] == rhs.data[i])) {
            return false;
        }
    }
    return true;
}

template <typename Scalar, int N>
bool operator!=(const Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    return !(lhs == rhs);
}

template <typename Scalar, int N>
Scalar dot(const Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    Scalar result = Scalar(0);
    for (int i = 0; i < N; ++i) {
        result += lhs.data[i] * rhs.data[i];
    }
    return result;
}

template <typename Scalar>
Vector<Scalar, 3> cross(const Vector<Scalar, 3>& lhs, const Vector<Scalar, 3>& rhs) {
    Vector<Scalar, 3> result;
    result.data[0] = lhs.data[1] * rhs.data[2] - lhs.data[2] * rhs.data[1];
    result.data[1] = lhs.data[2] * rhs.data[0] - lhs.data[0] * rhs.data[2];
    result.data[2] = lhs.data[0] * rhs.data[1] - lhs.data[1] * rhs.data[0];
    return result;
}

// 逐元素相乘（粘性阻尼与角速度）
template <typename Scalar, int N>
Vector<Scalar, N> elementwise(const Vector<Scalar, N>& lhs, const Vector<Scalar, N>& rhs) {
    Vector<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
        result.data[i] = lhs.data[i] * rhs.data[i];
    }
    return result;
}

// ---- 矩阵运算 ----
template <typename Scalar, int R, int C>
Matrix<Scalar, R, C> operator+(const Matrix<Scalar, R, C>& lhs, const Matrix<Scalar, R, C>& rhs) {
    Matrix<Scalar, R, C> result;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            result.data[r][c] = lhs.data[r][c] + rhs.data[r][c];
        }
    }
    return result;
}

template <typename Scalar, int R, int C>
Matrix<Scalar, R, C> operator-(const Matrix<Scalar, R, C>& lhs, const Matrix<Scalar, R, C>& rhs) {
    Matrix<Scalar, R, C> result;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            result.data[r][c] = lhs.data[r][c] - rhs.data[r][c];
        }
    }
    return result;
}

template <typename Scalar, int R, int C>
Matrix<Scalar, R, C> operator*(const Matrix<Scalar, R, C>& value, const Scalar& factor) {
    Matrix<Scalar, R, C> result;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            result.data[r][c] = value.data[r][c] * factor;
        }
    }
    return result;
}

template <typename Scalar, int R, int C>
Matrix<Scalar, R, C> operator*(const Scalar& factor, const Matrix<Scalar, R, C>& value) {
    return value * factor;
}

template <typename Scalar, int R, int C>
Matrix<Scalar, R, C>& operator+=(Matrix<Scalar, R, C>& lhs, const Matrix<Scalar, R, C>& rhs) {
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            lhs.data[r][c] += rhs.data[r][c];
        }
    }
    return lhs;
}

template <typename Scalar, int R, int C>
Matrix<Scalar, R, C>& operator-=(Matrix<Scalar, R, C>& lhs, const Matrix<Scalar, R, C>& rhs) {
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            lhs.data[r][c] -= rhs.data[r][c];
        }
    }
    return lhs;
}

template <typename Scalar, int R, int C>
Matrix<Scalar, C, R> transpose(const Matrix<Scalar, R, C>& value) {
    Matrix<Scalar, C, R> result;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            result.data[c][r] = value.data[r][c];
        }
    }
    return result;
}

template <typename Scalar, int R, int C, int K>
Matrix<Scalar, R, K> operator*(const Matrix<Scalar, R, C>& lhs, const Matrix<Scalar, C, K>& rhs) {
    Matrix<Scalar, R, K> result = Matrix<Scalar, R, K>::zero();
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            for (int k = 0; k < K; ++k) {
                result.data[r][k] += lhs.data[r][c] * rhs.data[c][k];
            }
        }
    }
    return result;
}

// 矩阵乘向量
template <typename Scalar, int R, int C>
Vector<Scalar, R> operator*(const Matrix<Scalar, R, C>& lhs, const Vector<Scalar, C>& rhs) {
    Vector<Scalar, R> result = Vector<Scalar, R>::zero();
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            result.data[r] += lhs.data[r][c] * rhs.data[c];
        }
    }
    return result;
}

// 行向量乘矩阵：a^T * M
template <typename Scalar, int R, int C>
Vector<Scalar, C> operator*(const Vector<Scalar, R>& lhs, const Matrix<Scalar, R, C>& rhs) {
    Vector<Scalar, C> result = Vector<Scalar, C>::zero();
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            result.data[c] += lhs.data[r] * rhs.data[r][c];
        }
    }
    return result;
}

// 4x4 对称正定求解（Cholesky 分解）
inline Vector4<double> solve4(const Matrix4<double>& mass, const Vector4<double>& rhs) {
    double lower[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0},
                          {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = mass.data[i][j];
            for (int k = 0; k < j; ++k) {
                sum -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (!(sum > 0.0)) {
                    return Vector4<double>::zero();
                }
                lower[i][j] = std::sqrt(sum);
            } else {
                lower[i][j] = sum / lower[j][j];
            }
        }
    }
    double intermediate[4];
    for (int i = 0; i < 4; ++i) {
        double sum = rhs.data[i];
        for (int k = 0; k < i; ++k) {
            sum -= lower[i][k] * intermediate[k];
        }
        intermediate[i] = sum / lower[i][i];
    }
    Vector4<double> result;
    for (int i = 3; i >= 0; --i) {
        double sum = intermediate[i];
        for (int k = i + 1; k < 4; ++k) {
            sum -= lower[k][i] * result.data[k];
        }
        result.data[i] = sum / lower[i][i];
    }
    return result;
}

}  // namespace math
}  // namespace gimbal
