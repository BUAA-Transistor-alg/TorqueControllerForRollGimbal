# 全向底盘 + 头颈四自由度串链动力学模型

## 1 模型对象

由五个刚体、四个转动关节组成的串链：

    B0 --(θ1)--> B1 --(θ2)--> B2 --(θ3)--> B3 --(θ4)--> B4

| 刚体 | 名称 | 与上一刚体的关节 |
|---|---|---|
| B0 | 底盘 | —（运动由外部给定） |
| B1 | 平台 | θ1  大 yaw |
| B2 | 小 yaw 电机节点 | θ2  roll |
| B3 | 颈部节点 | θ3  小 yaw |
| B4 | 头部 | θ4  pitch |

建模范围：全部刚体为刚体；关节为理想转动副，带粘性阻尼；每个关节施加已知电机力矩；底盘的全部运动为已知输入，不求解底盘方程。

## 2 记号

- 粗体小写为三维列向量，粗体大写为 3×3 旋转矩阵；向量分量按其所属坐标系的轴给出。
- 旋转一律为主动旋转、右乘（体固轴）：R' = R·R_local。
- 基元旋转矩阵（角度为弧度）：

$$R_x(a)=\begin{bmatrix}1&0&0\\0&\cos a&-\sin a\\0&\sin a&\cos a\end{bmatrix},\quad
R_y(a)=\begin{bmatrix}\cos a&0&\sin a\\0&1&0\\-\sin a&0&\cos a\end{bmatrix},\quad
R_z(a)=\begin{bmatrix}\cos a&-\sin a&0\\\sin a&\cos a&0\\0&0&1\end{bmatrix}$$

- 对任意 q 的函数 X，记号 ∂_q X 表示逐分量偏导数：$(∂_qX)_{j}=\partial X/\partial\theta_j$。
- 下标 i∈{1,2,3,4} 表示第 i 个刚体（B1…B4），下标 j 表示第 j 个关节。
- 反对称矩阵记号 $[\mathbf u]_\times$ 不出现于本文公式，叉乘一律写作 $\mathbf u\times\mathbf v$。

## 3 世界系与地面平面系

世界系 W：x 向右、y 向前、z 向上，右手系。重力加速度

$$\mathbf g_W=(0,\ 0,\ -g)^T$$

地面法向 $\mathbf n$（W 中的单位常向量），用坡角 λ 与坡向 γ 参数化：

$$\mathbf n=(\sin\lambda\cos\gamma,\ \sin\lambda\sin\gamma,\ \cos\lambda)^T$$

地面平面系 P 由 W 沿最短旋转转到法向 $\mathbf n$：

$$R_G=R_z(\gamma)R_y(\lambda)R_z(-\gamma),\qquad R_G\mathbf e_z=\mathbf n$$

底盘姿态由其相对 P 的 yaw 角 ψ 给出：

$$R_0=R_G R_z(\psi)$$

底盘角速度与角加速度：

$$\boldsymbol\omega_0=\dot\psi\,\mathbf n,\qquad \boldsymbol\alpha_0=\ddot\psi\,\mathbf n$$

## 4 几何参数

### 4.1 坐标系

- $B_0$ 原点为底盘参考点。
- $B_i$ 原点位于第 i 条关节轴线上（i=1…4）。
- $\theta_i=0$ 时 $B_i$ 与 $B_{i-1}$ 的轴平行且同向。

### 4.2 轴间安装距离

四个常向量，给出相邻两条关节轴线之间的相对位置：

| 向量 | 起点 | 终点 | 分量所在坐标系 |
|---|---|---|---|
| $\mathbf o_1$ | 底盘参考点 | 大 yaw 轴 | $B_0$ |
| $\mathbf o_2$ | 大 yaw 轴 | roll 轴 | $B_1$ |
| $\mathbf o_3$ | roll 轴 | 小 yaw 轴 | $B_2$ |
| $\mathbf o_4$ | 小 yaw 轴 | pitch 轴 | $B_3$ |

### 4.3 刚体惯量参数

对 i=1…4：

- $m_i$：质量；
- $\mathbf c_i$：质心相对 $B_i$ 原点的偏置，分量在 $B_i$ 中；
- $\mathbf I_i$：绕质心的 3×3 惯量张量，分量在 $B_i$ 轴上（可为非对角，对称正定）。

$B_0$ 不作为刚体进入动力学：其质量与惯量不是参数，也不出现在任何方程中。

### 4.4 关节轴

各轴在其父坐标系与子坐标系中的分量相同：

$$\mathbf a_1=(0,0,1),\quad \mathbf a_2=(0,1,0),\quad \mathbf a_3=(0,0,1),\quad \mathbf a_4=(1,0,0)$$

## 5 广义坐标与输入

广义坐标与速度：

$$q=(\theta_1,\theta_2,\theta_3,\theta_4)^T,\qquad \dot q$$

已知输入：

- 关节电机力矩 $\tau\in\mathbb R^4$，$\tau_j$ 的正方向为绕 $+\mathbf a_j$；
- 粘性阻尼系数 $d_j\ge 0$，$D=\mathrm{diag}(d_1,d_2,d_3,d_4)$；
- 底盘量 $\psi,\ \dot\psi,\ \ddot\psi,\ \mathbf a_0$（$\mathbf a_0$ 为底盘参考点的经典线加速度，W 中表示）；
- 底盘参考点的位置 $p_0$ 与速度 $\mathbf v_0$：二者只出现在 6.1、6.3 的中间量中，不出现在第 7、8 节的公式里；
- 地形量 $\lambda,\ \gamma$；
- 参数 $g$、$m_i,\mathbf c_i,\mathbf I_i$、$\mathbf o_1..\mathbf o_4$、$d_j$。

待求量：$\ddot q\in\mathbb R^4$。

## 6 运动学

### 6.1 位姿

$$p_1=p_0+R_0\mathbf o_1,\qquad p_i=p_{i-1}+R_{i-1}\mathbf o_i\ (i=2,3,4)$$

$$R_1=R_0R_z(\theta_1),\quad R_2=R_1R_y(\theta_2),\quad R_3=R_2R_z(\theta_3),\quad R_4=R_3R_x(\theta_4)$$

其中 $p_0$ 为底盘参考点位置（只出现在 $p_i$ 的定义中，不出现在最终方程）。

质心：

$$\mathbf g_i=p_i+R_i\mathbf c_i,\qquad G_i=\mathbf g_i-p_0$$

### 6.2 关节轴的世界方向

$$\mathbf a_1^{W}=R_0\mathbf e_z,\quad \mathbf a_2^{W}=R_1\mathbf e_y,\quad
\mathbf a_3^{W}=R_2\mathbf e_z,\quad \mathbf a_4^{W}=R_3\mathbf e_x$$

以下简记 $\mathbf a_j$ 为 $\mathbf a_j^{W}$。

### 6.3 速度与雅可比

$$\boldsymbol\omega_i=\boldsymbol\omega_0+\sum_{j\le i}\mathbf a_j\dot\theta_j$$

$$\mathbf v_{\mathbf g_i}=\mathbf v_0+\boldsymbol\omega_0\times G_i+\sum_{j\le i}\mathbf a_j\times(\mathbf g_i-p_j)\,\dot\theta_j$$

$$J^{v}_i[:,j]=\begin{cases}\mathbf a_j\times(\mathbf g_i-p_j), & j\le i\\ \mathbf 0,& j>i\end{cases},
\qquad
J^{\omega}_i[:,j]=\begin{cases}\mathbf a_j, & j\le i\\ \mathbf 0,& j>i\end{cases}$$

### 6.4 加速度

以 $\boldsymbol\omega_0,\ \boldsymbol\alpha_0,\ \mathbf a_{p_0}=\mathbf a_0$ 为初值：

$$\mathbf r_i=R_{i-1}\mathbf o_i$$

$$\mathbf a_{p_i}=\mathbf a_{p_{i-1}}+\boldsymbol\alpha_{i-1}\times\mathbf r_i+\boldsymbol\omega_{i-1}\times(\boldsymbol\omega_{i-1}\times\mathbf r_i)$$

$$\boldsymbol\alpha_i=\boldsymbol\alpha_{i-1}+\mathbf a_i^{W}\ddot\theta_i+\boldsymbol\omega_{i-1}\times(\mathbf a_i^{W}\dot\theta_i)$$

$$\mathbf a_{\mathbf g_i}=\mathbf a_{p_i}+\boldsymbol\alpha_i\times(R_i\mathbf c_i)+\boldsymbol\omega_i\times\big(\boldsymbol\omega_i\times(R_i\mathbf c_i)\big)$$

## 7 动力学

### 7.1 功能量

$$T=\frac12\sum_{i=1}^{4}\Big[m_i\|\mathbf v_{\mathbf g_i}\|^2+\boldsymbol\omega_i^T R_i\mathbf I_iR_i^T\boldsymbol\omega_i\Big],
\qquad
V=-\sum_{i=1}^{4}m_i\,\mathbf g_W\cdot\mathbf g_i$$

### 7.2 惯量矩阵

$$M=\sum_{i=1}^{4}\Big[m_iJ^{vT}_iJ^{v}_i+J^{\omega T}_iR_i\mathbf I_iR_i^TJ^{\omega}_i\Big]$$

$$M_{\theta\psi}=\sum_{i=1}^{4}\Big[m_iJ^{vT}_i(\mathbf n\times G_i)+J^{\omega T}_iR_i\mathbf I_iR_i^T\mathbf n\Big]$$

$$M_{\psi\psi}=\sum_{i=1}^{4}\Big[m_i\|\mathbf n\times G_i\|^2+\mathbf n^TR_i\mathbf I_iR_i^T\mathbf n\Big]$$

$$J_g=\sum_{i=1}^{4}m_iJ^{v}_i\in\mathbb R^{3\times4}$$

$M$ 为 4×4 对称正定矩阵。

### 7.3 已知项

$$C_{qq}(q,\dot q)\big|_j=\sum_{k,l}\Big[\partial_kM_{jl}-\frac12\partial_jM_{kl}\Big]\dot\theta_k\dot\theta_l$$

$$C_{q\psi}(q,\dot q)\big|_j=\dot\psi\sum_k\Big[\partial_kM_{\theta\psi,j}-\partial_jM_{\theta\psi,k}\Big]\dot\theta_k$$

$$b=-J_g^T\mathbf g_W+J_g^T\mathbf a_0+M_{\theta\psi}\ddot\psi
-\frac12\big(\partial_qM_{\psi\psi}\big)\dot\psi^2+C_{qq}+C_{q\psi}$$

### 7.4 运动方程

$$M(q)\,\ddot q+D\,\dot q+b=\tau$$

$$\ddot q=M^{-1}(q)\Big[\tau-D\dot q-b\Big]$$

## 8 计算步骤

给定 $(q,\dot q,\tau,\psi,\dot\psi,\ddot\psi,\mathbf a_0,\lambda,\gamma)$ 与全部参数：

1. 计算 $R_G=R_z(\gamma)R_y(\lambda)R_z(-\gamma)$、$\mathbf n=R_G\mathbf e_z$、$R_0=R_G R_z(\psi)$、$\boldsymbol\omega_0=\dot\psi\mathbf n$、$\boldsymbol\alpha_0=\ddot\psi\mathbf n$。
2. 按 6.1 逐级计算 $p_i,R_i$，再计算 $\mathbf g_i$ 与 $G_i=\mathbf g_i-p_0$。
3. 按 6.2 计算 $\mathbf a_1..\mathbf a_4$。
4. 按 6.3 组装 $J^{v}_i,J^{\omega}_i$；按 7.2 计算 $M,M_{\theta\psi},M_{\psi\psi},J_g$。
5. 计算 7.3 中所需的偏导 $\partial_kM_{jl}$、$\partial_kM_{\theta\psi,j}$、$\partial_kM_{\psi\psi}$（解析表达式或复步长精确求值），组装 $C_{qq},C_{q\psi},b$。
6. 计算 $\ddot q=M^{-1}(\tau-D\dot q-b)$。

递推等价形式（用于直接求 $\tau$ 或 $b$ 与 $M$）：由 6.4 得到 $(\mathbf a_{\mathbf g_i},\boldsymbol\alpha_i)$ 后取

$$\mathbf F_i=m_i(\mathbf a_{\mathbf g_i}-\mathbf g_W),\qquad
\mathbf N_i=R_i\mathbf I_iR_i^T\boldsymbol\alpha_i+\boldsymbol\omega_i\times(R_i\mathbf I_iR_i^T\boldsymbol\omega_i)$$

从 B4 向 B1 递推合力与合力矩（子链的量平移到本关节原点），关节力矩由 $\tau_i=\mathbf a_i^{W}\cdot\mathbf m_i$ 给出（$\mathbf m_i$ 为该关节处的合力矩）。以 $\ddot q=\mathbf 0$ 求值得到 $b$；令 $\dot q=\boldsymbol\omega_0=\boldsymbol\alpha_0=\mathbf a_0=\mathbf 0$，以 $\ddot q=\mathbf e_j$ 求值再减去 $b$，得到 $M$ 的第 $j$ 列。
