import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体 (WSL/Ubuntu: WenQuanYi, Noto Sans CJK)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def simulate_brownian_bridge(a, b, T, n_steps):
    """
    模拟从a到b的布朗桥
    a: 起点
    b: 终点
    T: 总时间
    n_steps: 时间步数
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps)

    # 生成标准布朗运动
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))

    # 构造布朗桥
    B = W - (t / T) * W[-1]

    # 一般布朗桥：从a到b
    X = a + (t / T) * (b - a) + B

    return t, X


# 模拟10条布朗桥路径
a = 0
b = 1
T = 1
n_steps = 200

plt.figure(figsize=(10, 6))
for _ in range(10):
    t, X = simulate_brownian_bridge(a, b, T, n_steps)
    plt.plot(t, X, alpha=0.7)

plt.plot(t, a + (t / T) * (b - a), "k--", label="均值(直线插值)")
plt.scatter([0, T], [a, b], c="red", s=100, zorder=5, label="端点")
plt.xlabel("时间t")
plt.ylabel("状态X(t)")
plt.title("从0到1的布朗桥模拟(10条路径)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
