import math
import matplotlib.pyplot as plt
from sympy import factorint
import numpy as np

# 计算C_x的第一部分：奇素因子乘积
def compute_Cx_part1(x):
    factors = factorint(x)
    primes = [p for p in factors.keys() if p > 2]
    product = 1.0
    for p in primes:
        product *= (p - 1) / (p - 2)
    return product

# 常数C_x的第二部分（已知近似值）
Cx_part2 = 0.66016

# 计算下界值
def lower_bound(x):
    if x % 2 != 0:
        return 0  # 仅处理偶数
    part1 = compute_Cx_part1(x)
    Cx = part1 * Cx_part2
    log_x = math.log(x)
    bound = (0.67 * x * Cx) / (log_x ** 2)
    return bound

# 生成x的范围（从10^3到10^50，以指数间隔取点）
x_values = [int(10**i) for i in np.linspace(3, 50, 1000)]
x_values = [x if x % 2 == 0 else x+1 for x in x_values]  # 确保偶数

# 计算对应的下界值
bounds = [lower_bound(x) for x in x_values]

# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(x_values, bounds, color='darkred', linewidth=2, label="Chen's Lower Bound")
plt.axhline(y=1, color='gray', linestyle='--', label="Lower Bound = 1")

# Set logarithmic scale for axes
plt.xscale('log')
plt.yscale('log')

# Add labels and title
plt.title("Chen's Theorem Lower Bound vs Even Number $x$\n(Formula: $P_x(1, 2) \\geq \\frac{0.67 x C_x}{(\\log x)^2}$)", fontsize=14)
plt.xlabel("Even Number $x$", fontsize=12)
plt.ylabel("Lower Bound Value", fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()