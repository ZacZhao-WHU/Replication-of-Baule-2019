import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
import math
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

#%%% 检查

np.random.seed(123)

mu = np.array([0.10, 0.10])        # 期望（年化或单位一致）
sigma = np.array([0.25, 0.25])     # 标准差
rho = 0.5
gamma = 3.0
alpha = 1.0                         # 纯 regret case (alpha=1)；可改为 0.5, 0
n_sims = 2000_000

# 构造协方差矩阵
cov = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]], [rho*sigma[0]*sigma[1], sigma[1]**2]])

# 生成多元正态模拟（每次模拟是两资产的 realized returns）
R = np.random.multivariate_normal(mean=mu, cov=cov, size=n_sims)  # shape (n_sims, 2)

# return-regret: Z_ret_{j,t} = R_{j,t} - alpha * R_max_t
R_max = R.max(axis=1)                         
Z_ret = R - alpha * R_max[:, None]            # shape (n_sims,2)

# 估计 regret-adjusted 均值与协方差
mu_ra_ret = Z_ret.mean(axis=0)
cov_ra_ret = np.cov(Z_ret, rowvar=False, bias=False)

# preference-regret:
pi_vals = R - gamma * (sigma**2)[None, :]     # shape (n_sims,2)
pi_max = pi_vals.max(axis=1)
# Z_pref_{j,t} = R_{j,t} - alpha * pi_max_t - alpha * gamma * sigma_j^2
Z_pref = R - alpha * pi_max[:, None] - alpha * gamma * (sigma**2)[None, :]

mu_ra_pref = Z_pref.mean(axis=0)
cov_ra_pref = np.cov(Z_pref, rowvar=False, bias=False)

# ---- 求解带预算约束的 μ-σ 最优权重（闭式，无卖空限制）
def mean_variance_weights(mu_vec, cov_mat, gamma):
    invS = np.linalg.inv(cov_mat)
    A = invS.dot(mu_vec)
    B = invS.dot(np.ones_like(mu_vec))
    denom = np.ones_like(mu_vec).dot(B)
    lam = (np.ones_like(mu_vec).dot(A) - 2*gamma) / denom
    w = (A - lam * B) / (2*gamma)
    return w

w_markowitz = mean_variance_weights(mu, cov, gamma)
w_ret = mean_variance_weights(mu_ra_ret, cov_ra_ret, gamma)
w_pref = mean_variance_weights(mu_ra_pref, cov_ra_pref, gamma)

# 输出对照
print("Markowitz weights:", w_markowitz)
print("Pure return-regret weights:", w_ret)
print("Pure preference-regret weights:", w_pref)


#%%% Figure 1_Part A

# 按 Property 2 实现 mu 的调整来融入后悔效应，同时保持原始协方差矩阵不变。
np.random.seed(1114)
sigma = np.array([0.25, 0.25])
rho = 0.5
cov_base = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],[rho*sigma[0]*sigma[1], sigma[1]**2]])
gamma = 3
alpha = 1
mu2 = 0.1
# 当资产1的预期收益从0%变化到20%时，最优权重如何变化
mu1_grid = np.linspace(0, 0.2, 25)
n_grid = len(mu1_grid)
n_sims = 2000000

w_mk = np.zeros(n_grid)
w_ret = np.zeros(n_grid)
w_pref = np.zeros(n_grid)

# 遍历不同的资产1预期收益
for i, mu1 in enumerate(mu1_grid): 
    mu = np.array([mu1, mu2])
    R = np.random.multivariate_normal(mean=mu, cov=cov_base, size=n_sims)  # 对每个mu1值，生成二元正态收益数据
    R_max = R.max(axis=1)
    # 计算每个资产收益与最佳资产收益的协方差
    cov_R_Rmax = np.array([np.cov(R[:,0], R_max, bias=False)[0,1], np.cov(R[:,1], R_max, bias=False)[0,1]]) # 提取协方差矩阵中第0行、第1列的元素
    
    # (sigma**2)[None, :] 将形状从 (2,) 变为 (1, 2)
    pi_values = R - gamma * (sigma**2)[None, :] # sigma**2 是一个形状为 (2,) 的一维数组  R 是一个形状为 (n_sims, 2) 的二维数组；
    pi_max = pi_values.max(axis=1)
    cov_R_pimax = np.array([np.cov(R[:,0], pi_max, bias=False)[0,1], np.cov(R[:,1], pi_max, bias=False)[0,1]])

    # 传统 Markowitz
    invS = np.linalg.inv(cov_base)                         
    A = invS.dot(mu)                                       
    B = invS.dot(np.ones_like(mu))                         
    denom = np.ones_like(mu).dot(B)                        
    lam = (np.ones_like(mu).dot(A) - 2*gamma) / denom      
    w = (A - lam * B) / (2*gamma)
    w_mk[i] = w[0]    # 存储的是在不同μ₁值下，资产1的最优权重

    # 收益后悔 仍使用原始收益(未调整）的协方差矩阵的逆。
    mu_ra_ret = mu + 2*alpha*gamma*cov_R_Rmax              # Property 2 公式9                    
    A_ret = invS.dot(mu_ra_ret)
    lam_ret = (np.ones_like(mu).dot(A_ret) - 2*gamma) / denom
    w_r = (A_ret - lam_ret * B) / (2*gamma)
    w_ret[i] = w_r[0]

    # 偏好后悔 仍使用原始收益(未调整）的协方差矩阵的逆。
    mu_ra_pref = mu + 2*alpha*gamma*cov_R_pimax
    gamma_pref = gamma * (1 + alpha)                       # 在偏好后悔下，投资者对风险更加厌恶                    
    A_pref = invS.dot(mu_ra_pref)
    lam_pref = (np.ones_like(mu).dot(A_pref) - 2*gamma_pref) / denom
    w_p = (A_pref - lam_pref * B) / (2*gamma_pref)
    w_pref[i] = w_p[0]
    
plt.figure(figsize=(5.5, 3))
plt.plot(mu1_grid, w_mk,   linestyle='-',   label='Markowitz')
plt.plot(mu1_grid, w_ret,  linestyle='--',  label='Pure Regret(Ret)')
plt.plot(mu1_grid, w_pref, linestyle=':', linewidth=2.5, label='Pure Regret(Pref)')
plt.xlim(0.05, 0.15)
plt.ylim(0.2, 0.8)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.xlabel('资产1预期收益', fontsize=13, fontweight='heavy')
plt.ylabel('资产1权重',     fontsize=13, fontweight='heavy')
plt.title('Part A: 资产1组合权重', fontsize=15, fontweight='heavy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%% Figure 1_Part B

np.random.seed(1114)
sigma = np.array([0.25, 0.25])
rho = 0.5
cov_base = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],[rho*sigma[0]*sigma[1], sigma[1]**2]])

mu2 = 0.10
mu1_grid = np.linspace(0.00, 0.20, 25)
n_sims = 2000000 

# 用来记录 Part B 的两个协方差序列
cov_R1_Rmax = np.zeros(len(mu1_grid))
cov_R2_Rmax = np.zeros(len(mu1_grid))

# 逐点循环
for i, mu1 in enumerate(mu1_grid):
    mu = np.array([mu1, mu2])  # 当前期望
    R = np.random.multivariate_normal(mean=mu, cov=cov_base, size=n_sims)  # 模拟 R（shape: n_sims x 2）
    R_max = R.max(axis=1)
    # 计算两个协方差： Cov(R1, R_max), Cov(R2, R_max)
    cov_R1_Rmax[i] = np.cov(R[:,0], R_max, bias=False)[1,0]
    cov_R2_Rmax[i] = np.cov(R[:,1], R_max, bias=False)[0,1]
    
# 绘图（Part B）
plt.figure(figsize=(5.5,3))
plt.plot(mu1_grid, cov_R1_Rmax, linestyle='-', label='Cov(R1, R_max)')
plt.plot(mu1_grid, cov_R2_Rmax, linestyle='--', label='Cov(R2, R_max)')
plt.xlim(0.05, 0.15)
plt.ylim(0.04, 0.06)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.005))
plt.xlabel('资产1预期收益', fontsize=13, fontweight='heavy')
plt.ylabel('Cov(R, R_max', fontsize=13, fontweight='heavy')
plt.title('Part B: 与事后最佳投资组合的协方差', fontsize=15, fontweight='heavy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%% Figure 2_Part A

np.random.seed(1114)
mu = np.array([0.10, 0.10])   # 两资产相同期望收益
sigma2 = 0.25                 # asset2 固定的标准差
rho = 0.5
gamma = 3.0
alpha = 1.0

# sigma1 网格（资产1 标准差从 0.05 到 0.50）
sigma1_grid = np.linspace(0.05, 0.50, 25)
n_grid = len(sigma1_grid)
n_sims = 2000000

w_mk = np.zeros(n_grid)
w_ret = np.zeros(n_grid)
w_pref = np.zeros(n_grid)

# 逐点循环
for i, s1 in enumerate(sigma1_grid):
    cov = np.array([[s1**2, rho*s1*sigma2], [rho*s1*sigma2, sigma2**2]])  # 当前的协方差矩阵（随 s1 变化）
    R = np.random.multivariate_normal(mean=mu, cov=cov, size=n_sims)
    R_max = R.max(axis=1)
    # 计算 Cov(R_j, R_max) 的向量
    cov_R_Rmax = np.array([np.cov(R[:,0], R_max, bias=False)[0,1], np.cov(R[:,1], R_max, bias=False)[0,1]])
    pi_vals = R - gamma * np.array([s1**2, sigma2**2])[None, :]
    pi_max = pi_vals.max(axis=1)
    cov_R_pimax = np.array([np.cov(R[:,0], pi_max, bias=False)[0,1], np.cov(R[:,1], pi_max, bias=False)[0,1]])

    # Markowitz 权重
    invS = np.linalg.inv(cov)
    A = invS.dot(mu)
    B = invS.dot(np.ones_like(mu))
    lam = (np.ones_like(mu).dot(A) - 2*gamma) / (np.ones_like(mu).dot(B))
    w = (A - lam * B) / (2*gamma)
    w_mk[i] = w[0]

    # 收益后悔
    mu_ra_ret = mu + 2*alpha*gamma*cov_R_Rmax
    A_ret = invS.dot(mu_ra_ret)
    lam_ret = (np.ones_like(mu).dot(A_ret) - 2*gamma) / (np.ones_like(mu).dot(B))
    w_r = (A_ret - lam_ret * B) / (2*gamma)
    w_ret[i] = w_r[0]

    # 偏好后悔
    mu_ra_pref = mu + 2*alpha*gamma*cov_R_pimax
    gamma_pref = gamma * (1 + alpha)
    A_pref = invS.dot(mu_ra_pref)
    lam_pref = (np.ones_like(mu).dot(A_pref) - 2*gamma_pref) / (np.ones_like(mu).dot(B))
    w_p = (A_pref - lam_pref * B) / (2*gamma_pref)
    w_pref[i] = w_p[0]
    
# 绘图（Part A）
plt.figure(figsize=(5.5,3))
plt.plot(sigma1_grid, w_mk, color='black', linewidth=2, label='Markowitz')
plt.plot(sigma1_grid, w_ret, linestyle='--', linewidth=2, label='Pure Regret(Ret)')
plt.plot(sigma1_grid, w_pref, color='orange', linestyle=':', linewidth=2.5, label='Pure Regret(Pref)')
plt.xlim(0.1, 0.4)
plt.ylim(0, 1)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.xlabel('资产1标准差', fontsize=13, fontweight='heavy')
plt.ylabel('资产1权重', fontsize=13, fontweight='heavy')
plt.title('Part A: 资产1组合权重', fontsize=16, fontweight='heavy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%% Figure 2_Part B

np.random.seed(1115)
mu = np.array([0.10, 0.10])
sigma2 = 0.25
rho = 0.5
gamma = 3.0
sigma1_grid = np.linspace(0.05, 0.50, 25)
n_sims = 2000000

cov_R1_Rmax = np.zeros(len(sigma1_grid))
cov_R2_Rmax = np.zeros(len(sigma1_grid))
cov_R1_pimax = np.zeros(len(sigma1_grid))
cov_R2_pimax = np.zeros(len(sigma1_grid))

# 逐点计算
for i, s1 in enumerate(sigma1_grid):
    cov = np.array([[s1**2, rho*s1*sigma2],[rho*s1*sigma2, sigma2**2]])
    
    R = np.random.multivariate_normal(mean=mu, cov=cov, size=n_sims)
    R_max = R.max(axis=1)
    cov_R1_Rmax[i] = np.cov(R[:,0], R_max, bias=False)[0,1]
    cov_R2_Rmax[i] = np.cov(R[:,1], R_max, bias=False)[0,1]
    
    pi_vals = R - gamma * np.array([s1**2, sigma2**2])[None, :]
    pi_max = pi_vals.max(axis=1)
    cov_R1_pimax[i] = np.cov(R[:,0], pi_max, bias=False)[0,1]
    cov_R2_pimax[i] = np.cov(R[:,1], pi_max, bias=False)[0,1]
    
# 绘图 
plt.figure(figsize=(5.5,3))
plt.plot(sigma1_grid, cov_R1_Rmax, color='black', linewidth=2, label='Cov(R1, R_max(Ret))')
plt.plot(sigma1_grid, cov_R2_Rmax, color='black', linestyle=':', linewidth=2.5, label='Cov(R2, R_max(Ret))')
plt.plot(sigma1_grid, cov_R1_pimax, color='orange', linestyle='-', linewidth=1.5, label='Cov(R1, pi_max(Pref))')
plt.plot(sigma1_grid, cov_R2_pimax, color='orange', linestyle=':', linewidth=2.5, label='Cov(R2, pi_max(Pref))')
plt.xlim(0.1, 0.4)
plt.ylim(0, 0.12)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))

plt.xlabel('资产1标准差')
plt.ylabel('Cov(R,R_max)')
plt.title('Part B: 与事后最佳投资组合的协方差')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%% Fig 3

np.random.seed(1115)

# 预期收益不再相同
mu = np.array([0.05, 0.10])   # 和 Fig.2 最大区别：mu1=5%, mu2=10%
sigma2 = 0.25                 # asset2 固定波动率
rho = 0.5
gamma = 3.0
alpha = 1.0
# sigma1 网格（资产1的标准差）
sigma1_grid = np.linspace(0.05, 0.50, 25)
n_sims = 2000000
w_mk = np.zeros(len(sigma1_grid))
w_ret = np.zeros(len(sigma1_grid))
w_pref = np.zeros(len(sigma1_grid))

# 循环逐点计算
for i, s1 in enumerate(sigma1_grid):
    cov = np.array([[s1**2, rho*s1*sigma2], [rho*s1*sigma2, sigma2**2]])
    
    R = np.random.multivariate_normal(mean=mu, cov=cov, size=n_sims)
    R_max = R.max(axis=1)
    cov_R_Rmax = np.array([np.cov(R[:,0], R_max, bias=False)[0,1], np.cov(R[:,1], R_max, bias=False)[0,1]])
    
    pi_vals = R - gamma * np.array([s1**2, sigma2**2])[None, :]
    pi_max = pi_vals.max(axis=1)
    cov_R_pimax = np.array([np.cov(R[:,0], pi_max, bias=False)[0,1], np.cov(R[:,1], pi_max, bias=False)[0,1]])

    # Markowitz
    invS = np.linalg.inv(cov)
    A = invS.dot(mu)
    B = invS.dot(np.ones_like(mu))
    lam = (np.ones_like(mu).dot(A) - 2*gamma) / (np.ones_like(mu).dot(B))
    w = (A - lam * B) / (2*gamma)
    w_mk[i] = w[0]

    # 收益后悔
    mu_ra_ret = mu + 2*alpha*gamma*cov_R_Rmax
    A_ret = invS.dot(mu_ra_ret)
    lam_ret = (np.ones_like(mu).dot(A_ret) - 2*gamma) / (np.ones_like(mu).dot(B))
    w_r = (A_ret - lam_ret * B) / (2*gamma)
    w_ret[i] = w_r[0]

    # 偏好后悔
    mu_ra_pref = mu + 2*alpha*gamma*cov_R_pimax
    gamma_pref = gamma * (1 + alpha)
    A_pref = invS.dot(mu_ra_pref)
    lam_pref = (np.ones_like(mu).dot(A_pref) - 2*gamma_pref) / (np.ones_like(mu).dot(B))
    w_p = (A_pref - lam_pref * B) / (2*gamma_pref)
    w_pref[i] = w_p[0]
    
# 绘图
plt.figure(figsize=(5.5, 3))
plt.plot(sigma1_grid, w_mk,  linewidth=2, label='Markowitz')
plt.plot(sigma1_grid, w_ret, linestyle='--', linewidth=2, label='Pure Regret(Ret)')
plt.plot(sigma1_grid, w_pref, linestyle=':',  linewidth=2.5, label='Pure Regret(Pref)')

plt.xlim(0.1, 0.4)
plt.ylim(0, 1)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.xlabel('资产1标准差')
plt.ylabel('资产1权重')
plt.title('资产1组合权重： (mu1=5%, mu2=10%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%%% Fig 4

np.random.seed(1116)
mu = np.array([0.10, 0.10])      # 两资产收益相同
sigma = np.array([0.25, 0.25])   # 两资产标准差相同
rho = 0.5
gamma = 3.0
alpha = 1.0

# 偏度设置（-1 到 +1）
skew_grid = np.linspace(-1.0, 1.0, 25)
n_sims = 2000000
w_mk   = np.zeros(len(skew_grid))
w_ret  = np.zeros(len(skew_grid))
w_pref = np.zeros(len(skew_grid))

# 协方差矩阵固定（资产2 正态；资产1 方差不变，仅改变偏度）
cov_base = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],[rho*sigma[0]*sigma[1], sigma[1]**2]])

for i, skew in enumerate(skew_grid):
    # 生成相关正态 Z1, Z2
    L = np.linalg.cholesky(np.array([[1, rho],[rho, 1]]))
    Z = np.random.randn(n_sims, 2).dot(L.T)
    Z1 = Z[:,0]
    Z2 = Z[:,1]

    # 生成 Gram–Charlier 偏度随机变量
    c = 0.75
    X = Z1 + c * skew * (Z1**2 - 1)
    X_std = (X - np.mean(X)) / np.std(X)   # 标准化以确保 Var(X)=1
    R1 = mu[0] + sigma[0] * X_std          # 转换为资产 1 的收益
    R2 = mu[1] + sigma[1] * Z2             # 资产 2 正态
    R = np.column_stack([R1, R2])
    R_max = R.max(axis=1)                  # 计算 R_max 与 pi_max

    cov_R_Rmax = np.array([np.cov(R[:,0], R_max, bias=False)[0,1], np.cov(R[:,1], R_max, bias=False)[0,1]])
    pi_vals = R - gamma * (sigma**2)[None,:]
    pi_max = pi_vals.max(axis=1)
    cov_R_pimax = np.array([np.cov(R[:,0], pi_max, bias=False)[0,1],np.cov(R[:,1], pi_max, bias=False)[0,1]])

    invS = np.linalg.inv(cov_base)
    B = invS.dot(np.ones_like(mu))

    # Markowitz
    A = invS.dot(mu)
    lam = (np.ones_like(mu).dot(A) - 2*gamma) / (np.ones_like(mu).dot(B))
    w = (A - lam * B) / (2*gamma)
    w_mk[i] = w[0]

    # 收益后悔
    mu_ra_ret = mu + 2*alpha*gamma*cov_R_Rmax
    A_ret = invS.dot(mu_ra_ret)
    lam_ret = (np.ones_like(mu).dot(A_ret) - 2*gamma) / (np.ones_like(mu).dot(B))
    w_r = (A_ret - lam_ret * B) / (2*gamma)
    w_ret[i] = w_r[0]

    # 偏好后悔
    mu_ra_pref = mu + 2*alpha*gamma*cov_R_pimax
    gamma_pref = gamma * (1 + alpha)
    A_pref = invS.dot(mu_ra_pref)
    lam_pref = (np.ones_like(mu).dot(A_pref) - 2*gamma_pref) / (np.ones_like(mu).dot(B))
    w_p = (A_pref - lam_pref * B) / (2*gamma_pref)
    w_pref[i] = w_p[0]
    
plt.figure(figsize=(5.5, 3))
plt.plot(skew_grid, w_mk,  linewidth=2, label='Markowitz')
plt.plot(skew_grid, w_ret, linestyle='--', linewidth=2, label='Pure Regret(Ret)')
plt.plot(skew_grid, w_pref, linestyle=':',  linewidth=2.5, label='Pure Regret(Pref)')

plt.ylim(0.3, 0.7)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))

plt.xlabel('资产1偏度')
plt.ylabel('资产1权重')
plt.title('Part A: 资产1组合权重')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%% Fig 5

np.random.seed(1117)
n_assets = 3
mu = np.full(n_assets, 0.10)    # 全部资产均值 10%
sigma = np.full(n_assets, 0.25) # 全部资产波动 25%
rho_other = 0.5                 # 其他资产之间的相关系数（除 asset1 外）
gamma = 3.0
alpha = 1.0

# rho 网格：改变资产1与其他每个资产的相关性
rho_grid = np.linspace(0.0, 0.95, 25)  # 从 0 到 0.95，25 点
n_grid = len(rho_grid)
n_sims = 2000000

# 资产1的权重
w_mk = np.zeros(n_grid)
w_ret = np.zeros(n_grid)
w_pref = np.zeros(n_grid)

# 逐点循环
for i, rho_var in enumerate(rho_grid):
    # 构造相关矩阵 Corr（3x3）
    Corr = np.full((n_assets, n_assets), rho_other)  # 先填充 0.5
    np.fill_diagonal(Corr, 1.0)                      # 对角为 1
    Corr[0,1] = rho_var # 把资产1与其余资产的相关性改为 rho_var
    Corr[1,0] = rho_var
    Corr[0,2] = rho_var
    Corr[2,0] = rho_var # Corr[1,2] 与 Corr[2,1] 保持 rho_other (0.5)
    
    Sigma = np.zeros((n_assets, n_assets)) # 根据 Corr 与 sigma 构造协方差矩阵 Sigma
    for a in range(n_assets):
        for b in range(n_assets):
            Sigma[a,b] = sigma[a] * sigma[b] * Corr[a,b]
    R = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_sims)
    R_max = R.max(axis=1)  # 计算事后最优资产收益 R_max （每一期在 n_assets 中取最大）

    # 计算向量 Cov(R_j, R_max)
    cov_R_Rmax = np.zeros(n_assets)
    for j in range(n_assets):
        cov_R_Rmax[j] = np.cov(R[:, j], R_max, bias=False)[0,1]

    # 计算 pi = R - gamma*sigma_j^2，并取 pi_max
    pi_vals = R - gamma * (sigma**2)[None, :]
    pi_max = pi_vals.max(axis=1)

    # 计算 Cov(R_j, pi_max)
    cov_R_pimax = np.zeros(n_assets)
    for j in range(n_assets):
        cov_R_pimax[j] = np.cov(R[:, j], pi_max, bias=False)[0,1]

    # 准备线性代数项（用于闭式解）
    invS = np.linalg.inv(Sigma)      # 论文在 Property2 中用原始 Sigma
    one_vec = np.ones(n_assets)
    B = invS.dot(one_vec)

    # Markowitz: A = invS * mu，求 lambda 与 w
    A = invS.dot(mu)
    lam = (one_vec.dot(A) - 2*gamma) / (one_vec.dot(B))
    w = (A - lam * B) / (2*gamma)
    w_mk[i] = w[0]   # asset1 权重

    # 收益后悔: mu_ra = mu + 2*alpha*gamma*cov_R_Rmax，gamma 不变
    mu_ra_ret = mu + 2*alpha*gamma*cov_R_Rmax
    A_ret = invS.dot(mu_ra_ret)
    lam_ret = (one_vec.dot(A_ret) - 2*gamma) / (one_vec.dot(B))
    w_ret_vec = (A_ret - lam_ret * B) / (2*gamma)
    w_ret[i] = w_ret_vec[0]

    # 偏好后悔: mu_ra = mu + 2*alpha*gamma*cov_R_pimax，gamma_pref = gamma*(1+alpha)
    mu_ra_pref = mu + 2*alpha*gamma*cov_R_pimax
    gamma_pref = gamma * (1 + alpha)
    A_pref = invS.dot(mu_ra_pref)
    lam_pref = (one_vec.dot(A_pref) - 2*gamma_pref) / (one_vec.dot(B))
    w_pref_vec = (A_pref - lam_pref * B) / (2*gamma_pref)
    w_pref[i] = w_pref_vec[0]

# 绘图（Fig.5 Part A）
plt.figure(figsize=(5.5,3))
plt.plot(rho_grid, w_mk, label='Markowitz', linewidth=2)
plt.plot(rho_grid, w_ret, linestyle='--', label='Pure Regret(Ret)', linewidth=2)
plt.plot(rho_grid, w_pref, linestyle=':', label='Pure Regret(Pref)', linewidth=2.5)

plt.ylim(0, 0.45)
plt.xlim(0.25, 0.75)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))

plt.xlabel('与资产1的相关性')
plt.ylabel('资产1权重')
plt.title('Part A: 资产1组合权重: 三资产情境')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%% Fig 6

np.random.seed(1118)

mu = np.array([0.05, 0.15])
sigma = np.array([0.10, 0.40])
rho = 0.5
alpha = 1.0
cov = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],[rho*sigma[0]*sigma[1], sigma[1]**2]])
gamma_grid = np.linspace(0.0, 10.0, 25)
n_sims = 2000000

w_mk = np.zeros(len(gamma_grid))
w_ret = np.zeros(len(gamma_grid))
w_pref = np.zeros(len(gamma_grid))

# 预先计算 invS, B（Sigma 不随 gamma 变）
invS = np.linalg.inv(cov)
one = np.ones(2)
B = invS.dot(one)

for idx, gamma in enumerate(gamma_grid):
    # 解析边界：gamma==0
    if gamma == 0.0:
        best = np.argmax(mu)
        w = np.zeros(2); w[best] = 1.0
        w_mk[idx] = w_ret[idx] = w_pref[idx] = w[0]
        continue

    R = np.random.multivariate_normal(mean=mu, cov=cov, size=n_sims)
    R_max = R.max(axis=1)
    cov_R_Rmax = np.array([np.cov(R[:,0], R_max, bias=False)[0,1], np.cov(R[:,1], R_max, bias=False)[0,1]])

    pi = R - gamma * (sigma**2)[None,:]
    pi_max = pi.max(axis=1)
    cov_R_pimax = np.array([np.cov(R[:,0], pi_max, bias=False)[0,1], np.cov(R[:,1], pi_max, bias=False)[0,1]])

    # Markowitz
    A = invS.dot(mu)
    lam = (one.dot(A) - 2*gamma) / (one.dot(B))
    w = (A - lam * B) / (2*gamma)
    w_mk[idx] = w[0]

    # 收益后悔
    mu_ra_ret = mu + 2*alpha*gamma*cov_R_Rmax
    A_ret = invS.dot(mu_ra_ret)
    lam_ret = (one.dot(A_ret) - 2*gamma) / (one.dot(B))
    w_r = (A_ret - lam_ret * B) / (2*gamma)
    w_ret[idx] = w_r[0]

    # 偏好后悔
    mu_ra_pref = mu + 2*alpha*gamma*cov_R_pimax
    gamma_pref = gamma * (1 + alpha)
    A_pref = invS.dot(mu_ra_pref)
    lam_pref = (one.dot(A_pref) - 2*gamma_pref) / (one.dot(B))
    w_p = (A_pref - lam_pref * B) / (2*gamma_pref)
    w_pref[idx] = w_p[0]

plt.figure(figsize=(5.5, 3))
plt.plot(gamma_grid, w_mk,  linewidth=2, label='Markowitz')
plt.plot(gamma_grid, w_ret, linestyle='--', linewidth=2, label='Pure Regret(Ret)')
plt.plot(gamma_grid, w_pref, linestyle=':',  linewidth=2, label='Pure Regret(Pref)')

plt.ylim(0, 1)
plt.xlim(0, 5)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.xlabel('风险厌恶系数')
plt.ylabel('资产1权重')
plt.title('风险厌恶系数对权重的影响')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()    


#%%% Fig 7

np.random.seed(202402)
mu = np.array([0.05, 0.15])
sigma = np.array([0.10, 0.40])
rho = 0.5
gamma = 3.0
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]    
n_sims = 2000000                      

Sigma = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]], [rho*sigma[0]*sigma[1], sigma[1]**2]])

# 模拟收益序列
R = np.random.multivariate_normal(mu, Sigma, n_sims)

# j* = argmax(mu - gamma*sigma²) 并取 R_opt = 单资产最优收益序列
phi = mu - gamma*(sigma**2)
j_star = int(np.argmax(phi))
R_opt = R[:, j_star]

cov_vec = np.array([np.cov(R[:,0], R_opt)[0,1], np.cov(R[:,1], R_opt)[0,1]])

# 横轴sigma, 纵轴mu^ra
w1 = np.linspace(-1.0,2.0,400)

plt.figure(figsize=(7, 6))
for alpha in alphas:
    mu_adj = mu - 2*alpha*gamma*cov_vec
    stds=[]
    rets=[]
    for w in w1:
        wv = np.array([w,1-w])
        stds.append(np.sqrt(wv@Sigma@wv))
        rets.append(wv@mu_adj)
    plt.plot(stds,rets,label=f"α={alpha}",linewidth=2)

plt.xlim(0, 0.45)
plt.xticks(np.arange(0, 0.46, 0.05))
plt.ylim(-0.2, 0.2)
plt.yticks(np.arange(-0.2, 0.21, 0.05))

plt.xlabel('标准差')
plt.ylabel('预期收益')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()