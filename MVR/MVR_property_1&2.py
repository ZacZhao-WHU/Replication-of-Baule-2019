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


#%% Property 1

T = 100
N = 5
R = np.random.randn(T, N)

def MVR_cov_ret(R, alpha):
    # 每个时期的事后最佳资产收益率
    R_max = R.max(axis=1)            # 形状 (T,)

    # 后悔调整后的收益
    Z = R - alpha * R_max[:, None]   # 形状 (T, N)

    # 后悔调整后收益的协方差
    Sigma_ra = np.cov(Z, rowvar=False, ddof=1)

    return Sigma_ra


def MVR_cov_pref(R, alpha, gamma):
    T, N = R.shape

    # 初始协方差矩阵
    Sigma = np.cov(R, rowvar=False, ddof=1)

    # 事前方差 
    var_assets = np.var(R, axis=0, ddof=1)   # 形状 (N,)

    # 事后偏好值
    # pi_{i,t} = R_{i,t} - gamma * sigma_i^2
    pi = R - gamma * var_assets[None, :]     # 形状 (T, N)

    # 事后最优偏好
    pi_max = pi.max(axis=1)                  # 形状 (T,)

    # 后悔调整随机分量
    Z_pref = R - alpha * pi_max[:, None]

    # 后悔调整后收益的协方差
    Sigma_ra_pref = np.cov(Z_pref, rowvar=False, ddof=1)

    # Property 1
    Sigma_pref = alpha * Sigma + Sigma_ra_pref

    return Sigma_pref


### Return Regret 
# 检验 1：α = 0 退化为 Markowitz
Sigma_0 = MVR_cov_ret(R, alpha=0.0)
np.allclose(Sigma_0, np.cov(R, rowvar=False, ddof=1))

# 检验 2：对称 & 正定
np.allclose(Sigma_0, Sigma_0.T)
np.linalg.eigvalsh(Sigma_0).min() > 0


### Preference Regret 
# 检验 1: α = 0 退化为 Markowitz是否为 True
Sigma_pref_0 = MVR_cov_pref(R, alpha=0.0, gamma=3)
np.allclose(Sigma_pref_0, np.cov(R, rowvar=False, ddof=1))

# 检验 2: 对称性 & 正定性
Sigma_pref = MVR_cov_pref(R, alpha=1, gamma=3)

np.allclose(Sigma_pref, Sigma_pref.T)
np.linalg.eigvalsh(Sigma_pref).min() > 0

# 检验 3: gamma → 0 时，检查preference-regret → return-regret
Sigma_pref_small_gamma = MVR_cov_pref(R, alpha=1, gamma=1e-8)

Sigma_ret = MVR_cov_ret(R, alpha=1)
alpha = 1
np.allclose(Sigma_pref_small_gamma, alpha * np.cov(R, rowvar=False) + Sigma_ret,atol=1e-6)



#%% 按 Property 2 实现 mu 的调整来融入后悔效应，同时保持原始协方差矩阵不变。
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
