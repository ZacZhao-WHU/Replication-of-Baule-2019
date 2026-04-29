import cvxpy as cp
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


#%% 数据处理

# 导入CRSP数据
mom = pd.read_csv('E:/Replications/2025_Fall/Baule_2019_Markowitz with regret/F-F_Momentum_Factor_daily.csv')
ff3 = pd.read_csv('E:/Replications/2025_Fall/Baule_2019_Markowitz with regret/F-F_Research_Data_Factors_daily.csv')
ff3["date"] = pd.to_datetime(ff3["date"], format="%Y%m%d")
mom["date"] = pd.to_datetime(mom["date"], format="%Y%m%d")

ff4 = pd.merge(ff3, mom, on="date", how="inner")
start_date = "1926-11-01"
end_date   = "2016-12-31"
ff4 = ff4[(ff4["date"] >= start_date) & (ff4["date"] <= end_date)]

# 选择用于 Table 3 的四个风格列（已为超额收益）
assets = ['Mkt-RF', 'SMB', 'HML', 'MOM']
df = ff4.copy()

# 确保按日期排序且去掉含 NaN 的行
df = df.sort_values('date').reset_index(drop=True)
df_assets = df[assets].astype(float).dropna()
df_assets = df_assets/100
df_assets


#%% Table3 PanelA_stats
# 计算每个资产的基本统计量
annual_factor = 252
assets = df_assets.columns
stats_calculations = [('Mean Return', df_assets.mean() * annual_factor), 
                      ('Std. Deviation', df_assets.std() * np.sqrt(annual_factor)),
                      ('Skewness', df_assets.skew())]

stats_data = []
index = []

for stat_name, values in stats_calculations:
    stats_data.append(values.round(4))
    index.append(stat_name)
    deviations = values - values.mean()
    stats_data.append(pd.Series([f"({dev:.4f})" for dev in deviations], index=assets))
    index.append('')

panel_a_stats = pd.DataFrame(stats_data, index=index, columns=assets)


# Table3 PanelA_regret_adjustment

R = df_assets.values        # 形状 (T, 4)
asset_names = df_assets.columns.tolist()
T, N = R.shape
alpha = 1.0
gamma = 3.0
R_max = R.max(axis=1)       # 形状 (T,)

def deviation(x):
    return x - x.mean()

#%%% 两种后悔视角
# Ret Regret 
# 协方差矩阵 [R, R_max]
ret_ra = np.array([np.cov(R[:, i], R_max, ddof=1)[0, 1] for i in range(N)]) * annual_factor
Z_ret = R - alpha * R_max[:, None]
std_regret_ret = Z_ret.std(axis=0, ddof=1) * np.sqrt(annual_factor)

panelA = pd.DataFrame({'Regret adjustment μ': ret_ra,
                       'Regret-adjusted std': std_regret_ret},
                      index=asset_names)

panelA_dev = panelA.apply(deviation)

panelA_table3 = pd.DataFrame(
    np.vstack([panelA['Regret adjustment μ'].values, 
               panelA_dev['Regret adjustment μ'].values, 
               panelA['Regret-adjusted std'].values, 
               panelA_dev['Regret-adjusted std'].values]),
    index=['Ret-adj μ', '(deviation)', 'Ret-adj std', '(deviation)'],columns=asset_names)

panelA_table3.round(4)


# Pref Regret 
var_assets = np.var(R, axis=0, ddof=1)    # (N,)
pi = R - gamma * var_assets[None, :]      # (T, N)
# 事后最优pi
pi_max = pi.max(axis=1)                   # (T,)
pref_ra = np.array([np.cov(R[:, i], pi_max, ddof=1)[0, 1]for i in range(N)]) * annual_factor
Z_pref = R - alpha * pi_max[:, None]
std_regret_pref = Z_pref.std(axis=0, ddof=1) * np.sqrt(annual_factor)

panelA_pref = pd.DataFrame({'Regret adjustment μ': pref_ra,
                            'Regret-adjusted std': std_regret_pref},
                           index=asset_names)

panelA_pref_dev = panelA_pref.apply(deviation)

panelA_pref_table3 = pd.DataFrame(
    np.vstack([panelA_pref['Regret adjustment μ'].values, panelA_pref_dev['Regret adjustment μ'].values,
               panelA_pref['Regret-adjusted std'].values, panelA_pref_dev['Regret-adjusted std'].values]),
    index=['Pref-adj μ (u–σ)', '(deviation)', 'Pref-adj std (u–σ)', '(deviation)'],
    columns=asset_names)
panelA_pref_table3.round(4)

# 检查
gamma_small = 1e-8

var_assets = np.var(R, axis=0, ddof=1)
pi_small = R - gamma_small * var_assets[None, :]
pi_max_small = pi_small.max(axis=1)

# preference → return
np.allclose(pi_max_small, R.max(axis=1), atol=1e-6)


#%%% Min-Var

# 协方差矩阵
Sigma = np.cov(R, rowvar=False, ddof=1)

# Min-Var 权重
ones = np.ones(N)
Sigma_inv = np.linalg.inv(Sigma)
w_mv = Sigma_inv @ ones
w_mv = w_mv / (ones @ Sigma_inv @ ones)

# Min-Var 基准收益
R_mv = R @ w_mv    # (T,)

pref_ra_minvar = np.array([np.cov(R[:, i], R_mv, ddof=1)[0, 1] for i in range(N)]) * annual_factor

Z_pref_mv = R - alpha * R_mv[:, None]
std_regret_pref_mv = Z_pref_mv.std(axis=0, ddof=1) * np.sqrt(annual_factor)

panelA_pref_mv = pd.DataFrame({'Regret adjustment μ': pref_ra_minvar,
                               'Regret-adjusted std': std_regret_pref_mv},
                              index=asset_names)

panelA_pref_mv_dev = panelA_pref_mv.apply(deviation)

panelA_pref_mv_table3 = pd.DataFrame(
    np.vstack([panelA_pref_mv['Regret adjustment μ'].values,
               panelA_pref_mv_dev['Regret adjustment μ'].values,
               panelA_pref_mv['Regret-adjusted std'].values,
               panelA_pref_mv_dev['Regret-adjusted std'].values]),
    index=['Pref-adj μ (Min-Var)','(deviation)',
           'Pref-adj std (Min-Var)', '(deviation)'],
    columns=asset_names)

panelA_pref_mv_table3.round(4)

#%% Panel A 汇总

panelA_table3 = panelA_table3.round(4)
panelA_pref_table3 = panelA_pref_table3.round(4)
panelA_pref_mv_table3 = panelA_pref_mv_table3.round(4)
df_all = (pd.concat([panel_a_stats, 
                     panelA_table3, 
                     panelA_pref_table3, 
                     panelA_pref_mv_table3],
                    axis=0, ignore_index=False))
df_all.round(4)

#%% Panel B

mu = R.mean(axis=0) * annual_factor               # (4,)
Sigma = np.cov(R, rowvar=False, ddof=1) * annual_factor  # (4,4)

#%%% 传统MV和Min_Var

# Markowitz
N = R.shape[1]
w = cp.Variable(N)
objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma))

constraints = [cp.sum(w) == 1,w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
w_markowitz = w.value

dev_w_markowitz = deviation(w_markowitz)
panelB_markowit = pd.DataFrame(np.vstack([w_markowitz, dev_w_markowitz]),
    index=['Markowitz (μ–σ)','(deviation)'],columns=asset_names)
panelB_markowitz = panelB_markowit.round(4)


# Min-Var
Sigma = np.cov(R, rowvar=False, ddof=1) * annual_factor
N = R.shape[1]
w = cp.Variable(N)
objective = cp.Minimize(cp.quad_form(w, Sigma))
constraints = [cp.sum(w) == 1,w >= 0]

# 求解
problem = cp.Problem(objective, constraints)
problem.solve()
# 提取权重
w_minvar = w.value

dev_w_minvar = deviation(w_minvar)
panelB_w_minvar = pd.DataFrame(np.vstack([w_minvar, dev_w_minvar]),
    index=['Markowitz (Min-Var)','(deviation)'],columns=asset_names)
panelB_minvar = panelB_w_minvar.round(4)

#%%% 纯后悔-两种视角

# Pure regret · return view · μ–σ (γ=3)
R_max = R.max(axis=1)
Z_ret = R - R_max[:, None]
Sigma_ra = np.cov(Z_ret, rowvar=False, ddof=1) * annual_factor

objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma_ra))
constraints = [cp.sum(w) == 1,w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
w_pure_ret_mu_sigma = w.value
# 非负
(w_pure_ret_mu_sigma >= -1e-10).all()

dev_w_pure_ret_mu_sigma = deviation(w_pure_ret_mu_sigma)
panelB_pure_ret_mu_sigma = pd.DataFrame(np.vstack([w_pure_ret_mu_sigma, dev_w_pure_ret_mu_sigma]),
                                        index=['Ret regret(μ–σ)','(deviation)'], 
                                        columns=asset_names)
panelB_ret_regret_mu_sigma = panelB_pure_ret_mu_sigma.round(4)


# Pure regret · return view  Min Var
objective = cp.Minimize(cp.quad_form(w, Sigma_ra))
constraints = [cp.sum(w) == 1,w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
w_pure_ret_minvar = w.value

dev_pure_ret_minvar = deviation(w_pure_ret_minvar)
panelB_pure_ret_minvar = pd.DataFrame(np.vstack([w_pure_ret_minvar, dev_pure_ret_minvar]),
                                      index=['Ret regret(Min-Var)', '(deviation)'],
                                      columns=asset_names)
panelB_Ret_regret_minvar = panelB_pure_ret_minvar.round(4)


# Pure regret · pref view · μ–σ (γ=3)
gamma = 3.0
var_assets = np.var(R, axis=0, ddof=1)
pi = R - gamma * var_assets[None, :]
pi_max = pi.max(axis=1)
Z_pref = R - pi_max[:, None]
Sigma_pref = np.cov(Z_pref, rowvar=False, ddof=1) * annual_factor
Sigma_pref_ra = Sigma + Sigma_pref

w = cp.Variable(N)
objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma_pref_ra))
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
w_pure_pref_mu_sigma = w.value

dev_pure_pref_mu_sigma = deviation(w_pure_pref_mu_sigma)
panelB_pure_pref_mu_sigma = pd.DataFrame(np.vstack([w_pure_pref_mu_sigma, dev_pure_pref_mu_sigma]),
                                      index=['Pref regret(μ–σ)', '(deviation)'],
                                      columns=asset_names)

panelB_Pref_regret_minvar = panelB_pure_pref_mu_sigma.round(4)

#%%% Min_Var

# Cov(R)
Sigma_R = np.cov(R, rowvar=False, ddof=1) * annual_factor

# Var(pi_max)
var_pi_max = np.var(pi_max, ddof=1) * annual_factor

# Cov(R, pi_max)
cov_R_pi = np.array([
    np.cov(R[:, i], pi_max, ddof=1)[0, 1]
    for i in range(N)
]) * annual_factor

# Preference-regret covariance (paper definition)
Sigma_pref = (
    Sigma_R
    + var_pi_max * np.ones((N, N))
    - np.outer(cov_R_pi, np.ones(N))
    - np.outer(np.ones(N), cov_R_pi)
)
Sigma_pref_ra = alpha * Sigma + Sigma_pref

objective = cp.Minimize(cp.quad_form(w, Sigma_pref_ra))
constraints = [cp.sum(w) == 1,w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()

w_pure_pref_minvar = w.value

dev_pure_pref_minvar = deviation(w_pure_pref_minvar)
panelB_pure_pref_minvar = pd.DataFrame(np.vstack([w_pure_pref_minvar, dev_pure_pref_minvar]),
                                       index=['Pref regret(Min-Var)','(deviation)'],
                                       columns=asset_names)

panelB_pure_pref_minvar.round(4)


#%% Panel B 汇总
df_all = (pd.concat([panelB_markowitz, 
                     panelB_minvar, 
                     panelB_ret_regret_mu_sigma, 
                     panelB_Ret_regret_minvar, 
                     panelB_Pref_regret_minvar, 
                     panelB_pure_pref_minvar],
                    axis=0, ignore_index=False))
df_all.round(4)