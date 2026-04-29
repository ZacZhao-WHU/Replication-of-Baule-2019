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

#%% Panel A_stats

# 计算每个资产的基本统计量
annual_factor = 252
assets = df_assets.columns
stats_calculations = [('Mean Return', df_assets.mean() * annual_factor), 
                      ('Std. Deviation', df_assets.std() * np.sqrt(annual_factor)),
                      ('Skewness', df_assets.skew())]

stats_data = []
index = []

for stat_name, values in stats_calculations:
    # 添加统计量行
    stats_data.append(values.round(4))
    index.append(stat_name)
    # 添加偏差行
    deviations = values - values.mean()
    stats_data.append(pd.Series([f"({dev:.4f})" for dev in deviations], index=assets))
    index.append('')

panel_a_stats = pd.DataFrame(stats_data, index=index, columns=assets)
panel_a_stats

#%%% Panel A ret regret adj

alpha = 1
gamma = 3
std_ret_adj = {}
covariances_ret = {}
regret_adjustment_mu = {}
cov = {'Mkt-RF': 0.0032,'SMB': -0.0009,'HML': 0.0002,'MOM': -0.0006}
df_asset = df_assets.copy()
df_assets_ret_adj = df_assets.copy()
df_asset['R_max'] = df_asset[['Mkt-RF', 'SMB', 'HML', 'MOM']].max(axis=1)

for asset in assets:
    df_assets_ret_adj[f'{asset}_ret_adj'] = df_asset[asset] - alpha * df_asset['R_max']
    cov_daily = np.cov(df_asset[asset], df_asset['R_max'])[0, 1]
    cov_annual = cov_daily * 252  # 年化
    covariances_ret[asset] = cov_annual
    std_daily = df_assets_ret_adj[f'{asset}_ret_adj'].std()
    std_annual = std_daily * np.sqrt(252)  # 年化标准差
    std_ret_adj[asset] = std_annual
    adjustment = 2 * alpha * gamma * cov[asset]
    regret_adjustment_mu[asset] = adjustment #* covariances_ret[asset]


adj_values = [regret_adjustment_mu[asset] for asset in assets]
adj_mean = np.mean(adj_values)
adj_deviations = [val - adj_mean for val in adj_values]

std_values = [std_ret_adj[asset] for asset in assets]
std_mean = np.mean(std_values)
std_deviations = [val - std_mean for val in std_values]

data_dict = {}
for i, asset in enumerate(assets):
    data_dict[asset] = [f"{regret_adjustment_mu[asset]:.4f}",
                        f"({adj_deviations[i]:.4f})",
                        f"{std_ret_adj[asset]:.4f}",
                        f"({std_deviations[i]:.4f})"]

ret_regret_df = pd.DataFrame(data_dict)
ret_regret_df.index = ['Regret adjustment μ',
                       '',
                       'Regret-adj. std. dev.',
                       '']
ret_regret_df


#%%% Panel A Pref regret adj

# 计算μ-σ情况 (γ=3) 下的Preference-regret
df_assets_pi = df_assets.copy()
variances_daily = {}
df_assets_pref_mu_sigma = df_assets.copy()
std_pref_adj = {}

for asset in assets:
    # 计算每个资产的方差（日度）
    variances_daily[asset] = df_assets[asset].var()
    # 计算每个时期的偏好值 π = R - γσ²
    df_assets_pi[f'pi_{asset}'] = df_assets[asset] - gamma * variances_daily[asset]
# 计算每个因子的π_max
df_assets_pi['pi_max'] = df_assets_pi[['pi_Mkt-RF', 'pi_SMB', 'pi_HML', 'pi_MOM']].max(axis=1)

# 计算每个资产与π_max的协方差 (用于μ调整)
cov_pref = {}
for asset in assets:
    cov_pref[asset] = np.cov(df_assets[asset], df_assets_pi['pi_max'])[0, 1]
    # 计算Preference-regret调整后收益 Z_pref = R - απ_max - αγσ²
    df_assets_pref_mu_sigma[f'Z_pref_{asset}'] = (df_assets[asset] - 
                                                  alpha * df_assets_pi['pi_max'] - 
                                                  alpha * gamma * variances_daily[asset])
    # 计算调整后收益的标准差
    std_pref_adj[asset] = df_assets_pref_mu_sigma[f'Z_pref_{asset}'].std()
    
annual_factor = 252
sqrt_annual_factor = np.sqrt(annual_factor)
# 年化协方差&标准差
cov_pref_mu_sigma = {asset: cov * annual_factor for asset, cov in cov_pref.items()}
std_pref_adj_mu_sigma = {asset: std * sqrt_annual_factor for asset, std in std_pref_adj.items()}

# 计算各统计量的平均值
avg_cov_pref_mu_sigma = np.mean(list(cov_pref_mu_sigma.values()))
avg_std_pref_mu_sigma = np.mean(list(std_pref_adj_mu_sigma.values()))

# 计算偏离值
deviations_cov_pref_mu_sigma = {asset: cov - avg_cov_pref_mu_sigma for asset, cov in cov_pref_mu_sigma.items()}
deviations_std_pref_mu_sigma = {asset: std - avg_std_pref_mu_sigma for asset, std in std_pref_adj_mu_sigma.items()}


data_dict = {}
for asset in assets:
    data_dict[asset] = [f"{cov_pref_mu_sigma[asset]:.4f}",
                        f"({deviations_cov_pref_mu_sigma[asset]:.4f})",
                        f"{std_pref_adj_mu_sigma[asset]:.4f}",
                        f"({deviations_std_pref_mu_sigma[asset]:.4f})"]

pref_mu_sigma_df = pd.DataFrame(data_dict)
pref_mu_sigma_df.index = ['Regret adjustment μ',
                          '',
                          'Regret-adj. std. dev.',
                          '']
pref_mu_sigma_df


#%%% Min-Var

# 最小方差情况下偏好后悔模型的计算
def min_variance_pref_regret(df_assets, assets, alpha=1, gamma=3):
    cov_matrix_daily = df_assets[assets].cov()
    n_assets = len(assets)
    ones = np.ones(n_assets) # 最小方差组合优化
    inv_cov = np.linalg.inv(cov_matrix_daily)
    min_var_weights = inv_cov.dot(ones) / ones.dot(inv_cov.dot(ones))
    
    # 计算min-var的收益序列
    df_assets_minvar = df_assets.copy()
    df_assets_minvar['R_minvar'] = df_assets[assets].dot(min_var_weights)
    minvar_variance_daily = min_var_weights.dot(cov_matrix_daily.dot(min_var_weights))
    minvar_variance_annual = minvar_variance_daily * 252 # 年化
    
    # 计算每个资产的偏好值
    df_assets_minvar['pi_minvar'] = (df_assets_minvar['R_minvar'] - gamma * minvar_variance_daily)
    std_pref_adj_minvar = {}
    covariances_pref_minvar = {}
    regret_adjustment_mu_minvar = {}
    
    # 为每个资产计算调整量
    for asset in assets:
        cov_daily = np.cov(df_assets_minvar[asset], df_assets_minvar['pi_minvar'])[0, 1]
        cov_annual = cov_daily * 252  # 年化
        covariances_pref_minvar[asset] = cov_annual
        regret_adjustment_mu_minvar[asset] = covariances_pref_minvar[asset]
        df_assets_minvar[f'{asset}_pref_adj_minvar'] = (df_assets_minvar[asset] - alpha * df_assets_minvar['pi_minvar'])
        
        std_daily = df_assets_minvar[f'{asset}_pref_adj_minvar'].std()
        std_annual = std_daily * np.sqrt(252)  # 年化标准差
        std_pref_adj_minvar[asset] = std_annual
    
    return (regret_adjustment_mu_minvar, std_pref_adj_minvar, covariances_pref_minvar, min_var_weights)


(mu_minvar, std_minvar, cov_minvar, min_var_weights) = min_variance_pref_regret(df_assets, assets, alpha=1, gamma=3)
# 计算偏离均值
adj_values_minvar = [mu_minvar[asset] for asset in assets]
adj_mean_minvar = np.mean(adj_values_minvar)
adj_deviations_minvar = [val - adj_mean_minvar for val in adj_values_minvar]
std_values_minvar = [std_minvar[asset] for asset in assets]
std_mean_minvar = np.mean(std_values_minvar)
std_deviations_minvar = [val - std_mean_minvar for val in std_values_minvar]

# 创建结果表格
data_dict_minvar = {}
for i, asset in enumerate(assets):
    data_dict_minvar[asset] = [f"{mu_minvar[asset]:.4f}",
                               f"({adj_deviations_minvar[i]:.4f})",
                               f"{std_minvar[asset]:.4f}",
                               f"({std_deviations_minvar[i]:.4f})"]

pref_minvar_df = pd.DataFrame(data_dict_minvar)
pref_minvar_df.index = ['Regret adjustment μ (Min-Var)',
                        '',
                        'Regret-adj. std. dev. (Min-Var)',
                        '']
pref_minvar_df

#%% Panel A 汇总
df_all = (pd.concat([panel_a_stats,
                     ret_regret_df,
                     pref_mu_sigma_df, 
                     pref_minvar_df],
                    axis=0, ignore_index=False))
df_all


#%% Panel B

gamma = 1.5
def portfolio_optimizer(mu, cov_matrix, method='mu_sigma', gamma=1.5):
    n_assets = len(mu)
    if method == 'mu_sigma':
        objective = lambda w: -(w.dot(mu) - gamma * w.dot(cov_matrix.dot(w)))
    else:  # min_var
        objective = lambda w: w.dot(cov_matrix.dot(w))

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    x0 = np.ones(n_assets) / n_assets
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
    
    return result.x if result.success else x0


# 计算年化统计量
def get_annualized_stats(df, assets):
    mu_annual = df[assets].mean() * 252
    cov_annual = df[assets].cov() * 252
    return mu_annual, cov_annual


# 计算权重与均值的偏离
def calculate_deviations(weights):
    return weights - 1.0 / len(weights)

# 均值方差权重
def markowitz_weights(df, assets, gamma=1.5):
    mu, cov = get_annualized_stats(df, assets)
    mu_sigma_w = portfolio_optimizer(mu, cov, 'mu_sigma', gamma)
    min_var_w = portfolio_optimizer(mu, cov, 'min_var')
    
    return (mu_sigma_w, calculate_deviations(mu_sigma_w), min_var_w, calculate_deviations(min_var_w))

# 收益后悔权重
def return_regret_weights(df, assets, alpha=1, gamma=1.5):
    mu, cov = get_annualized_stats(df, assets)
    df_temp = df.copy()
    df_temp['R_max'] = df_temp[assets].max(axis=1)
    cov_adjustments = np.array([np.cov(df_temp[asset], df_temp['R_max'])[0, 1] * 252 for asset in assets])
    mu_adjusted = mu + 2 * alpha * gamma * cov_adjustments
    
    # μ-σ优化
    mu_sigma_w = portfolio_optimizer(mu_adjusted, cov, 'mu_sigma', gamma)
    # 最小方差优化（使用调整后的协方差矩阵）
    Z_ret = df_temp[assets].subtract(alpha * df_temp['R_max'], axis=0)
    cov_adjusted = Z_ret.cov() * 252
    min_var_w = portfolio_optimizer(np.zeros(len(assets)), cov_adjusted, 'min_var')
    
    return (mu_sigma_w, calculate_deviations(mu_sigma_w), min_var_w, calculate_deviations(min_var_w))

# 偏好min-var
def preference_minvar(df, assets, alpha=1, gamma_original=1.5, gamma_adjusted=None):
    # 使用日度数据计算，避免年化转换问题
    cov_daily = df[assets].cov()
    n_assets = len(assets)
    ones = np.ones(n_assets)
    inv_cov = np.linalg.inv(cov_daily)
    min_var_w = inv_cov.dot(ones) / ones.dot(inv_cov.dot(ones))
    
    # 计算最小方差组合的日度收益和方差
    df_temp = df.copy()
    df_temp['R_minvar'] = df_temp[assets].dot(min_var_w)
    minvar_var_daily = min_var_w.dot(cov_daily.dot(min_var_w))
    
    # 使用日度方差计算偏好值
    df_temp['pi_minvar'] = df_temp['R_minvar'] - gamma_original * minvar_var_daily
    
    # 计算调整后的日度收益
    Z_pref = df_temp[assets].copy()
    for i, asset in enumerate(assets):
        asset_var_daily = df_temp[asset].var()
        Z_pref[asset] = (df_temp[asset] - alpha * df_temp['pi_minvar'] - alpha * gamma_original * asset_var_daily)
    
    # 计算调整后的日度协方差矩阵并年化
    cov_Z_pref_daily = Z_pref.cov()
    cov_adjusted_daily = alpha * cov_daily + cov_Z_pref_daily
    cov_adjusted = cov_adjusted_daily * 252  # 年化
    
    # 计算调整后的最小方差组合
    inv_cov_adj = np.linalg.inv(cov_adjusted)
    adj_min_var_w = inv_cov_adj.dot(ones) / ones.dot(inv_cov_adj.dot(ones))
    
    return adj_min_var_w, calculate_deviations(adj_min_var_w)

# 偏好后悔权重
def preference_regret_weights(df, assets, alpha=1, gamma=1.5):
    mu, cov = get_annualized_stats(df, assets)
    variances_daily = df[assets].var()
    pi_values = df[assets].copy()
    
    for i, asset in enumerate(assets):
        pi_values[asset] = df[asset] - gamma * variances_daily[i]
    
    pi_values['pi_max'] = pi_values[assets].max(axis=1)
    
    # 计算调整后的期望收益
    cov_adjustments = np.array([np.cov(df[asset], pi_values['pi_max'])[0, 1] * 252 for asset in assets])
    mu_adjusted = mu + 2 * alpha * gamma * cov_adjustments
    
    # μ-σ优化（使用调整后的风险厌恶系数）
    gamma_adjusted = gamma * (1 + alpha)
    mu_sigma_w = portfolio_optimizer(mu_adjusted, cov, 'mu_sigma', gamma_adjusted)
    # 最小方差优化
    min_var_w, min_var_dev = preference_minvar(df, assets, alpha, gamma_original=gamma, gamma_adjusted=gamma_adjusted)
    
    return (mu_sigma_w, calculate_deviations(mu_sigma_w), min_var_w, min_var_dev)

#%% 汇总Panel B
def create_panel_B(df_assets, assets, alpha=1, gamma=gamma):
    results = {}
    results['markowitz'] = markowitz_weights(df_assets, assets, gamma)
    results['return_regret'] = return_regret_weights(df_assets, assets, alpha, gamma)
    results['pref_regret'] = preference_regret_weights(df_assets, assets, alpha, gamma)
    
    # 创建结果表格
    strategies = [('Markowitz: μ-σ', results['markowitz'][0:2]), ('Markowitz: Min-Var', results['markowitz'][2:4]),
                  ('Pure Regret Return: μ-σ', results['return_regret'][0:2]), ('Pure Regret Return: Min-Var', results['return_regret'][2:4]),
                  ('Pure Regret Preference: μ-σ', results['pref_regret'][0:2]), ('Pure Regret Preference: Min-Var', results['pref_regret'][2:4])]
    
    # 构建数据
    data = {}
    for strategy, (weights, deviations) in strategies:
        for i, asset in enumerate(assets):
            if asset not in data:
                data[asset] = []
            data[asset].extend([f"{weights[i]:.4f}", f"({deviations[i]:.4f})"])
    
    # 创建多级索引
    index_tuples = []
    for strategy, _ in strategies:
        index_tuples.append((strategy, ' '))
        index_tuples.append((strategy, ' '))
    
    multi_index = pd.MultiIndex.from_tuples(index_tuples)
    return pd.DataFrame(data, index=multi_index)

panel_B = create_panel_B(df_assets, assets, alpha=1, gamma=gamma)
panel_B