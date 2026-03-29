import numpy as np
from typing import Optional, Union


def calculate_nse(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算NSE系数 (Nash-Sutcliffe Efficiency)

    NSE = 1 - Σ(Q_sim - Q_obs)² / Σ(Q_obs - Q_mean)²

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        NSE值 (范围 -∞ 到 1)
    """
    if mask is not None:
        sim = sim[mask]
        obs = obs[mask]

    obs_mean = np.mean(obs)

    numerator = np.sum((sim - obs) ** 2)
    denominator = np.sum((obs - obs_mean) ** 2)

    if denominator < 1e-10:
        return np.nan

    nse = 1 - numerator / denominator
    return float(nse)


def calculate_rmse(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算RMSE (Root Mean Square Error)

    RMSE = √[Σ(Q_sim - Q_obs)² / n]

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        RMSE值
    """
    if mask is not None:
        sim = sim[mask]
        obs = obs[mask]

    mse = np.mean((sim - obs) ** 2)
    return float(np.sqrt(mse))


def calculate_bias(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算偏差 (Bias)

    Bias = Σ(Q_sim - Q_obs) / n

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        偏差值
    """
    if mask is not None:
        sim = sim[mask]
        obs = obs[mask]

    return float(np.mean(sim - obs))


def calculate_pbias(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算百分偏差 (Percent Bias)

    PBIAS = 100 * Σ(Q_sim - Q_obs) / Σ(Q_obs)

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        百分偏差值（%）
    """
    if mask is not None:
        sim = sim[mask]
        obs = obs[mask]

    numerator = np.sum(sim - obs)
    denominator = np.sum(obs)

    if abs(denominator) < 1e-10:
        return np.nan

    return float(100 * numerator / denominator)


def calculate_kge(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算KGE系数 (Kling-Gupta Efficiency)

    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        KGE值
    """
    if mask is not None:
        sim = sim[mask]
        obs = obs[mask]

    r = np.corrcoef(sim, obs)[0, 1]

    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(kge)


def calculate_rsr(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算RSR (RMSE-observations standard deviation Ratio)

    RSR = RMSE / STDEV_obs

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        RSR值
    """
    rmse = calculate_rmse(sim, obs, mask)
    obs_std = np.std(obs[mask] if mask is not None else obs)

    if obs_std < 1e-10:
        return np.nan

    return float(rmse / obs_std)


def calculate_persistence(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算模型效率系数（过程相对误差）

    关注峰现时间和过程形状

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        效率系数
    """
    if mask is not None:
        sim = sim[mask]
        obs = obs[mask]

    sim_diff = np.diff(sim)
    obs_diff = np.diff(obs)

    numerator = np.sum((sim_diff - obs_diff) ** 2)
    denominator = np.sum(obs_diff ** 2)

    if denominator < 1e-10:
        return np.nan

    return float(1 - numerator / denominator)


def peak_timing_error(
    sim: np.ndarray,
    obs: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    计算峰现时间误差

    Args:
        sim: 模拟值
        obs: 观测值
        dt: 时间步长

    Returns:
        峰现时间误差（时间单位）
    """
    sim_peak_idx = np.argmax(sim)
    obs_peak_idx = np.argmax(obs)

    return float((sim_peak_idx - obs_peak_idx) * dt)


def evaluate_all(
    sim: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> dict:
    """
    计算所有评价指标

    Args:
        sim: 模拟值
        obs: 观测值
        mask: 有效数据掩码

    Returns:
        指标字典
    """
    return {
        'NSE': calculate_nse(sim, obs, mask),
        'KGE': calculate_kge(sim, obs, mask),
        'RMSE': calculate_rmse(sim, obs, mask),
        'BIAS': calculate_bias(sim, obs, mask),
        'PBIAS': calculate_pbias(sim, obs, mask),
        'RSR': calculate_rsr(sim, obs, mask),
        'PERSISTENCE': calculate_persistence(sim, obs, mask),
        'PEAK_TIMING': peak_timing_error(sim, obs)
    }
