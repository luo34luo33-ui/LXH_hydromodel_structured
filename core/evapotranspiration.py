"""
Evapotranspiration Sub-Model
Based on Liuxihe Model principles (Section 2.2.3)

Formulas:
    Es = λ * Ep                     when θ > θfc
    Es = (1-λ) * Ep * (θ-θw)/(θfc-θw)  when θw < θ ≤ θfc
    Es = 0                          when θ ≤ θw
"""

import numpy as np
from typing import Tuple


class Evapotranspiration:
    """蒸散发计算模型"""
    
    def __init__(
        self,
        theta_fc: np.ndarray,
        theta_w: np.ndarray,
        lambda_coef: np.ndarray,
        mask: np.ndarray = None
    ):
        """
        Args:
            theta_fc: 田间持水率 (fraction 0-1)
            theta_w: 凋萎含水率 (fraction 0-1)
            lambda_coef: 蒸发系数,水面=1,其他<1
            mask: 有效网格掩码
        """
        self.theta_fc = theta_fc
        self.theta_w = theta_w
        self.lambda_coef = lambda_coef
        self.mask = mask
    
    def compute(
        self,
        Ep: np.ndarray,
        theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算实际蒸散发量
        
        Args:
            Ep: 潜在蒸发率 (mm/h)
            theta: 土壤当前含水率 (fraction)
            
        Returns:
            (Es, Ep_adj): 实际蒸散发量(mm/h), 调整后潜在蒸发率(mm/h)
        """
        Es = np.zeros_like(Ep, dtype=np.float64)
        
        cond1 = theta > self.theta_fc
        cond2 = (theta > self.theta_w) & (theta <= self.theta_fc)
        cond3 = theta <= self.theta_w
        
        Es[cond1] = self.lambda_coef[cond1] * Ep[cond1]
        Es[cond2] = (1 - self.lambda_coef[cond2]) * Ep[cond2] * (
            (theta[cond2] - self.theta_w[cond2]) / 
            (self.theta_fc[cond2] - self.theta_w[cond2])
        )
        Es[cond3] = 0.0
        
        Ep_adj = Ep - Es
        
        return Es, Ep_adj
    
    def compute_layer(
        self,
        Ep: np.ndarray,
        theta: np.ndarray,
        layer_depth: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分层计算蒸散发
        
        Args:
            Ep: 潜在蒸发率 (mm/h)
            theta: 土壤当前含水率 (fraction)
            layer_depth: 土层深度 (mm)
            
        Returns:
            (Es, water_loss): 蒸散发量, 水分损失
        """
        Es, Ep_adj = self.compute(Ep, theta)
        water_loss = Es * layer_depth / 1000.0
        
        return Es, water_loss


def calculate_et(
    rainfall: np.ndarray,
    temperature: np.ndarray = None,
    Ep_method: str = 'hargreaves',
    Ep: np.ndarray = None
) -> np.ndarray:
    """
    便捷函数: 计算潜在蒸发量
    
    Args:
        rainfall: 降雨量 (mm/h)
        temperature: 温度 (°C), 可选
        Ep_method: 蒸发计算方法
        Ep: 直接输入的潜在蒸发率
        
    Returns:
        Ep: 潜在蒸发率 (mm/h)
    """
    if Ep is not None:
        return Ep
    
    if Ep_method == 'hargreaves' and temperature is not None:
        Ra = 15.0
        Ep = 0.0023 * Ra * (temperature + 17.8) * (rainfall ** 0.5)
    else:
        Ep = np.ones_like(rainfall) * 0.1
    
    return Ep
