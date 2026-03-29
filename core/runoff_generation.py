"""
Runoff Generation Sub-Model (产流计算子模型)
Based on Liuxihe Model principles (Section 2.2.4)

Processes:
1. Saturation Excess Runoff (蓄满产流)
2. Interflow/Subsurface Flow (壤中流) - Campbell formula
3. Percolation to Groundwater (渗漏量)

Formulas:
    Net rainfall = P - Es
    When θ < θsat: infiltrate, no runoff
    When θ ≥ θsat: produce saturation excess runoff
    
    Interflow: Qlat = vlat * L * z (Darcy's law)
    Percolation: Qper = K * L * L
"""

import numpy as np
from typing import Tuple, Dict


class RunoffGeneration:
    """
    产流计算模型 - 蓄满产流模式
    
    包含:
    - 地表径流计算
    - 壤中流计算 (Campbell公式)
    - 渗漏量计算
    """
    
    def __init__(
        self,
        theta_sat: np.ndarray,
        theta_fc: np.ndarray,
        theta_w: np.ndarray,
        theta_init: np.ndarray = None,
        K_sat: np.ndarray = None,
        b_exp: np.ndarray = None,
        soil_depth: np.ndarray = None,
        slope: np.ndarray = None,
        cell_size: float = 90.0,
        mask: np.ndarray = None
    ):
        """
        Args:
            theta_sat: 饱和含水率 (fraction)
            theta_fc: 田间持水率 (fraction)
            theta_w: 凋萎含水率 (fraction)
            theta_init: 初始土壤含水率 (fraction)
            K_sat: 饱和水力传导率 (mm/h)
            b_exp: Campbell公式参数
            soil_depth: 土层厚度 (mm)
            slope: 坡度 (ratio)
            cell_size: 网格大小 (m)
            mask: 有效网格掩码
        """
        self.theta_sat = theta_sat
        self.theta_fc = theta_fc
        self.theta_w = theta_w
        self.theta = theta_init if theta_init is not None else theta_fc.copy()
        self.K_sat = K_sat if K_sat is not None else np.ones_like(theta_sat) * 10.0
        self.b_exp = b_exp if b_exp is not None else np.ones_like(theta_sat) * 4.0
        self.soil_depth = soil_depth if soil_depth is not None else np.ones_like(theta_sat) * 100.0
        self.slope = slope
        self.cell_size = cell_size
        self.mask = mask
    
    def compute(
        self,
        P: np.ndarray,
        Es: np.ndarray,
        theta_prev: np.ndarray = None,
        dt: float = 3600.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算产流量
        
        Args:
            P: 降雨量 (mm/h)
            Es: 实际蒸散发量 (mm/h)
            theta_prev: 前期土壤含水率 (fraction)
            dt: 时间步长 (s)
            
        Returns:
            (Rs, Qint, Qper, theta_new):
            Rs: 地表径流量 (mm/h)
            Qint: 壤中流量 (mm/h)
            Qper: 渗漏量 (mm/h)
            theta_new: 更新后的土壤含水率 (fraction)
        """
        if theta_prev is not None:
            self.theta = theta_prev.copy()
        
        theta = self.theta.copy()
        dt_h = dt / 3600.0
        
        net_rain = P - Es
        net_rain = np.maximum(net_rain, 0.0)
        
        storage_capacity = (self.theta_sat - theta) * self.soil_depth
        excess = np.maximum(net_rain - storage_capacity, 0.0)
        
        Rs = excess.copy()
        theta_after_excess = theta + (net_rain - excess) / self.soil_depth
        theta_after_excess = np.minimum(theta_after_excess, self.theta_sat)
        
        Qint = np.zeros_like(P)
        Qper = np.zeros_like(P)
        
        cond_interflow = theta_after_excess > self.theta_fc
        
        if self.slope is not None and np.any(cond_interflow):
            theta_ratio = np.where(
                cond_interflow,
                theta_after_excess / self.theta_sat,
                0.0
            )
            
            K_current = self.K_sat * (theta_ratio ** (2 * self.b_exp + 3))
            
            S0 = np.maximum(self.slope, 0.0001)
            v_lat = K_current * S0
            
            L = self.cell_size
            Qint = v_lat * L * self.soil_depth / 1000.0
            
            Qint = np.where(cond_interflow, Qint, 0.0)
            
            perc_condition = theta_after_excess > self.theta_fc
            Qper = np.where(perc_condition, K_current * L * L / 1000.0, 0.0)
        
        Q_total_out = Rs + Qint + Qper
        water_balance = net_rain - Q_total_out
        theta_new = theta_after_excess + water_balance / self.soil_depth
        theta_new = np.clip(theta_new, self.theta_w, self.theta_sat)
        
        self.theta = theta_new.copy()
        
        return Rs, Qint, Qper, theta_new
    
    def compute_wetness_index(self) -> np.ndarray:
        """
        计算前期土壤湿润指标 ( antecedent precipitation index )
        
        Returns:
            API: 前期影响雨量
        """
        K = 0.85
        theta_ratio = (self.theta - self.theta_w) / (self.theta_sat - self.theta_w)
        API = theta_ratio * 100.0
        return API
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """获取当前状态变量"""
        return {
            'theta': self.theta.copy(),
            'theta_sat': self.theta_sat.copy(),
            'theta_fc': self.theta_fc.copy(),
            'theta_w': self.theta_w.copy()
        }
    
    def set_state(self, state: Dict[str, np.ndarray]):
        """设置状态变量"""
        if 'theta' in state:
            self.theta = state['theta'].copy()


def compute_runoff_fast(
    P: np.ndarray,
    theta: np.ndarray,
    theta_sat: np.ndarray,
    theta_fc: np.ndarray,
    dt: float = 3600.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    快速产流计算 (简化版)
    
    Returns:
        (R, theta_new): 产流量(mm/h), 更新后含水率
    """
    dt_h = dt / 3600.0
    storage = (theta_sat - theta) * 100.0
    
    excess = np.maximum(P - storage, 0.0)
    theta_new = theta + (P - excess) / 100.0
    theta_new = np.clip(theta_new, theta_fc, theta_sat)
    
    return excess, theta_new
