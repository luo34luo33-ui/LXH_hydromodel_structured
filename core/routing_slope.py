"""
Slope Routing (边坡汇流计算)
Based on Liuxihe Model principles (Section 2.3.3)

Method: Kinematic Wave Method (运动波法)

Formulas:
    Q = (L/n) * h^(5/3) * S0^(1/2)
    Kinematic wave equation with Newton-Raphson solution
"""

import numpy as np
from typing import Tuple, Optional


class SlopeRouter:
    """
    边坡汇流模型 - 运动波法
    
    采用一维运动波法进行边坡汇流计算
    """
    
    def __init__(
        self,
        slope: np.ndarray,
        mannings_n: np.ndarray,
        cell_size: float = 90.0,
        mask: np.ndarray = None
    ):
        """
        Args:
            slope: 坡度 (ratio, not angle)
            mannings_n: 曼宁系数
            cell_size: 网格大小 (m)
            mask: 有效网格掩码
        """
        self.slope = np.maximum(slope, 0.0001)
        self.n = np.maximum(mannings_n, 0.01)
        self.cell_size = cell_size
        self.mask = mask
        
        self.L = cell_size
        self.a_coef = (self.n / self.L * self.slope ** (-0.5)) ** (3.0 / 5.0)
        self.b_exp = 3.0 / 5.0
    
    def route(
        self,
        Q_in: np.ndarray,
        q_source: np.ndarray,
        dt: float = 3600.0
    ) -> np.ndarray:
        """
        边坡汇流计算
        
        Args:
            Q_in: 上游入流量 (m³/s)
            q_source: 单元产流量 (mm/h) -> 转换为 m³/s
            dt: 时间步长 (s)
            
        Returns:
            Q_out: 出流量 (m³/s)
        """
        q_source_m3s = q_source * self.cell_size * self.cell_size / 1000.0 / 3600.0
        
        Q_upstream = Q_in.copy()
        
        Q_cell = q_source_m3s + Q_upstream
        
        dx = self.cell_size
        
        C = dt * self.a_coef * self.b_exp / dx
        
        Q_out = np.zeros_like(Q_cell)
        
        valid_mask = q_source_m3s > 0
        for i in range(Q_cell.shape[0]):
            for j in range(Q_cell.shape[1]):
                if not valid_mask[i, j]:
                    continue
                    
                Q_est = Q_cell[i, j]
                Q_prev = Q_out[i, j] if Q_out[i, j] > 0 else Q_est * 0.5
                
                for _ in range(20):
                    h = (self.n[i, j] / self.L * self.slope[i, j] ** (-0.5)) ** (3.0/5.0) * Q_est ** (3.0/5.0)
                    
                    f = Q_est - Q_prev - dt * (q_source_m3s[i, j] + Q_upstream[i, j] - Q_est / dx * h)
                    
                    df = 1 + dt * h / dx * 3.0 / 5.0 / Q_est ** (2.0/5.0) * (self.n[i, j] / self.L * self.slope[i, j] ** (-0.5)) ** (3.0/5.0)
                    
                    Q_new = Q_est - f / df
                    
                    if abs(Q_new - Q_est) < 1e-6 or Q_new < 0:
                        Q_out[i, j] = max(Q_new, 0)
                        break
                    
                    Q_est = Q_new
                
                Q_out[i, j] = max(Q_est, 0)
        
        return Q_out
    
    def route_cell(
        self,
        Q_up: float,
        R: float,
        dt: float
    ) -> float:
        """
        单网格边坡汇流计算
        
        Args:
            Q_up: 上游入流量 (m³/s)
            R: 产流深 (mm/h)
            dt: 时间步长 (s)
            
        Returns:
            Q_out: 出流量 (m³/s)
        """
        R_m3s = R * self.cell_size * self.cell_size / 1000.0 / 3600.0
        Q_in = Q_up + R_m3s
        
        n = self.n.flat[0] if self.n.ndim > 0 else self.n
        S0 = self.slope.flat[0] if self.slope.ndim > 0 else self.slope
        
        a = (n / self.L * S0 ** (-0.5)) ** (3.0 / 5.0)
        
        Q_est = Q_in
        Q_prev = 0
        
        for _ in range(30):
            h = a * Q_est ** (3.0/5.0)
            
            C1 = dt / (2 * self.cell_size)
            C2 = self.L * a * Q_est ** (3.0/5.0)
            
            Q_new = (C1 * Q_in + C2 + R_m3s * dt / 2) / (C1 + a * Q_est ** (3.0/5.0) * dt / 2)
            
            if abs(Q_new - Q_est) < 1e-8:
                break
            
            Q_est = max(Q_new, 0)
        
        return Q_est
    
    def compute_travel_time(self) -> np.ndarray:
        """
        计算坡面水流传播时间
        
        Returns:
            tt: 传播时间 (s)
        """
        velocity = (1.0 / self.n) * (self.slope ** 0.5) * (0.001 ** (2.0/3.0))
        velocity = np.maximum(velocity, 1e-6)
        tt = self.cell_size / velocity
        return tt


def route_hillslope(
    Q_in: np.ndarray,
    R: np.ndarray,
    slope: np.ndarray,
    n: np.ndarray,
    cell_size: float,
    dt: float = 3600.0
) -> np.ndarray:
    """
    便捷函数: 坡面汇流计算
    
    Args:
        Q_in: 入流量 (m³/s)
        R: 产流量 (mm/h)
        slope: 坡度
        n: 曼宁系数
        cell_size: 网格大小
        dt: 时间步长
        
    Returns:
        Q_out: 出流量 (m³/s)
    """
    router = SlopeRouter(slope, n, cell_size)
    return router.route(Q_in, R, dt)
