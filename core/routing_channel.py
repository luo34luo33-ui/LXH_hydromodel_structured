"""
Channel Routing (河道汇流计算)
Based on Liuxihe Model principles (Section 2.3.4)

Method: Diffusive Wave Method (扩散波法)

Formulas:
    A = c * Q^(3/5)  (trapezoidal cross-section)
    Newton-Raphson solution for Saint-Venant equations
"""

import numpy as np
from typing import Optional, Tuple


class ChannelRouter:
    """
    河道汇流模型 - 扩散波法
    
    采用一维扩散波法进行河道汇流计算
    河道断面概化为梯形
    """
    
    def __init__(
        self,
        flow_dir: np.ndarray,
        slope: np.ndarray,
        mannings_n: np.ndarray,
        channel_width: np.ndarray = None,
        channel_side_slope: float = 2.0,
        cell_size: float = 90.0,
        mask: np.ndarray = None
    ):
        """
        Args:
            flow_dir: D8流向 (1-8)
            slope: 河道底坡 (ratio)
            mannings_n: 曼宁系数
            channel_width: 河道底宽 (m), 默认20m
            channel_side_slope: 河道侧坡 (H:V)
            cell_size: 网格大小 (m)
            mask: 有效网格掩码
        """
        self.flow_dir = flow_dir
        self.slope = np.maximum(slope, 0.0001)
        self.n = np.maximum(mannings_n, 0.01)
        self.channel_width = channel_width if channel_width is not None else np.ones_like(slope) * 20.0
        self.side_slope = channel_side_slope
        self.cell_size = cell_size
        self.mask = mask
        
        self.D8_DELTAS = np.array([
            [1, 1, 0, -1, -1, -1, 0, 1],
            [0, 1, 1, 1, 0, -1, -1, -1]
        ], dtype=np.int64)
    
    def get_upstream_cells(self, i: int, j: int) -> list:
        """
        获取上游邻接单元
        
        Args:
            i, j: 网格坐标
            
        Returns:
            [(ni, nj), ...]: 上游单元列表
        """
        upstream = []
        for d in range(8):
            ni = i + int(self.D8_DELTAS[0, d])
            nj = j + int(self.D8_DELTAS[1, d])
            
            if 0 <= ni < self.flow_dir.shape[0] and 0 <= nj < self.flow_dir.shape[1]:
                d_down = int(self.flow_dir[ni, nj]) - 1
                if 0 <= d_down < 8:
                    ni_down = ni + int(self.D8_DELTAS[0, d_down])
                    nj_down = nj + int(self.D8_DELTAS[1, d_down])
                    if ni_down == i and nj_down == j:
                        upstream.append((ni, nj))
        
        return upstream
    
    def route(
        self,
        Q_in: np.ndarray,
        Q_upstream: np.ndarray,
        lateral_inflow: np.ndarray = None,
        dt: float = 3600.0
    ) -> np.ndarray:
        """
        河道汇流计算
        
        Args:
            Q_in: 单元入流量 (m³/s)
            Q_upstream: 上游来流量 (m³/s)
            lateral_inflow: 侧向入流 (m³/s)
            dt: 时间步长 (s)
            
        Returns:
            Q_out: 出流量 (m³/s)
        """
        if lateral_inflow is None:
            lateral_inflow = np.zeros_like(Q_in)
        
        Q_total = Q_in + Q_upstream + lateral_inflow
        
        Q_out = np.zeros_like(Q_total)
        
        for i in range(self.flow_dir.shape[0]):
            for j in range(self.flow_dir.shape[1]):
                if self.flow_dir[i, j] == 0:
                    continue
                
                if self.mask is not None and not self.mask[i, j]:
                    continue
                
                w = max(self.channel_width[i, j], 1.0)
                S0 = self.slope[i, j]
                n = self.n[i, j]
                beta = self.side_slope
                
                Q_in_cell = Q_total[i, j]
                
                Q_up = 0
                for ui, uj in self.get_upstream_cells(i, j):
                    Q_up += Q_out[ui, uj]
                
                Q_prev = Q_out[i, j] if Q_out[i, j] > 0 else Q_in_cell * 0.5
                
                Q_est = Q_in_cell + Q_up
                
                for _ in range(30):
                    h = self._compute_depth(Q_est, w, S0, n, beta)
                    
                    chi = w + 2 * h / np.sin(np.arctan(1/beta))
                    R = w * h / chi if chi > 0 else 0.1
                    
                    v = (1.0 / n) * (R ** (2.0/3.0)) * (S0 ** 0.5)
                    c = v * chi / (3600 * n * R ** (2.0/3.0) * S0 ** 0.5)
                    
                    Sf = S0 - (h - 0) / self.cell_size
                    Sf = max(Sf, 0.0001)
                    
                    v_new = (1.0 / n) * (R ** (2.0/3.0)) * (Sf ** 0.5)
                    c_new = v_new * chi / (3600 * n * R ** (2.0/3.0) * Sf ** 0.5)
                    
                    dt_dx = dt / self.cell_size
                    
                    numerator = dt_dx * Q_in_cell + c * h + lateral_inflow[i, j] * dt
                    denominator = dt_dx + c_new
                    
                    Q_new = numerator / denominator
                    
                    if abs(Q_new - Q_est) < 1e-6 or Q_new < 0:
                        Q_out[i, j] = max(Q_new, 0)
                        break
                    
                    Q_est = max(Q_new, 0)
                
                Q_out[i, j] = max(Q_est, 0)
        
        return Q_out
    
    def _compute_depth(
        self,
        Q: float,
        w: float,
        S0: float,
        n: float,
        beta: float
    ) -> float:
        """
        根据流量计算水深 (梯形断面)
        
        Args:
            Q: 流量 (m³/s)
            w: 底宽 (m)
            S0: 底坡
            n: 曼宁系数
            beta: 侧坡
            
        Returns:
            h: 水深 (m)
        """
        if Q <= 0:
            return 0.001
        
        chi_coef = (3600 * n / (w ** (5.0/3.0) * S0 ** 0.5)) ** (3.0/5.0)
        
        A = w * 0.5 + chi_coef * Q ** (3.0/5.0)
        
        discriminant = w ** 2 + 4 * A * beta
        h = (-w + np.sqrt(discriminant)) / (2 * beta)
        
        return max(h, 0.001)
    
    def route_cell(
        self,
        Q_in: float,
        Q_prev: float,
        lateral: float,
        dt: float
    ) -> float:
        """
        单网格河道汇流计算
        
        Args:
            Q_in: 入流量 (m³/s)
            Q_prev: 前期出流 (m³/s)
            lateral: 侧向入流 (m³/s)
            dt: 时间步长 (s)
            
        Returns:
            Q_out: 出流量 (m³/s)
        """
        w = max(self.channel_width.flat[0], 1.0) if self.channel_width.ndim > 0 else max(self.channel_width, 1.0)
        S0 = max(self.slope.flat[0], 0.0001) if self.slope.ndim > 0 else max(self.slope, 0.0001)
        n = max(self.n.flat[0], 0.01) if self.n.ndim > 0 else max(self.n, 0.01)
        
        Q_est = Q_in + lateral
        Q_prev_est = max(Q_prev, Q_est * 0.5)
        
        for _ in range(50):
            h = self._compute_depth(Q_est, w, S0, n, self.side_slope)
            
            chi = w + 2 * h / np.sin(np.arctan(1/self.side_slope))
            R = w * h / chi
            
            Sf = S0 - h / self.cell_size
            Sf = max(Sf, 0.0001)
            
            v = (1.0 / n) * (R ** (2.0/3.0)) * (Sf ** 0.5)
            c = v * chi / (3600 * n * R ** (2.0/3.0) * Sf ** 0.5)
            
            C0 = dt * c / self.cell_size / (1 + dt * c / self.cell_size)
            C1 = 1.0 / (1 + dt * c / self.cell_size)
            
            Q_new = C0 * (Q_in + lateral) + C1 * Q_prev_est
            
            if abs(Q_new - Q_est) < 1e-8:
                return max(Q_new, 0)
            
            Q_est = max(Q_new, 0)
            Q_prev_est = Q_new
        
        return max(Q_est, 0)


def route_channel(
    Q_in: np.ndarray,
    Q_prev: np.ndarray,
    flow_dir: np.ndarray,
    slope: np.ndarray,
    n: np.ndarray,
    channel_width: np.ndarray,
    dt: float,
    cell_size: float
) -> np.ndarray:
    """
    便捷函数: 河道汇流计算
    """
    router = ChannelRouter(flow_dir, slope, n, channel_width, cell_size=cell_size)
    return router.route(Q_in, Q_prev, dt=dt)
