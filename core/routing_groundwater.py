"""
Groundwater Routing (地下径流汇流计算)
Based on Liuxihe Model principles (Section 2.3.7)

Method: Linear Reservoir Method

Formula:
    Qg(t+dt) = omega * Qg(t) + (1-omega) * Qper
"""

import numpy as np
from typing import Tuple, Dict


class GroundwaterRouter:
    """地下径流汇流模型 - 线性水库法"""
    
    def __init__(self, omega: float = 0.997, initial_flow: float = 0.0):
        self.omega = omega
        self.Qg = initial_flow
    
    def route(self, Q_per: float, dt: float = 3600.0) -> Tuple[float, float]:
        """地下径流汇流计算"""
        Qg_new = self.omega * self.Qg + (1 - self.omega) * Q_per
        self.Qg = Qg_new
        return Qg_new, Qg_new
    
    def compute_basin_percolation(self, Q_per_grid: np.ndarray, cell_area: float) -> float:
        """计算流域总渗漏量"""
        Q_per_mm = np.sum(Q_per_grid)
        Q_per_total = Q_per_mm * cell_area / 1000.0 / 3600.0
        return Q_per_total
    
    def get_state(self) -> Dict[str, float]:
        return {'Qg': self.Qg, 'omega': self.omega}
    
    def set_state(self, Qg: float = None, omega: float = None):
        if Qg is not None:
            self.Qg = Qg
        if omega is not None:
            self.omega = omega


def route_groundwater(Qg_prev: float, Q_per: float, omega: float = 0.997) -> float:
    """便捷函数: 地下径流汇流计算"""
    return omega * Qg_prev + (1 - omega) * Q_per
