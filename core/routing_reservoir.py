"""
Reservoir Routing (水库汇流计算)
Based on Liuxihe Model principles (Section 2.3.5)

Method: Water Balance Method (水量平衡法)
"""

import numpy as np
from typing import Dict


class ReservoirRouter:
    """水库汇流模型 - 水量平衡法"""
    
    def __init__(
        self,
        reservoir_mask: np.ndarray,
        stage_storage: Dict[float, float] = None,
        storage_discharge: Dict[float, float] = None,
        initial_stage: float = 0.0
    ):
        self.reservoir_mask = reservoir_mask
        self.stage_storage = stage_storage or {}
        self.storage_discharge = storage_discharge or {}
        self.stage = initial_stage
        self.storage = 0.0
        if self.stage_storage:
            self.storage = self.stage_storage.get(initial_stage, 0.0)
    
    def route(self, Q_in: np.ndarray, dt: float = 3600.0):
        Q_reservoir = np.zeros_like(Q_in)
        
        if self.reservoir_mask is not None:
            Q_reservoir[self.reservoir_mask] = Q_in[self.reservoir_mask]
        
        Q_total = float(np.sum(Q_reservoir))
        Q_out = self._compute_outflow()
        
        if self.storage_discharge:
            sorted_items = sorted(self.storage_discharge.items(), reverse=True)
            for storage, discharge in sorted_items:
                if self.storage >= storage:
                    Q_out = discharge
                    break
        
        dV = (Q_total - Q_out) * dt
        self.storage = max(self.storage + dV, 0)
        
        Q_reservoir[np.where(self.reservoir_mask)] = Q_out
        
        return Q_reservoir, self.stage, self.storage
    
    def _compute_outflow(self):
        if not self.storage_discharge:
            return float(np.sum(self.reservoir_mask))
        
        sorted_items = sorted(self.storage_discharge.items(), reverse=True)
        for storage, discharge in sorted_items:
            if self.storage >= storage:
                return discharge
        return 0.0
    
    def get_state(self):
        return {'stage': self.stage, 'storage': self.storage}


def route_reservoir_simple(Q_in: np.ndarray, reservoir_mask: np.ndarray) -> np.ndarray:
    """简单水库汇流: 直接累加入库流量"""
    Q_out = Q_in.copy()
    if reservoir_mask is not None:
        Q_out[reservoir_mask] = np.sum(Q_in[reservoir_mask])
    return Q_out
