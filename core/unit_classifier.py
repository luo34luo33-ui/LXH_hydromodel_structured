"""
Unit Classifier (单元分类)
Based on Liuxihe Model principles (Section 2.2.2, 2.4)

Classifies cells into:
- 0: Slope unit (边坡单元)
- 1: River unit (河道单元)
- 2: Reservoir unit (水库单元)
"""

import numpy as np
from typing import Tuple


class UnitClassifier:
    """
    流域单元分类器
    
    根据累积流阈值和水库掩码将网格分为:
    - 边坡单元
    - 河道单元
    - 水库单元
    """
    
    UNIT_SLOPE = 0
    UNIT_RIVER = 1
    UNIT_RESERVOIR = 2
    
    def __init__(
        self,
        flow_accumulation: np.ndarray,
        river_threshold: float = 10.0,
        reservoir_mask: np.ndarray = None
    ):
        """
        Args:
            flow_accumulation: 累积流数组
            river_threshold: 河道识别阈值 (累积流值)
            reservoir_mask: 水库单元掩码
        """
        self.flow_accumulation = flow_accumulation
        self.river_threshold = river_threshold
        self.reservoir_mask = reservoir_mask
    
    def classify(
        self,
        method: str = 'threshold'
    ) -> np.ndarray:
        """
        单元分类
        
        Args:
            method: 'threshold' or 'strahler'
            
        Returns:
            unit_type: 单元类型矩阵
        """
        if method == 'threshold':
            return self._classify_by_threshold()
        elif method == 'strahler':
            return self._classify_by_strahler()
        else:
            return self._classify_by_threshold()
    
    def _classify_by_threshold(self) -> np.ndarray:
        """基于阈值的分类方法"""
        unit_type = np.where(
            self.flow_accumulation >= self.river_threshold,
            self.UNIT_RIVER,
            self.UNIT_SLOPE
        )
        
        if self.reservoir_mask is not None:
            unit_type[self.reservoir_mask] = self.UNIT_RESERVOIR
        
        return unit_type
    
    def _classify_by_strahler(self,strahler_order: np.ndarray = None) -> np.ndarray:
        """基于Strahler分级的分类方法"""
        if strahler_order is None:
            from spatial.flow_network import strahler_order
            unit_type = np.zeros_like(self.flow_accumulation)
        else:
            unit_type = np.where(
                strahler_order >= 3,
                self.UNIT_RIVER,
                self.UNIT_SLOPE
            )
        
        if self.reservoir_mask is not None:
            unit_type[self.reservoir_mask] = self.UNIT_RESERVOIR
        
        return unit_type
    
    def get_river_cells(self) -> np.ndarray:
        """获取河道单元掩码"""
        return self.flow_accumulation >= self.river_threshold
    
    def get_slope_cells(self) -> np.ndarray:
        """获取边坡单元掩码"""
        river = self.get_river_cells()
        slope = ~river
        if self.reservoir_mask is not None:
            slope = slope & ~self.reservoir_mask
        return slope
    
    def get_reservoir_cells(self) -> np.ndarray:
        """获取水库单元掩码"""
        if self.reservoir_mask is not None:
            return self.reservoir_mask
        return np.zeros_like(self.flow_accumulation, dtype=bool)


def classify_units(
    flow_acc: np.ndarray,
    threshold: float = 10.0,
    reservoir_mask: np.ndarray = None
) -> np.ndarray:
    """
    便捷函数: 单元分类
    
    Args:
        flow_acc: 累积流
        threshold: 河道阈值
        reservoir_mask: 水库掩码
        
    Returns:
        unit_type: 单元类型 (0=边坡, 1=河道, 2=水库)
    """
    classifier = UnitClassifier(flow_acc, threshold, reservoir_mask)
    return classifier.classify()
