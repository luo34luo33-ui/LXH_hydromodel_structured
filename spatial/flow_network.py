import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import xarray as xr

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

D8_DELTAS = np.array([[1, 1, 0, -1, -1, -1, 0, 1],
                       [0, 1, 1, 1, 0, -1, -1, -1]], dtype=np.int64)


def _accumulate_flow_flat(
    flow_dir: np.ndarray,
    area: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    累积流计算（基于拓扑排序）

    基于D8流向，使用拓扑排序从上游到下游累加贡献面积

    Args:
        flow_dir: 流向编码 (1-8)
        area: 初始面积（通常为1）
        width, height: 栅格尺寸

    Returns:
        累积流（贡献面积）
    """
    acc = area.copy().astype(np.float64)
    acc_flat = acc.ravel()
    
    n = height * width
    in_degree = np.zeros(n, dtype=np.int32)
    out_neighbors = [[] for _ in range(n)]
    
    valid_mask = (flow_dir > 0)
    valid_flat = valid_mask.ravel()
    
    for i in range(height):
        for j in range(width):
            if not valid_mask[i, j]:
                continue
            
            idx = i * width + j
            fd = flow_dir[i, j]
            
            d = int(fd) - 1
            ni = i + int(D8_DELTAS[0, d])
            nj = j + int(D8_DELTAS[1, d])
            
            if 0 <= ni < height and 0 <= nj < width and valid_mask[ni, nj]:
                down_idx = ni * width + nj
                out_neighbors[idx].append(down_idx)
                in_degree[down_idx] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0 and valid_flat[i]])
    
    while queue:
        node = queue.popleft()
        
        for down_idx in out_neighbors[node]:
            acc_flat[down_idx] += acc_flat[node]
            in_degree[down_idx] -= 1
            if in_degree[down_idx] == 0:
                queue.append(down_idx)
    
    return acc


def _strahler_order_flat(
    flow_dir: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Strahler河流分级（numba加速版）

    Args:
        flow_dir: 流向编码
        width, height: 栅格尺寸

    Returns:
        Strahler分级 (1为源头, 2+为主流)
    """
    order = np.zeros_like(flow_dir, dtype=np.int16)
    
    def get_order_from_upstream(i, j, depth=0):
        if depth > height * width:
            return 0
        
        upstream = []
        for d in range(8):
            ni = i + D8_DELTAS[0, d]
            nj = j + D8_DELTAS[1, d]
            if 0 <= ni < height and 0 <= nj < width:
                d_down = flow_dir[ni, nj] - 1
                if d_down >= 0:
                    ni_down = ni + D8_DELTAS[0, d_down]
                    nj_down = nj + D8_DELTAS[1, d_down]
                    if ni_down == i and nj_down == j:
                        upstream.append((ni, nj))
        
        if not upstream:
            return 1
        
        max_order = 0
        for ni, nj in upstream:
            upstream_order = get_order_from_upstream(ni, nj, depth + 1)
            if upstream_order > max_order:
                max_order = upstream_order
        
        return max_order
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if flow_dir[i, j] == 0:
                continue
            order[i, j] = get_order_from_upstream(i, j)
    
    return order


@dataclass
class RiverReach:
    """河道河段"""
    reach_id: int
    order: int
    cells: List[Tuple[int, int]]
    length: float
    slope: float
    width: float


class FlowNetwork:
    """汇流网络构建器"""

    UNIT_US = 0  # 边坡单元 (Upslope)
    UNIT_RN = 1  # 河道单元 (River)
    UNIT_RV = 2  # 水库单元 (Reservoir)

    def __init__(
        self,
        dem: np.ndarray,
        flow_dir: np.ndarray,
        slope: Optional[np.ndarray] = None,
        river_threshold: float = 0.01,
        reservoir_threshold: float = 0.1
    ):
        """
        Args:
            dem: 高程数据
            flow_dir: D8流向
            slope: 坡度数据
            river_threshold: 河道识别阈值（累积面积比例）
            reservoir_threshold: 水库识别阈值
        """
        self.dem = dem
        self.flow_dir = flow_dir
        self.slope = slope
        self.height, self.width = dem.shape
        
        self.accumulation = None
        self.strahler = None
        self.unit_type = None
        self.river_threshold = river_threshold
        self.reservoir_threshold = reservoir_threshold

    def accumulate_flow(self) -> np.ndarray:
        """计算累积流"""
        if self.accumulation is None:
            area = np.ones_like(self.dem, dtype=np.float64)
            self.accumulation = _accumulate_flow_flat(
                self.flow_dir, area, self.width, self.height
            )
        return self.accumulation

    def strahler_order(self) -> np.ndarray:
        """计算Strahler河流分级"""
        if self.strahler is None:
            self.strahler = _strahler_order_flat(
                self.flow_dir, self.width, self.height
            )
        return self.strahler

    def classify_units(
        self,
        acc_threshold: float = 0.01,
        reservoir_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        网格单元分类

        Args:
            acc_threshold: 河道识别累积面积阈值（面积比例）
            reservoir_mask: 水库位置掩码

        Returns:
            unit_type: 单元类型矩阵 (0=边坡, 1=河道, 2=水库)
        """
        acc = self.accumulate_flow()
        max_acc = acc.max()
        threshold = acc_threshold * max_acc
        
        valid_mask = self.flow_dir > 0
        
        if reservoir_mask is not None:
            self.unit_type = np.where(
                valid_mask & (acc >= threshold), self.UNIT_RN,
                np.where(
                    valid_mask & reservoir_mask,
                    self.UNIT_RV,
                    np.where(valid_mask, self.UNIT_US, -1)
                )
            )
        else:
            self.unit_type = np.where(
                valid_mask & (acc >= threshold), self.UNIT_RN,
                np.where(valid_mask, self.UNIT_US, -1)
            )
        
        return self.unit_type

    def get_downstream(self, i: int, j: int) -> Optional[Tuple[int, int]]:
        """获取下游单元"""
        d = self.flow_dir[i, j]
        if d == 0:
            return None
        d -= 1
        ni = i + D8_DELTAS[0, d]
        nj = j + D8_DELTAS[1, d]
        return (ni, nj)

    def get_upstream(self, i: int, j: int) -> List[Tuple[int, int]]:
        """获取上游邻接单元列表"""
        upstream = []
        for d in range(8):
            ni = i + D8_DELTAS[0, d]
            nj = j + D8_DELTAS[1, d]
            if 0 <= ni < self.height and 0 <= nj < self.width:
                d_down = self.flow_dir[ni, nj] - 1
                if d_down >= 0:
                    ni_down = ni + D8_DELTAS[0, d_down]
                    nj_down = nj + D8_DELTAS[1, d_down]
                    if ni_down == i and nj_down == j:
                        upstream.append((ni, nj))
        return upstream

    def find_outlet(self) -> Tuple[int, int]:
        """查找流域出口（最大累积流位置）"""
        acc = self.accumulate_flow()
        idx = np.argmax(acc)
        i = idx // self.width
        j = idx % self.width
        return (i, j)

    def extract_reaches(self, min_order: int = 2) -> List[RiverReach]:
        """提取河道河段"""
        if self.strahler is None:
            self.strahler_order()
        
        reaches = []
        visited = np.zeros_like(self.dem, dtype=bool)
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if visited[i, j] or self.strahler[i, j] < min_order:
                    continue
                
                reach_cells = []
                order = self.strahler[i, j]
                ci, cj = i, j
                
                while True:
                    visited[ci, cj] = True
                    reach_cells.append((ci, cj))
                    
                    downstream = self.get_downstream(ci, cj)
                    if downstream is None or visited[downstream[0], downstream[1]]:
                        break
                    
                    if self.strahler[downstream[0], downstream[1]] < order:
                        break
                    
                    ci, cj = downstream
                
                if len(reach_cells) > 0:
                    reach_id = len(reaches) + 1
                    length = len(reach_cells)
                    slope_avg = self.slope[reach_cells[0]] if self.slope is not None else 0.01
                    reaches.append(RiverReach(
                        reach_id=reach_id,
                        order=order,
                        cells=reach_cells,
                        length=length,
                        slope=slope_avg,
                        width=10.0
                    ))
        
        return reaches

    def to_xarray(self) -> xr.Dataset:
        """转换为xarray Dataset"""
        return xr.Dataset({
            'dem': (['y', 'x'], self.dem),
            'flow_direction': (['y', 'x'], self.flow_dir),
            'accumulation': (['y', 'x'], self.accumulate_flow()),
            'strahler_order': (['y', 'x'], self.strahler_order() if self.strahler is not None else np.zeros_like(self.dem)),
            'unit_type': (['y', 'x'], self.unit_type if self.unit_type is not None else np.zeros_like(self.dem, dtype=int)),
            'slope': (['y', 'x'], self.slope if self.slope is not None else np.zeros_like(self.dem))
        })


def strahler_order(flow_dir: np.ndarray) -> np.ndarray:
    """便捷函数：Strahler分级"""
    height, width = flow_dir.shape
    return _strahler_order_flat(flow_dir, width, height)


def accumulate_flow(flow_dir: np.ndarray) -> np.ndarray:
    """便捷函数：累积流计算"""
    height, width = flow_dir.shape
    area = np.ones_like(flow_dir, dtype=np.float64)
    return _accumulate_flow_flat(flow_dir, area, width, height)
