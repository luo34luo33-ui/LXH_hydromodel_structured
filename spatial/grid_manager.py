import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import deque
import xarray as xr

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


D8_DELTAS = np.array([[1, 1, 0, -1, -1, -1, 0, 1],
                       [0, 1, 1, 1, 0, -1, -1, -1]], dtype=np.int64)


class TopologicalSort:
    """汇流网络拓扑排序"""

    def __init__(self, flow_dir: np.ndarray):
        """
        Args:
            flow_dir: D8流向矩阵
        """
        self.flow_dir = flow_dir
        self.height, self.width = flow_dir.shape
        self._topo_order = None
        self._reverse_topo = None

    def sort(self) -> np.ndarray:
        """
        计算拓扑排序（从上游到下游）

        Returns:
            topo_order: 拓扑序列（展平索引）
        """
        if self._topo_order is not None:
            return self._topo_order
        
        n = self.height * self.width
        in_degree = np.zeros(n, dtype=np.int32)
        adj_list = [[] for _ in range(n)]
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                idx = i * self.width + j
                d = self.flow_dir[i, j]
                
                if d > 0:
                    d = int(d) - 1
                    ni = int(i) + int(D8_DELTAS[0, d])
                    nj = int(j) + int(D8_DELTAS[1, d])
                    if 0 <= ni < self.height and 0 <= nj < self.width:
                        down_idx = ni * self.width + nj
                        adj_list[down_idx].append(idx)
                        in_degree[idx] += 1
        
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        topo = []
        
        while queue:
            node = queue.popleft()
            topo.append(node)
            
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        self._topo_order = np.array(topo, dtype=np.int32)
        return self._topo_order

    def sort_by_accumulation(self, acc: np.ndarray) -> np.ndarray:
        """
        按累积流排序（优先处理上游小流域）

        Args:
            acc: 累积流矩阵

        Returns:
            排序后的索引数组
        """
        flat_acc = acc.flatten()
        sorted_idx = np.argsort(flat_acc)
        return sorted_idx[::-1]

    def get_processing_order(self, method: str = 'topo') -> np.ndarray:
        """
        获取处理顺序

        Args:
            method: 'topo' (拓扑排序) 或 'acc' (按累积流)

        Returns:
            处理顺序索引
        """
        if method == 'topo':
            return self.sort()
        elif method == 'acc':
            acc = np.ones_like(self.flow_dir, dtype=np.float64) * 1000
            return self.sort_by_accumulation(acc)
        else:
            raise ValueError(f"Unknown method: {method}")


def _compute_receiving_water_fraction(
    flow_dir: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    计算分水线方向的水量分配比例

    每个网格的上游来水按比例分配到各下游邻接网格

    Returns:
        fraction: (8, height, width) 各方向分配比例
    """
    fraction = np.zeros((8, height, width), dtype=np.float64)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            d = flow_dir[i, j]
            if d == 0:
                continue
            
            d -= 1
            ni = i + D8_DELTAS[0, d]
            nj = j + D8_DELTAS[1, d]
            
            if 0 <= ni < height and 0 <= nj < width:
                slope = 0.0
                for k in range(8):
                    ki = i + D8_DELTAS[0, k]
                    kj = j + D8_DELTAS[1, k]
                    if 0 <= ki < height and 0 <= kj < width:
                        dist = 1.0 if k % 2 == 0 else np.sqrt(2.0)
                        slope += max(0.0, (flow_dir[ki, kj] - 1) / dist)
                
                fraction[d, ni, nj] = 1.0 / max(slope, 1.0)
    
    return fraction


class GridManager:
    """网格管理器 - 统一管理分布式水文模型的网格数据"""

    def __init__(
        self,
        flow_dir: np.ndarray,
        unit_type: Optional[np.ndarray] = None,
        accumulation: Optional[np.ndarray] = None,
        use_gpu: bool = False
    ):
        """
        Args:
            flow_dir: D8流向
            unit_type: 单元类型 (0=边坡, 1=河道, 2=水库)
            accumulation: 累积流
            use_gpu: 是否使用GPU
        """
        self.flow_dir = flow_dir
        self.unit_type = unit_type
        self.accumulation = accumulation
        self.use_gpu = use_gpu
        self.height, self.width = flow_dir.shape
        self.n_cells = self.height * self.width
        
        self.topo_sorter = TopologicalSort(flow_dir)
        self._flat_idx_to_2d = None
        self._2d_to_flat_idx = None

    def initialize(self):
        """初始化网格管理器"""
        self.topo_order = self.topo_sorter.sort()
        self._build_index_maps()
        
        if self.accumulation is not None:
            self._sort_by_accumulation()

    def _build_index_maps(self):
        """构建索引映射"""
        self._flat_idx_to_2d = np.zeros((self.n_cells, 2), dtype=np.int32)
        self._2d_to_flat_idx = np.zeros((self.height, self.width), dtype=np.int32)
        
        idx = 0
        for i in range(self.height):
            for j in range(self.width):
                self._flat_idx_to_2d[idx] = [i, j]
                self._2d_to_flat_idx[i, j] = idx
                idx += 1

    def _sort_by_accumulation(self):
        """按累积流重新排序（从上游到下游，小到大）"""
        flat_acc = self.accumulation.flatten()
        self.topo_order = np.argsort(flat_acc)

    def flat_to_2d(self, flat_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """展平索引转2D索引"""
        if isinstance(flat_idx, (int, np.int32, np.int64)):
            return self._flat_idx_to_2d[flat_idx]
        
        coords = self._flat_idx_to_2d[flat_idx]
        return coords[:, 0], coords[:, 1]

    def d2_to_flat(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """2D索引转展平索引"""
        return self._2d_to_flat_idx[i, j]

    def get_downstream(self, i: int, j: int) -> Optional[Tuple[int, int]]:
        """获取下游网格坐标"""
        d = self.flow_dir[i, j]
        if d == 0:
            return None
        d -= 1
        ni = i + D8_DELTAS[0, d]
        nj = j + D8_DELTAS[1, d]
        return (ni, nj)

    def get_upstream(self, i: int, j: int) -> List[Tuple[int, int]]:
        """获取上游邻接网格列表"""
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

    def iter_upstream(self, i: int, j: int):
        """迭代上游网格"""
        return iter(self.get_upstream(i, j))

    def get_unit_type(self, i: int, j: int) -> int:
        """获取单元类型"""
        if self.unit_type is None:
            return 0
        return self.unit_type[i, j]

    def get_river_cells(self) -> np.ndarray:
        """获取所有河道单元"""
        if self.unit_type is None:
            return np.array([], dtype=np.int32)
        mask = self.unit_type == 1
        return np.where(mask.flatten())[0]

    def get_slope_cells(self) -> np.ndarray:
        """获取所有边坡单元"""
        if self.unit_type is None:
            return np.arange(self.n_cells)
        mask = self.unit_type == 0
        return np.where(mask.flatten())[0]

    def get_reservoir_cells(self) -> np.ndarray:
        """获取所有水库单元"""
        if self.unit_type is None:
            return np.array([], dtype=np.int32)
        mask = self.unit_type == 2
        return np.where(mask.flatten())[0]

    def create_state_arrays(self, dtype=np.float64) -> Dict[str, np.ndarray]:
        """
        创建状态变量数组

        Returns:
            包含所有状态变量的字典
        """
        return {
            'soil_moisture': np.zeros((self.height, self.width), dtype=dtype),
            'surface_runoff': np.zeros((self.height, self.width), dtype=dtype),
            'subsurface_flow': np.zeros((self.height, self.width), dtype=dtype),
            'channel_flow': np.zeros((self.height, self.width), dtype=dtype),
            'groundwater': np.zeros((self.height, self.width), dtype=dtype),
            'reservoir_level': np.zeros((self.height, self.width), dtype=dtype),
        }

    def to_xarray(self) -> xr.Dataset:
        """导出为xarray Dataset"""
        data_vars = {
            'flow_direction': (['y', 'x'], self.flow_dir),
            'unit_type': (['y', 'x'], self.unit_type) if self.unit_type is not None else None,
            'accumulation': (['y', 'x'], self.accumulation) if self.accumulation is not None else None,
        }
        data_vars = {k: v for k, v in data_vars.items() if v is not None}
        
        return xr.Dataset(data_vars)

    def get_subcatchment(
        self,
        outlet_i: int,
        outlet_j: int
    ) -> np.ndarray:
        """
        提取子流域

        Args:
            outlet_i, outlet_j: 出口位置

        Returns:
            子流域掩码
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        queue = deque([(outlet_i, outlet_j)])
        
        while queue:
            i, j = queue.popleft()
            if mask[i, j]:
                continue
            mask[i, j] = True
            
            for ni, nj in self.get_upstream(i, j):
                if not mask[ni, nj]:
                    queue.append((ni, nj))
        
        return mask
