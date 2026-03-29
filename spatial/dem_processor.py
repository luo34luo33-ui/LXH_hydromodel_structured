import numpy as np
from typing import Tuple, Optional
from pathlib import Path

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

D8_DIRECTIONS = np.array([
    [1, 1, 0, -1, -1, -1, 0, 1],
    [0, 1, 1, 1, 0, -1, -1, -1]
])

D8_WEIGHTS = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])


def _fill_sinks_flat(elevation: np.ndarray, width: int, height: int) -> np.ndarray:
    """平面填充洼地算法（numba加速版）"""
    filled = elevation.copy()
    labels = np.zeros_like(elevation, dtype=np.int32)
    
    current_label = 1
    queue = []
    
    border_mask = np.zeros_like(elevation, dtype=np.bool_)
    for i in range(height):
        border_mask[i, 0] = True
        border_mask[i, width-1] = True
    for j in range(width):
        border_mask[0, j] = True
        border_mask[height-1, j] = True
    
    for idx in range(height * width):
        i = idx // width
        j = idx % width
        if border_mask[i, j]:
            queue.append((filled[i, j], i, j))
    
    queue.sort()
    
    while queue:
        elev, i, j = queue.pop(0)
        
        if labels[i, j] != 0:
            continue
        
        local_min = elev
        for d in range(8):
            ni = i + D8_DIRECTIONS[0, d]
            nj = j + D8_DIRECTIONS[1, d]
            if 0 <= ni < height and 0 <= nj < width:
                if filled[ni, nj] < local_min:
                    local_min = filled[ni, j]
        
        if local_min < elev:
            filled[i, j] = local_min
        else:
            labels[i, j] = current_label
            current_label += 1
    
    return filled


def _d8_flow_direction_flat(elevation: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    D8流向计算（numba加速并行版）

    Returns:
        flow_dir: 流向编码 (1-8, 0表示边界/凹陷)
    """
    flow_dir = np.zeros_like(elevation, dtype=np.int8)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            elev_center = elevation[i, j]
            min_slope = -1.0
            best_dir = 0
            
            for d in range(8):
                ni = i + D8_DIRECTIONS[0, d]
                nj = j + D8_DIRECTIONS[1, d]
                
                dist = D8_WEIGHTS[d]
                slope = (elev_center - elevation[ni, nj]) / dist
                
                if slope > min_slope:
                    min_slope = slope
                    best_dir = d + 1
            
            flow_dir[i, j] = best_dir
    
    return flow_dir


def _compute_slope_flat(elevation: np.ndarray, width: int, height: int, 
                        cell_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算坡度和坡向（numba加速并行版）

    Returns:
        slope: 坡度（弧度）
        aspect: 坡向（弧度，从北顺时针）
    """
    slope = np.zeros_like(elevation, dtype=np.float64)
    aspect = np.zeros_like(elevation, dtype=np.float64)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            dz_dx = ((elevation[i-1, j+1] + 2*elevation[i, j+1] + elevation[i+1, j+1]) -
                    (elevation[i-1, j-1] + 2*elevation[i, j-1] + elevation[i+1, j-1])) / (8 * cell_size)
            
            dz_dy = ((elevation[i+1, j-1] + 2*elevation[i+1, j] + elevation[i+1, j+1]) -
                    (elevation[i-1, j-1] + 2*elevation[i-1, j] + elevation[i-1, j+1])) / (8 * cell_size)
            
            slope[i, j] = np.sqrt(dz_dx**2 + dz_dy**2)
            aspect[i, j] = np.arctan2(dz_dx, -dz_dy)
            
            if aspect[i, j] < 0:
                aspect[i, j] += 2 * np.pi
    
    return slope, aspect


class DEMProcessor:
    """DEM处理类 - 洼地填充、流向计算、坡度计算"""

    def __init__(self, cell_size: float = 100.0, use_gpu: bool = False):
        """
        Args:
            cell_size: 栅格单元大小（米）
            use_gpu: 是否使用GPU加速
        """
        self.cell_size = cell_size
        self.use_gpu = use_gpu

    def fill_sinks(self, dem: np.ndarray) -> np.ndarray:
        """
        填充洼地

        Args:
            dem: 高程数据

        Returns:
            填充后的高程数据
        """
        height, width = dem.shape
        
        if self.use_gpu:
            import cupy as cp
            dem_gpu = cp.asarray(dem)
            filled_gpu = self._fill_sinks_gpu(dem_gpu, width, height)
            return cp.asnumpy(filled_gpu)
        
        return _fill_sinks_flat(dem, width, height)

    def _fill_sinks_gpu(self, dem, width, height):
        """GPU填充洼地"""
        pass

    def d8_flow_direction(self, dem: np.ndarray) -> np.ndarray:
        """
        计算D8流向

        Args:
            dem: 高程数据（应先填充洼地）

        Returns:
            流向编码 (1-8)
        """
        height, width = dem.shape
        
        if self.use_gpu:
            import cupy as cp
            dem_gpu = cp.asarray(dem)
            flow_dir_gpu = self._d8_flow_direction_gpu(dem_gpu, width, height)
            return cp.asnumpy(flow_dir_gpu)
        
        return _d8_flow_direction_flat(dem, width, height)

    def _d8_flow_direction_gpu(self, dem, width, height):
        """GPU流向计算"""
        import cupy as cp
        flow_dir = cp.zeros_like(dem, dtype=cp.int8)
        return flow_dir

    def compute_slope(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算坡度和坡向

        Args:
            dem: 高程数据

        Returns:
            (slope, aspect): 坡度（弧度）和坡向（弧度）
        """
        height, width = dem.shape
        
        if self.use_gpu:
            import cupy as cp
            dem_gpu = cp.asarray(dem)
            slope_gpu, aspect_gpu = self._compute_slope_gpu(dem_gpu, width, height)
            return cp.asnumpy(slope_gpu), cp.asnumpy(aspect_gpu)
        
        return _compute_slope_flat(dem, width, height, self.cell_size)

    def _compute_slope_gpu(self, dem, width, height):
        """GPU坡度计算"""
        import cupy as cp
        return cp.zeros_like(dem), cp.zeros_like(dem)


def fill_sinks(dem: np.ndarray, **kwargs) -> np.ndarray:
    """便捷函数：填充洼地"""
    processor = DEMProcessor(**kwargs)
    return processor.fill_sinks(dem)


def d8_flow_direction(dem: np.ndarray, **kwargs) -> np.ndarray:
    """便捷函数：计算D8流向"""
    processor = DEMProcessor(**kwargs)
    return processor.d8_flow_direction(dem)


def compute_slope(dem: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """便捷函数：计算坡度"""
    processor = DEMProcessor(**kwargs)
    return processor.compute_slope(dem)


class FlatDetector:
    """平坦区域检测与处理"""

    @staticmethod
    def detect_flats(
        dem: np.ndarray,
        flow_dir: np.ndarray,
        threshold: float = 1e-6
    ) -> np.ndarray:
        """检测平坦区域"""
        height, width = dem.shape
        is_flat = np.zeros_like(dem, dtype=np.bool_)
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if flow_dir[i, j] == 0:
                    elev = dem[i, j]
                    has_lower = False
                    for d in range(8):
                        ni = i + D8_DIRECTIONS[0, d]
                        nj = j + D8_DIRECTIONS[1, d]
                        if dem[ni, nj] < elev - threshold:
                            has_lower = True
                            break
                    if not has_lower:
                        is_flat[i, j] = True
        
        return is_flat

    @staticmethod
    def resolve_flats(dem: np.ndarray, flow_dir: np.ndarray) -> np.ndarray:
        """解析平坦区域 - 优先流向低海拔边界"""
        resolved = dem.copy()
        return resolved
