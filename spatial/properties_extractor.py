import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional
from pathlib import Path


class PropertiesExtractor:
    """流域属性参数提取器 - 从LULC/土壤类型提取模型参数"""

    SOIL_HYDRAULIC_PARAMS = {
        'theta_sat': {
            'clay': 0.41, 'sandy_clay': 0.38, 'silt_clay': 0.47,
            'sand': 0.39, 'loamy_sand': 0.41, 'sandy_loam': 0.41,
            'loam': 0.43, 'silt_loam': 0.45, 'sandy_clay_loam': 0.39,
            'clay_loam': 0.41, 'silt_clay_loam': 0.43, 'peat': 0.55,
            'default': 0.40
        },
        'theta_fc': {
            'clay': 0.32, 'sandy_clay': 0.29, 'silt_clay': 0.37,
            'sand': 0.10, 'loamy_sand': 0.13, 'sandy_loam': 0.21,
            'loam': 0.27, 'silt_loam': 0.33, 'sandy_clay_loam': 0.27,
            'clay_loam': 0.34, 'silt_clay_loam': 0.36, 'peat': 0.42,
            'default': 0.25
        },
        'theta_wp': {
            'clay': 0.20, 'sandy_clay': 0.17, 'silt_clay': 0.24,
            'sand': 0.04, 'loamy_sand': 0.05, 'sandy_loam': 0.08,
            'loam': 0.10, 'silt_loam': 0.16, 'sandy_clay_loam': 0.13,
            'clay_loam': 0.19, 'silt_clay_loam': 0.22, 'peat': 0.26,
            'default': 0.10
        },
        'K_sat': {
            'clay': 10, 'sandy_clay': 20, 'silt_clay': 5,
            'sand': 500, 'loamy_sand': 400, 'sandy_loam': 200,
            'loam': 100, 'silt_loam': 50, 'sandy_clay_loam': 80,
            'clay_loam': 30, 'silt_clay_loam': 20, 'peat': 200,
            'default': 100
        },
        'b_exp': {
            'clay': 6.4, 'sandy_clay': 4.4, 'silt_clay': 8.4,
            'sand': 2.8, 'loamy_sand': 3.3, 'sandy_loam': 3.6,
            'loam': 4.2, 'silt_loam': 4.8, 'sandy_clay_loam': 5.2,
            'clay_loam': 5.8, 'silt_clay_loam': 6.8, 'peat': 3.5,
            'default': 4.0
        }
    }

    LANDUSE_PARAMS = {
        'vegetation_coverage': {
            1: 0.9, 2: 0.8, 3: 0.6, 4: 0.3, 5: 0.1, 6: 0.0,
            'default': 0.5
        },
        'root_depth': {
            1: 1.5, 2: 1.2, 3: 0.8, 4: 0.4, 5: 0.1, 6: 0.0,
            'default': 0.5
        },
        'mannings_n': {
            1: 0.1, 2: 0.15, 3: 0.25, 4: 0.35, 5: 0.05, 6: 0.02,
            'default': 0.1
        }
    }

    def __init__(self, cell_size: float = 100.0):
        self.cell_size = cell_size

    def extract_soil_params(
        self,
        soil_type: np.ndarray,
        soil_type_map: Optional[Dict[int, str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        从土壤类型提取水力参数

        Args:
            soil_type: 土壤类型编码
            soil_type_map: 类型编码到类型名称的映射

        Returns:
            参数字典 (theta_sat, theta_fc, theta_wp, K_sat, b_exp)
        """
        if soil_type_map is None:
            soil_type_map = {i: str(i) for i in range(1, 13)}
        
        params = {}
        for param_name, param_table in self.SOIL_HYDRAULIC_PARAMS.items():
            param_grid = np.zeros_like(soil_type, dtype=np.float64)
            
            for code, soil_name in soil_type_map.items():
                mask = soil_type == code
                value = param_table.get(soil_name, param_table['default'])
                param_grid[mask] = value
            
            params[param_name] = param_grid
        
        return params

    def extract_landuse_params(
        self,
        lulc: np.ndarray,
        lulc_map: Optional[Dict[int, str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        从土地利用提取参数

        Args:
            lulc: 土地利用类型编码
            lulc_map: 类型编码到名称的映射

        Returns:
            参数字典 (veg_coverage, root_depth, mannings_n)
        """
        if lulc_map is None:
            lulc_map = {1: 'forest', 2: 'grass', 3: 'cropland', 
                       4: 'urban', 5: 'water', 6: 'bare'}
        
        params = {}
        for param_name, param_table in self.LANDUSE_PARAMS.items():
            param_grid = np.zeros_like(lulc, dtype=np.float64)
            
            for code, lulc_name in lulc_map.items():
                mask = lulc == code
                value = param_table.get(code, param_table['default'])
                param_grid[mask] = value
            
            params[param_name] = param_grid
        
        return params

    def extract_all_params(
        self,
        dem: np.ndarray,
        slope: np.ndarray,
        soil_type: Optional[np.ndarray] = None,
        lulc: Optional[np.ndarray] = None,
        soil_type_map: Optional[Dict[int, str]] = None,
        lulc_map: Optional[Dict[int, str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        提取所有模型参数

        Args:
            dem: 高程
            slope: 坡度
            soil_type: 土壤类型
            lulc: 土地利用
            soil_type_map: 土壤类型映射
            lulc_map: 土地利用映射

        Returns:
            完整参数字典
        """
        params = {
            'elevation': dem,
            'slope': slope,
            'cell_area': np.ones_like(dem) * self.cell_size ** 2,
        }
        
        if soil_type is not None:
            soil_params = self.extract_soil_params(soil_type, soil_type_map)
            params.update(soil_params)
        else:
            params.update({
                'theta_sat': np.ones_like(dem) * 0.40,
                'theta_fc': np.ones_like(dem) * 0.25,
                'theta_wp': np.ones_like(dem) * 0.10,
                'K_sat': np.ones_like(dem) * 100.0,
                'b_exp': np.ones_like(dem) * 4.0,
            })
        
        if lulc is not None:
            lulc_params = self.extract_landuse_params(lulc, lulc_map)
            params.update(lulc_params)
        else:
            params.update({
                'veg_coverage': np.ones_like(dem) * 0.5,
                'root_depth': np.ones_like(dem) * 0.5,
                'mannings_n': np.ones_like(dem) * 0.1,
            })
        
        return params

    def to_dataclass(self, params: Dict[str, np.ndarray]) -> 'ModelParameters':
        """转换为参数据类"""
        return ModelParameters(**params)


class ModelParameters:
    """模型参数据类 - 便于传递给物理引擎"""
    
    def __init__(
        self,
        elevation: np.ndarray,
        slope: np.ndarray,
        cell_area: np.ndarray,
        theta_sat: np.ndarray,
        theta_fc: np.ndarray,
        theta_wp: np.ndarray,
        K_sat: np.ndarray,
        b_exp: np.ndarray,
        veg_coverage: np.ndarray,
        root_depth: np.ndarray,
        mannings_n: np.ndarray,
        WMM: float = 10.0,
        K: float = 0.4,
        B: float = 4.0,
        IM: float = 0.01,
        KE: float = 15.0,
        XE: float = 0.5,
        KG: float = 0.1,
        CG: float = 0.98,
        KKG: float = 0.15,
        C: float = 0.15,
        CS: float = 0.5,
    ):
        self.elevation = elevation
        self.slope = slope
        self.cell_area = cell_area
        self.theta_sat = theta_sat
        self.theta_fc = theta_fc
        self.theta_wp = theta_wp
        self.K_sat = K_sat
        self.b_exp = b_exp
        self.veg_coverage = veg_coverage
        self.root_depth = root_depth
        self.mannings_n = mannings_n
        
        self.WMM = WMM
        self.K = K
        self.B = B
        self.IM = IM
        self.KE = KE
        self.XE = XE
        self.KG = KG
        self.CG = CG
        self.KKG = KKG
        self.C = C
        self.CS = CS

    def to_dict(self) -> Dict[str, np.ndarray]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_config(
        cls,
        params: Dict,
        extractor: PropertiesExtractor,
        dem: np.ndarray,
        slope: np.ndarray,
        soil_type: Optional[np.ndarray] = None,
        lulc: Optional[np.ndarray] = None
    ) -> 'ModelParameters':
        """从配置和空间数据创建"""
        spatial_params = extractor.extract_all_params(
            dem, slope, soil_type, lulc
        )
        spatial_params.update(params)
        return cls(**spatial_params)
