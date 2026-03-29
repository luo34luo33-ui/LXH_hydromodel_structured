import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ConfigLoader:
    """配置加载器，支持YAML和JSON格式"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self._config = None
        self._params = None
        self._ranges = None

    def load_model_config(self) -> Dict[str, Any]:
        """加载模型配置文件"""
        if self._config is None:
            config_path = self.config_dir / "model_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config

    def load_params(self) -> Dict[str, float]:
        """加载默认参数"""
        if self._params is None:
            params_path = self.config_dir / "default_params.json"
            with open(params_path, 'r', encoding='utf-8') as f:
                self._params = json.load(f)
        return self._params

    def load_param_ranges(self) -> Dict[str, Dict[str, float]]:
        """加载参数范围"""
        if self._ranges is None:
            ranges_path = self.config_dir / "param_ranges.json"
            with open(ranges_path, 'r', encoding='utf-8') as f:
                self._ranges = json.load(f)
        return self._ranges

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        config = self.load_model_config()
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


def load_params(param_file: Optional[str] = None) -> Dict[str, float]:
    """便捷函数：加载模型参数"""
    loader = ConfigLoader()
    params = loader.load_params()
    
    if param_file:
        with open(param_file, 'r', encoding='utf-8') as f:
            user_params = json.load(f)
            params.update(user_params)
    
    return params


def validate_params(params: Dict[str, float], 
                   ranges: Optional[Dict[str, Dict[str, float]]] = None) -> bool:
    """验证参数是否在有效范围内"""
    if ranges is None:
        loader = ConfigLoader()
        ranges = loader.load_param_ranges()
    
    for name, value in params.items():
        if name in ranges:
            r = ranges[name]
            if not (r['min'] <= value <= r['max']):
                raise ValueError(
                    f"Parameter {name}={value} out of range "
                    f"[{r['min']}, {r['max']}]"
                )
    return True


class ParamMapper:
    """流溪河模型参数映射器 - 与原模型兼容"""

    PARAMS_11 = ['WMM', 'K', 'B', 'IM', 'KE', 'XE', 'KG', 'CG', 'KKG', 'C', 'CS']

    @staticmethod
    def to_physical_units(params: Dict[str, float]) -> Dict[str, Any]:
        """转换为物理单位参数"""
        return {
            'WMM': params['WMM'] * 1000,  # mm -> m
            'K': params['K'],
            'B': params['B'],
            'IM': params['IM'],
            'KE': params['KE'],
            'XE': params['XE'],
            'KG': params['KG'],
            'CG': params['CG'],
            'KKG': params['KKG'],
            'C': params['C'],
            'CS': params['CS'],
        }

    @staticmethod
    def from_original_format(param_array: np.ndarray) -> Dict[str, float]:
        """从原模型参数格式转换"""
        if len(param_array) != 11:
            raise ValueError("原模型应有11个参数")
        
        return dict(zip(ParamMapper.PARAMS_11, param_array.tolist()))
