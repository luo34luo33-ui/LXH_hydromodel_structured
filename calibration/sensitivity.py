import numpy as np
from typing import Dict, List, Tuple, Callable
import json
from pathlib import Path


class SensitivityAnalyzer:
    """
    参数敏感性分析

    使用Bohensizein方法识别高度敏感/敏感/不敏感参数
    """

    PARAM_NAMES = ['WMM', 'K', 'B', 'IM', 'KE', 'XE', 'KG', 'CG', 'KKG', 'C', 'CS']

    def __init__(
        self,
        param_ranges: Dict[str, Dict[str, float]],
        model_runner: Callable
    ):
        """
        Args:
            param_ranges: 参数范围字典
            model_runner: 模型运行函数，接收参数字典返回目标函数值
        """
        self.param_ranges = param_ranges
        self.model_runner = model_runner
        self.results = {}

    def first_order_sensitivity(
        self,
        base_params: Dict[str, float],
        obs: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        一阶敏感性分析

        使用Monte Carlo采样估计各参数的一阶敏感性指数

        Args:
            base_params: 基准参数
            obs: 观测值
            n_samples: 采样次数

        Returns:
            参数字典，值为敏感性指数
        """
        base_value = self.model_runner(base_params)

        variances = {}
        for param_name in self.PARAM_NAMES:
            if param_name not in base_params:
                continue

            param_range = self.param_ranges.get(param_name, {'min': 0, 'max': 1})
            min_val = param_range['min']
            max_val = param_range['max']

            values = np.linspace(min_val, max_val, n_samples)
            outputs = []

            for val in values:
                test_params = base_params.copy()
                test_params[param_name] = val
                output = self.model_runner(test_params)
                outputs.append(output)

            outputs = np.array(outputs)
            variances[param_name] = np.var(outputs)

        total_variance = sum(variances.values())
        if total_variance == 0:
            return {k: 0 for k in variances}

        sensitivity = {k: v / total_variance for k, v in variances.items()}
        return sensitivity

    def regional_sensitivity_analysis(
        self,
        base_params: Dict[str, float],
        obs: np.ndarray,
        threshold: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """
        区域敏感性分析（RSA）

        识别高度敏感、敏感和不敏感参数

        Args:
            base_params: 基准参数
            obs: 观测值
            threshold: 目标函数阈值

        Returns:
            敏感性分类结果
        """
        sensitivity = self.first_order_sensitivity(base_params, obs)

        sorted_params = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)

        classification = {
            'highly_sensitive': [],
            'sensitive': [],
            'insensitive': []
        }

        total = sum(s.values())
        if total > 0:
            cumulative = 0
            for name, value in sorted_params:
                cumulative += value / total
                if cumulative < 0.5:
                    classification['highly_sensitive'].append(name)
                elif cumulative < 0.8:
                    classification['sensitive'].append(name)
                else:
                    classification['insensitive'].append(name)
        else:
            classification['insensitive'] = list(sensitivity.keys())

        return classification

    def sobol_indices(
        self,
        base_params: Dict[str, float],
        n_samples: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Sobol敏感性指数

        计算一阶和总阶敏感性指数

        Args:
            base_params: 基准参数
            n_samples: 采样次数

        Returns:
            参数字典，值为(一阶指数, 总阶指数)元组
        """
        from scipy.stats import qmc

        n_params = len(self.PARAM_NAMES)
        sampler = qmc.Sobol(d=n_params, scramble=False)
        sample = sampler.random_base2(m=int(np.log2(n_samples)))

        param_names = [p for p in self.PARAM_NAMES if p in base_params]
        bounds = []
        for p in param_names:
            r = self.param_ranges.get(p, {'min': 0, 'max': 1})
            bounds.append([r['min'], r['max']])

        bounds = np.array(bounds)
        scaled_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

        outputs = []
        for i in range(n_samples):
            params = base_params.copy()
            for j, p in enumerate(param_names):
                params[p] = scaled_samples[i, j]
            outputs.append(self.model_runner(params))

        outputs = np.array(outputs)

        first_order = {}
        total_order = {}

        for j, p in enumerate(param_names):
            A = outputs[::2]
            B = outputs[1::2]

            var_A = np.var(A)
            var_B = np.var(B)
            var_total = np.var(outputs)

            if var_total < 1e-10:
                first_order[p] = 0.0
                total_order[p] = 0.0
                continue

            first_order[p] = 1 - np.mean((A - B) ** 2) / (2 * var_total)
            total_order[p] = np.mean(B * (outputs[::2] - outputs[1::2])) / (2 * var_total)

        return {p: (first_order.get(p, 0), total_order.get(p, 0)) for p in param_names}

    def save_results(self, filepath: str):
        """保存敏感性分析结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)


def load_sensitivity_results(filepath: str) -> Dict:
    """加载敏感性分析结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
