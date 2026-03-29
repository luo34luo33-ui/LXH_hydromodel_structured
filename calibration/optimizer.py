import numpy as np
from typing import Dict, Callable, Optional, List
from scipy.optimize import differential_evolution, minimize
import json


class Optimizer:
    """
    模型参数优化器

    支持多种优化算法：SCE-UA, DE, LM等
    """

    def __init__(
        self,
        objective_func: Callable,
        param_ranges: Dict[str, Dict[str, float]],
        bounds: Optional[List[tuple]] = None,
        param_names: Optional[List[str]] = None
    ):
        """
        Args:
            objective_func: 目标函数，接受参数字典返回目标值（越小越好）
            param_ranges: 参数范围
            bounds: 优化边界
            param_names: 参数名称列表
        """
        self.objective_func = objective_func
        self.param_ranges = param_ranges
        self.bounds = bounds
        self.param_names = param_names or list(param_ranges.keys())

        if bounds is None:
            self.bounds = [
                (param_ranges[p]['min'], param_ranges[p]['max'])
                for p in self.param_names
            ]

        self.best_params = None
        self.best_value = np.inf
        self.history = []

    def optimize_de(
        self,
        max_iter: int = 1000,
        pop_size: int = 20,
        tol: float = 1e-6,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        差分进化算法 (Differential Evolution)

        Args:
            max_iter: 最大迭代次数
            pop_size: 种群大小
            tol: 收敛容差
            seed: 随机种子

        Returns:
            最优参数
        """
        def wrapper(x):
            params = dict(zip(self.param_names, x))
            value = self.objective_func(params)
            self.history.append({'params': params, 'value': value})

            if value < self.best_value:
                self.best_value = value
                self.best_params = params.copy()

            return value

        result = differential_evolution(
            wrapper,
            self.bounds,
            maxiter=max_iter,
            popsize=pop_size,
            tol=tol,
            seed=seed,
            polish=True,
            disp=True
        )

        self.best_params = dict(zip(self.param_names, result.x))
        self.best_value = result.fun

        return self.best_params

    def optimize_sce(
        self,
        max_iter: int = 1000,
        n_complexes: int = 5,
        complex_size: int = 10
    ) -> Dict[str, float]:
        """
        SCE-UA算法 (Shuffled Complex Evolution)

        Args:
            max_iter: 最大迭代次数
            n_complexes: 复合形数量
            complex_size: 每个复合形大小

        Returns:
            最优参数
        """
        n_params = len(self.param_names)

        population = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (complex_size * n_complexes, n_params)
        )

        for _ in range(max_iter):
            populations = np.array_split(population, n_complexes)

            for complex_pop in populations:
                complex_pop = self._cce(complex_pop)

            population = np.vstack(populations)

            np.random.shuffle(population)

            params = dict(zip(self.param_names, population[0]))
            value = self.objective_func(params)
            self.history.append({'params': params, 'value': value})

            if value < self.best_value:
                self.best_value = value
                self.best_params = params.copy()

        return self.best_params

    def _cce(self, complex_pop: np.ndarray) -> np.ndarray:
        """竞争复合形进化 (CCE)"""
        n_params = len(self.param_names)

        for _ in range(n_params):
            if len(complex_pop) < 2:
                break

            sorted_pop = complex_pop[np.argsort([
                self.objective_func(dict(zip(self.param_names, p)))
                for p in complex_pop
            ])]

            centroid = np.mean(sorted_pop[:-1], axis=0)

            worst = sorted_pop[-1]

            reflected = 2 * centroid - worst
            reflected = np.clip(reflected,
                             [b[0] for b in self.bounds],
                             [b[1] for b in self.bounds])

            f_reflected = self.objective_func(dict(zip(self.param_names, reflected)))
            f_worst = self.objective_func(dict(zip(self.param_names, worst)))

            if f_reflected < f_worst:
                complex_pop[-1] = reflected
            else:
                contracted = (centroid + worst) / 2
                f_contracted = self.objective_func(dict(zip(self.param_names, contracted)))
                if f_contracted < f_worst:
                    complex_pop[-1] = contracted

        return complex_pop

    def optimize_lm(
        self,
        initial_params: Dict[str, float],
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        Levenberg-Marquardt算法

        适用于目标函数可导的情况

        Args:
            initial_params: 初始参数
            max_iter: 最大迭代次数

        Returns:
            最优参数
        """
        x0 = np.array([initial_params[p] for p in self.param_names])

        def wrapper(x):
            params = dict(zip(self.param_names, x))
            return self.objective_func(params)

        result = minimize(
            wrapper,
            x0,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': max_iter}
        )

        self.best_params = dict(zip(self.param_names, result.x))
        self.best_value = result.fun

        return self.best_params

    def save_history(self, filepath: str):
        """保存优化历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, default=str)

    def load_history(self, filepath: str):
        """加载优化历史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
