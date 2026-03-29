import numpy as np
from typing import Callable, Optional, Union
import cupy as cp


def newton_raphson(
    f: Callable[[np.ndarray], np.ndarray],
    df: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    use_gpu: bool = False
) -> np.ndarray:
    """
    通用牛顿迭代法求解器（支持张量化计算）

    适用于边坡汇流和河道汇流中求解圣维南方程离散后的非线性方程

    Args:
        f: 方程函数 f(x) = 0
        df: 导数函数 df/dx
        x0: 初始猜测值（可张量）
        tol: 收敛容差
        max_iter: 最大迭代次数
        use_gpu: 是否使用GPU加速

    Returns:
        x: 收敛解

    Example:
        >>> # 求解 Q^2 + a*Q - b = 0
        >>> f = lambda Q: Q**2 + a*Q - b
        >>> df = lambda Q: 2*Q + a
        >>> Q = newton_raphson(f, df, x0)
    """
    xp = cp if use_gpu else np
    x = xp.array(x0, dtype=xp.float64)

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        dfx = xp.where(xp.abs(dfx) < 1e-12, 1e-12, dfx)
        
        delta = fx / dfx
        x = x - delta

        if xp.all(xp.abs(delta) < tol):
            break

    return cp.asnumpy(x) if use_gpu else x


def newton_raphson_vectorized(
    equations: np.ndarray,
    jacobians: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    use_gpu: bool = False
) -> np.ndarray:
    """
    向量化牛顿迭代（同时求解多个方程）

    用于批量网格的并行求解

    Args:
        equations: 函数值矩阵 (n_grids, n_vars)
        jacobians: 雅可比矩阵 (n_grids, n_vars, n_vars)
        x0: 初始值
        tol: 收敛容差
        max_iter: 最大迭代次数
        use_gpu: 是否使用GPU加速

    Returns:
        x: 收敛解矩阵
    """
    xp = cp if use_gpu else np
    x = xp.array(x0, dtype=xp.float64)

    for iteration in range(max_iter):
        delta = xp.linalg.solve(jacobians, equations)
        x = x - delta

        equations_new = equations + jacobians @ delta
        if xp.all(xp.abs(equations_new) < tol):
            break

    return cp.asnumpy(x) if use_gpu else x


def hybrid_solver(
    f: Callable,
    df: Callable,
    x0: float,
    bounds: tuple = (1e-10, 1e10),
    tol: float = 1e-8,
    max_iter: int = 50
) -> float:
    """
    混合求解器：牛顿法 + 二分法（保证收敛）

    当牛顿法发散时自动切换到二分法

    Args:
        f: 目标函数
        df: 导数函数
        x0: 初始猜测
        bounds: 搜索边界
        tol: 收敛精度
        max_iter: 最大迭代次数

    Returns:
        x: 收敛解
    """
    lo, hi = bounds
    
    for _ in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        
        if abs(dfx) < 1e-12:
            x0 = (lo + hi) / 2
            continue
            
        x_new = x0 - fx / dfx
        
        if x_new < lo or x_new > hi or abs(dfx) < 1e-6:
            x_new = (lo + hi) / 2
            if f(lo) * f(hi) > 0:
                break
            mid = (lo + hi) / 2
            if f(mid) * f(lo) < 0:
                hi = mid
            else:
                lo = mid
            x0 = (lo + hi) / 2
        else:
            x0 = x_new
            
        if abs(f(x0)) < tol:
            break
            
    return float(x0)
