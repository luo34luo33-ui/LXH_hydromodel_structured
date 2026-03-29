import numpy as np
import cupy as cp
from typing import Union


def tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Thomas算法求解三对角方程组 (Tri-Diagonal Matrix Algorithm)

    用于扩散波方程的隐式求解

    Args:
        a: 下对角线元素 (n-1,)
        b: 主对角线元素 (n,)
        c: 上对角线元素 (n-1,)
        d: 右端向量 (n,)

    Returns:
        x: 求解向量 (n,)

    Example:
        >>> a = np.array([1, 1])
        >>> b = np.array([2, 2, 2])
        >>> c = np.array([1, 1])
        >>> d = np.array([3, 4, 3])
        >>> x = tdma(a, b, c, d)
    """
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom
    
    d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / \
                     (b[n - 1] - a[n - 2] * c_prime[n - 2])
    
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x


def thomas_algorithm(
    a: Union[np.ndarray, cp.ndarray],
    b: Union[np.ndarray, cp.ndarray],
    c: Union[np.ndarray, cp.ndarray],
    d: Union[np.ndarray, cp.ndarray],
    use_gpu: bool = False
) -> Union[np.ndarray, cp.ndarray]:
    """
    Thomas算法的向量化版本（支持GPU）

    用于批量三对角系统的并行求解

    Args:
        a, b, c, d: 系数矩阵和右端向量
        use_gpu: 是否使用GPU

    Returns:
        x: 求解向量
    """
    if use_gpu:
        xp = cp
        a, b, c, d = map(lambda arr: xp.asarray(arr), [a, b, c, d])
    else:
        xp = np

    batch_size = d.shape[0] if d.ndim > 1 else 1
    n = d.shape[-1] if d.ndim > 1 else len(d)
    
    c_prime = xp.zeros_like(c)
    d_prime = xp.zeros_like(d)
    
    c_prime[..., 0] = c[..., 0] / b[..., 0]
    d_prime[..., 0] = d[..., 0] / b[..., 0]
    
    for i in range(1, n):
        denom = b[..., i] - a[..., i-1] * c_prime[..., i-1]
        c_prime[..., i] = c[..., i] / denom
        d_prime[..., i] = (d[..., i] - a[..., i-1] * d_prime[..., i-1]) / denom
    
    x = xp.zeros_like(d)
    x[..., n-1] = d_prime[..., n-1]
    for i in range(n - 2, -1, -1):
        x[..., i] = d_prime[..., i] - c_prime[..., i] * x[..., i + 1]
    
    return x


def banded_matrix_solve(
    ab: np.ndarray,
    nr: int,
    nb: int,
    b: np.ndarray
) -> np.ndarray:
    """
    求解带状矩阵方程 (LAPACK接口)

    Args:
        ab: 带状矩阵 (2*nr-1, nb)
        nr: 半带宽
        nb: 方程数
        b: 右端向量

    Returns:
        x: 解向量
    """
    from scipy.linalg import solve_banded
    
    ab_lapack = np.zeros((2 * nr - 1, nb))
    ab_lapack[nr-1:, :nr] = ab[:nr, :].T
    ab_lapack[:nr-1, nr:] = ab[nr:, nr:].T
    
    return solve_banded((nr - 1, nr - 1), ab_lapack, b)
