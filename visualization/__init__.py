"""
可视化模块

提供水文模型结果的绘图功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Tuple, List, Union
import xarray as xr


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def create_colormap():
    """创建水文模型专用色带"""
    colors = ['#FFFFFF', '#A6CEE3', '#66C2A5', '#2CA02C', '#006400',
              '#FFD700', '#FF8C00', '#FF4500', '#8B0000', '#4B0082']
    return LinearSegmentedColormap.from_list('hydro', colors, N=256)


def plot_dem(dem: np.ndarray, 
             header: Optional[dict] = None,
             figsize: Tuple[int, int] = (10, 8),
             save_path: Optional[str] = None,
             show: bool = True):
    """
    绘制DEM高程图

    Args:
        dem: 高程数据
        header: ASC头信息
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    extent = [0, dem.shape[1], 0, dem.shape[0]]
    if header:
        extent = [
            header.get('xllcorner', 0),
            header.get('xllcorner', 0) + dem.shape[1] * header.get('cellsize', 1),
            header.get('yllcorner', 0),
            header.get('yllcorner', 0) + dem.shape[0] * header.get('cellsize', 1)
        ]

    im = ax.imshow(dem, cmap='terrain', extent=extent, origin='lower')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Digital Elevation Model')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_flow_direction(flow_dir: np.ndarray,
                        dem: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None,
                        show: bool = True):
    """
    绘制流向图

    Args:
        flow_dir: 流向数据
        dem: 高程数据（用于背景）
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    if dem is not None:
        ax.imshow(dem, cmap='gray_r', alpha=0.3, origin='lower')

    D8_DELTAS = np.array([[1, 1, 0, -1, -1, -1, 0, 1],
                           [0, 1, 1, 1, 0, -1, -1, -1]])

    step = 5
    y_coords, x_coords = np.where((flow_dir > 0) & (flow_dir <= 8))
    y_valid = y_coords[::step]
    x_valid = x_coords[::step]

    for y, x in zip(y_valid, x_valid):
        d = int(flow_dir[y, x]) - 1
        if 0 <= d < 8:
            dx = D8_DELTAS[1, d] * 0.8
            dy = D8_DELTAS[0, d] * 0.8
            ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2,
                    fc='blue', ec='blue', linewidth=0.5)

    ax.set_xlim(-0.5, flow_dir.shape[1] - 0.5)
    ax.set_ylim(-0.5, flow_dir.shape[0] - 0.5)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_title('D8 Flow Direction')
    ax.set_aspect('equal')
    ax.invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_accumulation(acc: np.ndarray,
                      header: Optional[dict] = None,
                      log_scale: bool = True,
                      figsize: Tuple[int, int] = (10, 8),
                      save_path: Optional[str] = None,
                      show: bool = True):
    """
    绘制累积流图

    Args:
        acc: 累积流数据
        header: ASC头信息
        log_scale: 是否使用对数刻度
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    acc_valid = np.where(acc > 0, acc, np.nan)

    if log_scale:
        acc_log = np.log10(acc_valid + 1)
        vmin, vmax = np.nanmin(acc_log), np.nanmax(acc_log)
        im = ax.imshow(acc_log, cmap='Blues', origin='lower',
                      vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(acc_valid, cmap='Blues', origin='lower')

    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_title('Flow Accumulation' + (' (log scale)' if log_scale else ''))
    plt.colorbar(im, ax=ax, label='Accumulation (cells)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_unit_classification(unit_type: np.ndarray,
                            dem: Optional[np.ndarray] = None,
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None,
                            show: bool = True):
    """
    绘制网格单元分类图

    Args:
        unit_type: 单元类型 (0=边坡, 1=河道, 2=水库)
        dem: 高程数据（用于背景）
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    if dem is not None:
        ax.imshow(dem, cmap='gray_r', alpha=0.2, origin='lower')

    colors = ['#90EE90', '#4169E1', '#FF6347']
    labels = ['Upslope (US)', 'River (RN)', 'Reservoir (RV)']

    cmap = mcolors.ListedColormap(colors)
    im = ax.imshow(unit_type, cmap=cmap, vmin=0, vmax=2, origin='lower')

    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_title('Grid Unit Classification')

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=l)
                      for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_hydrograph(simulated: np.ndarray,
                    observed: Optional[np.ndarray] = None,
                    time_index: Optional[np.ndarray] = None,
                    dt_minutes: int = 60,
                    title: str = 'Hydrograph',
                    labels: Tuple[str, str] = ('Simulated', 'Observed'),
                    figsize: Tuple[int, int] = (12, 6),
                    save_path: Optional[str] = None,
                    show: bool = True):
    """
    绘制流量过程线

    Args:
        simulated: 模拟流量
        observed: 实测流量（可选）
        time_index: 时间索引
        dt_minutes: 时间步长（分钟）
        title: 图形标题
        labels: 图例标签
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    if time_index is not None:
        t = time_index
    else:
        t = np.arange(len(simulated)) * dt_minutes / 60

    ax.plot(t, simulated, 'b-', linewidth=1.5, label=labels[0])

    if observed is not None:
        ax.plot(t, observed, 'r--', linewidth=1.5, label=labels[1])

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_spatial_result(data: np.ndarray,
                        header: Optional[dict] = None,
                        cmap: str = 'viridis',
                        title: str = 'Spatial Result',
                        label: str = '',
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None,
                        show: bool = True):
    """
    绘制空间分布结果

    Args:
        data: 空间数据
        header: ASC头信息
        cmap: 色带
        title: 标题
        label: 颜色条标签
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    extent = None
    if header:
        extent = [
            header.get('xllcorner', 0),
            header.get('xllcorner', 0) + data.shape[1] * header.get('cellsize', 1),
            header.get('yllcorner', 0),
            header.get('yllcorner', 0) + data.shape[0] * header.get('cellsize', 1)
        ]

    im = ax.imshow(data, cmap=cmap, extent=extent, origin='lower')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    if label:
        plt.colorbar(im, ax=ax, label=label)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_comparison(simulated: np.ndarray,
                    observed: np.ndarray,
                    time_index: Optional[np.ndarray] = None,
                    dt_minutes: int = 60,
                    title: str = 'Comparison',
                    figsize: Tuple[int, int] = (12, 8),
                    save_path: Optional[str] = None,
                    show: bool = True):
    """
    绘制模拟与观测对比图

    Args:
        simulated: 模拟值
        observed: 观测值
        time_index: 时间索引
        dt_minutes: 时间步长
        title: 标题
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    if time_index is not None:
        t = time_index
    else:
        t = np.arange(len(simulated)) * dt_minutes / 60

    axes[0].plot(t, simulated, 'b-', linewidth=1.5, label='Simulated')
    axes[0].plot(t, observed, 'r--', linewidth=1.5, label='Observed')
    axes[0].set_ylabel('Discharge (m³/s)')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = simulated - observed
    axes[1].bar(t, residuals, width=dt_minutes/60*0.8, color='gray', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Residual (m³/s)')
    axes[1].set_title('Residuals')
    axes[1].grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_scatter_compare(simulated: np.ndarray,
                         observed: np.ndarray,
                         title: str = 'Scatter Plot',
                         figsize: Tuple[int, int] = (8, 8),
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    绘制散点对比图

    Args:
        simulated: 模拟值
        observed: 观测值
        title: 标题
        figsize: 图形大小
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(observed, simulated, alpha=0.5, s=20)

    vmin = min(observed.min(), simulated.min())
    vmax = max(observed.max(), simulated.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='1:1 Line')

    ax.set_xlabel('Observed (m³/s)')
    ax.set_ylabel('Simulated (m³/s)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
