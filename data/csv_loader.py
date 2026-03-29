"""
CSV格式雨量数据加载器
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
from datetime import datetime


class CSVRainfallLoader:
    """
    CSV格式雨量数据加载器

    支持多种CSV格式：
    1. 逐站点逐时刻格式
    2. 逐时刻逐站点矩阵格式
    3. 站号+降雨量格式
    """

    def __init__(self, cell_size: float = 90.0, dt_minutes: int = 60):
        """
        Args:
            cell_size: 栅格大小（米）
            dt_minutes: 时间步长（分钟）
        """
        self.cell_size = cell_size
        self.dt_minutes = dt_minutes

    def load_from_stations(
        self,
        csv_path: str,
        station_coords: Dict[int, Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        method: str = 'IDW',
        power: float = 2.0,
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
        """
        从站点数据插值到网格

        Args:
            csv_path: CSV文件路径
            station_coords: 站点坐标 {站号: (x, y)}
            bbox: 边界 (xmin, ymin, xmax, ymax)
            method: 插值方法 ('IDW', 'Kriging', 'Nearest')
            power: IDW幂指数
            grid_shape: 网格形状 (ny, nx)

        Returns:
            (rainfall_grid, stations_df)
        """
        df = pd.read_csv(csv_path)

        xmin, ymin, xmax, ymax = bbox

        if grid_shape is None:
            nx = int((xmax - xmin) / self.cell_size)
            ny = int((ymax - ymin) / self.cell_size)
        else:
            ny, nx = grid_shape

        rainfall_grid = np.zeros((len(df), ny, nx))

        x_coords = np.linspace(xmin, xmax, nx)
        y_coords = np.linspace(ymin, ymax, ny)

        if method == 'IDW':
            rainfall_grid = self._idw_interpolation(
                df, station_coords, x_coords, y_coords, power
            )
        elif method == 'Nearest':
            rainfall_grid = self._nearest_interpolation(
                df, station_coords, x_coords, y_coords
            )
        else:
            rainfall_grid = self._idw_interpolation(
                df, station_coords, x_coords, y_coords, power
            )

        time_index = pd.to_datetime(df.iloc[:, 0]) if len(df.columns) > 1 else pd.RangeIndex(len(df))

        rainfall_da = xr.DataArray(
            rainfall_grid.astype(np.float32),
            dims=['time', 'y', 'x'],
            coords={
                'time': time_index,
                'y': y_coords,
                'x': x_coords
            },
            attrs={
                'units': 'mm',
                'dt_minutes': self.dt_minutes,
                'method': method
            }
        )

        return rainfall_da, df

    def _idw_interpolation(
        self,
        df: pd.DataFrame,
        station_coords: Dict[int, Tuple[float, float]],
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        power: float = 2.0
    ) -> np.ndarray:
        """反距离加权插值"""
        n_times = len(df)
        ny, nx = len(y_coords), len(x_coords)
        rainfall = np.zeros((n_times, ny, nx))

        time_col = df.columns[0]
        value_cols = [c for c in df.columns if c != time_col]

        for t in range(n_times):
            values = {}
            for i, col in enumerate(value_cols):
                station_id = int(col) if col.isdigit() else i + 1
                if station_id in station_coords:
                    values[station_id] = df[col].iloc[t]

            if not values:
                continue

            for iy in range(ny):
                for ix in range(nx):
                    x, y = x_coords[ix], y_coords[iy]
                    weight_sum = 0
                    value_sum = 0

                    for station_id, val in values.items():
                        sx, sy = station_coords[station_id]
                        dist = np.sqrt((x - sx)**2 + (y - sy)**2)

                        if dist < 0.1:
                            rainfall[t, iy, ix] = val
                            break

                        weight = 1.0 / (dist ** power)
                        weight_sum += weight
                        value_sum += weight * val
                    else:
                        if weight_sum > 0:
                            rainfall[t, iy, ix] = value_sum / weight_sum

        return rainfall

    def _nearest_interpolation(
        self,
        df: pd.DataFrame,
        station_coords: Dict[int, Tuple[float, float]],
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> np.ndarray:
        """最近邻插值"""
        n_times = len(df)
        ny, nx = len(y_coords), len(x_coords)
        rainfall = np.zeros((n_times, ny, nx))

        time_col = df.columns[0]
        value_cols = [c for c in df.columns if c != time_col]

        for t in range(n_times):
            values = {}
            for i, col in enumerate(value_cols):
                station_id = int(col) if col.isdigit() else i + 1
                if station_id in station_coords:
                    values[station_id] = df[col].iloc[t]

            if not values:
                continue

            for iy in range(ny):
                for ix in range(nx):
                    x, y = x_coords[ix], y_coords[iy]

                    min_dist = float('inf')
                    nearest_val = 0

                    for station_id, val in values.items():
                        sx, sy = station_coords[station_id]
                        dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_val = val

                    rainfall[t, iy, ix] = nearest_val

        return rainfall

    def load_grid_csv(
        self,
        csv_path: str,
        time_format: str = '%Y-%m-%d %H:%M',
        has_header: bool = True
    ) -> xr.DataArray:
        """
        加载网格化CSV数据（时刻为行，网格位置为列）

        Args:
            csv_path: CSV文件路径
            time_format: 时间格式
            has_header: 是否有表头

        Returns:
            xarray.DataArray
        """
        df = pd.read_csv(csv_path)

        if has_header and df.columns[0] == 'time':
            df.set_index('time', inplace=True)
            time_index = pd.to_datetime(df.index, format=time_format)
        else:
            time_index = pd.RangeIndex(len(df))

        data = df.values.astype(np.float32)

        ny, nx = self._infer_grid_shape(data.shape[1])
        n_times = len(df)

        rainfall = data.reshape(n_times, ny, nx)

        return xr.DataArray(
            rainfall,
            dims=['time', 'y', 'x'],
            coords={'time': time_index},
            attrs={'units': 'mm', 'dt_minutes': self.dt_minutes}
        )

    def _infer_grid_shape(self, n_cells: int) -> Tuple[int, int]:
        """推断网格形状"""
        ny = int(np.sqrt(n_cells))
        nx = n_cells // ny
        if ny * nx == n_cells:
            return ny, nx

        for ny in range(int(np.sqrt(n_cells)), 0, -1):
            nx = n_cells // ny
            if ny * nx == n_cells:
                return ny, nx

        return 1, n_cells

    def load_station_csv(
        self,
        csv_path: str,
        station_col: str = 'station',
        rainfall_col: str = 'rainfall',
        time_col: str = 'datetime',
        dt_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        加载站点CSV数据

        Args:
            csv_path: CSV文件路径
            station_col: 站号列名
            rainfall_col: 雨量列名
            time_col: 时间列名
            dt_minutes: 时间步长（分钟）

        Returns:
            DataFrame: (time, station, rainfall) 长格式
        """
        df = pd.read_csv(csv_path)

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])

        return df

    def to_grid(
        self,
        station_data: pd.DataFrame,
        station_coords: Dict[int, Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        method: str = 'IDW'
    ) -> xr.DataArray:
        """
        将站点数据转换为网格数据

        Args:
            station_data: 站点数据 (长格式)
            station_coords: 站点坐标
            bbox: 边界
            method: 插值方法

        Returns:
            网格化降雨数据
        """
        time_index = station_data['datetime'].unique()
        xmin, ymin, xmax, ymax = bbox
        nx = int((xmax - xmin) / self.cell_size)
        ny = int((ymax - ymin) / self.cell_size)

        rainfall = np.zeros((len(time_index), ny, nx))

        for t, t_val in enumerate(time_index):
            t_data = station_data[station_data['datetime'] == t_val]

            values = {}
            for _, row in t_data.iterrows():
                sid = row.get('station', 0)
                if sid in station_coords:
                    values[sid] = row.get('rainfall', 0)

            x_coords = np.linspace(xmin, xmax, nx)
            y_coords = np.linspace(ymin, ymax, ny)

            for iy in range(ny):
                for ix in range(nx):
                    x, y = x_coords[ix], y_coords[iy]
                    val = self._interpolate_point(x, y, values, station_coords, method)
                    rainfall[t, iy, ix] = val

        return xr.DataArray(
            rainfall.astype(np.float32),
            dims=['time', 'y', 'x'],
            coords={
                'time': time_index,
                'y': y_coords,
                'x': x_coords
            }
        )

    def _interpolate_point(
        self,
        x: float,
        y: float,
        values: Dict,
        station_coords: Dict,
        method: str = 'IDW'
    ) -> float:
        """单点插值"""
        if not values:
            return 0.0

        if method == 'IDW':
            weight_sum = 0
            value_sum = 0
            power = 2.0

            for sid, val in values.items():
                sx, sy = station_coords[sid]
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < 0.1:
                    return val
                weight = 1.0 / (dist ** power)
                weight_sum += weight
                value_sum += weight * val

            return value_sum / weight_sum if weight_sum > 0 else 0.0

        else:
            min_dist = float('inf')
            nearest = 0
            for sid, val in values.items():
                sx, sy = station_coords[sid]
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = val
            return nearest


def load_csv_rainfall(
    csv_path: str,
    format: str = 'auto',
    **kwargs
) -> Union[xr.DataArray, pd.DataFrame]:
    """
    便捷函数：加载CSV雨量数据

    Args:
        csv_path: CSV文件路径
        format: 数据格式 ('station', 'grid', 'auto')
        **kwargs: 其他参数

    Returns:
        降雨数据
    """
    loader = CSVRainfallLoader(**kwargs)

    if format == 'auto':
        df = pd.read_csv(csv_path, nrows=5)

        if df.shape[1] <= 3 and any(c in ['station', 'time', 'datetime'] for c in df.columns):
            format = 'station'
        else:
            format = 'grid'

    if format == 'station':
        if 'station_coords' in kwargs:
            return loader.load_from_stations(csv_path, **kwargs)
        else:
            return loader.load_station_csv(csv_path, **kwargs)
    else:
        return loader.load_grid_csv(csv_path, **kwargs)
