import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, Optional, Tuple, List
from datetime import datetime

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


class ForcingLoader:
    """气象强迫数据加载器"""

    def __init__(
        self,
        rainfall_dir: str,
        dt: int = 3600,
        use_gpu: bool = False
    ):
        """
        Args:
            rainfall_dir: 降雨数据目录
            dt: 时间步长（秒）
            use_gpu: 是否使用GPU
        """
        self.rainfall_dir = Path(rainfall_dir)
        self.dt = dt
        self.use_gpu = use_gpu

    def load_rainfall_txt(
        self,
        path: str,
        time_col: int = 0,
        value_col: int = 1,
        date_format: str = '%Y-%m-%d %H:%M:%S'
    ) -> pd.DataFrame:
        """
        读取文本格式降雨数据

        Args:
            path: 文件路径
            time_col: 时间列索引
            value_col: 数值列索引
            date_format: 日期格式

        Returns:
            DataFrame with datetime index and rainfall values
        """
        df = pd.read_csv(path)
        
        if isinstance(time_col, int):
            df.iloc[:, time_col] = pd.to_datetime(
                df.iloc[:, time_col], format=date_format
            )
            df.set_index(df.columns[time_col], inplace=True)
        
        df['P'] = df.iloc[:, value_col].astype(float)
        return df[['P']]

    def load_rainfall_grid(
        self,
        grid_dir: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        filename_pattern: str = 'rain_{date}.tif'
    ) -> xr.DataArray:
        """
        加载网格降雨数据（逐时刻GeoTIFF）

        Args:
            grid_dir: 栅格数据目录
            grid_dir: 数据目录
            start_date: 开始日期
            end_date: 结束日期
            filename_pattern: 文件名模式，{date}替换为日期

        Returns:
            xarray.DataArray with time dimension
        """
        from .raster_io import RasterIO
        
        grid_dir = Path(grid_dir)
        dates = pd.date_range(start_date, end_date, freq=f'{self.dt}s')
        
        io = RasterIO()
        data_list = []
        
        for date in dates:
            date_str = date.strftime('%Y%m%d%H%M')
            filename = filename_pattern.format(date=date_str)
            filepath = grid_dir / filename
            
            if filepath.exists():
                data, meta = io.read_tiff(str(filepath))
                data_list.append(data)
            else:
                if data_list:
                    data_list.append(np.zeros_like(data_list[-1]))
                else:
                    data_list.append(np.zeros((100, 100)))
        
        stacked = np.stack(data_list, axis=0)
        
        return xr.DataArray(
            stacked,
            dims=['time', 'y', 'x'],
            coords={'time': dates}
        )

    def load_station_rainfall(
        self,
        stations_file: str,
        rainfall_file: str,
        method: str = 'IDW',
        power: float = 2.0
    ) -> xr.DataArray:
        """
        从雨量站数据插值到网格

        Args:
            stations_file: 站点坐标文件 (lon, lat, x, y)
            rainfall_file: 降雨时序文件
            method: 插值方法 (IDW/Kriging/Thiessen)
            power: IDW幂指数

        Returns:
            网格化降雨数据
        """
        stations = pd.read_csv(stations_file)
        rainfall = pd.read_csv(rainfall_file, index_col=0, parse_dates=True)
        
        if method == 'IDW':
            return self._idw_interpolation(stations, rainfall, power)
        elif method == 'Thiessen':
            return self._thiessen_interpolation(stations, rainfall)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _idw_interpolation(
        self,
        stations: pd.DataFrame,
        rainfall: pd.DataFrame,
        power: float
    ) -> xr.DataArray:
        """反距离加权插值"""
        pass

    def _thiessen_interpolation(
        self,
        stations: pd.DataFrame,
        rainfall: pd.DataFrame
    ) -> xr.DataArray:
        """泰森多边形插值"""
        pass


class RadarRainfallLoader:
    """雷达降雨数据加载器"""

    def __init__(self, radar_dir: str, use_gpu: bool = False):
        self.radar_dir = Path(radar_dir)
        self.use_gpu = use_gpu

    def load_radar_grid(
        self,
        path: str,
        calibration_factor: float = 1.0
    ) -> np.ndarray:
        """
        加载雷达降雨数据并转换为地面降雨

        Args:
            path: 雷达数据文件路径
            calibration_factor: 校准因子 (Z-R关系)

        Returns:
            降雨强度 (mm/h)
        """
        raw_data = self._read_radar_format(path)
        
        Z = np.power(10, (np.log10(raw_data) - 1.15) / 1.56)
        R = np.power(Z / 200, 1/1.56) * calibration_factor
        
        return np.maximum(R, 0)


def load_rainfall(
    data_path: str,
    format: str = 'auto',
    **kwargs
) -> Union[np.ndarray, xr.DataArray, pd.DataFrame]:
    """
    便捷函数：加载降雨数据

    Args:
        data_path: 数据路径
        format: 数据格式 (txt/csv/tif/netcdf/auto)
        **kwargs: 其他参数传递给对应加载器

    Returns:
        降雨数据
    """
    path = Path(data_path)
    suffix = path.suffix.lower()
    
    if format == 'auto':
        if suffix in ['.txt', '.csv']:
            format = 'txt'
        elif suffix in ['.tif', '.tiff']:
            format = 'tif'
        elif suffix in ['.nc', '.netcdf']:
            format = 'netcdf'
    
    if format == 'txt':
        loader = ForcingLoader(rainfall_dir=str(path.parent), **kwargs)
        return loader.load_rainfall_txt(str(path))
    elif format == 'tif':
        from .raster_io import RasterIO
        return RasterIO.read_tiff(str(path))[0]
    elif format == 'netcdf':
        from .raster_io import RasterIO
        return RasterIO.read_netcdf(str(path))[0]
    else:
        raise ValueError(f"Unsupported format: {format}")
