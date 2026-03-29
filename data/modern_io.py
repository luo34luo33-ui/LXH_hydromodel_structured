"""
现代数据格式模块

支持NetCDF4、GeoTIFF、Zarr等现代科学数据格式
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import json


class ModernDataFormat:
    """
    现代数据格式统一接口

    支持格式：
    - NetCDF4: 时空数据的标准格式
    - GeoTIFF: 栅格地理数据
    - Zarr: 大规模并行读取
    - JSON: 配置和元数据
    """

    @staticmethod
    def read_netcdf(
        path: str,
        var_name: Optional[str] = None,
        time_slice: Optional[Tuple[int, int]] = None
    ) -> Tuple[xr.DataArray, Dict]:
        """
        读取NetCDF格式

        Args:
            path: 文件路径
            var_name: 变量名（默认第一个）
            time_slice: 时间切片 (start, end)

        Returns:
            (data, metadata)
        """
        ds = xr.open_dataset(path, engine='netcdf4')

        if var_name is None:
            var_name = list(ds.data_vars)[0] if len(ds.data_vars) > 0 else list(ds.keys())[0]

        data = ds[var_name]

        if time_slice is not None:
            data = data.isel(time=slice(*time_slice))

        metadata = {
            'dims': dict(data.dims),
            'coords': {k: str(v) for k, v in data.coords.items()},
            'attrs': {k: str(v) for k, v in data.attrs.items()},
            'shape': data.shape,
            'dtype': str(data.dtype)
        }

        return data, metadata

    @staticmethod
    def write_netcdf(
        path: str,
        data: xr.DataArray,
        encoding: Optional[Dict] = None,
        compress: bool = True
    ):
        """
        写入NetCDF格式

        Args:
            path: 输出路径
            data: 数据数组
            encoding: 编码参数
            compress: 是否压缩
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if encoding is None:
            encoding = {}

        if compress and 'zlib' not in encoding:
            encoding['zlib'] = True
            encoding['complevel'] = 4

        ds = data.to_dataset()
        ds.to_netcdf(path, format='NETCDF4', encoding=encoding)

    @staticmethod
    def create_watershed_dataset(
        dem: np.ndarray,
        flow_dir: np.ndarray,
        flow_acc: np.ndarray,
        slope: np.ndarray,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        crs: str = 'EPSG:4326',
        cell_size: float = 90.0,
        nodata: float = -9999.0
    ) -> xr.Dataset:
        """
        创建流域数据集

        将多个空间数据整合为统一的xarray Dataset

        Args:
            dem: 高程数据
            flow_dir: 流向数据
            flow_acc: 累积流数据
            slope: 坡度数据
            x_coords: X坐标（可选，自动生成）
            y_coords: Y坐标（可选，自动生成）
            crs: 坐标系
            cell_size: 栅格大小
            nodata: 无数据值

        Returns:
            xarray.Dataset
        """
        ny, nx = dem.shape

        if x_coords is None:
            x_coords = np.arange(nx) * cell_size
        if y_coords is None:
            y_coords = np.arange(ny) * cell_size

        ds = xr.Dataset({
            'dem': (['y', 'x'], np.where(dem == nodata, np.nan, dem)),
            'flow_direction': (['y', 'x'], flow_dir.astype(np.int8)),
            'flow_accumulation': (['y', 'x'], flow_acc),
            'slope': (['y', 'x'], slope),
        }, coords={
            'x': ('x', x_coords),
            'y': ('y', y_coords),
            'y_coords': (['y'], y_coords),
            'x_coords': (['x'], x_coords),
        }, attrs={
            'crs': crs,
            'cell_size': cell_size,
            'nodata': nodata,
            'title': 'Watershed Spatial Data',
            'history': f'Created by Liuxihe Model Python'
        })

        return ds

    @staticmethod
    def create_forcing_dataset(
        rainfall: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        humidity: Optional[np.ndarray] = None,
        radiation: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None,
        dt_minutes: int = 60
    ) -> xr.Dataset:
        """
        创建气象强迫数据集

        Args:
            rainfall: 降雨数据 (time, y, x)
            temperature: 温度数据
            humidity: 湿度数据
            radiation: 辐射数据
            time: 时间坐标
            dt_minutes: 时间步长（分钟）

        Returns:
            xarray.Dataset
        """
        if time is None:
            time = np.arange(rainfall.shape[0])

        data_vars = {'rainfall': (['time', 'y', 'x'], rainfall)}

        if temperature is not None:
            data_vars['temperature'] = (['time', 'y', 'x'], temperature)
        if humidity is not None:
            data_vars['humidity'] = (['time', 'y', 'x'], humidity)
        if radiation is not None:
            data_vars['radiation'] = (['time', 'y', 'x'], radiation)

        attrs = {
            'rainfall_units': 'mm',
            'temperature_units': 'Celsius',
            'humidity_units': 'percent',
            'radiation_units': 'W/m2',
            'dt_minutes': dt_minutes
        }

        ds = xr.Dataset(data_vars, coords={'time': time}, attrs=attrs)
        return ds


def asc_to_netcdf(
    dem_path: str,
    output_path: str,
    flow_dir_path: Optional[str] = None,
    flow_acc_path: Optional[str] = None,
    **kwargs
):
    """
    将ASC格式转换为NetCDF

    Args:
        dem_path: DEM文件路径
        output_path: 输出NetCDF路径
        flow_dir_path: 流向文件路径
        flow_acc_path: 累积流文件路径
    """
    from .asc_io import read_asc

    dem, header = read_asc(dem_path)

    ds = xr.Dataset({
        'dem': (['y', 'x'], dem)
    }, attrs={
        'xllcorner': header.get('xllcorner', 0),
        'yllcorner': header.get('yllcorner', 0),
        'cellsize': header.get('cellsize', 1),
        'nodata': header.get('NODATA_value', -9999),
        'ncols': header.get('ncols', dem.shape[1]),
        'nrows': header.get('nrows', dem.shape[0])
    })

    if flow_dir_path:
        flow_dir, _ = read_asc(flow_dir_path)
        ds['flow_direction'] = (['y', 'x'], flow_dir)

    if flow_acc_path:
        flow_acc, _ = read_asc(flow_acc_path)
        ds['flow_accumulation'] = (['y', 'x'], flow_acc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, format='NETCDF4', encoding={'dem': {'zlib': True, 'complevel': 4}})


def netcdf_to_asc(
    nc_path: str,
    var_name: str,
    output_path: str
):
    """
    将NetCDF转换为ASC格式

    Args:
        nc_path: NetCDF文件路径
        var_name: 变量名
        output_path: 输出ASC路径
    """
    ds = xr.open_dataset(nc_path)
    data = ds[var_name].values

    header = {
        'ncols': data.shape[1],
        'nrows': data.shape[0],
        'xllcorner': float(ds.attrs.get('xllcorner', 0)),
        'yllcorner': float(ds.attrs.get('yllcorner', 0)),
        'cellsize': float(ds.attrs.get('cellsize', 1)),
        'NODATA_value': float(ds.attrs.get('nodata', -9999))
    }

    from .asc_io import write_asc
    write_asc(output_path, data, header)


class WatershedDataLoader:
    """
    流域数据加载器

    统一的现代数据加载接口
    """

    def __init__(
        self,
        data_dir: str,
        crs: str = 'EPSG:4326'
    ):
        """
        Args:
            data_dir: 数据目录
            crs: 坐标系
        """
        self.data_dir = Path(data_dir)
        self.crs = crs

    def load_spatial_data(self) -> xr.Dataset:
        """加载空间数据"""
        nc_files = list(self.data_dir.glob('**/*.nc')) + list(self.data_dir.glob('**/*.nc4'))

        spatial_nc = None
        for f in nc_files:
            if 'spatial' in f.stem.lower() or 'static' in f.stem.lower():
                spatial_nc = f
                break

        if spatial_nc and spatial_nc.exists():
            return xr.open_dataset(spatial_nc)

        dem_files = list(self.data_dir.glob('**/dem.tif')) + list(self.data_dir.glob('**/dem.asc'))
        if dem_files:
            from .raster_io import RasterIO
            from .asc_io import read_asc

            if dem_files[0].suffix == '.asc':
                dem, header = read_asc(str(dem_files[0]))
            else:
                dem, header = RasterIO.read_tiff(str(dem_files[0]))

            return ModernDataFormat.create_watershed_dataset(dem, np.zeros_like(dem), np.zeros_like(dem), np.zeros_like(dem))

        raise FileNotFoundError(f"No spatial data found in {self.data_dir}")

    def load_forcing_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> xr.Dataset:
        """加载气象强迫数据"""
        nc_files = list(self.data_dir.glob('**/*forcing*.nc')) + \
                   list(self.data_dir.glob('**/*meteo*.nc')) + \
                   list(self.data_dir.glob('**/*rain*.nc'))

        if nc_files:
            ds = xr.open_dataset(str(nc_files[0]))
            if start_date and end_date:
                ds = ds.sel(time=slice(start_date, end_date))
            return ds

        rainfall_files = list(self.data_dir.glob('**/rain*.asc'))
        if rainfall_files:
            from .asc_io import read_asc
            data, _ = read_asc(str(rainfall_files[0]))
            return ModernDataFormat.create_forcing_dataset(data[np.newaxis, ...])

        raise FileNotFoundError(f"No forcing data found in {self.data_dir}")

    def save_config(self, output_path: str, **kwargs):
        """保存数据配置"""
        config = {
            'data_dir': str(self.data_dir),
            'crs': self.crs,
            'kwargs': kwargs
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def create_sample_config(data_dir: str, output_path: str):
    """
    创建示例配置文件

    Args:
        data_dir: 数据目录
        output_path: 输出路径
    """
    config = {
        "data_sources": {
            "spatial": {
                "format": "NetCDF4",
                "files": ["spatial/watershed.nc"],
                "variables": ["dem", "flow_direction", "flow_accumulation", "slope"]
            },
            "forcing": {
                "format": "NetCDF4",
                "files": ["forcing/rainfall.nc"],
                "variables": ["rainfall", "temperature", "humidity"]
            }
        },
        "model_config": {
            "dt_minutes": 60,
            "cell_size": 90.0,
            "crs": "EPSG:4326",
            "river_threshold": 0.01
        },
        "parameters": {
            "WMM": 10.0,
            "K": 0.4,
            "B": 4.0,
            "IM": 0.01,
            "KE": 15.0,
            "XE": 0.5,
            "KG": 0.1,
            "CG": 0.98,
            "KKG": 0.15,
            "C": 0.15,
            "CS": 0.5
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config
