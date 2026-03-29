import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import pandas as pd


class RasterIO:
    """栅格数据读写器 - 支持GeoTIFF和NetCDF格式"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    @staticmethod
    def read_tiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取GeoTIFF文件

        Args:
            path: 文件路径

        Returns:
            data: 栅格数据数组
            metadata: 元数据 (transform, crs, nodata)
        """
        with rasterio.open(path) as src:
            data = src.read(1)
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'nodata': src.nodata,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'res': src.res
            }
        return data, metadata

    @staticmethod
    def write_tiff(
        path: str,
        data: np.ndarray,
        metadata: Dict[str, Any],
        compress: str = 'lzw'
    ):
        """
        写入GeoTIFF文件

        Args:
            path: 输出路径
            data: 栅格数据
            metadata: 元数据字典
            compress: 压缩方式
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = data.ndim == 3 and data.shape[0] or 1
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=data.shape[1],
            width=data.shape[2],
            count=count,
            dtype=data.dtype,
            crs=metadata.get('crs', 'EPSG:4326'),
            transform=metadata.get('transform'),
            compress=compress,
            nodata=metadata.get('nodata', -9999)
        ) as dst:
            dst.write(data)

    @staticmethod
    def read_netcdf(
        path: str,
        var_name: Optional[str] = None
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        """读取NetCDF格式"""
        ds = xr.open_dataset(path)
        
        if var_name is None:
            var_name = list(ds.data_vars)[0]
        
        data = ds[var_name]
        metadata = {
            'dims': data.dims,
            'coords': dict(data.coords),
            'attrs': dict(data.attrs)
        }
        return data, metadata

    @staticmethod
    def write_netcdf(
        path: str,
        data: xr.DataArray,
        encoding: Optional[Dict] = None
    ):
        """写入NetCDF格式"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        ds = data.to_dataset()
        ds.to_netcdf(path, encoding=encoding)

    @staticmethod
    def array_to_datarray(
        data: np.ndarray,
        transform: rasterio.transform.Affine,
        crs: str = 'EPSG:4326',
        nodata: float = -9999
    ) -> xr.DataArray:
        """
        将numpy数组转换为xarray DataArray并附加空间坐标

        Args:
            data: 栅格数据
            transform: 仿射变换矩阵
            crs: 坐标系
            nodata: 无数据值

        Returns:
            xr.DataArray: 带坐标的DataArray
        """
        height, width = data.shape
        
        xs = np.arange(width) * transform.a + transform.c + transform.a / 2
        ys = np.arange(height) * transform.e + transform.f + transform.e / 2
        
        da = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={
                'y': ys[::-1],
                'x': xs,
                'spatial_ref': xr.DataArray(
                    0,
                    attrs={
                        'crs_wkt': rasterio.crs.CRS.from_epsg(
                            crs.split(':')[-1]
                        ).to_wkt() if ':' in crs else crs
                    }
                )
            },
            attrs={'nodata': nodata}
        )
        
        return da

    @staticmethod
    def resample_grid(
        data: np.ndarray,
        src_transform: rasterio.transform.Affine,
        dst_transform: rasterio.transform.Affine,
        dst_shape: Tuple[int, int],
        method: Optional[str] = 'bilinear'
    ) -> np.ndarray:
        """
        栅格重采样

        Args:
            data: 源数据
            src_transform: 源仿射变换
            dst_transform: 目标仿射变换
            dst_shape: 目标形状
            method: 重采样方法 (nearest/bilinear/cubic)

        Returns:
            重采样后的数据
        """
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        
        method_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic
        }
        
        dst_data = np.zeros(dst_shape, dtype=data.dtype)
        
        reproject(
            source=data,
            destination=dst_data,
            src_transform=src_transform,
            dst_transform=dst_transform,
            resampling=method_map.get(method, Resampling.bilinear)
        )
        
        return dst_data


def read_raster(path: str, as_datarray: bool = True) -> Union[np.ndarray, xr.DataArray]:
    """便捷函数：读取栅格数据"""
    io = RasterIO()
    data, metadata = io.read_tiff(path)
    
    if as_datarray:
        return io.array_to_datarray(data, metadata['transform'], 
                                   str(metadata['crs']), metadata['nodata'])
    return data


def write_raster(
    path: str,
    data: Union[np.ndarray, xr.DataArray],
    reference_path: Optional[str] = None,
    **kwargs
):
    """便捷函数：写入栅格数据"""
    io = RasterIO()
    
    if isinstance(data, xr.DataArray):
        if reference_path:
            _, ref_meta = io.read_tiff(reference_path)
            transform = ref_meta['transform']
            crs = str(ref_meta['crs'])
        else:
            transform = from_bounds(0, 0, data.x.size, data.y.size,
                                   data.x.size, data.y.size)
            crs = 'EPSG:4326'
        
        metadata = {'transform': transform, 'crs': crs}
        io.write_tiff(path, data.values, metadata, **kwargs)
    else:
        raise ValueError("Data must be xarray.DataArray or provide reference_path")


class OriginalModelAdapter:
    """
    原模型输入格式适配器

    兼容流溪河模型原有数据格式（文本格式/特定二进制格式）
    """

    @staticmethod
    def read_txt_raster(path: str, skip_header: int = 0) -> np.ndarray:
        """读取文本格式栅格数据"""
        data = np.loadtxt(path, skiprows=skip_header)
        return data

    @staticmethod
    def read_binary_raster(
        path: str,
        shape: Tuple[int, int],
        dtype: str = 'float32'
    ) -> np.ndarray:
        """读取二进制格式栅格数据"""
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int16': np.int16
        }
        data = np.fromfile(path, dtype=dtype_map.get(dtype, np.float32))
        return data.reshape(shape)

    @staticmethod
    def write_binary_raster(path: str, data: np.ndarray, dtype: str = 'float32'):
        """写入二进制格式栅格数据"""
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
        }
        data.astype(dtype_map.get(dtype, np.float32)).tofile(path)
