"""
数据格式转换工具

支持ASC、NetCDF、GeoTIFF之间的相互转换
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Union, List
import json
from datetime import datetime


def create_sample_data(
    output_dir: str,
    ny: int = 169,
    nx: int = 129,
    nt: int = 100,
    cell_size: float = 90.0
):
    """
    创建示例数据（用于测试）

    Args:
        output_dir: 输出目录
        ny, nx: 空间维
        nt: 时间维
        cell_size: 栅格大小
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    time = np.arange(nt)

    dem = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            dem[i, j] = 700 + (ny - i) * 2 + (nx - j) * 1.5 + np.random.randn() * 5
    dem = np.where(dem < 0, 0, dem)

    dem_ds = xr.Dataset({
        'dem': (['y', 'x'], dem)
    }, coords={
        'x': ('x', x),
        'y': ('y', y)
    }, attrs={
        'description': 'Sample DEM',
        'units': 'meters',
        'nodata': -9999.0
    })
    dem_ds.to_netcdf(output_path / 'dem.nc')
    dem_ds.to_dataarray().rio.write_transform().rio.set_spatial_dims(x_dim='x', y_dim='y').rio.to_raster(output_path / 'dem.tif')

    flow_dir = np.random.randint(1, 9, size=(ny, nx)).astype(np.int8)
    flow_acc = np.random.randint(1, 1000, size=(ny, nx))
    flow_acc = np.where(dem == 0, 0, flow_acc)

    flow_acc_ds = xr.Dataset({
        'flow_direction': (['y', 'x'], flow_dir),
        'flow_accumulation': (['y', 'x'], flow_acc)
    }, coords={
        'x': ('x', x),
        'y': ('y', y)
    })
    flow_acc_ds.to_netcdf(output_path / 'flow.nc')

    rainfall = np.random.exponential(scale=0.5, size=(nt, ny, nx))
    rainfall[rainfall < 0.1] = 0

    rainfall_ds = xr.Dataset({
        'rainfall': (['time', 'y', 'x'], rainfall.astype(np.float32))
    }, coords={
        'time': time,
        'x': ('x', x),
        'y': ('y', y)
    }, attrs={
        'units': 'mm',
        'dt_minutes': 60
    })
    rainfall_ds.to_netcdf(output_path / 'rainfall.nc', encoding={
        'rainfall': {'zlib': True, 'complevel': 4}
    })

    print(f"Sample data created in {output_path}")
    print(f"  - dem.nc, dem.tif: DEM data")
    print(f"  - flow.nc: Flow direction and accumulation")
    print(f"  - rainfall.nc: Rainfall forcing data")


def create_watershed_template(
    output_path: str,
    description: str = "Watershed Spatial Data"
) -> Dict:
    """
    创建流域空间数据模板配置

    Args:
        output_path: 输出路径
        description: 描述

    Returns:
        配置字典
    """
    template = {
        "metadata": {
            "title": "Watershed Spatial Data",
            "description": description,
            "created": datetime.now().isoformat(),
            "model": "Liuxihe Distributed Hydrological Model",
            "version": "1.0"
        },
        "spatial": {
            "crs": "EPSG:4326",
            "resolution": 90.0,
            "nodata": -9999.0,
            "variables": [
                {"name": "dem", "units": "m", "description": "Digital Elevation Model"},
                {"name": "flow_direction", "units": "D8_code", "description": "D8 flow direction (1-8)"},
                {"name": "flow_accumulation", "units": "cells", "description": "Flow accumulation"},
                {"name": "slope", "units": "radians", "description": "Terrain slope"},
                {"name": "landuse", "units": "class_id", "description": "Land use classification"},
                {"name": "soiltype", "units": "class_id", "description": "Soil type classification"}
            ]
        },
        "forcing": {
            "dt_minutes": 60,
            "variables": [
                {"name": "rainfall", "units": "mm", "description": "Precipitation"},
                {"name": "temperature", "units": "Celsius", "description": "Air temperature"},
                {"name": "humidity", "units": "percent", "description": "Relative humidity"},
                {"name": "radiation", "units": "W/m2", "description": "Solar radiation"},
                {"name": "wind_speed", "units": "m/s", "description": "Wind speed"},
                {"name": "pet", "units": "mm", "description": "Potential evapotranspiration"}
            ]
        },
        "output": {
            "variables": [
                {"name": "discharge", "units": "m3/s", "description": "Streamflow at outlet"},
                {"name": "soil_moisture", "units": "mm", "description": "Soil water content"},
                {"name": "surface_runoff", "units": "mm", "description": "Surface runoff"},
                {"name": "subsurface_flow", "units": "mm", "description": "Subsurface runoff"},
                {"name": "evapotranspiration", "units": "mm", "description": "Actual evapotranspiration"}
            ]
        }
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    return template


def batch_convert_asc_to_netcdf(
    input_dir: str,
    output_dir: str,
    variables: Optional[Dict[str, str]] = None
):
    """
    批量转换ASC文件为NetCDF

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        variables: 变量名映射 {'dem': 'dem', 'flowdir': 'flow_direction'}
    """
    from .asc_io import read_asc

    if variables is None:
        variables = {
            'dem': 'dem',
            'flowdir': 'flow_direction',
            'flowacc': 'flow_accumulation',
            'slope': 'slope'
        }

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combined_data = {}
    header_info = {}

    for asc_file in input_path.glob('*.asc'):
        var_name = asc_file.stem.lower()

        data, header = read_asc(str(asc_file))

        for pattern, standard_name in variables.items():
            if pattern in var_name:
                combined_data[standard_name] = data
                header_info[standard_name] = header
                print(f"  {asc_file.name} -> {standard_name}")

    if combined_data:
        ny, nx = next(iter(combined_data.values())).shape
        x = np.arange(nx) * header_info.get('cellsize', 1)
        y = np.arange(ny) * header_info.get('cellsize', 1)

        data_vars = {k: (['y', 'x'], v) for k, v in combined_data.items()}

        ds = xr.Dataset(
            data_vars,
            coords={
                'x': ('x', x),
                'y': ('y', y)
            },
            attrs={
                'xllcorner': header_info[next(iter(header_info))].get('xllcorner', 0),
                'yllcorner': header_info[next(iter(header_info))].get('yllcorner', 0),
                'cellsize': header_info[next(iter(header_info))].get('cellsize', 1),
                'nodata': header_info[next(iter(header_info))].get('NODATA_value', -9999)
            }
        )

        output_file = output_path / 'spatial_combined.nc'
        ds.to_netcdf(output_file, format='NETCDF4')
        print(f"Combined spatial data saved to {output_file}")


def export_forcing_to_standard_format(
    rainfall_file: str,
    output_file: str,
    format: str = 'netcdf'
):
    """
    导出标准格式的气象强迫数据

    Args:
        rainfall_file: 降雨数据文件
        output_file: 输出文件
        format: 输出格式 ('netcdf', 'zarr')
    """
    from .asc_io import read_flood_data

    rainfall_path = Path(rainfall_file)

    if rainfall_path.suffix == '.txt':
        rainfall, header = read_flood_data(str(rainfall_path))

        rainfall_3d = rainfall['discharge'].values.reshape(-1, 169, 129)

        ds = xr.Dataset({
            'rainfall': (['time', 'y', 'x'], rainfall_3d.astype(np.float32))
        }, attrs={
            'dt_minutes': header.get('dt_minutes', 60),
            'start_time': header.get('start_time', ''),
            'end_time': header.get('end_time', '')
        })

        if format == 'netcdf':
            ds.to_netcdf(output_file)
        elif format == 'zarr':
            ds.to_zarr(output_file)
    else:
        ds = xr.open_dataset(rainfall_file)

        if format == 'netcdf':
            ds.to_netcdf(output_file)
        elif format == 'zarr':
            ds.to_zarr(output_file)


def create_data_catalog(
    data_dir: str,
    output_path: str
):
    """
    创建数据目录清单（STAC兼容格式）

    Args:
        data_dir: 数据目录
        output_path: 输出JSON路径
    """
    data_path = Path(data_dir)

    catalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": "liuxihe-watershed-data",
        "title": "Liuxihe Watershed Data",
        "description": "Hydrological model input and output data",
        "providers": [],
        "links": [],
        "assets": {}
    }

    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in ['.nc', '.nc4', '.tif', '.tiff', '.json']:

                asset_info = {
                    "href": str(file_path.relative_to(data_path)),
                    "type": get_mime_type(suffix),
                    "title": file_path.stem
                }

                catalog["assets"][file_path.stem] = asset_info
                catalog["links"].append({
                    "rel": "item",
                    "href": str(file_path.relative_to(data_path))
                })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Data catalog created: {output_path}")
    print(f"  Total assets: {len(catalog['assets'])}")


def get_mime_type(suffix: str) -> str:
    """获取MIME类型"""
    mime_types = {
        '.nc': 'application/x-netcdf',
        '.nc4': 'application/x-netcdf',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff',
        '.json': 'application/json',
        '.asc': 'text/plain'
    }
    return mime_types.get(suffix.lower(), 'application/octet-stream')


def validate_data_format(file_path: str) -> Dict:
    """
    验证数据格式

    Args:
        file_path: 文件路径

    Returns:
        验证结果字典
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    result = {
        'file': str(path),
        'format': None,
        'valid': False,
        'checks': [],
        'warnings': []
    }

    if suffix == '.nc':
        try:
            ds = xr.open_dataset(path)
            result['format'] = 'NetCDF4'
            result['valid'] = True
            result['checks'].append(f"Variables: {list(ds.data_vars)}")
            result['checks'].append(f"Dimensions: {dict(ds.dims)}")
            ds.close()
        except Exception as e:
            result['checks'].append(f"Error: {e}")

    elif suffix in ['.tif', '.tiff']:
        try:
            from .raster_io import RasterIO
            data, meta = RasterIO.read_tiff(path)
            result['format'] = 'GeoTIFF'
            result['valid'] = True
            result['checks'].append(f"Shape: {data.shape}")
            result['checks'].append(f"CRS: {meta.get('crs')}")
        except Exception as e:
            result['checks'].append(f"Error: {e}")

    elif suffix == '.asc':
        try:
            from .asc_io import read_asc
            data, header = read_asc(path)
            result['format'] = 'ESRI ASCII Grid'
            result['valid'] = True
            result['checks'].append(f"Shape: {data.shape}")
            result['checks'].append(f"Header: {header}")
        except Exception as e:
            result['checks'].append(f"Error: {e}")

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Format Tools')
    parser.add_argument('command', choices=['create-sample', 'create-template', 'batch-convert', 'catalog', 'validate'])
    parser.add_argument('--input', '-i', help='Input directory/file')
    parser.add_argument('--output', '-o', help='Output directory/file')
    parser.add_argument('--format', '-f', default='netcdf', choices=['netcdf', 'zarr'])

    args = parser.parse_args()

    if args.command == 'create-sample':
        create_sample_data(args.output or 'Output/sample_data')
    elif args.command == 'create-template':
        create_watershed_template(args.output or 'Output/watershed_template.json')
    elif args.command == 'batch-convert':
        batch_convert_asc_to_netcdf(args.input, args.output)
    elif args.command == 'catalog':
        create_data_catalog(args.input, args.output or 'Output/data_catalog.json')
    elif args.command == 'validate':
        result = validate_data_format(args.input)
        print(json.dumps(result, indent=2))
