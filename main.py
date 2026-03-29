"""
流溪河分布式水文模型主程序

支持现代数据格式（NetCDF4、GeoTIFF、Zarr）
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from datetime import datetime
import time
import json

from utils import ConfigLoader, setup_logger, get_logger
from utils.config_loader import ParamMapper
from spatial import (
    DEMProcessor, FlowNetwork, GridManager, PropertiesExtractor
)
from data import (
    RasterIO, ForcingLoader, read_asc, ModernDataFormat,
    WatershedDataLoader, asc_to_netcdf
)
from core import (
    Evapotranspiration, RunoffGeneration,
    SlopeRouter, ChannelRouter, ReservoirRouter, GroundwaterRouter
)


class LiuxiheModel:
    """
    流溪河分布式水文模型主类

    整合所有子模型，实现完整的水文循环模拟
    支持多种数据格式：NetCDF4、GeoTIFF、ASC
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Args:
            config_path: 配置文件路径 (YAML/JSON)
            use_gpu: 是否使用GPU加速
            **kwargs: 其他配置参数
        """
        self.use_gpu = use_gpu
        self.logger = get_logger()

        self.config = ConfigLoader()
        if config_path:
            config_file = Path(config_path)
            if config_file.suffix == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self.params = config_data.get('parameters', self.config.load_params())
                    self.param_ranges = config_data.get('param_ranges', self.config.load_param_ranges())
            else:
                self.config = ConfigLoader(config_file.parent)
                self.params = self.config.load_params()
                self.param_ranges = self.config.load_param_ranges()
        else:
            self.params = self.config.load_params()
            self.param_ranges = self.config.load_param_ranges()

        self.dem = None
        self.flow_dir = None
        self.slope = None
        self.accumulation = None
        self.unit_type = None
        self.grid_manager = None
        self.prop_extractor = None
        self.header = {}

        self.dt = kwargs.get('dt', 3600)
        self.cell_size = kwargs.get('cell_size', 100)

        self.state = {}
        self.output = {}

    def setup_from_nc(self, spatial_nc: str, **kwargs):
        """
        从NetCDF格式设置模型

        Args:
            spatial_nc: 空间数据NetCDF文件路径
            **kwargs: 额外参数
        """
        self.logger.info(f"Loading spatial data from NetCDF: {spatial_nc}")

        ds = xr.open_dataset(spatial_nc)

        if 'dem' in ds:
            self.dem = ds['dem'].values
        else:
            raise ValueError("NetCDF must contain 'dem' variable")

        self.header = dict(ds.attrs)

        if 'flow_direction' in ds:
            self.flow_dir = ds['flow_direction'].values
        else:
            self._compute_flow_from_dem()

        if 'slope' in ds:
            self.slope = ds['slope'].values
        else:
            dem_processor = DEMProcessor(cell_size=self.cell_size, use_gpu=self.use_gpu)
            dem_filled = dem_processor.fill_sinks(self.dem)
            self.slope, _ = dem_processor.compute_slope(dem_filled)

        if 'flow_accumulation' in ds:
            self.accumulation = ds['flow_accumulation'].values
        else:
            self._compute_accumulation()

        ds.close()
        self._setup_common()

    def setup_from_tiff(self, dem_path: str, **kwargs):
        """
        从GeoTIFF格式设置模型

        Args:
            dem_path: DEM GeoTIFF文件路径
            **kwargs: 额外参数
        """
        self.logger.info(f"Loading DEM from GeoTIFF: {dem_path}")

        io = RasterIO(use_gpu=self.use_gpu)
        self.dem, dem_meta = io.read_tiff(dem_path)

        self.header = {
            'cellsize': abs(dem_meta['transform'].a),
            'xllcorner': dem_meta['transform'].c,
            'yllcorner': dem_meta['transform'].f,
            'crs': str(dem_meta['crs']) if dem_meta.get('crs') else 'EPSG:4326'
        }

        self._compute_flow_from_dem()

        self._compute_accumulation()

        self._setup_common()

    def setup_from_asc(self, dem_path: str, **kwargs):
        """
        从ASC格式设置模型（原模型格式）

        Args:
            dem_path: DEM ASC文件路径
            **kwargs: 额外参数
        """
        self.logger.info(f"Loading DEM from ASC: {dem_path}")

        self.dem, self.header = read_asc(dem_path)

        flow_dir_path = kwargs.get('flow_dir_path')
        flow_acc_path = kwargs.get('flow_acc_path')

        if flow_dir_path:
            self.flow_dir, _ = read_asc(flow_dir_path)
        else:
            self._compute_flow_from_dem()

        if flow_acc_path:
            self.accumulation, _ = read_asc(flow_acc_path)
        else:
            self._compute_accumulation()

        slope_path = kwargs.get('slope_path')
        if slope_path:
            self.slope, _ = read_asc(slope_path)
        else:
            dem_processor = DEMProcessor(cell_size=self.header.get('cellsize', self.cell_size), use_gpu=self.use_gpu)
            dem_filled = dem_processor.fill_sinks(self.dem)
            self.slope, _ = dem_processor.compute_slope(dem_filled)

        self._setup_common()

    def setup(self, dem_path: str, **kwargs):
        """
        自动检测格式并设置模型

        Args:
            dem_path: DEM数据路径（支持 .nc, .tif, .tiff, .asc）
            **kwargs: 额外参数
        """
        path = Path(dem_path)
        suffix = path.suffix.lower()

        if suffix in ['.nc', '.nc4']:
            self.setup_from_nc(dem_path, **kwargs)
        elif suffix in ['.tif', '.tiff']:
            self.setup_from_tiff(dem_path, **kwargs)
        elif suffix == '.asc':
            self.setup_from_asc(dem_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _compute_flow_from_dem(self):
        """从DEM计算流向"""
        dem_processor = DEMProcessor(cell_size=self.cell_size, use_gpu=self.use_gpu)
        dem_filled = dem_processor.fill_sinks(self.dem)
        self.flow_dir = dem_processor.d8_flow_direction(dem_filled)
        self.slope, _ = dem_processor.compute_slope(dem_filled)

    def _compute_accumulation(self):
        """计算累积流"""
        flow_network = FlowNetwork(
            dem=self.dem,
            flow_dir=self.flow_dir,
            slope=self.slope
        )
        self.accumulation = flow_network.accumulate_flow()

    def _setup_common(self, **kwargs):
        """通用设置"""
        self.logger.info("Building flow network...")
        self.flow_network = FlowNetwork(
            dem=self.dem,
            flow_dir=self.flow_dir,
            slope=self.slope
        )

        if self.accumulation is None:
            self.accumulation = self.flow_network.accumulate_flow()

        self.logger.info("Classifying grid units...")
        self.unit_type = self.flow_network.classify_units(
            acc_threshold=kwargs.get('river_threshold', 0.01)
        )

        self.logger.info("Initializing grid manager...")
        self.grid_manager = GridManager(
            flow_dir=self.flow_dir,
            unit_type=self.unit_type,
            accumulation=self.accumulation,
            use_gpu=self.use_gpu
        )
        self.grid_manager.initialize()

        self.logger.info("Extracting properties...")
        self.prop_extractor = PropertiesExtractor(cell_size=self.cell_size)

        soil_type = kwargs.get('soil_type')
        lulc = kwargs.get('lulc')

        if soil_type is not None and lulc is not None:
            self.spatial_params = self.prop_extractor.extract_all_params(
                self.dem, self.slope, soil_type, lulc
            )
        else:
            self.spatial_params = self.prop_extractor.extract_all_params(
                self.dem, self.slope
            )

        self._init_state()
        self.logger.info("Model initialization completed.")

    def _init_state(self):
        """初始化状态变量"""
        self.state = {
            'soil_moisture': self.spatial_params.get('theta_fc', np.ones_like(self.dem) * 0.25),
            'surface_runoff': np.zeros_like(self.dem),
            'subsurface_flow': np.zeros_like(self.dem),
            'channel_flow': np.zeros_like(self.dem),
            'groundwater': np.zeros_like(self.dem),
            'Q_out': np.zeros_like(self.dem),
        }

        self.evap_model = Evapotranspiration(
            theta_wp=self.spatial_params.get('theta_wp', np.ones_like(self.dem) * 0.1),
            theta_fc=self.spatial_params.get('theta_fc', np.ones_like(self.dem) * 0.25),
            theta_sat=self.spatial_params.get('theta_sat', np.ones_like(self.dem) * 0.4),
            veg_coverage=self.spatial_params.get('veg_coverage'),
            root_depth=self.spatial_params.get('root_depth')
        )

        self.runoff_model = RunoffGeneration(
            theta_sat=self.spatial_params['theta_sat'],
            theta_fc=self.spatial_params['theta_fc'],
            theta_wp=self.spatial_params.get('theta_wp', self.spatial_params['theta_fc'] * 0.4),
            K_sat=self.spatial_params['K_sat'],
            b_exp=self.spatial_params['b_exp'],
            slope=self.slope,
            cell_size=self.cell_size
        )

        self.slope_router = SlopeRouter(
            slope=self.slope,
            mannings_n=self.spatial_params.get('mannings_n', np.ones_like(self.dem) * 0.1),
            cell_size=self.cell_size,
            use_gpu=self.use_gpu
        )

        self.channel_router = ChannelRouter(
            flow_dir=self.flow_dir,
            slope=self.slope,
            mannings_n=self.spatial_params.get('mannings_n', np.ones_like(self.dem) * 0.1),
            cell_size=self.cell_size,
            use_gpu=self.use_gpu
        )

        self.gw_router = GroundwaterRouter(
            kg=self.params['KG'],
            cg=self.params['CG'],
            kkg=self.params['KKG']
        )

    def step(
        self,
        P: np.ndarray,
        PET: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        单步模拟

        Args:
            P: 降雨量 (mm/dt)
            PET: 潜在蒸散发 (mm/dt)

        Returns:
            状态更新后的字典
        """
        theta = self.state['soil_moisture']

        ETa, _ = self.evap_model.calculate(PET, theta, self.dt)

        R, Ss, Perc, theta_new = self.runoff_model.compute(
            P, ETa, theta, self.dt
        )

        self.state['soil_moisture'] = theta_new

        Q_in = np.zeros_like(self.dem)
        for idx in self.grid_manager.topo_order:
            i, j = self.grid_manager.flat_to_2d(idx)
            unit_type = self.grid_manager.get_unit_type(i, j)

            if unit_type == 0:
                Q_cell = self.slope_router.route(
                    Q_in[i, j], R[i, j], self.dt
                )
            else:
                Q_cell = self.channel_router.route(
                    Q_in[i, j], self.dt, self.state['Q_out']
                )

            downstream = self.grid_manager.get_downstream(i, j)
            if downstream:
                di, dj = downstream
                Q_in[di, dj] += Q_cell

            self.state['Q_out'][i, j] = Q_cell

        self.state['surface_runoff'] = R
        self.state['subsurface_flow'] = Ss

        return self.state.copy()

    def run(
        self,
        rainfall: Union[xr.DataArray, np.ndarray],
        pet: Optional[Union[xr.DataArray, np.ndarray]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> xr.Dataset:
        """
        批量模拟

        Args:
            rainfall: 降雨数据 (time, y, x) 或 (y, x) 用于单步
            pet: 潜在蒸散发数据（可选，默认0）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            模拟结果 Dataset
        """
        if isinstance(rainfall, np.ndarray):
            if rainfall.ndim == 2:
                rainfall = rainfall[np.newaxis, ...]
            rainfall = xr.DataArray(rainfall, dims=['time', 'y', 'x'])

        if pet is None:
            pet = xr.zeros_like(rainfall)
        elif isinstance(pet, np.ndarray):
            if pet.ndim == 2:
                pet = pet[np.newaxis, ...]
            pet = xr.DataArray(pet, dims=['time', 'y', 'x'])

        n_steps = rainfall.shape[0]
        self.logger.info(f"Running simulation for {n_steps} timesteps...")

        results = {
            'discharge': [],
            'surface_runoff': [],
            'subsurface_flow': [],
            'soil_moisture': []
        }

        start_time = time.time()

        for t in range(n_steps):
            P = rainfall[t].values
            PET = pet[t].values

            state = self.step(P, PET)

            results['discharge'].append(state['Q_out'])
            results['surface_runoff'].append(state['surface_runoff'])
            results['subsurface_flow'].append(state['subsurface_flow'])
            results['soil_moisture'].append(state['soil_moisture'])

            if (t + 1) % 100 == 0 or t == 0:
                elapsed = time.time() - start_time
                rate = (t + 1) / elapsed if elapsed > 0 else 0
                eta = (n_steps - t - 1) / rate if rate > 0 else 0
                self.logger.info(
                    f"Progress: {t+1}/{n_steps} ({100*(t+1)/n_steps:.1f}%) "
                    f"Rate: {rate:.1f} it/s ETA: {eta:.0f}s"
                )

        self.logger.info("Simulation completed.")

        time_coords = rainfall.coords.get('time', np.arange(n_steps))

        self.output = xr.Dataset({
            'discharge': (['time', 'y', 'x'], np.array(results['discharge'])),
            'surface_runoff': (['time', 'y', 'x'], np.array(results['surface_runoff'])),
            'subsurface_flow': (['time', 'y', 'x'], np.array(results['subsurface_flow'])),
            'soil_moisture': (['time', 'y', 'x'], np.array(results['soil_moisture']))
        }, coords={'time': time_coords}, attrs={
            'model': 'Liuxihe Distributed Hydrological Model',
            'dt_seconds': self.dt,
            'cell_size': self.cell_size,
            'created': datetime.now().isoformat()
        })

        return self.output

    def save_output(self, path: str, format: str = 'netcdf'):
        """
        保存输出结果

        Args:
            path: 输出路径
            format: 输出格式 ('netcdf', 'zarr', 'tiff')
        """
        if not self.output:
            self.logger.warning("No output to save.")
            return

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'netcdf':
            self.output.to_netcdf(path, format='NETCDF4', 
                                 encoding={v: {'zlib': True, 'complevel': 4} 
                                          for v in self.output.data_vars})
        elif format == 'zarr':
            self.output.to_zarr(path)
        else:
            self.output.to_netcdf(path, format='NETCDF4')

        self.logger.info(f"Output saved to {path}")

    def get_outlet_hydrograph(self) -> np.ndarray:
        """获取出口断面的流量过程线"""
        outlet_i, outlet_j = self.flow_network.find_outlet()
        return self.output['discharge'].values[:, outlet_i, outlet_j]

    def set_params(self, params: Dict[str, float]):
        """更新参数"""
        self.params.update(params)
        ParamMapper.validate_params(params, self.param_ranges)

    def get_state(self) -> Dict:
        """获取当前状态"""
        return self.state.copy()

    def to_xarray(self) -> xr.Dataset:
        """导出为xarray Dataset"""
        return xr.Dataset({
            'dem': (['y', 'x'], self.dem),
            'flow_direction': (['y', 'x'], self.flow_dir),
            'flow_accumulation': (['y', 'x'], self.accumulation),
            'slope': (['y', 'x'], self.slope),
            'unit_type': (['y', 'x'], self.unit_type)
        }, attrs=self.header)


def main():
    """主函数入口"""
    import argparse

    parser = argparse.ArgumentParser(description='Liuxihe Distributed Hydrological Model')
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file (YAML/JSON)')
    parser.add_argument('--dem', '-d', type=str,
                       help='DEM data path (.nc/.tif/.asc)')
    parser.add_argument('--spatial', '-s', type=str,
                       help='Spatial NetCDF file (contains dem, flow_dir, slope)')
    parser.add_argument('--rainfall', '-r', type=str,
                       help='Rainfall data path (.nc)')
    parser.add_argument('--pet', '-p', type=str,
                       help='PET data path (optional, .nc)')
    parser.add_argument('--output', '-o', type=str, default='Output/result.nc',
                       help='Output file path')
    parser.add_argument('--format', '-f', type=str, default='netcdf',
                       choices=['netcdf', 'zarr'],
                       help='Output format')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--dt', type=int, default=3600,
                       help='Time step in seconds')

    args = parser.parse_args()

    logger = setup_logger(log_file='logs/model.log')

    model = LiuxiheModel(
        config_path=args.config,
        use_gpu=args.gpu,
        dt=args.dt
    )

    if args.spatial:
        model.setup_from_nc(args.spatial)
    elif args.dem:
        model.setup(args.dem)
    else:
        logger.error("Please provide --dem or --spatial parameter")
        return

    logger.info("Loading forcing data...")
    if args.rainfall:
        rainfall_data = xr.open_dataset(args.rainfall)
        if 'rainfall' in rainfall_data:
            rainfall = rainfall_data['rainfall']
        elif 'precipitation' in rainfall_data:
            rainfall = rainfall_data['precipitation']
        else:
            rainfall = rainfall_data[list(rainfall_data.data_vars)[0]]
    else:
        rainfall = xr.zeros_like(model.to_xarray()['dem'])[np.newaxis, ...]

    if args.pet:
        pet_data = xr.open_dataset(args.pet)
        if 'pet' in pet_data:
            pet = pet_data['pet']
        else:
            pet = xr.zeros_like(rainfall)
    else:
        pet = xr.zeros_like(rainfall)

    results = model.run(rainfall, pet)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save_output(args.output, format=args.format)

    logger.info("Done.")


if __name__ == '__main__':
    main()
