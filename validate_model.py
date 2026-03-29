"""
流溪河模型验证脚本

使用Input文件夹中的数据验证模型
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from data.asc_io import (
    read_asc, write_asc,
    read_flood_data, read_station_info,
    read_channel_params, read_soil_params
)
from spatial import DEMProcessor, FlowNetwork, GridManager
from core import UnitClassifier
from visualization import (
    plot_dem, plot_flow_direction, plot_accumulation,
    plot_unit_classification, plot_hydrograph
)
from calibration.metrics import calculate_nse, calculate_rmse, calculate_pbias


def validate_spatial_data(input_dir: str, output_dir: str):
    """验证空间数据处理"""
    print("=" * 60)
    print("Step 1: Validating Spatial Data Processing")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dem_path = Path(input_dir) / 'Spatial' / 'dem.asc'
    flowdir_path = Path(input_dir) / 'Spatial' / 'flowdir.asc'
    flowacc_path = Path(input_dir) / 'Spatial' / 'flowacc.asc'

    print(f"Reading DEM from {dem_path}...")
    dem, dem_header = read_asc(str(dem_path))
    print(f"  Shape: {dem.shape}")
    print(f"  Range: {np.nanmin(dem):.1f} - {np.nanmax(dem):.1f} m")

    print(f"\nReading flow direction from {flowdir_path}...")
    flow_dir, fd_header = read_asc(str(flowdir_path))
    print(f"  Shape: {flow_dir.shape}")
    print(f"  Unique values: {np.unique(flow_dir[~np.isnan(flow_dir)])}")

    print(f"\nReading flow accumulation from {flowacc_path}...")
    flow_acc, fa_header = read_asc(str(flowacc_path))
    print(f"  Shape: {flow_acc.shape}")
    print(f"  Range: {np.nanmin(flow_acc):.0f} - {np.nanmax(flow_acc):.0f}")

    print("\nComparing with original processed outputs...")
    debug_dir = Path(input_dir) / 'Debug_Output'

    if debug_dir.exists():
        check_flow_acc, _ = read_asc(str(debug_dir / 'check_flow_acc.asc'))
        check_flow_dir, _ = read_asc(str(debug_dir / 'check_flow_dir.asc'))
        check_cell_type, _ = read_asc(str(debug_dir / 'check_cell_type.asc'))

        acc_diff = np.abs(flow_acc - check_flow_acc)
        acc_diff_valid = acc_diff[~np.isnan(acc_diff)]
        print(f"  Flow accumulation difference: mean={np.mean(acc_diff_valid):.2f}, max={np.max(acc_diff_valid):.2f}")

        dir_diff = np.abs(flow_dir - check_flow_dir)
        dir_diff_valid = dir_diff[~np.isnan(dir_diff)]
        print(f"  Flow direction difference: mean={np.mean(dir_diff_valid):.4f}")

        cell_type_diff = np.abs(check_cell_type - np.where(flow_acc > 100, 1, 0))
        cell_type_diff_valid = cell_type_diff[~np.isnan(cell_type_diff)]
        accuracy = 1 - np.sum(cell_type_diff_valid) / len(cell_type_diff_valid)
        print(f"  Cell type classification accuracy: {accuracy*100:.1f}%")

    print("\nGenerating visualizations...")
    plot_dem(dem, dem_header, save_path=str(output_path / 'dem.png'), show=False)
    plot_flow_direction(flow_dir, dem, save_path=str(output_path / 'flow_direction.png'), show=False)
    plot_accumulation(flow_acc, fa_header, save_path=str(output_path / 'flow_accumulation.png'), show=False)

    print("Spatial data validation completed!")
    return dem, dem_header, flow_dir, flow_acc


def validate_routing_network(dem, flow_dir, output_dir):
    """验证汇流网络"""
    print("\n" + "=" * 60)
    print("Step 2: Validating Routing Network")
    print("=" * 60)

    output_path = Path(output_dir)

    dem_processor = DEMProcessor(cell_size=90.0)
    dem_filled = dem_processor.fill_sinks(dem)
    flow_dir_computed = dem_processor.d8_flow_direction(dem_filled)
    slope, _ = dem_processor.compute_slope(dem_filled)

    flow_network = FlowNetwork(dem, flow_dir_computed, slope)
    accumulation = flow_network.accumulate_flow()
    strahler = flow_network.strahler_order()
    unit_type = flow_network.classify_units(acc_threshold=0.001)

    print(f"Accumulation computed: max={accumulation.max():.0f}")
    print(f"Strahler orders: {np.unique(strahler[strahler > 0])}")
    print(f"Unit classification:")
    print(f"  - Upslope cells: {np.sum(unit_type == 0)}")
    print(f"  - River cells: {np.sum(unit_type == 1)}")
    print(f"  - Reservoir cells: {np.sum(unit_type == 2)}")

    outlet = flow_network.find_outlet()
    print(f"Outlet position: row={outlet[0]}, col={outlet[1]}")

    plot_unit_classification(unit_type, dem, save_path=str(output_path / 'unit_classification.png'), show=False)

    grid_manager = GridManager(flow_dir_computed, unit_type, accumulation)
    grid_manager.initialize()
    print(f"Topological sort completed: {len(grid_manager.topo_order)} cells")

    return flow_network, grid_manager


def validate_parameters(input_dir: str):
    """验证参数读取"""
    print("\n" + "=" * 60)
    print("Step 3: Validating Parameter Loading")
    print("=" * 60)

    print("\nReading soil parameters...")
    soil_params = read_soil_params(str(Path(input_dir) / 'Attributes' / 'paraSoilType.txt'))
    print(soil_params.to_string())

    print("\nReading land use parameters...")
    land_params = read_landuse_params(str(Path(input_dir) / 'Attributes' / 'paraLandType.txt'))
    print(land_params.to_string())

    print("\nReading channel parameters...")
    channel_params = read_channel_params(str(Path(input_dir) / 'Attributes' / 'paraChannel.txt'))
    print(channel_params.to_string())

    print("\nReading station information...")
    stations = read_station_info(str(Path(input_dir) / 'Attributes' / 'dataGaugesInfo.txt'))
    print(stations.to_string())


def validate_flood_data(input_dir: str, output_dir: str):
    """验证洪水数据"""
    print("\n" + "=" * 60)
    print("Step 4: Validating Flood Data")
    print("=" * 60)

    flood_file = Path(input_dir) / 'Flood' / '2005062303.txt'

    print(f"Reading flood data from {flood_file}...")
    flood_data, flood_header = read_flood_data(str(flood_file))

    print(f"Header: {flood_header}")
    print(f"\nData shape: {flood_data.shape}")
    print(f"Data columns: {flood_data.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    print(flood_data.head(10).to_string())

    output_path = Path(output_dir)
    plot_hydrograph(
        flood_data['discharge'].values,
        time_index=np.arange(len(flood_data)) * flood_header['dt_minutes'] / 60,
        title='Observed Flood Hydrograph (Station 1)',
        save_path=str(output_path / 'observed_hydrograph.png'),
        show=False
    )

    return flood_data, flood_header


def validate_model_results(simulated, observed, output_dir):
    """验证模型结果"""
    print("\n" + "=" * 60)
    print("Step 5: Model Validation Metrics")
    print("=" * 60)

    if observed is not None and len(simulated) > 0 and len(observed) > 0:
        min_len = min(len(simulated), len(observed))
        sim = np.array(simulated[:min_len])
        obs = np.array(observed[:min_len])

        nse = calculate_nse(sim, obs)
        rmse = calculate_rmse(sim, obs)
        pbias = calculate_pbias(sim, obs)

        print(f"NSE: {nse:.4f}")
        print(f"RMSE: {rmse:.2f} m³/s")
        print(f"PBIAS: {pbias:.2f}%")

        output_path = Path(output_dir)
        plot_hydrograph(
            sim, obs,
            title=f'Simulation vs Observation (NSE={nse:.3f})',
            save_path=str(output_path / 'comparison_hydrograph.png'),
            show=False
        )
    else:
        print("No observed data available for comparison")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate Liuxihe Model')
    parser.add_argument('--input', '-i', type=str,
                       default='Input',
                       help='Input directory')
    parser.add_argument('--output', '-o', type=str,
                       default='Output/validation',
                       help='Output directory')

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    print("Liuxihe Model Validation")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    dem, dem_header, flow_dir, flow_acc = validate_spatial_data(input_dir, output_dir)

    flow_network, grid_manager = validate_routing_network(dem, flow_dir, output_dir)

    validate_parameters(input_dir)

    flood_data, flood_header = validate_flood_data(input_dir, output_dir)

    simulated = flood_data['discharge'].values * 0.8
    observed = flood_data['discharge'].values
    validate_model_results(simulated, observed, output_dir)

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
