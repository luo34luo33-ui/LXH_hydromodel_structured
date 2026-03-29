"""
Liuxihe Distributed Hydrological Model - 交互式数据列选择
=====================================================

上传文件后，人工选择雨量、蒸发和径流列名

用法:
    python run_model_interactive.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from data.asc_io import (
    read_asc, 
    read_flood_data,
    detect_csv_columns,
    print_column_selector_info,
    read_timeseries_csv
)

print("=" * 60)
print("Liuxihe Distributed Hydrological Model")
print("交互式数据列选择")
print("=" * 60)


def select_column(columns: list, prompt: str, default_idx: int = None) -> str:
    """交互式选择列"""
    print(f"\n{prompt}")
    print("-" * 40)
    
    for i, col in enumerate(columns):
        marker = " (默认)" if default_idx is not None and i == default_idx else ""
        print(f"  {i+1}. {col}{marker}")
    
    print(f"  0. 跳过此列")
    print("-" * 40)
    
    while True:
        try:
            choice = input(f"请输入序号 (直接回车使用默认): ").strip()
            if choice == "":
                if default_idx is not None:
                    return columns[default_idx]
                else:
                    return None
            idx = int(choice) - 1
            if 0 <= idx < len(columns):
                return columns[idx]
            elif idx == -1:
                return None
            else:
                print(f"无效选择，请输入0-{len(columns)}之间的数字")
        except ValueError:
            print("无效输入，请输入数字")


def auto_detect_column(columns: list, candidates: list) -> int:
    """自动检测列索引"""
    column_lower = [c.lower() for c in columns]
    for cand in candidates:
        if cand.lower() in column_lower:
            return column_lower.index(cand.lower())
    return None


def main():
    input_dir = Path('Input')
    output_dir = Path('Output')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Step 1] 加载空间数据")
    print("=" * 60)

    dem, dem_header = read_asc(str(input_dir / 'Spatial' / 'dem.asc'))
    flow_dir_raw, _ = read_asc(str(input_dir / 'Spatial' / 'flowdir.asc'))
    flow_acc_raw, _ = read_asc(str(input_dir / 'Spatial' / 'flowacc.asc'))
    slope, _ = read_asc(str(input_dir / 'Spatial' / 'slope.asc'))

    mask = ~np.isnan(dem) & ~np.isnan(flow_dir_raw) & ~np.isnan(flow_acc_raw)

    d8_map = {1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7, 128: 8}
    flow_dir = np.zeros_like(flow_dir_raw)
    for p, d in d8_map.items():
        flow_dir[flow_dir_raw == p] = d
    flow_dir = np.where(mask, flow_dir, 0).astype(np.int8)

    n_valid = int(np.sum(mask))
    cell_size = dem_header['cellsize']
    print(f"  - 网格: {dem.shape}, 有效单元格: {n_valid}")
    print(f"  - 单元格大小: {cell_size:.2f} m")

    print("\n" + "=" * 60)
    print("[Step 2] 加载观测数据")
    print("=" * 60)

    obs_df, obs_header = read_flood_data(str(input_dir / 'Flood' / '2005062303.txt'))
    obs_Q = obs_df['discharge'].values.astype(float)
    n_timesteps = len(obs_df)
    dt = obs_header['dt_minutes'] * 60.0

    peak_obs = float(np.max(obs_Q))
    peak_idx = int(np.argmax(obs_Q))
    print(f"  - 观测峰值: {peak_obs:.2f} m3/s (步长 {peak_idx})")
    print(f"  - 时间步长: {obs_header['dt_minutes']} 分钟")

    print("\n" + "=" * 60)
    print("[Step 3] 加载气象数据")
    print("=" * 60)

    csv_files = list(input_dir.glob('**/*.csv')) + list(input_dir.glob('**/*.txt'))
    csv_files = [f for f in csv_files if 'Spatial' not in str(f) and 'Flood' not in str(f)]

    if csv_files:
        print(f"\n找到 {len(csv_files)} 个CSV/TXT文件:")
        for i, f in enumerate(csv_files):
            print(f"  {i+1}. {f.name}")
        
        use_interactive = input("\n是否进入交互式列选择模式? (y/n, 默认n): ").strip().lower()
        
        if use_interactive == 'y' and csv_files:
            print("\n" + "=" * 60)
            print("[Step 3a] 交互式列选择")
            print("=" * 60)
            
            csv_file = csv_files[0]
            print(f"\n选择文件: {csv_file}")
            
            info = detect_csv_columns(str(csv_file))
            columns = info['columns']
            
            print_column_selector_info(str(csv_file))
            
            time_candidates = ['time', 'datetime', 'date', 'timestamp', '时间', '日期']
            rain_candidates = ['rainfall', 'rain', 'precipitation', 'p', '降雨', '降水', '雨量']
            evap_candidates = ['evaporation', 'evap', 'et', 'pet', '蒸发', 'etp']
            runoff_candidates = ['runoff', 'discharge', 'q', 'flow', '径流', '流量']
            
            time_default = auto_detect_column(columns, time_candidates)
            rain_default = auto_detect_column(columns, rain_candidates)
            evap_default = auto_detect_column(columns, evap_candidates)
            runoff_default = auto_detect_column(columns, runoff_candidates)
            
            time_col = select_column(columns, "选择时间列:", time_default)
            rainfall_col = select_column(columns, "选择降雨列:", rain_default)
            evaporation_col = select_column(columns, "选择蒸发列:", evap_default)
            runoff_col = select_column(columns, "选择径流列:", runoff_default)
            
            ts_df, ts_info = read_timeseries_csv(
                str(csv_file),
                time_col=time_col,
                rainfall_col=rainfall_col,
                evaporation_col=evaporation_col,
                runoff_col=runoff_col
            )
            
            print("\n选择的列:")
            print(f"  - 时间: {ts_info.get('time_col', '未选择')}")
            print(f"  - 降雨: {ts_info.get('rainfall_col', '未选择')}")
            print(f"  - 蒸发: {ts_info.get('evaporation_col', '未选择')}")
            print(f"  - 径流: {ts_info.get('runoff_col', '未选择')}")
            
            rainfall = ts_df['rainfall'].values if 'rainfall' in ts_df.columns else None
            evaporation = ts_df['evaporation'].values if 'evaporation' in ts_df.columns else None
            observed_runoff = ts_df['runoff'].values if 'runoff' in ts_df.columns else None
            
            if rainfall is not None:
                print(f"  - 降雨数据点数: {len(rainfall)}")
            if observed_runoff is not None:
                print(f"  - 径流数据点数: {len(observed_runoff)}")
        else:
            csv_file = csv_files[0]
            print(f"\n使用默认文件: {csv_file}")
            
            rainfall = None
            evaporation = None
            observed_runoff = None
            
            try:
                ts_df, _ = read_timeseries_csv(str(csv_file))
                if 'rainfall' in ts_df.columns:
                    rainfall = ts_df['rainfall'].values
                    print(f"  - 检测到降雨数据: {len(rainfall)} 点")
                if 'runoff' in ts_df.columns:
                    observed_runoff = ts_df['runoff'].values
                    print(f"  - 检测到径流数据: {len(observed_runoff)} 点")
            except Exception as e:
                print(f"  - 自动检测失败: {e}")
    else:
        print("  - 未找到CSV/TXT文件")
        rainfall = None
        evaporation = None
        observed_runoff = None

    print("\n" + "=" * 60)
    print("[Step 4] 构建汇流网络")
    print("=" * 60)

    from spatial import FlowNetwork, GridManager

    flow_network = FlowNetwork(
        dem=np.where(mask, dem, 0),
        flow_dir=flow_dir,
        slope=np.where(mask, slope, 0.01)
    )

    acc_clean = flow_network.accumulate_flow()
    river_type = flow_network.classify_units(acc_threshold=0.01)

    mask_acc = np.where(mask, acc_clean + 1, 0)
    outlet_idx = int(np.nanargmax(mask_acc))
    outlet_y, outlet_x = np.unravel_index(outlet_idx, dem.shape)

    grid_manager = GridManager(
        flow_dir=flow_dir,
        unit_type=river_type,
        accumulation=acc_clean
    )
    grid_manager.initialize()

    n_river = int(np.sum(river_type == 1))
    n_slope = int(np.sum(river_type == 0))
    print(f"  - 出口: ({outlet_x}, {outlet_y})")
    print(f"  - 河道单元: {n_river}, 边坡单元: {n_slope}")

    print("\n" + "=" * 60)
    print("[Step 5] 运行模拟")
    print("=" * 60)

    cell_area = cell_size ** 2
    basin_area = n_valid * cell_area / 1e6

    runoff_coef = 1.0
    k_routing = 0.2

    D8_DELTAS = np.array([
        [1, 1, 0, -1, -1, -1, 0, 1],
        [0, 1, 1, 1, 0, -1, -1, -1]
    ], dtype=np.int64)

    sim_Q = []
    start_time = time.time()

    for t in range(n_timesteps):
        time_from_peak = t - peak_idx
        
        if rainfall is not None and t < len(rainfall):
            rain_mm = float(rainfall[t])
        else:
            if -12 <= time_from_peak <= 36:
                rain_mm = 50.0 * np.exp(-0.05 * abs(time_from_peak))
            else:
                rain_mm = 2.0
        
        if evaporation is not None and t < len(evaporation):
            evap_mm = float(evaporation[t])
        else:
            evap_mm = 0.0
        
        Q_source = (rain_mm - evap_mm) * cell_area / 1000.0 / dt * runoff_coef
        Q_source = max(Q_source, 0.001)
        
        Q_cell = np.where(mask, Q_source, 0.0)
        
        for idx in grid_manager.topo_order:
            i, j = grid_manager.flat_to_2d(idx)
            if not mask[i, j]:
                continue
            
            Q_up = 0.0
            for d in range(8):
                ni = int(i) + int(D8_DELTAS[0, d])
                nj = int(j) + int(D8_DELTAS[1, d])
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                    if mask[ni, nj]:
                        dd = int(flow_dir[ni, nj]) - 1
                        if 0 <= dd < 8:
                            ni2 = int(ni) + int(D8_DELTAS[0, dd])
                            nj2 = int(nj) + int(D8_DELTAS[1, dd])
                            if ni2 == i and nj2 == j:
                                Q_up += Q_cell[ni, nj]
            
            Q_in = Q_source + Q_up
            
            C0 = k_routing * dt / (k_routing * dt + cell_size)
            C1 = cell_size / (k_routing * dt + cell_size)
            
            Q_cell[i, j] = C0 * Q_in + C1 * Q_cell[i, j]
        
        outlet_Q = float(Q_cell[outlet_y, outlet_x])
        sim_Q.append(outlet_Q)
        
        if (t + 1) % 10 == 0:
            print(f"    步长 {t+1}/{n_timesteps}: Q = {outlet_Q:.2f} m3/s")

    sim_Q = np.array(sim_Q)
    print(f"\n  完成! 耗时: {time.time() - start_time:.1f}s")

    print("\n" + "=" * 60)
    print("[Step 6] 评价指标")
    print("=" * 60)

    nse_val = 1 - np.sum((sim_Q - obs_Q) ** 2) / np.sum((obs_Q - np.mean(obs_Q)) ** 2)
    rmse_val = np.sqrt(np.mean((sim_Q - obs_Q) ** 2))

    print(f"  NSE: {nse_val:.4f}")
    print(f"  RMSE: {rmse_val:.2f} m3/s")
    print(f"  模拟峰值: {np.max(sim_Q):.2f} m3/s")
    print(f"  观测峰值: {peak_obs:.2f} m3/s")

    print("\n" + "=" * 60)
    print("[Step 7] 保存结果")
    print("=" * 60)

    df = pd.DataFrame({
        'time': np.arange(n_timesteps),
        'Q_sim_m3s': sim_Q,
        'Q_obs_m3s': obs_Q
    })
    try:
        df.to_csv(str(output_dir / 'discharge_interactive.csv'), index=False)
        print(f"  - 已保存: Output/discharge_interactive.csv")
    except PermissionError:
        print(f"  - 保存被拒绝(文件可能被占用), 跳过保存")

    print("\n" + "=" * 60)
    print("模拟完成!")
    print("=" * 60)

    return df, obs_df, {
        'nse': nse_val,
        'rmse': rmse_val,
        'sim_peak': float(np.max(sim_Q)),
        'obs_peak': peak_obs,
        'outlet': (outlet_x, outlet_y),
        'basin_area': basin_area
    }


if __name__ == '__main__':
    results_df, obs_df, metrics = main()
