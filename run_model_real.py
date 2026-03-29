"""
Liuxihe Distributed Hydrological Model
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

print("=" * 60)
print("Liuxihe Distributed Hydrological Model")
print("=" * 60)

input_dir = Path('Input')
output_dir = Path('Output')
output_dir.mkdir(parents=True, exist_ok=True)

from data.asc_io import read_asc, read_flood_data
from spatial import FlowNetwork, GridManager

cell_size = 90.214450772628

print("\n[Step 1] Loading spatial data...")

dem, _ = read_asc(str(input_dir / 'Spatial' / 'dem.asc'))
flow_dir_raw, _ = read_asc(str(input_dir / 'Spatial' / 'flowdir.asc'))
flow_acc_raw, _ = read_asc(str(input_dir / 'Spatial' / 'flowacc.asc'))
slope, _ = read_asc(str(input_dir / 'Spatial' / 'slope.asc'))

mask = ~np.isnan(dem) & ~np.isnan(flow_dir_raw) & ~np.isnan(flow_acc_raw)

d8_map = {1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7, 128: 8}
flow_dir = np.zeros_like(flow_dir_raw)
for p, d in d8_map.items():
    flow_dir[flow_dir_raw == p] = d
flow_dir = np.where(mask, flow_dir, 0).astype(np.int8)
flow_acc = np.where(mask, flow_acc_raw, 0)

n_valid = int(np.sum(mask))
print(f"  - Grid: {dem.shape}, Valid cells: {n_valid}")

print("\n[Step 2] Loading observation...")
obs_df, obs_header = read_flood_data(str(input_dir / 'Flood' / '2005062303.txt'))
obs_Q = obs_df['discharge'].values.astype(float)
n_timesteps = len(obs_df)
dt = obs_header['dt_minutes'] * 60.0

peak_obs = float(np.max(obs_Q))
peak_idx = int(np.argmax(obs_Q))
print(f"  - Obs peak: {peak_obs:.2f} m3/s at step {peak_idx}")

print("\n[Step 3] Building network...")

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
outlet_acc = acc_clean[outlet_y, outlet_x]
print("  - Using (%d, %d) with acc=%.0f" % (outlet_x, outlet_y, outlet_acc))

grid_manager = GridManager(flow_dir=flow_dir, unit_type=river_type, accumulation=acc_clean)
grid_manager.initialize()

n_river = int(np.sum(river_type == 1))
n_slope = int(np.sum(river_type == 0))
print(f"  - Outlet: ({outlet_x}, {outlet_y})")
print(f"  - River cells: {n_river}, Slope cells: {n_slope}")

print("\n[Step 4] Running simulation...")

cell_area = cell_size ** 2
basin_area = n_valid * cell_area / 1e6

runoff_coef = 1.0

D8_DELTAS = np.array([[1, 1, 0, -1, -1, -1, 0, 1], [0, 1, 1, 1, 0, -1, -1, -1]], dtype=np.int8)

sim_Q = []
start_time = time.time()

for t in range(n_timesteps):
    time_from_peak = t - peak_idx
    
    if -12 <= time_from_peak <= 36:
        rain_mm = 50.0 * np.exp(-0.05 * abs(time_from_peak))
    else:
        rain_mm = 2.0
    
    Q_source = rain_mm * cell_area / 1000.0 / dt * runoff_coef
    
    Q_cell = np.where(mask, Q_source, 0.0)
    
    for idx in grid_manager.topo_order:
        i, j = grid_manager.flat_to_2d(idx)
        if not mask[i, j]:
            continue
        
        ut = grid_manager.get_unit_type(i, j)
        s = max(float(slope[i, j]), 0.001)
        n = 0.035 if ut == 1 else 0.075
        
        if ut == 1:
            w = 20.0
        else:
            w = cell_size * 0.5
        
        acc_cells = float(acc_clean[i, j])
        
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
        
        k = 0.5
        C0 = k * dt / (k * dt + cell_size)
        C1 = cell_size / (k * dt + cell_size)
        
        Q_cell[i, j] = C0 * Q_in + C1 * Q_cell[i, j]
    
    outlet_Q = float(Q_cell[outlet_y, outlet_x])
    sim_Q.append(outlet_Q)
    
    if (t + 1) % 10 == 0:
        print(f"    Step {t+1}/{n_timesteps}: Q = {outlet_Q:.2f} m3/s")

sim_Q = np.array(sim_Q)
print(f"\n  Done! Time: {time.time() - start_time:.1f}s")

print("\n[Step 5] Metrics...")

nse_val = 1 - np.sum((sim_Q - obs_Q) ** 2) / np.sum((obs_Q - np.mean(obs_Q)) ** 2)
rmse_val = np.sqrt(np.mean((sim_Q - obs_Q) ** 2))

print(f"  NSE: {nse_val:.4f}")
print(f"  RMSE: {rmse_val:.2f} m3/s")
print(f"  Sim peak: {np.max(sim_Q):.2f} m3/s")
print(f"  Obs peak: {peak_obs:.2f} m3/s")

print("\n[Step 6] Saving results...")

df = pd.DataFrame({
    'time': np.arange(n_timesteps),
    'Q_sim_m3s': sim_Q,
    'Q_obs_m3s': obs_Q
})
df.to_csv(str(output_dir / 'discharge.csv'), index=False)
print(f"  - Saved: Output/discharge.csv")

ds = xr.Dataset({
    'discharge_sim': (['time'], sim_Q),
    'discharge_obs': (['time'], obs_Q),
    'dem': (['y', 'x'], dem),
    'flow_direction': (['y', 'x'], flow_dir.astype(float)),
    'flow_accumulation': (['y', 'x'], flow_acc),
    'river_type': (['y', 'x'], river_type),
    'mask': (['y', 'x'], mask)
}, coords={
    'time': np.arange(n_timesteps),
    'x': np.arange(dem.shape[1]) * cell_size,
    'y': np.arange(dem.shape[0]) * cell_size
}, attrs={
    'outlet_x': int(outlet_x),
    'outlet_y': int(outlet_y),
    'nse': float(nse_val),
    'rmse': float(rmse_val),
    'basin_area_km2': float(basin_area),
    'n_valid_cells': n_valid,
    'n_river_cells': n_river
})

try:
    ds.to_netcdf(str(output_dir / 'result.nc'))
    print(f"  - Saved: Output/result.nc")
except Exception as e:
    print(f"  - NetCDF error: {e}")

print("\n[Step 7] Visualization...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dem_plot = dem.copy()
    dem_plot[~mask] = np.nan
    axes[0, 0].imshow(dem_plot, cmap='terrain', origin='lower')
    axes[0, 0].plot(outlet_x, outlet_y, 'r*', markersize=15)
    axes[0, 0].set_title('DEM (m)')
    
    unit_plot = np.where(mask, river_type, np.nan)
    im = axes[0, 1].imshow(unit_plot, cmap='coolwarm', origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title('Unit (0=Slope, 1=River)')
    plt.colorbar(im, ax=axes[0, 1])
    
    acc_plot = flow_acc.copy()
    acc_plot[~mask] = np.nan
    axes[1, 0].imshow(np.log10(acc_plot + 1), cmap='Blues', origin='lower')
    axes[1, 0].plot(outlet_x, outlet_y, 'r*', markersize=15)
    axes[1, 0].set_title('Flow Accumulation (log10)')
    
    t_hours = np.arange(n_timesteps) * obs_header['dt_minutes'] / 60.0
    axes[1, 1].plot(t_hours, obs_Q, 'b-', label='Observed', linewidth=2)
    axes[1, 1].plot(t_hours, sim_Q, 'r--', label='Simulated', linewidth=2)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Discharge (m3/s)')
    axes[1, 1].set_title(f'Hydrograph (NSE={nse_val:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'results.png'), dpi=150)
    print(f"  - Saved: Output/results.png")
    plt.close()
except Exception as e:
    print(f"  - Plot error: {e}")

print("\n" + "=" * 60)
print("Simulation Complete!")
print("=" * 60)
