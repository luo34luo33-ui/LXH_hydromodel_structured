"""
运行模型脚本

生成示例数据并运行模型
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import xarray as xr
from pathlib import Path
import time

print("=" * 60)
print("流溪河分布式水文模型 - 运行脚本")
print("=" * 60)

# 创建输出目录
output_dir = Path('Output')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 生成示例数据
print("\n[Step 1] 生成示例数据...")

ny, nx = 100, 100  # 网格大小
cell_size = 90.0    # 栅格大小（米）

np.random.seed(42)

# 生成DEM
print(f"  - 生成DEM ({ny}x{nx})...")
x_coords = np.arange(nx) * cell_size
y_coords = np.arange(ny) * cell_size

dem = np.zeros((ny, nx))
for i in range(ny):
    for j in range(nx):
        dem[i, j] = 700 + (ny - i) * 3 + (nx - j) * 2 + np.random.randn() * 10
dem = np.maximum(dem, 100)

# 生成流向（简化的随机流向）
print(f"  - 生成流向数据...")
flow_dir = np.random.randint(1, 9, size=(ny, nx)).astype(np.int8)

# 计算累积流
print(f"  - 计算累积流...")
flow_acc = np.zeros((ny, nx))
for i in range(1, ny-1):
    for j in range(1, nx-1):
        d = flow_dir[i, j] - 1
        if d == 0:
            ni, nj = i+1, j
        elif d == 1:
            ni, nj = i+1, j+1
        elif d == 2:
            ni, nj = i, j+1
        elif d == 3:
            ni, nj = i-1, j+1
        elif d == 4:
            ni, nj = i-1, j
        elif d == 5:
            ni, nj = i-1, j-1
        elif d == 6:
            ni, nj = i, j-1
        else:
            ni, nj = i+1, j-1
        flow_acc[i, j] = 1 + flow_acc[ni, nj]

# 坡度
slope = np.ones((ny, nx)) * 0.01 + np.random.rand(ny, nx) * 0.02

# 单元分类
print(f"  - 分类网格单元...")
unit_type = np.where(flow_acc > 10, 1, 0)  # 河道
unit_type[0, :] = 0
unit_type[-1, :] = 0
unit_type[:, 0] = 0
unit_type[:, -1] = 0

# 保存空间数据为NetCDF
print(f"  - 保存空间数据...")
spatial_ds = xr.Dataset({
    'dem': (['y', 'x'], dem),
    'flow_direction': (['y', 'x'], flow_dir),
    'flow_accumulation': (['y', 'x'], flow_acc),
    'slope': (['y', 'x'], slope),
    'unit_type': (['y', 'x'], unit_type)
}, coords={
    'x': ('x', x_coords),
    'y': ('y', y_coords)
}, attrs={
    'crs': 'EPSG:4326',
    'cell_size': cell_size,
    'nodata': -9999.0
})
spatial_ds.to_netcdf(str(output_dir / 'spatial.nc'), format='NETCDF4')
print(f"    保存到: Output/spatial.nc")

# 生成降雨数据
print(f"  - 生成降雨数据...")
nt = 48  # 时间步数（小时）
rainfall = np.zeros((nt, ny, nx))
for t in range(nt):
    if 10 <= t <= 20:  # 降雨事件
        base_rain = 5.0 + np.random.rand() * 3
        for i in range(ny):
            for j in range(nx):
                rainfall[t, i, j] = base_rain * (1 + np.random.rand() * 0.3)
    else:
        rainfall[t, :, :] = np.random.rand(ny, nx) * 0.2  # 小雨

time_index = np.arange(nt)
forcing_ds = xr.Dataset({
    'rainfall': (['time', 'y', 'x'], rainfall.astype(np.float32))
}, coords={
    'time': time_index,
    'x': ('x', x_coords),
    'y': ('y', y_coords)
}, attrs={
    'units': 'mm',
    'dt_minutes': 60
})
forcing_ds.to_netcdf(str(output_dir / 'rainfall.nc'), format='NETCDF4', 
                    encoding={'rainfall': {'zlib': True, 'complevel': 4}})
print(f"    保存到: Output/rainfall.nc")

# 2. 运行模型
print("\n[Step 2] 初始化模型...")

from data import read_asc
from spatial import DEMProcessor, FlowNetwork, GridManager
from core import RunoffGeneration, SlopeRouter, ChannelRouter

# 使用示例数据
print("  - 读取DEM...")
model_dem = dem.copy()
model_flow_dir = flow_dir.copy()
model_slope = slope.copy()

print("  - 构建汇流网络...")
flow_network = FlowNetwork(
    dem=model_dem,
    flow_dir=model_flow_dir,
    slope=model_slope
)
accumulation = flow_network.accumulate_flow()
river_type = flow_network.classify_units(acc_threshold=0.01)

print("  - 初始化网格管理器...")
grid_manager = GridManager(
    flow_dir=model_flow_dir,
    unit_type=river_type,
    accumulation=accumulation
)
grid_manager.initialize()
print(f"    拓扑排序: {len(grid_manager.topo_order)} 单元")

# 初始化物理模型
print("  - 初始化物理模型...")

# 土壤参数
theta_sat = np.ones_like(model_dem) * 0.40
theta_fc = np.ones_like(model_dem) * 0.25
theta_wp = np.ones_like(model_dem) * 0.10
K_sat = np.ones_like(model_dem) * 50.0
b_exp = np.ones_like(model_dem) * 4.0
mannings_n = np.ones_like(model_dem) * 0.1

runoff_model = RunoffGeneration(
    theta_sat=theta_sat,
    theta_fc=theta_fc,
    theta_wp=theta_wp,
    K_sat=K_sat,
    b_exp=b_exp,
    slope=model_slope,
    cell_size=cell_size
)

slope_router = SlopeRouter(
    slope=model_slope,
    mannings_n=mannings_n,
    cell_size=cell_size
)

channel_router = ChannelRouter(
    flow_dir=model_flow_dir,
    slope=model_slope,
    mannings_n=mannings_n,
    cell_size=cell_size
)

# 状态变量
dt = 3600.0
theta = theta_fc.copy()
Q_out = np.zeros_like(model_dem)
surface_runoff = np.zeros_like(model_dem)
subsurface_flow = np.zeros_like(model_dem)

# 3. 运行模拟
print("\n[Step 3] 运行模拟...")
print(f"  - 时间步数: {nt}")
print(f"  - 网格数量: {ny*nx}")

start_time = time.time()
results = {
    'discharge': [],
    'surface_runoff': [],
    'subsurface_flow': [],
    'soil_moisture': []
}

for t in range(nt):
    P = rainfall[t]
    PET = np.zeros_like(P)
    
    # 蒸散发
    ET = PET.copy()
    
    # 产流
    R, Ss, Perc, theta_new = runoff_model.compute(P, ET, theta, dt)
    theta = theta_new
    
    # 汇流
    Q_in = np.zeros_like(model_dem)
    
    for idx in grid_manager.topo_order:
        i, j = grid_manager.flat_to_2d(idx)
        unit_type = grid_manager.get_unit_type(i, j)
        
        if unit_type == 0:  # 边坡单元
            Q_cell = slope_router.route(Q_in[i, j], R[i, j], dt)
        else:  # 河道单元
            Q_cell = channel_router.route(Q_in[i, j], dt, Q_out)
        
        downstream = grid_manager.get_downstream(i, j)
        if downstream:
            di, dj = downstream
            Q_in[di, dj] += Q_cell
        
        Q_out[i, j] = Q_cell
    
    # 保存结果
    results['discharge'].append(Q_out.copy())
    results['surface_runoff'].append(R)
    results['subsurface_flow'].append(Ss)
    results['soil_moisture'].append(theta.copy())
    
    if (t + 1) % 10 == 0 or t == 0:
        elapsed = time.time() - start_time
        rate = (t + 1) / elapsed if elapsed > 0 else 0
        peak_Q = Q_out.max()
        print(f"    Step {t+1}/{nt}: Peak Q = {peak_Q:.4f} m³/s, Rate = {rate:.1f} it/s")

elapsed_total = time.time() - start_time
print(f"\n  模拟完成! 总耗时: {elapsed_total:.2f}秒")

# 4. 保存结果
print("\n[Step 4] 保存结果...")

output_ds = xr.Dataset({
    'discharge': (['time', 'y', 'x'], np.array(results['discharge'])),
    'surface_runoff': (['time', 'y', 'x'], np.array(results['surface_runoff'])),
    'subsurface_flow': (['time', 'y', 'x'], np.array(results['subsurface_flow'])),
    'soil_moisture': (['time', 'y', 'x'], np.array(results['soil_moisture']))
}, coords={
    'time': time_index,
    'x': ('x', x_coords),
    'y': ('y', y_coords)
}, attrs={
    'model': 'Liuxihe Distributed Hydrological Model',
    'dt_seconds': dt,
    'cell_size': cell_size,
    'n_timesteps': nt,
    'n_cells': ny * nx,
    'peak_discharge': float(np.array(results['discharge']).max()),
    'total_runoff': float(np.array(results['surface_runoff']).sum())
})

output_ds.to_netcdf(str(output_dir / 'result.nc'), format='NETCDF4',
                    encoding={
                        'discharge': {'zlib': True, 'complevel': 4},
                        'surface_runoff': {'zlib': True, 'complevel': 4}
                    })

print(f"  - 结果保存到: Output/result.nc")

# 5. 统计信息
print("\n" + "=" * 60)
print("模拟结果统计")
print("=" * 60)

all_Q = np.array(results['discharge'])
all_R = np.array(results['surface_runoff'])

print(f"  总径流量: {all_R.sum():.2f} mm")
print(f"  峰值流量: {all_Q.max():.4f} m³/s")
print(f"  平均流量: {all_Q.mean():.4f} m³/s")
print(f"  出口位置: 最大累积流处")

# 6. 生成可视化
print("\n[Step 5] 生成可视化...")

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # DEM
    im1 = axes[0, 0].imshow(dem, cmap='terrain', origin='lower')
    axes[0, 0].set_title('Digital Elevation Model (m)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 流向
    axes[0, 1].imshow(model_flow_dir, cmap='tab10', origin='lower')
    axes[0, 1].set_title('Flow Direction (D8)')
    
    # 单元分类
    colors = ['#90EE90', '#4169E1']
    cmap_custom = matplotlib.colors.ListedColormap(colors)
    axes[1, 0].imshow(river_type, cmap=cmap_custom, vmin=0, vmax=1, origin='lower')
    axes[1, 0].set_title('Unit Classification (Green=Slope, Blue=River)')
    
    # 出口流量过程
    outlet_Q = all_Q[:, ny//2, nx//2]
    axes[1, 1].plot(time_index, outlet_Q, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Discharge (m³/s)')
    axes[1, 1].set_title('Outlet Hydrograph')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'model_results.png'), dpi=150, bbox_inches='tight')
    print(f"  - 可视化保存到: Output/model_results.png")
    
    plt.close()

except Exception as e:
    print(f"  - 可视化跳过: {e}")

print("\n" + "=" * 60)
print("运行完成!")
print("=" * 60)
print(f"\n输出文件:")
print(f"  - Output/spatial.nc     (空间数据)")
print(f"  - Output/rainfall.nc    (降雨数据)")
print(f"  - Output/result.nc       (模拟结果)")
print(f"  - Output/model_results.png (可视化)")
