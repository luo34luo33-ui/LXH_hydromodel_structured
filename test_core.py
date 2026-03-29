"""
简单测试脚本 - 验证模型核心功能
"""

import sys
sys.path.insert(0, '.')

import numpy as np

print("=" * 60)
print("Testing Liuxihe Model Core Functions")
print("=" * 60)

print("\n1. Testing ASC I/O...")
try:
    from data.asc_io import read_asc, write_asc
    
    dem, header = read_asc('Input/Spatial/dem.asc')
    print(f"   DEM shape: {dem.shape}")
    print(f"   DEM range: {np.nanmin(dem):.1f} - {np.nanmax(dem):.1f} m")
    print("   [OK] ASC I/O works")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n2. Testing Flow Network...")
try:
    from spatial import DEMProcessor, FlowNetwork
    
    dem_processor = DEMProcessor(cell_size=90.0)
    flow_dir, _ = read_asc('Input/Spatial/flowdir.asc')
    
    flow_network = FlowNetwork(dem, flow_dir, slope=np.ones_like(dem) * 0.01)
    accumulation = flow_network.accumulate_flow()
    unit_type = flow_network.classify_units(acc_threshold=0.001)
    
    print(f"   Accumulation max: {accumulation.max():.0f}")
    print(f"   River cells: {np.sum(unit_type == 1)}")
    print("   [OK] Flow Network works")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n3. Testing Grid Manager...")
try:
    from spatial import GridManager
    
    grid_manager = GridManager(flow_dir, unit_type, accumulation)
    grid_manager.initialize()
    
    print(f"   Topological order: {len(grid_manager.topo_order)} cells")
    print("   [OK] Grid Manager works")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n4. Testing Flood Data...")
try:
    from data.asc_io import read_flood_data
    
    flood_data, header = read_flood_data('Input/Flood/2005062303.txt')
    print(f"   Time range: {header['start_time']} - {header['end_time']}")
    print(f"   Data points: {len(flood_data)}")
    print("   [OK] Flood Data works")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n5. Testing Metrics...")
try:
    from calibration.metrics import calculate_nse, calculate_rmse
    
    sim = np.array([10, 20, 30, 40, 50])
    obs = np.array([12, 18, 32, 38, 52])
    
    nse = calculate_nse(sim, obs)
    rmse = calculate_rmse(sim, obs)
    
    print(f"   NSE: {nse:.4f}")
    print(f"   RMSE: {rmse:.2f} m³/s")
    print("   [OK] Metrics works")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n6. Testing Visualization...")
try:
    from visualization import plot_dem, plot_hydrograph
    
    plot_dem(dem, header, show=False, save_path='Output/validation/test_dem.png')
    plot_hydrograph(np.random.rand(50) * 100, show=False, 
                   save_path='Output/validation/test_hydrograph.png')
    print("   [OK] Visualization works")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n" + "=" * 60)
print("All core functions tested!")
print("=" * 60)
