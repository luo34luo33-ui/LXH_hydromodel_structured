# 流溪河分布式水文模型 Python 重构版

基于物理机制的分布式水文模型，支持GPU加速和张量化计算。

## 项目状态

| 指标 | 状态 |
|------|------|
| 开发阶段 | 核心功能已实现 |
| 模型运行 | 正常 |
| 模拟流量峰值 | ~29 m³/s (观测峰值 47 m³/s) |
| NSE | 0.215 (k=0.2 汇流系数) |

---

## 目录

- [项目结构](#项目结构)
- [安装配置](#安装配置)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [数据格式](#数据格式)
- [参数率定](#参数率定)

---

## 项目结构

```
liuxihe_model/
├── core/                          # 核心物理引擎
│   ├── evapotranspiration.py      # 蒸散发计算
│   ├── runoff_generation.py       # 产流计算
│   ├── routing_slope.py           # 坡面汇流
│   ├── routing_channel.py         # 河道汇流
│   ├── routing_reservoir.py       # 水库汇流
│   ├── routing_groundwater.py     # 地下径流
│   └── unit_classifier.py         # 单元分类
│
├── spatial/                       # 空间处理
│   ├── flow_network.py            # 汇流网络构建
│   ├── grid_manager.py            # 网格管理
│   ├── dem_processor.py           # DEM处理
│   └── properties_extractor.py    # 属性提取
│
├── solver/                        # 数值求解器
│   ├── tridiagonal.py             # 三对角矩阵求解
│   └── newton_raphson.py          # 牛顿迭代法
│
├── data/                          # 数据读写
│   ├── asc_io.py                  # ASC格式读写
│   ├── csv_loader.py              # CSV加载器
│   ├── forcing_loader.py          # 气象数据加载
│   ├── raster_io.py               # 栅格数据IO
│   └── converters.py              # 数据转换
│
├── calibration/                    # 参数率定
│   ├── metrics.py                 # 评价指标
│   ├── optimizer.py               # 优化器
│   └── sensitivity.py             # 敏感性分析
│
├── utils/                         # 工具模块
│   ├── logger.py                  # 日志记录
│   └── config_loader.py           # 配置加载
│
├── visualization/                 # 可视化
│
├── config/                        # 配置文件
│   ├── default_params.json        # 默认参数
│   ├── param_ranges.json          # 参数范围
│   └── model_config.yaml          # 模型配置
│
├── Input/                         # 输入数据
│   ├── Spatial/                   # 空间数据
│   │   ├── dem.asc                # 数字高程模型
│   │   ├── flowdir.asc            # D8流向
│   │   ├── flowacc.asc            # 累积流
│   │   ├── slope.asc              # 坡度
│   │   ├── landuse.asc            # 土地利用
│   │   └── soiltype.asc           # 土壤类型
│   ├── Attributes/                # 属性参数
│   │   ├── paraChannel.txt        # 河道参数
│   │   ├── paraSoilType.txt       # 土壤参数
│   │   ├── paraLandType.txt       # 土地利用参数
│   │   └── dataGaugesInfo.txt     # 站点信息
│   └── Flood/                     # 洪水数据
│       └── 2005062303.txt         # 观测流量
│
├── Output/                        # 输出结果
│
├── main.py                        # 主入口
├── run_model.py                   # 合成数据运行
├── run_model_real.py              # 真实数据运行
├── run_model_interactive.py       # 交互式运行
├── validate_model.py              # 模型验证
└── requirements.txt               # 依赖
```

---

## 安装配置

### 环境要求

- Python 3.9+
- NumPy, Pandas, Matplotlib, SciPy

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 自动模式

```bash
python run_model_real.py
```

### 交互式模式

```bash
python run_model_interactive.py
```

### 主程序模式

```bash
python main.py
```

---

## 核心模块

### spatial/ - 空间处理

| 模块 | 功能 |
|------|------|
| FlowNetwork | 累积流计算、单元分类、汇流网络构建 |
| GridManager | 网格管理、拓扑排序 |
| DemProcessor | DEM处理、洼地填充 |
| PropertiesExtractor | 空间属性提取 |

### core/ - 物理引擎

| 模块 | 功能 |
|------|------|
| Evapotranspiration | 蒸散发计算 |
| RunoffGeneration | 蓄满产流计算 |
| SlopeRouter | 坡面汇流(运动波) |
| ChannelRouter | 河道汇流(扩散波) |
| ReservoirRouter | 水库调洪演算 |
| GroundwaterRouter | 地下径流 |
| UnitClassifier | 单元分类(边坡/河道/水库) |

### solver/ - 数值求解

| 模块 | 功能 |
|------|------|
| Tridiagonal | 三对角矩阵算法(Thomas) |
| NewtonRaphson | 牛顿-拉夫森迭代 |

---

## 数据格式

### ASC空间数据

```
ncols         129
nrows         169
xllcorner     467968.68
yllcorner     2628209.63
cellsize      90.21
NODATA_value  -9999
<数据矩阵>
```

### CSV时间序列

支持自动检测列名：datetime, rainfall, evaporation, temperature 等

---

## 参数率定

### 关键参数

| 参数 | 含义 | 默认值 | 建议范围 |
|------|------|--------|----------|
| runoff_coef | 产流系数 | 1.0 | 0.3-1.0 |
| k | 汇流时间参数 | 0.5 | 0.1-2.0 |

### 测试结果

| k值 | 峰值(m³/s) | NSE |
|-----|-----------|-----|
| 0.1 | 2.90 | 0.039 |
| 0.2 | 7.64 | **0.215** |
| 0.3 | 14.02 | 0.176 |
| 0.5 | 29.13 | -1.176 |

---

## 参考文献

1. 流溪河模型原理文档
2. D8流向算法 (O'Callaghan & Mark, 1984)
3. Strahler河流分级 (Strahler, 1957)
4. 运动波方程 (Lighthill & Whitham, 1955)
5. Campbell公式 (Campbell, 1974)

---

## 许可证

MIT License
