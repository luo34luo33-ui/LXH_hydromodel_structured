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

- [背景介绍](#背景介绍)
- [模型原理](#模型原理)
- [项目结构](#项目结构)
- [安装配置](#安装配置)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [数据格式](#数据格式)
- [参数率定](#参数率定)

---

## 背景介绍

### 流溪河概况

流溪河位于广东省广州市从化区，是珠江支流之一，发源于从化吕田镇桂蜂山，流经从化市区，最终汇入珠江口。流域面积约2300平方公里，干流全长约156公里。

流溪河水库（又称流溪河国家森林公园）是广州市的重要水源地，具有防洪、发电、灌溉、供水等多重功能。由于流域地处亚热带季风气候区，雨量充沛但时空分布不均，容易发生洪涝灾害，因此需要科学的水文模型进行洪水预报和水资源管理。

### 模型发展历程

流溪河模型是由**中山大学陈洋波教授**主持研发的开源模型。作为国内分布式水文模型研究的先行者，陈洋波教授及其团队于 2009 年首次提出该模型，并持续迭代优化至今。

**陈洋波教授简介**：
- 1964年6月生，湖北崇阳人
- 武汉大学水文学及水资源专业博士
- 曾赴美国普林斯顿大学、美国加州大学尔湾分校从事访问研究
- 中山大学地理科学与规划学院水资源与环境系教授
- 自然地理、GIS与遥感两专业博士生导师
- 水灾害管理与水利信息化实验室主任

该模型基于流域产汇流物理机制，采用网格化的分布式架构，能够精细模拟流域内不同位置的水文过程。

模型主要特点：
- **分布式架构**：将流域划分为高分辨率网格单元，考虑空间异质性
- **物理机制驱动**：基于水力学和水文学物理方程，非经验性模拟
- **多单元类型**：区分边坡、河道、水库等不同下垫面单元
- **实时预报能力**：适用于洪水预警和水资源调度

### 研究意义

1. **洪水预报**：为流溪河流域提供实时洪水预报服务，保障下游广州市防洪安全
2. **水资源管理**：支持流域水资源优化配置和调度决策
3. **气候变化适应**：评估气候变化对流域水资源的影响
4. **技术推广**：为其他中小流域提供模型构建和技术参考

---

## 模型原理

### 1. 分布式网格架构

模型采用**网格离散化**方法，将流域划分为多个计算单元（网格），每个网格具有独立的空间属性（高程、坡度、土地利用、土壤类型等）。

```
流域网格划分示意图：

┌─────────────────────────────────────────┐
│ 边坡单元(US) │ 边坡单元(US) │ 边坡单元 │ 边坡单元 │
├─────────────────────────────────────────┤
│ 边坡单元(US) │ 河道单元(RN) │ 边坡单元 │ 边坡单元 │
├─────────────────────────────────────────┤
│ 边坡单元(US) │ 河道单元(RN) │ 边坡单元 │ 边坡单元 │
├─────────────────────────────────────────┤
│ 水库单元(RV) │ 河道单元(RN) │ 边坡单元(US) │ 出口 │
└─────────────────────────────────────────┘

US: 边坡单元 (Hillslope) - 产流 + 坡面汇流
RN: 河道单元 (River) - 河道汇流
RV: 水库单元 (Reservoir) - 水库调洪演算
```

模型网格参数：
- 网格数量：129 × 169 = 21,801 个
- 有效网格：约 10,905 个（扣除边界和NODATA区域）
- 网格分辨率：约 90 米

### 2. 单元分类算法

基于累积流阈值和 Strahler 河流分级方法，将网格单元分为三类：

| 单元类型 | 累积流阈值 | 特征 |
|---------|-----------|------|
| 边坡单元(US) | acc < threshold | 坡面水文过程 |
| 河道单元(RN) | acc >= threshold 且非水库 | 河道水流演进 |
| 水库单元(RV) | 根据土地利用确定 | 水库调洪演算 |

**D8流向算法**：使用8方向水流流向确定每个网格的水流方向

```
D8流向编码：
  7  6  5
   \ | /
  8--0--4
   / | \
  1  2  3
```

### 3. 产流计算 - 蓄满产流模型

蓄满产流是我国南方湿润地区广泛使用的水文模型，适用于雨量充沛、地下水位较高的流域。

```
降雨输入 → 蒸散发计算 → 土壤含水量更新 → 蓄满判据 → 产流量
                                    ↓
                             壤中流(Campbell公式)
                                    ↓
                             深层渗漏 → 地下水补给
```

**水量平衡方程**：
$$P - E = \Delta W + R$$

其中：
- $P$：降雨量
- $E$：蒸散发量
- $\Delta W$：土壤含水量变化
- $R$：产流量

**蓄满判据**：当土壤含水量达到田间持水量时，降雨全部转化为产流

### 4. 坡面汇流 - 运动波模型

坡面汇流采用**运动波近似**（Kinematic Wave Approximation）模拟坡面水流演进。

**运动波方程**：
$$Q = \alpha \cdot h^{\beta}$$

其中：
- $Q$：流量 (m³/s)
- $h$：水深 (m)
- $\alpha$：与坡度、糙率相关的系数
- $\beta$：指数（通常取 5/3，基于曼宁公式）

**圣维南方程简化**：
$$\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = i$$

其中：
- $h$：水深
- $t$：时间
- $q$：单宽流量
- $x$：空间距离
- $i$：净雨强度

### 5. 河道汇流 - 扩散波模型

河道汇流采用**扩散波近似**（Diffusive Wave Approximation），考虑水面比降和槽蓄作用。

**扩散波方程**：
$$\frac{\partial Q}{\partial t} + c_k \frac{\partial Q}{\partial x} = D \frac{\partial^2 Q}{\partial x^2}$$

其中：
- $c_k$： kinematic wave celerity
- $D$：扩散系数

**梯形断面河道**：
```
     B (河底宽度)
   ┌───┐
   │   │  m·H (边坡投影)
  /     \
 /       \
─────────────
     B + 2mH  (水面宽度)
```

采用**牛顿迭代法**求解非线性方程组。

### 6. 水库调洪演算

对于流域内的水库单元，采用**水量平衡方程**进行调洪演算：

$$(Q_{in,1} + Q_{in,2}) \frac{\Delta t}{2} - (Q_{out,1} + Q_{out,2}) \frac{\Delta t}{2} = S_2 - S_1$$

其中：
- $Q_{in}$：入库流量
- $Q_{out}$：出库流量
- $S$：水库蓄量
- $\Delta t$：时间步长

出库流量与水库蓄量/水位的关系由水库特征曲线确定。

### 7. 汇流网络拓扑

基于D8流向算法构建汇流网络，采用**拓扑排序**保证"从上游到下游"的计算顺序：

```
原始网格顺序：
  ③ ──→ ② ──→ ①
         ↓
         ④ ──→ ⑤ ──→ 出口

拓扑排序后（升序，上游优先）：
  ③ → ② → ① → ④ → ⑤ → 出口

计算流程：
1. 首先计算最上游网格的产流量
2. 将上游出流作为下游入流的一部分
3. 依次类推，直到流域出口
```

### 8. 地下水模型

采用**线性水库**方法模拟地下水径流：

$$Q_g = k_g \cdot S_g$$

其中：
- $Q_g$：地下径流量
- $k_g$：地下水消退系数
- $S_g$：地下水储量

---

## 关于本项目

> **郑重声明**
>
> 本项目是流溪河模型的 **Python 非官方重构版本**，旨在学习和研究分布式水文模型的数值计算方法。我们对陈洋波教授及其团队多年来的辛勤付出和卓越贡献表示由衷的敬意和感谢。原模型的研发凝聚了团队大量的心血和智慧，本重构版本仅作为学术交流和技术探索之用。
>
> 如需使用流溪河模型进行正式研究或生产应用，请访问中山大学水灾害管理与水利信息化实验室获取官方版本或授权。

### 重构初衷

1. **现代化开发**：将原有 Fortran 代码重构为 Python，便于学术研究者快速上手和二次开发
2. **模块化设计**：采用面向对象的模块化架构，提高代码可读性和可维护性
3. **可视化支持**：集成 Matplotlib 实现实时可视化，方便结果分析和模型调试
4. **参数率定工具**：开发自动率定模块，支持多参数优化和敏感性分析
5. **跨平台运行**：Python 的跨平台特性使模型可在 Windows、Linux、MacOS 等系统运行

### 与原版的关系

| 对比项 | 原版 (Fortran) | 本重构版 (Python) |
|--------|---------------|------------------|
| 开发语言 | Fortran 77/90 | Python 3.9+ |
| 代码结构 | 过程式 | 面向对象 |
| 依赖环境 | Fortran 编译器 | NumPy, Pandas |
| 可视化 | 外部工具 | 内置 Matplotlib |
| 运行效率 | 较高 | 适中（可通过 NumPy 优化） |

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
2. D8流向算法 - O'Callaghan, J.F. & Mark, D.M. (1984). The extraction of drainage networks from digital elevation data
3. Strahler河流分级 - Strahler, A.N. (1957). Quantitative analysis of watershed geomorphology
4. 运动波方程 - Lighthill, M.J. & Whitham, G.B. (1955). On kinematic waves
5. 扩散波近似 - Ponce, V.M. (1990). Generalized propagation of flood waves
6. Campbell公式 - Campbell, G.S. (1974). A simple method for determining unsaturated conductivity from moisture retention data

---

## 许可证

MIT License
