import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd


def read_asc(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    读取ESRI ASCII Grid格式数据

    Args:
        filepath: ASC文件路径

    Returns:
        (data, header): 数据数组和头部信息
    """
    header = {}
    data_list = []

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i < 6:
                parts = line.split()
                if len(parts) == 2:
                    val = parts[1]
                    try:
                        header[parts[0]] = int(val)
                    except ValueError:
                        try:
                            header[parts[0]] = float(val)
                        except ValueError:
                            header[parts[0]] = val
            else:
                data_list.append([float(x) for x in line.split()])

    data = np.array(data_list)

    if header.get('NODATA_value'):
        nodata = header['NODATA_value']
        data = np.where(data == nodata, np.nan, data)

    return data, header


def write_asc(filepath: str, data: np.ndarray, header: Dict):
    """
    写入ESRI ASCII Grid格式

    Args:
        filepath: 输出路径
        data: 数据数组
        header: 头部信息
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"ncols         {header.get('ncols', data.shape[1])}\n")
        f.write(f"nrows         {header.get('nrows', data.shape[0])}\n")
        f.write(f"xllcorner     {header.get('xllcorner', 0)}\n")
        f.write(f"yllcorner     {header.get('yllcorner', 0)}\n")
        f.write(f"cellsize      {header.get('cellsize', 1)}\n")
        nodata = header.get('NODATA_value', -9999)
        f.write(f"NODATA_value  {nodata}\n")

        data_out = data.copy()
        data_out[np.isnan(data_out)] = nodata

        for row in data_out:
            f.write(' '.join(f'{v:.6f}' for v in row) + '\n')


def read_flood_data(filepath: str) -> Tuple[pd.DataFrame, Dict]:
    """
    读取洪水数据文件

    格式:
    第1行: 开始时间 结束时间 时间步长(分钟) 数据点数 出口数
    第2行: ID, 水位, X地址, Y地址, 流量

    Args:
        filepath: 洪水数据文件路径

    Returns:
        (data, header): 数据DataFrame和头部信息
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_line = lines[0].strip().split('\t')
    header = {
        'start_time': header_line[0],
        'end_time': header_line[1],
        'dt_minutes': int(header_line[2]),
        'n_points': int(header_line[3]),
        'n_outlets': int(header_line[4])
    }

    data_lines = lines[2:]

    records = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                records.append({
                    'ID': int(parts[0]),
                    'stage': float(parts[1]),
                    'X': int(parts[2]),
                    'Y': int(parts[3]),
                    'discharge': float(parts[4])
                })
            except ValueError:
                continue

    df = pd.DataFrame(records)

    return df, header


def read_station_info(filepath: str) -> pd.DataFrame:
    """读取站点信息"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_line = lines[1].strip().split('\t')
    expected_cols = len(header_line)

    records = []
    for line in lines[2:]:
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            try:
                records.append({
                    'ID': int(parts[0]),
                    'Name': parts[1],
                    'SID': parts[2],
                    'X': float(parts[3]),
                    'Y': float(parts[4]),
                    'EnName': parts[5] if len(parts) > 5 else ''
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(records)


def read_channel_params(filepath: str) -> pd.DataFrame:
    """读取河道参数"""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip()
    return df


def read_soil_params(filepath: str) -> pd.DataFrame:
    """读取土壤参数"""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip()
    return df


def read_landuse_params(filepath: str) -> pd.DataFrame:
    """读取土地利用参数"""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip()
    return df


def detect_csv_columns(filepath: str, delimiter: str = ',') -> Dict[str, list]:
    """
    检测CSV文件的列名和样本数据

    Args:
        filepath: CSV文件路径
        delimiter: 分隔符

    Returns:
        包含列名和样本数据的字典
    """
    df = pd.read_csv(filepath, nrows=5, delimiter=delimiter)
    columns = list(df.columns)
    sample_data = {}
    for col in columns:
        sample_data[col] = df[col].tolist()
    return {
        'columns': columns,
        'sample_data': sample_data,
        'n_rows': len(pd.read_csv(filepath, delimiter=delimiter))
    }


def read_timeseries_csv(
    filepath: str,
    time_col: Optional[str] = None,
    rainfall_col: Optional[str] = None,
    evaporation_col: Optional[str] = None,
    runoff_col: Optional[str] = None,
    delimiter: str = ',',
    datetime_format: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    读取时间序列CSV文件，支持手动指定列名

    Args:
        filepath: CSV文件路径
        time_col: 时间列名 (可选，自动检测)
        rainfall_col: 降雨列名 (可选)
        evaporation_col: 蒸发列名 (可选)
        runoff_col: 径流列名 (可选)
        delimiter: 分隔符
        datetime_format: 时间格式

    Returns:
        (data, info): DataFrame和附加信息
    """
    df = pd.read_csv(filepath, delimiter=delimiter)

    columns = list(df.columns)
    column_lower = [c.lower() for c in columns]

    info = {
        'columns': columns,
        'detected': {}
    }

    if time_col is None:
        time_candidates = ['time', 'datetime', 'date', 'timestamp', '时间', '日期']
        for cand in time_candidates:
            if cand.lower() in column_lower:
                idx = column_lower.index(cand.lower())
                time_col = columns[idx]
                info['detected']['time'] = time_col
                break

    if rainfall_col is None:
        rain_candidates = ['rainfall', 'rain', 'precipitation', 'p', '降雨', '降水', '雨量']
        for cand in rain_candidates:
            if cand.lower() in column_lower:
                idx = column_lower.index(cand.lower())
                rainfall_col = columns[idx]
                info['detected']['rainfall'] = rainfall_col
                break

    if evaporation_col is None:
        evap_candidates = ['evaporation', 'evap', 'et', 'pet', '蒸发', 'etp']
        for cand in evap_candidates:
            if cand.lower() in column_lower:
                idx = column_lower.index(cand.lower())
                evaporation_col = columns[idx]
                info['detected']['evaporation'] = evaporation_col
                break

    if runoff_col is None:
        runoff_candidates = ['runoff', 'discharge', 'q', 'flow', '径流', '流量']
        for cand in runoff_candidates:
            if cand.lower() in column_lower:
                idx = column_lower.index(cand.lower())
                runoff_col = columns[idx]
                info['detected']['runoff'] = runoff_col
                break

    if time_col and time_col in df.columns:
        if datetime_format:
            df[time_col] = pd.to_datetime(df[time_col], format=datetime_format)
        else:
            df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)

    result = pd.DataFrame()
    if time_col and time_col in df.columns:
        result['time'] = df[time_col]
    if rainfall_col and rainfall_col in df.columns:
        result['rainfall'] = df[rainfall_col].astype(float)
    if evaporation_col and evaporation_col in df.columns:
        result['evaporation'] = df[evaporation_col].astype(float)
    if runoff_col and runoff_col in df.columns:
        result['runoff'] = df[runoff_col].astype(float)

    info['time_col'] = time_col
    info['rainfall_col'] = rainfall_col
    info['evaporation_col'] = evaporation_col
    info['runoff_col'] = runoff_col

    return result, info


def print_column_selector_info(filepath: str, delimiter: str = ','):
    """
    打印CSV文件的列信息，用于手动选择列名

    Args:
        filepath: CSV文件路径
        delimiter: 分隔符
    """
    info = detect_csv_columns(filepath, delimiter)
    columns = info['columns']
    sample_data = info['sample_data']

    print(f"\n文件: {filepath}")
    print(f"总行数: {info['n_rows']}")
    print(f"\n可用列名 (共{len(columns)}列):")
    print("-" * 50)

    for i, col in enumerate(columns):
        samples = sample_data[col][:3]
        samples_str = ', '.join([str(s) for s in samples])
        print(f"  {i+1}. {col}")
        print(f"      样本: [{samples_str}]")

    print("-" * 50)
    print("\n自动检测结果:")
    print(f"  - 时间列: {info.get('detected', {}).get('time', '未检测到')}")
    print(f"  - 降雨列: {info.get('detected', {}).get('rainfall', '未检测到')}")
    print(f"  - 蒸发列: {info.get('detected', {}).get('evaporation', '未检测到')}")
    print(f"  - 径流列: {info.get('detected', {}).get('runoff', '未检测到')}")
