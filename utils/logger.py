import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str = 'liuxihe_model',
                level: str = 'INFO',
                log_file: Optional[str] = None) -> logging.Logger:
    """配置日志系统"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'liuxihe_model') -> logging.Logger:
    """获取已配置的logger"""
    return logging.getLogger(name)


class ProgressLogger:
    """进度日志记录器"""

    def __init__(self, total: int, logger: Optional[logging.Logger] = None):
        self.total = total
        self.current = 0
        self.logger = logger or get_logger()
        self.start_time = datetime.now()

    def update(self, step: int = 1, info: str = ''):
        """更新进度"""
        self.current += step
        pct = self.current / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if elapsed > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            self.logger.info(
                f"Progress: {self.current}/{self.total} ({pct:.1f}%) "
                f"Rate: {rate:.1f} it/s ETA: {eta:.0f}s {info}"
            )
        else:
            self.logger.info(f"Progress: {self.current}/{self.total} ({pct:.1f}%) {info}")

    def finish(self):
        """完成记录"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"Completed: {self.total}/{self.total} (100%) "
            f"Total time: {elapsed:.1f}s"
        )
