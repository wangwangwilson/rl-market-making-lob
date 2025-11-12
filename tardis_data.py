"""
Tardis加密货币数据获取和处理模块
支持下载高频orderbook数据并转换为策略需要的格式
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import gzip
import json
from pathlib import Path


class TardisDataFetcher:
    """Tardis数据获取器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.tardis.dev/v1"):
        """
        初始化Tardis数据获取器
        
        Args:
            api_key: Tardis API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    def get_exchanges(self) -> List[str]:
        """获取支持的交易所列表"""
        url = f"{self.base_url}/exchanges"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def download_orderbook_snapshot(
        self,
        exchange: str,
        symbol: str,
        date: str,
        data_type: str = "book_snapshot_25",
        output_dir: str = "./data"
    ) -> str:
        """
        下载orderbook快照数据
        
        Args:
            exchange: 交易所名称（如'binance'）
            symbol: 交易对（如'BTCUSDT'）
            date: 日期（YYYY-MM-DD格式）
            data_type: 数据类型（book_snapshot_25表示25档orderbook）
            output_dir: 输出目录
            
        Returns:
            下载的文件路径
        """
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 构建下载URL
        url = f"{self.base_url}/datasets/{exchange}/{data_type}"
        params = {
            "symbols": symbol,
            "from": date,
            "to": date
        }
        
        print(f"正在下载 {exchange} {symbol} {date} 的orderbook数据...")
        response = requests.get(url, headers=self.headers, params=params, stream=True)
        response.raise_for_status()
        
        # 保存文件
        filename = f"{exchange}_{symbol}_{date}_{data_type}.csv.gz"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"下载完成: {filepath}")
        return filepath


class OrderBookDataProcessor:
    """订单簿数据处理器"""
    
    @staticmethod
    def load_tardis_orderbook(filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        加载Tardis orderbook数据
        
        Args:
            filepath: 数据文件路径（支持.csv.gz或.csv）
            max_rows: 最大读取行数（用于测试）
            
        Returns:
            DataFrame包含orderbook数据
        """
        print(f"正在加载数据: {filepath}")
        
        # 根据文件扩展名决定是否解压
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, compression='gzip', nrows=max_rows)
        else:
            df = pd.read_csv(filepath, nrows=max_rows)
        
        print(f"加载完成，数据形状: {df.shape}")
        return df
    
    @staticmethod
    def convert_to_strategy_format(
        df: pd.DataFrame,
        num_levels: int = 5
    ) -> pd.DataFrame:
        """
        将Tardis orderbook格式转换为策略需要的格式
        
        Tardis格式通常包含:
        - timestamp: 时间戳
        - bids: [[price, size], ...]
        - asks: [[price, size], ...]
        
        策略需要格式:
        - Time: 时间戳
        - Bid Price 1-N, Bid Size 1-N
        - Ask Price 1-N, Ask Size 1-N
        
        Args:
            df: Tardis原始数据
            num_levels: 需要的orderbook层数
            
        Returns:
            转换后的DataFrame
        """
        print(f"正在转换数据格式，目标层数: {num_levels}")
        
        result_data = []
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"处理进度: {idx}/{len(df)}")
            
            record = {'Time': row.get('timestamp', row.get('local_timestamp', idx))}
            
            # 解析bids和asks（可能是JSON字符串）
            bids = row.get('bids', [])
            asks = row.get('asks', [])
            
            if isinstance(bids, str):
                bids = json.loads(bids)
            if isinstance(asks, str):
                asks = json.loads(asks)
            
            # 提取指定层数的bid和ask数据
            for i in range(num_levels):
                level = i + 1
                
                # Bid数据（按价格降序）
                if i < len(bids):
                    record[f'Bid Price {level}'] = float(bids[i][0])
                    record[f'Bid Size {level}'] = float(bids[i][1])
                else:
                    record[f'Bid Price {level}'] = np.nan
                    record[f'Bid Size {level}'] = 0
                
                # Ask数据（按价格升序）
                if i < len(asks):
                    record[f'Ask Price {level}'] = float(asks[i][0])
                    record[f'Ask Size {level}'] = float(asks[i][1])
                else:
                    record[f'Ask Price {level}'] = np.nan
                    record[f'Ask Size {level}'] = 0
            
            result_data.append(record)
        
        result_df = pd.DataFrame(result_data)
        
        # 转换时间戳
        if 'Time' in result_df.columns:
            result_df['Time'] = pd.to_datetime(result_df['Time'], unit='ms', errors='coerce')
        
        print(f"格式转换完成，输出形状: {result_df.shape}")
        return result_df


def create_test_dataset(
    api_key: str,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    date: str = None,
    num_levels: int = 5,
    max_rows: int = 10000,
    output_dir: str = "./data"
) -> pd.DataFrame:
    """
    创建测试数据集的便捷函数
    
    Args:
        api_key: Tardis API密钥
        exchange: 交易所
        symbol: 交易对
        date: 日期（默认为昨天）
        num_levels: orderbook层数
        max_rows: 最大行数
        output_dir: 输出目录
        
    Returns:
        处理后的DataFrame
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 1. 下载数据
    fetcher = TardisDataFetcher(api_key)
    filepath = fetcher.download_orderbook_snapshot(
        exchange=exchange,
        symbol=symbol,
        date=date,
        output_dir=output_dir
    )
    
    # 2. 加载和处理数据
    processor = OrderBookDataProcessor()
    raw_df = processor.load_tardis_orderbook(filepath, max_rows=max_rows)
    
    # 3. 转换格式
    processed_df = processor.convert_to_strategy_format(raw_df, num_levels=num_levels)
    
    # 4. 保存处理后的数据
    output_file = os.path.join(output_dir, f"{exchange}_{symbol}_{date}_processed.csv")
    processed_df.to_csv(output_file, index=False)
    print(f"处理后的数据已保存: {output_file}")
    
    return processed_df
