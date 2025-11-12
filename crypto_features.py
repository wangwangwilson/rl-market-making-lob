"""
加密货币数据特征工程模块
计算市场做市策略需要的各种特征
"""
import pandas as pd
import numpy as np
from typing import Optional


class CryptoFeatureEngine:
    """加密货币特征工程引擎"""
    
    @staticmethod
    def calculate_midprice(df: pd.DataFrame) -> pd.Series:
        """计算中间价"""
        return (df['Bid Price 1'] + df['Ask Price 1']) / 2
    
    @staticmethod
    def calculate_spread(df: pd.DataFrame) -> pd.Series:
        """计算买卖价差"""
        return df['Ask Price 1'] - df['Bid Price 1']
    
    @staticmethod
    def calculate_log_return(midprice: pd.Series, periods: int = 1) -> pd.Series:
        """计算对数收益率"""
        return np.log(midprice / midprice.shift(periods))
    
    @staticmethod
    def calculate_realized_volatility(log_returns: pd.Series, window: int = 300) -> pd.Series:
        """
        计算已实现波动率
        
        Args:
            log_returns: 对数收益率序列
            window: 滚动窗口大小（如5分钟 = 300秒）
        """
        return log_returns.rolling(window=window).std()
    
    @staticmethod
    def calculate_rsi(midprice: pd.Series, window: int = 300) -> pd.Series:
        """
        计算相对强弱指标(RSI)
        
        Args:
            midprice: 中间价序列
            window: 计算窗口（如5分钟 = 300秒）
        """
        delta = midprice.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_order_size_imbalance(df: pd.DataFrame, levels: int = 1) -> pd.Series:
        """
        计算订单量失衡指标
        OSI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Args:
            df: 包含bid/ask数据的DataFrame
            levels: 使用的orderbook层数
        """
        bid_volume = sum(df[f'Bid Size {i}'] for i in range(1, levels + 1))
        ask_volume = sum(df[f'Ask Size {i}'] for i in range(1, levels + 1))
        
        total_volume = bid_volume + ask_volume
        osi = (bid_volume - ask_volume) / total_volume.replace(0, np.nan)
        return osi
    
    @staticmethod
    def calculate_weighted_midprice(df: pd.DataFrame, levels: int = 3) -> pd.Series:
        """
        计算加权中间价（考虑多层orderbook）
        
        Args:
            df: orderbook数据
            levels: 使用的层数
        """
        total_bid_volume = sum(df[f'Bid Size {i}'] for i in range(1, levels + 1))
        total_ask_volume = sum(df[f'Ask Size {i}'] for i in range(1, levels + 1))
        
        weighted_bid = sum(
            df[f'Bid Price {i}'] * df[f'Bid Size {i}']
            for i in range(1, levels + 1)
        ) / total_bid_volume.replace(0, np.nan)
        
        weighted_ask = sum(
            df[f'Ask Price {i}'] * df[f'Ask Size {i}']
            for i in range(1, levels + 1)
        ) / total_ask_volume.replace(0, np.nan)
        
        return (weighted_bid + weighted_ask) / 2
    
    @staticmethod
    def add_all_features(
        df: pd.DataFrame,
        rv_window: int = 300,
        rsi_window: int = 300,
        osi_levels: int = 1
    ) -> pd.DataFrame:
        """
        为orderbook数据添加所有特征
        
        Args:
            df: orderbook数据
            rv_window: 波动率计算窗口（秒数）
            rsi_window: RSI计算窗口（秒数）
            osi_levels: OSI使用的orderbook层数
            
        Returns:
            添加特征后的DataFrame
        """
        print("正在计算特征...")
        df = df.copy()
        
        # 基础特征
        df['midprice'] = CryptoFeatureEngine.calculate_midprice(df)
        df['spread'] = CryptoFeatureEngine.calculate_spread(df)
        df['log_return'] = CryptoFeatureEngine.calculate_log_return(df['midprice'])
        
        # 高级特征
        df['RV_5min'] = CryptoFeatureEngine.calculate_realized_volatility(
            df['log_return'], window=rv_window
        )
        df['RSI_5min'] = CryptoFeatureEngine.calculate_rsi(
            df['midprice'], window=rsi_window
        )
        df['OSI_10s'] = CryptoFeatureEngine.calculate_order_size_imbalance(
            df, levels=osi_levels
        )
        
        # 填充NaN值
        df = df.bfill()
        df = df.ffill()
        
        print(f"特征计算完成，总共 {len(df.columns)} 列")
        return df


def prepare_strategy_data(
    orderbook_file: str,
    output_file: str,
    num_levels: int = 5,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    准备策略所需的完整数据
    
    Args:
        orderbook_file: orderbook CSV文件路径
        output_file: 输出文件路径
        num_levels: orderbook层数
        max_rows: 最大处理行数
        
    Returns:
        处理后的DataFrame
    """
    print(f"正在加载数据: {orderbook_file}")
    df = pd.read_csv(orderbook_file, nrows=max_rows)
    
    # 验证必要列是否存在
    required_cols = ['Bid Price 1', 'Ask Price 1', 'Bid Size 1', 'Ask Size 1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")
    
    # 添加特征
    df = CryptoFeatureEngine.add_all_features(df)
    
    # 选择策略需要的列
    feature_cols = [
        'Bid Price 1', 'Bid Size 1', 'Ask Price 1', 'Ask Size 1',
        'midprice', 'spread', 'log_return', 'RV_5min', 'RSI_5min', 'OSI_10s'
    ]
    
    # 添加更多层的数据（如果存在）
    for i in range(2, num_levels + 1):
        if f'Bid Price {i}' in df.columns:
            feature_cols.extend([
                f'Bid Price {i}', f'Bid Size {i}',
                f'Ask Price {i}', f'Ask Size {i}'
            ])
    
    df_output = df[feature_cols].copy()
    
    # 保存处理后的数据
    df_output.to_csv(output_file, index=False)
    print(f"策略数据已保存: {output_file}")
    print(f"数据形状: {df_output.shape}")
    
    return df_output


def create_scaled_data(
    df: pd.DataFrame,
    output_file: str,
    price_cols: Optional[list] = None,
    size_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    对数据进行标准化处理
    
    Args:
        df: 输入数据
        output_file: 输出文件路径
        price_cols: 价格列名列表
        size_cols: 数量列名列表
        
    Returns:
        标准化后的DataFrame
    """
    from sklearn.preprocessing import StandardScaler
    
    df_scaled = df.copy()
    
    # 自动识别价格和数量列
    if price_cols is None:
        price_cols = [col for col in df.columns if 'Price' in col or col == 'midprice']
    if size_cols is None:
        size_cols = [col for col in df.columns if 'Size' in col]
    
    # 标准化价格列
    if price_cols:
        scaler_price = StandardScaler()
        df_scaled[price_cols] = scaler_price.fit_transform(df[price_cols])
    
    # 标准化数量列（使用对数变换）
    if size_cols:
        for col in size_cols:
            df_scaled[col] = np.log1p(df[col])
        scaler_size = StandardScaler()
        df_scaled[size_cols] = scaler_size.fit_transform(df_scaled[size_cols])
    
    # 保存
    df_scaled.to_csv(output_file, index=False)
    print(f"标准化数据已保存: {output_file}")
    
    return df_scaled
