"""
加密货币数据集成测试脚本
验证Tardis数据获取、处理和策略运行
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入自定义模块
from tardis_data import TardisDataFetcher, OrderBookDataProcessor, create_test_dataset
from crypto_features import CryptoFeatureEngine, prepare_strategy_data, create_scaled_data


def test_data_pipeline(api_key: str, use_sample_data: bool = False):
    """
    测试完整的数据处理流程
    
    Args:
        api_key: Tardis API密钥
        use_sample_data: 是否使用样本数据（测试时可跳过实际下载）
    """
    print("=" * 60)
    print("开始测试加密货币数据处理流程")
    print("=" * 60)
    
    # 配置
    exchange = "binance"
    symbol = "BTCUSDT"
    date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    data_dir = "./crypto_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 步骤1: 获取少量测试数据
    if not use_sample_data:
        print("\n[步骤1] 下载少量Tardis数据进行测试...")
        try:
            df_raw = create_test_dataset(
                api_key=api_key,
                exchange=exchange,
                symbol=symbol,
                date=date,
                num_levels=5,
                max_rows=5000,  # 仅测试5000行
                output_dir=data_dir
            )
        except Exception as e:
            print(f"下载失败: {e}")
            print("将创建模拟数据进行测试...")
            use_sample_data = True
    
    # 使用模拟数据
    if use_sample_data:
        print("\n[步骤1] 创建模拟orderbook数据...")
        df_raw = create_sample_orderbook_data(5000)
        sample_file = os.path.join(data_dir, "sample_orderbook.csv")
        df_raw.to_csv(sample_file, index=False)
        print(f"模拟数据已保存: {sample_file}")
    
    # 步骤2: 验证数据格式
    print("\n[步骤2] 验证数据格式...")
    print(f"数据形状: {df_raw.shape}")
    print(f"列名: {df_raw.columns.tolist()}")
    print(f"\n前5行数据:")
    print(df_raw.head())
    
    # 验证必要列
    required_cols = ['Bid Price 1', 'Ask Price 1', 'Bid Size 1', 'Ask Size 1']
    missing = [col for col in required_cols if col not in df_raw.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    print("✓ 数据格式验证通过")
    
    # 步骤3: 添加特征
    print("\n[步骤3] 计算策略特征...")
    df_features = CryptoFeatureEngine.add_all_features(
        df_raw,
        rv_window=100,  # 测试用较小窗口
        rsi_window=100,
        osi_levels=1
    )
    
    print(f"特征计算完成，新增列:")
    new_cols = [col for col in df_features.columns if col not in df_raw.columns]
    print(new_cols)
    
    # 检查是否有异常值
    print("\n特征统计信息:")
    feature_cols = ['midprice', 'spread', 'log_return', 'RV_5min', 'RSI_5min', 'OSI_10s']
    print(df_features[feature_cols].describe())
    
    # 步骤4: 准备策略数据
    print("\n[步骤4] 准备策略所需数据...")
    strategy_file = os.path.join(data_dir, "strategy_data.csv")
    df_strategy = prepare_strategy_data(
        orderbook_file=os.path.join(data_dir, "sample_orderbook.csv") if use_sample_data 
                      else os.path.join(data_dir, f"{exchange}_{symbol}_{date}_processed.csv"),
        output_file=strategy_file,
        num_levels=5,
        max_rows=5000
    )
    
    print("✓ 策略数据准备完成")
    
    # 步骤5: 数据标准化
    print("\n[步骤5] 进行数据标准化...")
    scaled_file = os.path.join(data_dir, "scaled_data.csv")
    df_scaled = create_scaled_data(df_strategy, scaled_file)
    
    print("✓ 数据标准化完成")
    
    # 步骤6: 数据质量检查
    print("\n[步骤6] 数据质量检查...")
    check_data_quality(df_scaled)
    
    print("\n" + "=" * 60)
    print("数据处理流程测试完成！")
    print("=" * 60)
    
    return df_scaled


def create_sample_orderbook_data(num_rows: int = 5000) -> pd.DataFrame:
    """
    创建模拟orderbook数据用于测试
    
    Args:
        num_rows: 数据行数
        
    Returns:
        模拟的orderbook DataFrame
    """
    np.random.seed(42)
    
    # 模拟BTC价格在40000附近波动
    base_price = 40000
    price_volatility = 100
    
    data = {
        'Time': pd.date_range(start='2024-01-01', periods=num_rows, freq='1s')
    }
    
    # 生成价格随机游走
    midprice = base_price + np.cumsum(np.random.randn(num_rows) * price_volatility / 100)
    
    # 生成5层orderbook
    for level in range(1, 6):
        spread = 0.01 * base_price * level  # 价差随层数增加
        
        # Ask价格和数量
        data[f'Ask Price {level}'] = midprice + spread / 2 + (level - 1) * 5
        data[f'Ask Size {level}'] = np.random.uniform(0.1, 10, num_rows)
        
        # Bid价格和数量
        data[f'Bid Price {level}'] = midprice - spread / 2 - (level - 1) * 5
        data[f'Bid Size {level}'] = np.random.uniform(0.1, 10, num_rows)
    
    df = pd.DataFrame(data)
    return df


def check_data_quality(df: pd.DataFrame):
    """检查数据质量"""
    print("检查数据质量...")
    
    # 检查缺失值
    missing_count = df.isnull().sum()
    if missing_count.sum() > 0:
        print(f"⚠ 发现缺失值:\n{missing_count[missing_count > 0]}")
    else:
        print("✓ 无缺失值")
    
    # 检查无穷值
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_count.sum() > 0:
        print(f"⚠ 发现无穷值:\n{inf_count[inf_count > 0]}")
    else:
        print("✓ 无无穷值")
    
    # 检查价差是否为正
    if 'spread' in df.columns:
        negative_spreads = (df['spread'] < 0).sum()
        if negative_spreads > 0:
            print(f"⚠ 发现{negative_spreads}个负价差")
        else:
            print("✓ 价差全部为正")
    
    print("数据质量检查完成")


def test_with_strategy(data_file: str):
    """
    使用加密货币数据测试策略
    
    Args:
        data_file: 处理好的数据文件路径
    """
    print("\n" + "=" * 60)
    print("测试策略运行")
    print("=" * 60)
    
    # 导入策略环境
    from main_code import EnhancedMarketMakingEnv, RewardConfig, LSTMPolicyNetwork
    import torch
    
    # 加载数据
    print(f"加载数据: {data_file}")
    data = pd.read_csv(data_file)
    
    # 选择少量数据进行快速测试
    data_subset = data.iloc[:1000]
    print(f"使用数据子集: {data_subset.shape}")
    
    # 创建环境
    print("\n创建市场做市环境...")
    env = EnhancedMarketMakingEnv(
        data=data_subset,
        max_episode_steps=500,
        reward_config=RewardConfig(eta=0.3, zeta=0.005, transaction_cost=0.0001),
        prob_execution=True,
        use_path_signatures=False
    )
    
    print(f"观测空间维度: {env.observation_space.shape}")
    print(f"动作空间大小: {env.action_space.n}")
    
    # 测试环境重置和步进
    print("\n测试环境交互...")
    obs, info = env.reset()
    print(f"初始观测形状: {obs.shape}")
    
    # 执行几步随机动作
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            print(f"回合在第{step + 1}步结束")
            break
    
    print(f"✓ 环境测试通过，累计奖励: {total_reward:.4f}")
    
    # 测试策略网络
    print("\n测试策略网络...")
    state_dim = env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = LSTMPolicyNetwork(state_dim, hidden_dim=64, output_dim=9, lstm_layers=1)
    policy.to(device)
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
    action_probs, state_value = policy(obs_tensor)
    
    print(f"动作概率形状: {action_probs.shape}")
    print(f"状态价值形状: {state_value.shape}")
    print("✓ 策略网络测试通过")
    
    print("\n" + "=" * 60)
    print("策略集成测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # Tardis API配置
    TARDIS_API_KEY = "TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4"
    
    # 运行测试（先用模拟数据）
    print("使用模拟数据进行测试...")
    df_scaled = test_data_pipeline(TARDIS_API_KEY, use_sample_data=True)
    
    # 测试与策略的集成
    test_with_strategy("./crypto_data/scaled_data.csv")
    
    print("\n所有测试完成！")
