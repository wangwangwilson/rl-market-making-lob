"""
Tardis真实数据下载和处理示例
展示如何下载和使用真实的加密货币高频数据
"""
import os
from datetime import datetime, timedelta
from tardis_data import TardisDataFetcher, OrderBookDataProcessor
from crypto_features import prepare_strategy_data, create_scaled_data


# Tardis API配置
TARDIS_API_KEY = "TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4"


def download_binance_btc_data(date_str: str = None, max_rows: int = 50000):
    """
    下载Binance BTC/USDT数据示例
    
    Args:
        date_str: 日期字符串 (YYYY-MM-DD)，默认为昨天
        max_rows: 最大处理行数
    """
    if date_str is None:
        # 默认下载昨天的数据
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print("=" * 70)
    print(f"下载Binance BTC/USDT数据 - {date_str}")
    print("=" * 70)
    
    # 配置
    exchange = "binance"
    symbol = "BTCUSDT"
    data_dir = "./tardis_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. 下载数据
    print(f"\n[步骤1] 从Tardis下载数据...")
    fetcher = TardisDataFetcher(TARDIS_API_KEY)
    
    try:
        raw_file = fetcher.download_orderbook_snapshot(
            exchange=exchange,
            symbol=symbol,
            date=date_str,
            data_type="book_snapshot_25",  # 25档orderbook
            output_dir=data_dir
        )
        print(f"✓ 下载完成: {raw_file}")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. API密钥无效")
        print("2. 该日期数据不可用")
        print("3. 网络连接问题")
        print("\n请检查后重试，或使用模拟数据进行测试")
        return None
    
    # 2. 加载和转换数据
    print(f"\n[步骤2] 处理数据...")
    processor = OrderBookDataProcessor()
    
    # 加载原始数据（限制行数以加快处理）
    raw_df = processor.load_tardis_orderbook(raw_file, max_rows=max_rows)
    
    # 转换为策略格式
    processed_df = processor.convert_to_strategy_format(raw_df, num_levels=5)
    
    # 保存处理后的数据
    processed_file = os.path.join(data_dir, f"{exchange}_{symbol}_{date_str}_processed.csv")
    processed_df.to_csv(processed_file, index=False)
    print(f"✓ 处理后数据保存: {processed_file}")
    
    # 3. 计算特征
    print(f"\n[步骤3] 计算策略特征...")
    strategy_file = os.path.join(data_dir, f"{exchange}_{symbol}_{date_str}_strategy.csv")
    df_strategy = prepare_strategy_data(
        orderbook_file=processed_file,
        output_file=strategy_file,
        num_levels=5,
        max_rows=max_rows
    )
    
    # 4. 数据标准化
    print(f"\n[步骤4] 数据标准化...")
    scaled_file = os.path.join(data_dir, f"{exchange}_{symbol}_{date_str}_scaled.csv")
    df_scaled = create_scaled_data(df_strategy, scaled_file)
    
    # 5. 数据摘要
    print(f"\n" + "=" * 70)
    print("数据下载和处理完成！")
    print("=" * 70)
    print(f"\n数据摘要:")
    print(f"  - 原始数据: {raw_file}")
    print(f"  - 处理后数据: {processed_file}")
    print(f"  - 策略数据: {strategy_file}")
    print(f"  - 标准化数据: {scaled_file}")
    print(f"\n数据形状: {df_scaled.shape}")
    print(f"数据列: {df_scaled.columns.tolist()}")
    
    print(f"\n数据统计:")
    print(df_scaled[['midprice', 'spread', 'RV_5min', 'RSI_5min', 'OSI_10s']].describe())
    
    print(f"\n下一步:")
    print(f"1. 检查数据质量")
    print(f"2. 运行回测: python3 run_crypto_backtest.py")
    print(f"3. 分析结果")
    
    return scaled_file


def download_multiple_days(
    start_date: str,
    num_days: int = 7,
    exchange: str = "binance",
    symbol: str = "BTCUSDT"
):
    """
    下载多天数据
    
    Args:
        start_date: 起始日期 (YYYY-MM-DD)
        num_days: 下载天数
        exchange: 交易所
        symbol: 交易对
    """
    print("=" * 70)
    print(f"批量下载 {num_days} 天数据")
    print("=" * 70)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    files = []
    
    for i in range(num_days):
        date = start + timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        
        print(f"\n下载第 {i+1}/{num_days} 天: {date_str}")
        try:
            file = download_binance_btc_data(date_str)
            if file:
                files.append(file)
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print(f"批量下载完成！成功: {len(files)}/{num_days}")
    print("=" * 70)
    
    return files


def test_other_exchanges():
    """测试其他交易所数据下载"""
    
    fetcher = TardisDataFetcher(TARDIS_API_KEY)
    
    # 获取支持的交易所列表
    print("获取支持的交易所...")
    try:
        exchanges = fetcher.get_exchanges()
        print(f"\nTardis支持 {len(exchanges)} 个交易所")
        print("部分交易所列表:")
        for ex in exchanges[:10]:
            print(f"  - {ex}")
    except Exception as e:
        print(f"获取交易所列表失败: {e}")
    
    # 测试其他交易所
    test_configs = [
        ("binance-futures", "BTCUSDT-PERP"),
        ("coinbase", "BTC-USD"),
        ("kraken", "XBT/USD")
    ]
    
    date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    for exchange, symbol in test_configs:
        print(f"\n测试 {exchange} - {symbol}")
        try:
            file = fetcher.download_orderbook_snapshot(
                exchange=exchange,
                symbol=symbol,
                date=date_str,
                output_dir="./tardis_test"
            )
            print(f"✓ 下载成功: {file}")
        except Exception as e:
            print(f"✗ 下载失败: {e}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("Tardis真实数据下载示例")
    print("=" * 70)
    
    # 菜单
    print("\n请选择操作:")
    print("1. 下载单日BTC数据（推荐）")
    print("2. 下载多日BTC数据")
    print("3. 测试其他交易所")
    print("4. 退出")
    
    try:
        choice = input("\n输入选项 (1-4): ").strip()
        
        if choice == "1":
            # 下载单日数据
            date_str = input("输入日期 (YYYY-MM-DD, 回车使用昨天): ").strip()
            if not date_str:
                date_str = None
            
            scaled_file = download_binance_btc_data(date_str, max_rows=50000)
            
            if scaled_file:
                print(f"\n✓ 数据已准备好，可以运行回测了:")
                print(f"  python3 run_crypto_backtest.py")
        
        elif choice == "2":
            # 下载多日数据
            start_date = input("输入起始日期 (YYYY-MM-DD): ").strip()
            num_days = int(input("输入天数 (默认7): ").strip() or "7")
            
            files = download_multiple_days(start_date, num_days)
            print(f"\n✓ 成功下载 {len(files)} 天数据")
        
        elif choice == "3":
            # 测试其他交易所
            test_other_exchanges()
        
        elif choice == "4":
            print("退出")
            sys.exit(0)
        
        else:
            print("无效选项")
    
    except KeyboardInterrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n如果遇到API问题，可以先使用 test_crypto_data.py 进行测试")
