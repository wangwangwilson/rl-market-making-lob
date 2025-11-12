"""
加密货币市场做市策略完整回测脚本
使用少量真实数据进行训练和评估
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 导入模块
from main_code import (
    EnhancedMarketMakingEnv, 
    RewardConfig, 
    TrainingConfig,
    LSTMPolicyNetwork,
    MLPPolicyNetwork,
    actor_critic_train,
    test_and_visualize,
    baseline_fixed_spread_agent,
    baseline_avellaneda_stoikov_agent
)
from crypto_features import prepare_strategy_data, create_scaled_data


def run_crypto_backtest(
    data_file: str,
    output_dir: str = "./backtest_results",
    num_episodes: int = 20,
    max_episode_steps: int = 500
):
    """
    运行加密货币回测
    
    Args:
        data_file: 策略数据文件路径
        output_dir: 输出目录
        num_episodes: 训练回合数
        max_episode_steps: 每回合最大步数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("加密货币市场做市策略回测")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[步骤1] 加载和准备数据...")
    data = pd.read_csv(data_file)
    print(f"数据形状: {data.shape}")
    print(f"数据列: {data.columns.tolist()}")
    
    # 使用子集进行快速测试
    data_subset = data.iloc[:2000]
    print(f"使用数据子集: {data_subset.shape}")
    
    # 2. 配置
    reward_config = RewardConfig(
        eta=0.3,
        zeta=0.005,
        transaction_cost=0.0001
    )
    
    training_config = TrainingConfig(
        learning_rate=0.002,
        gamma=0.99,
        hidden_dim=64,  # 减小网络大小以加快训练
        use_lstm=True,
        lstm_layers=1,  # 减少LSTM层数
        entropy_coef=0.1,
        max_gradient_norm=0.5
    )
    
    # 3. 创建环境
    print("\n[步骤2] 创建市场做市环境...")
    env = EnhancedMarketMakingEnv(
        data=data_subset,
        max_episode_steps=max_episode_steps,
        reward_config=reward_config,
        prob_execution=True,
        use_path_signatures=False
    )
    
    print(f"观测空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.n}")
    
    # 4. 训练策略
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[步骤3] 训练策略 (设备: {device})...")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # LSTM策略
    print("\n训练LSTM策略...")
    lstm_policy = LSTMPolicyNetwork(
        state_dim, 
        training_config.hidden_dim, 
        action_dim, 
        training_config.lstm_layers
    )
    lstm_policy.to(device)
    lstm_optimizer = Adam(lstm_policy.parameters(), lr=training_config.learning_rate)
    
    lstm_rewards = actor_critic_train(
        env, lstm_policy, lstm_optimizer, num_episodes,
        gamma=training_config.gamma,
        entropy_coef=training_config.entropy_coef,
        max_grad_norm=training_config.max_gradient_norm,
        use_gae=training_config.use_gae,
        gae_lambda=training_config.gae_lambda,
        device=device
    )
    
    # 5. 评估策略
    print("\n[步骤4] 评估训练好的策略...")
    lstm_results = test_and_visualize(env, lstm_policy, num_episodes=10, device=device)
    
    # 6. 评估基准策略
    print("\n[步骤5] 评估基准策略...")
    print("评估Fixed Spread策略...")
    fixed_results = baseline_fixed_spread_agent(
        env, 
        fixed_bid_offset=-1.0, 
        fixed_ask_offset=1.0, 
        num_episodes=10
    )
    
    print("评估Avellaneda-Stoikov策略...")
    av_results = baseline_avellaneda_stoikov_agent(
        env,
        risk_aversion=0.1,
        k=1.0,
        num_episodes=10
    )
    
    # 7. 结果分析
    print("\n[步骤6] 分析回测结果...")
    analyze_results(
        lstm_rewards=lstm_rewards,
        lstm_results=lstm_results,
        fixed_results=fixed_results,
        av_results=av_results,
        output_dir=output_dir
    )
    
    # 8. 生成报告
    print("\n[步骤7] 生成回测报告...")
    generate_report(
        lstm_rewards=lstm_rewards,
        lstm_results=lstm_results,
        fixed_results=fixed_results,
        av_results=av_results,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("回测完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 70)


def analyze_results(lstm_rewards, lstm_results, fixed_results, av_results, output_dir):
    """分析回测结果"""
    
    # 计算统计指标
    lstm_pnls = [sum(ep) for ep in lstm_results['pnls']]
    fixed_pnls = [sum(ep) for ep in fixed_results['pnls']]
    av_pnls = [sum(ep) for ep in av_results['pnls']]
    
    stats = {
        'LSTM策略': {
            '平均PnL': np.mean(lstm_pnls),
            'PnL标准差': np.std(lstm_pnls),
            '最大PnL': np.max(lstm_pnls),
            '最小PnL': np.min(lstm_pnls),
            '平均训练奖励': np.mean(lstm_rewards[-10:])  # 最后10个回合
        },
        'Fixed Spread': {
            '平均PnL': np.mean(fixed_pnls),
            'PnL标准差': np.std(fixed_pnls),
            '最大PnL': np.max(fixed_pnls),
            '最小PnL': np.min(fixed_pnls)
        },
        'Avellaneda-Stoikov': {
            '平均PnL': np.mean(av_pnls),
            'PnL标准差': np.std(av_pnls),
            '最大PnL': np.max(av_pnls),
            '最小PnL': np.min(av_pnls)
        }
    }
    
    print("\n回测统计结果:")
    print("-" * 70)
    for strategy, metrics in stats.items():
        print(f"\n{strategy}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 保存统计结果
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(output_dir, 'backtest_stats.csv'))
    print(f"\n统计结果已保存: {output_dir}/backtest_stats.csv")


def generate_report(lstm_rewards, lstm_results, fixed_results, av_results, output_dir):
    """生成可视化报告"""
    
    # 1. 训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_rewards, label='LSTM训练奖励', linewidth=2)
    plt.title('训练奖励曲线', fontsize=14)
    plt.xlabel('回合', fontsize=12)
    plt.ylabel('回合奖励', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=300)
    plt.close()
    
    # 2. PnL对比
    lstm_pnls = [sum(ep) for ep in lstm_results['pnls']]
    fixed_pnls = [sum(ep) for ep in fixed_results['pnls']]
    av_pnls = [sum(ep) for ep in av_results['pnls']]
    
    plt.figure(figsize=(10, 6))
    x = range(len(lstm_pnls))
    plt.plot(x, lstm_pnls, 'o-', label='LSTM策略', linewidth=2)
    plt.plot(x, fixed_pnls, 's-', label='Fixed Spread', linewidth=2)
    plt.plot(x, av_pnls, '^-', label='Avellaneda-Stoikov', linewidth=2)
    plt.title('各策略PnL对比', fontsize=14)
    plt.xlabel('测试回合', fontsize=12)
    plt.ylabel('总PnL', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pnl_comparison.png'), dpi=300)
    plt.close()
    
    # 3. 累积PnL (最后一个回合)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(lstm_results['pnls'][-1]), label='LSTM', linewidth=2)
    plt.plot(np.cumsum(fixed_results['pnls'][-1]), label='Fixed Spread', linewidth=2)
    plt.plot(np.cumsum(av_results['pnls'][-1]), label='Avellaneda-Stoikov', linewidth=2)
    plt.title('累积PnL (最后回合)', fontsize=14)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('累积PnL', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_pnl.png'), dpi=300)
    plt.close()
    
    # 4. 库存变化
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_results['inventories'][-1], label='LSTM', linewidth=2)
    plt.plot(fixed_results['inventories'][-1], label='Fixed Spread', linewidth=2)
    plt.plot(av_results['inventories'][-1], label='Avellaneda-Stoikov', linewidth=2)
    plt.title('库存变化 (最后回合)', fontsize=14)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('库存', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inventory_dynamics.png'), dpi=300)
    plt.close()
    
    print("可视化报告已生成:")
    print(f"  - {output_dir}/training_curve.png")
    print(f"  - {output_dir}/pnl_comparison.png")
    print(f"  - {output_dir}/cumulative_pnl.png")
    print(f"  - {output_dir}/inventory_dynamics.png")


if __name__ == "__main__":
    # 运行回测
    data_file = "./crypto_data/scaled_data.csv"
    
    if not os.path.exists(data_file):
        print("数据文件不存在，请先运行 test_crypto_data.py 生成数据")
        exit(1)
    
    run_crypto_backtest(
        data_file=data_file,
        output_dir="./crypto_backtest_results",
        num_episodes=20,
        max_episode_steps=500
    )
