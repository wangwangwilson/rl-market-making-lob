import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns  # Added for plotting aesthetics
from functools import partial
from collections import Counter  # Added for action distribution
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
from dataclasses import dataclass

# Handle signatory import
try:
    import signatory
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False

@dataclass
class Config:
    total_timesteps: int = 10000
    max_episode_steps: int = 1000
    cash: float = 100.0
    max_price_history: int = 100
    episodes: int = 250
    debug: bool = False

@dataclass
class TrainingConfig:
    learning_rate: float = 0.002
    gamma: float = 0.99
    hidden_dim: int = 128
    use_lstm: bool = True
    lstm_layers: int = 2
    entropy_coef: float = 0.5
    batch_size: int = 64
    n_steps: int = 5
    use_gae: bool = True
    gae_lambda: float = 0.95
    max_gradient_norm: float = 0.5

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
sns.set(style="whitegrid", font_scale=1.2)  # Set Seaborn style

@dataclass
class RewardConfig:
    eta: float = 0.3
    zeta: float = 0.005
    transaction_cost: float = 0.0001

class EnhancedMarketMakingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, max_episode_steps: int = 1000,
                 reward_config: RewardConfig = RewardConfig(),
                 prob_execution: bool = True, use_path_signatures: bool = False):
        super().__init__()
        self.data = data
        self.data_np = data.values.astype(np.float32)
        self.columns = list(data.columns)
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config
        self.prob_execution = prob_execution
        self.use_path_signatures = use_path_signatures and HAS_SIGNATORY
        self.config = Config()

        # State variables
        self.current_step = 0
        self.inventory = 0
        self.cash = self.config.cash
        self.max_inventory = 10
        self.price_history = []
        self.trade_timestamps = []
        self.trade_window = 60
        self.max_trades_in_window = 5

        # Path signatures setup
        self.signature_depth = 3
        self.signature_dim = (signatory.signature_channels(2, self.signature_depth)
                              if self.use_path_signatures else 0)

        # Spaces
        self.action_space = spaces.Discrete(9) #9
        num_features = self.data_np.shape[1] + 1 + 2 + 1 + self.signature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

        # Metrics
        self.metrics = {
            'episode_reward': 0, 'trading_pnl': 0, 'inventory_costs': 0,
            'transaction_costs': 0, 'trades_executed': 0, 'bid_transactions': 0,
            'ask_transactions': 0, 'inventory_history': [], 'pnl_history': []
        }

        # Action mapping
        self.action_mapping = {
           0: (-1.0, -1.0), 1: (-1.0, 0.0), 2: (-1.0, 1.0),
           3: (0.0, -1.0), 4: (0.0, 0.0), 5: (0.0, 1.0),
            6: (1.0, -1.0), 7: (1.0, 0.0), 8: (1.0, 1.0)
       }
    

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        max_start = max(int(0.8 * len(self.data_np) - self.max_episode_steps), 1)
        self.current_step = self.np_random.integers(0, max_start) if max_start > 0 else 0
        self.inventory = 0
        self.cash = self.config.cash
        self.last_bid_offset = 0.0
        self.last_ask_offset = 0.0
        self.last_pnl = 0.0
        self.price_history = []
        self.trade_timestamps = []

        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = [] if isinstance(self.metrics[key], list) else 0

        # Initial portfolio value
        bid_idx, ask_idx = self.columns.index('Bid Price 1'), self.columns.index('Ask Price 1')
        market_bid = float(self.data_np[self.current_step][bid_idx])
        market_ask = float(self.data_np[self.current_step][ask_idx])
        self.prev_value = self.cash + self.inventory * (market_bid + market_ask) / 2

        return self._next_observation(), {}

    def _next_observation(self):
        obs_market = self.data_np[self.current_step]
        normalized_inventory = self.inventory / self.max_inventory
        recent_actions = np.array([self.last_bid_offset, self.last_ask_offset])
        recent_pnl = np.array([self.last_pnl])

        signature = (self._compute_signature() if self.use_path_signatures and len(self.price_history) >= 2
                     else np.zeros(self.signature_dim))

        return np.concatenate([obs_market, [normalized_inventory], recent_actions, recent_pnl, signature])

    def _compute_signature(self):
        try:
            times = np.arange(len(self.price_history))
            path = np.stack([times, self.price_history], axis=1)
            path_tensor = torch.tensor(path, dtype=torch.float32).unsqueeze(0)
            return signatory.signature(path_tensor, depth=self.signature_depth).squeeze(0).numpy()
        except Exception as e:
            print(f"Error in signature computation: {e}")
            return np.zeros(self.signature_dim)

    def step(self, action):
        current_time = float(self.data_np[self.current_step][0])
        self.trade_timestamps = [ts for ts in self.trade_timestamps if current_time - ts < self.trade_window]
        trade_allowed = len(self.trade_timestamps) < self.max_trades_in_window

        bid_offset, ask_offset = self.action_mapping[action]
        self.last_bid_offset, self.last_ask_offset = bid_offset, ask_offset

        market_bid, market_ask, current_mid = self._get_market_prices()
        self.price_history.append(current_mid)
        if len(self.price_history) > self.config.max_price_history:
            self.price_history.pop(0)

        bid_quote, ask_quote = self._calculate_quotes(current_mid, bid_offset, ask_offset)

        if bid_quote > ask_quote:
            reward = self._calculate_invalid_spread_reward()
            trading_pnl = transaction_costs = 0.0
        else:
            trading_pnl, transaction_costs = self._execute_trades(
                bid_quote, ask_quote, market_bid, market_ask, trade_allowed
            )
            reward = self._calculate_reward(trading_pnl, transaction_costs)

        self._update_metrics(trading_pnl, transaction_costs, reward)
        self.current_step += 1
        done = self.current_step >= min(self.max_episode_steps + self.current_step, len(self.data_np) - 1)
        truncated = False

        return self._next_observation(), reward, done, truncated, self.metrics

    def _get_market_prices(self):
        bid_idx = self.columns.index('Bid Price 1')
        ask_idx = self.columns.index('Ask Price 1')
        market_bid = float(self.data_np[self.current_step][bid_idx])
        market_ask = float(self.data_np[self.current_step][ask_idx])
        return market_bid, market_ask, (market_bid + market_ask) / 2

    def _calculate_quotes(self, mid_price, bid_offset, ask_offset):
        return mid_price - bid_offset, mid_price + ask_offset

    def _calculate_invalid_spread_reward(self):
        return -0.1  # Reduced penalty for clarity

    def _execute_trades(self, bid_quote, ask_quote, market_bid, market_ask, trade_allowed):
        trading_pnl = transaction_costs = 0.0
        current_time = float(self.data_np[self.current_step][0])

        if not trade_allowed:
            return trading_pnl, transaction_costs

        if self.prob_execution:
            trading_pnl, transaction_costs = self._execute_probabilistic_trades(
                bid_quote, ask_quote, market_bid, market_ask, current_time
            )
        else:
            trading_pnl, transaction_costs = self._execute_deterministic_trades(
                bid_quote, ask_quote, market_bid, market_ask, current_time
            )
        return trading_pnl, transaction_costs

    def _execute_probabilistic_trades(self, bid_quote, ask_quote, market_bid, market_ask, current_time):
        trading_pnl = transaction_costs = 0.0
        if bid_quote <= market_ask and self.inventory < self.max_inventory:
            bid_prob = np.exp(-2 * (market_ask - bid_quote) / market_ask)
            if self.np_random.random() < bid_prob:
                execution_price = min(bid_quote, market_ask)
                self._update_buy(execution_price, current_time)
                trading_pnl += (self.price_history[-1] - execution_price)
                transaction_costs += self.reward_config.transaction_cost * execution_price

        if ask_quote >= market_bid and self.inventory > -self.max_inventory:
            ask_prob = np.exp(-2 * (ask_quote - market_bid) / market_bid)
            if self.np_random.random() < ask_prob:
                execution_price = max(ask_quote, market_bid)
                self._update_sell(execution_price, current_time)
                trading_pnl += (execution_price - self.price_history[-1])
                transaction_costs += self.reward_config.transaction_cost * execution_price
        return trading_pnl, transaction_costs

    def _execute_deterministic_trades(self, bid_quote, ask_quote, market_bid, market_ask, current_time):
        trading_pnl = transaction_costs = 0.0
        if bid_quote >= market_ask and self.inventory < self.max_inventory:
            self._update_buy(market_ask, current_time)
            trading_pnl += (self.price_history[-1] - market_ask)
            transaction_costs += self.reward_config.transaction_cost * market_ask

        if ask_quote <= market_bid and self.inventory > -self.max_inventory:
            self._update_sell(market_bid, current_time)
            trading_pnl += (market_bid - self.price_history[-1])
            transaction_costs += self.reward_config.transaction_cost * market_bid
        return trading_pnl, transaction_costs

    def _update_buy(self, price, timestamp):
        self.cash -= price
        self.inventory += 1
        self.metrics['trades_executed'] += 1
        self.metrics['bid_transactions'] += 1
        self.trade_timestamps.append(timestamp)

    def _update_sell(self, price, timestamp):
        self.cash += price
        self.inventory -= 1
        self.metrics['trades_executed'] += 1
        self.metrics['ask_transactions'] += 1
        self.trade_timestamps.append(timestamp)

    def _calculate_reward(self, trading_pnl, transaction_costs):
        mid_price = self.price_history[-1]
        value = self.cash + self.inventory * mid_price
        delta_value = value - self.prev_value
        self.prev_value = value
    
        # Dampened upside only positive change is dampened
        dampened_value_change = delta_value * (1 - self.reward_config.eta) if delta_value > 0 else delta_value
    
        # Inventory penalty
        inventory_penalty = self.reward_config.zeta * (self.inventory ** 2) * 0.1
    
        # Total reward
        reward = 10 * dampened_value_change - inventory_penalty - 2 * transaction_costs
        reward = np.clip(reward,-10,10)

    
        self.last_pnl = reward
        return reward


    def _update_metrics(self, trading_pnl, transaction_costs, reward):
        self.metrics['trading_pnl'] += trading_pnl
        self.metrics['transaction_costs'] += transaction_costs
        self.metrics['episode_reward'] += reward
        self.metrics['pnl_history'].append(self.last_pnl)
        self.metrics['inventory_history'].append(self.inventory)

class LSTMPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lstm_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.hidden = None

    def reset_hidden(self, batch_size: int = 1, device: str = "cpu"):
        self.hidden = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device).contiguous(),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device).contiguous()
        )

    def forward(self, x: torch.Tensor, reset_hidden: bool = False):
        if reset_hidden or self.hidden is None or self.hidden[0].size(1) != x.size(0):
            self.reset_hidden(batch_size=x.size(0), device=x.device)

        if len(x.shape) == 2: 
            x = x.unsqueeze(1)  

        x = x.contiguous()  # Ensure input is contiguous
        lstm_out, self.hidden = self.lstm(x, self.hidden)  
        lstm_out = lstm_out.reshape(-1, self.hidden_dim)  

        policy_logits = self.policy_head(lstm_out)
        action_probs = F.softmax(policy_logits, dim=-1)
        state_values = self.value_head(lstm_out)

        action_probs = action_probs.view(x.size(0), x.size(1), -1)
        state_values = state_values.view(x.size(0), x.size(1), -1)
        return action_probs, state_values

class MLPPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, reset_hidden: bool = False):
        if len(x.shape) == 2:  # (batch_size, input_dim)
            shared_features = self.shared(x)
            policy_logits = self.policy_head(shared_features)
            action_probs = F.softmax(policy_logits, dim=-1)
            state_values = self.value_head(shared_features)
            return action_probs, state_values
        else:  # (batch_size, seq_len, input_dim)
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, x.size(-1))
            shared_features = self.shared(x_flat)
            policy_logits = self.policy_head(shared_features)
            action_probs = F.softmax(policy_logits, dim=-1)
            state_values = self.value_head(shared_features)
            return (action_probs.view(batch_size, seq_len, -1),
                    state_values.view(batch_size, seq_len, -1))



def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        returns.insert(0, gae + values[i])
        next_value = values[i]
    return returns

def actor_critic_train(env, policy_net, optimizer, num_episodes, gamma=0.99, entropy_coef=0.01, max_grad_norm=0.5, use_gae=True, gae_lambda=0.95, device="cpu"):
    episode_rewards = []
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        obs, _ = env.reset()
        if hasattr(policy_net, 'reset_hidden'):
            policy_net.reset_hidden(batch_size=1, device=device)
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        done = False
        episode_reward = 0.0
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            action_probs, state_value = policy_net(obs_tensor)
            action_probs, state_value = action_probs.squeeze(1), state_value.squeeze(1)
            m = Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            next_obs, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            states.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(state_value.item())
            dones.append(float(done))
            obs = next_obs
            episode_reward += reward

        # Process episode
        states_tensor = torch.FloatTensor(states).unsqueeze(0).to(device).contiguous()  # Ensure contiguous
        action_probs, state_values = policy_net(states_tensor)
        action_probs = action_probs.squeeze(0)
        state_values = state_values.squeeze(0).squeeze(-1)
        if use_gae:
            returns = compute_gae(rewards, values, 0.0, dones, gamma, gae_lambda)
        else:
            returns = []
            discounted_reward = 0
            for r, d in zip(reversed(rewards), reversed(dones)):
                discounted_reward = r + gamma * discounted_reward * (1 - d)
                returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)
        log_probs = torch.stack(log_probs)
        advantages = returns - state_values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(state_values, returns)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
        total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        optimizer.step()

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.4f}")

    return episode_rewards


def baseline_fixed_spread_agent(env, fixed_bid_offset=-0.1, fixed_ask_offset=0.1, num_episodes=20):
    results = {'pnls': [], 'rewards': [], 'inventories': []}
    action_map = env.action_mapping
    action = next(k for k, v in action_map.items() if v == (fixed_bid_offset, fixed_ask_offset))

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_pnl, episode_reward, episode_inventory = [], [], []

        while not done:
            next_obs, reward, done, _, info = env.step(action)
            episode_pnl.append(info['pnl_history'][-1])
            episode_reward.append(reward)
            episode_inventory.append(info['inventory_history'][-1])
            obs = next_obs

        results['pnls'].append(episode_pnl)
        results['rewards'].append(episode_reward)
        results['inventories'].append(episode_inventory)

    return results

def baseline_avellaneda_stoikov_agent(env, risk_aversion=0.1, k=1.0, num_episodes=20):
    results = {'pnls': [], 'rewards': [], 'inventories': []}
    action_map = env.action_mapping

    for _ in range(num_episodes):
        obs, _ = env.reset()  # Reset returns the initial observation
        done = False
        episode_pnl, episode_reward, episode_inventory = [], [], []

        while not done:
            # Calculate mid-price from the observation
            # Assuming obs[0] is 'Bid Price 1' and obs[2] is 'Ask Price 1'
            bid_price = obs[0]
            ask_price = obs[2]
            mid_price = (bid_price + ask_price) / 2

            # Get inventory
            inventory = env.inventory

            # Calculate volatility from price_history if enough data exists, else use default
            if len(env.price_history) >= 10:
                volatility = np.std(env.price_history[-10:])
            else:
                volatility = 0.01  # Default value when history is insufficient

            # Avellaneda-Stoikov calculations
            T = (env.max_episode_steps - env.current_step) / env.max_episode_steps
            reservation_price = mid_price - inventory * risk_aversion * volatility**2 * T
            spread = risk_aversion * volatility**2 * T + (2 / risk_aversion) * np.log(1 + risk_aversion / k)
            bid_offset = -(spread / 2)
            ask_offset = spread / 2

            # Choose action based on closest matching offsets
            action = min(action_map.keys(), key=lambda x: abs(action_map[x][0] - bid_offset) + abs(action_map[x][1] - ask_offset))
            
            # Take a step in the environment
            next_obs, reward, done, _, info = env.step(action)
            
            # Record results
            episode_pnl.append(info['pnl_history'][-1])
            episode_reward.append(reward)
            episode_inventory.append(info['inventory_history'][-1])
            
            # Update observation for the next iteration
            obs = next_obs

        # Store episode results
        results['pnls'].append(episode_pnl)
        results['rewards'].append(episode_reward)
        results['inventories'].append(episode_inventory)

    return results

def test_and_visualize(env, policy_net, num_episodes=10, device="cpu"):
    results = {'pnls': [], 'rewards': [], 'inventories': [], 'bid_offsets': [], 'ask_offsets': []}
    action_map = env.action_mapping
    for _ in range(num_episodes):
        obs, _ = env.reset()
        if hasattr(policy_net, 'reset_hidden'):
            policy_net.reset_hidden(batch_size=1, device=device)
        done = False
        episode_pnl, episode_reward, episode_inventory = [], [], []
        episode_bid_offsets, episode_ask_offsets = [], []
        step = 0
        while not done and step < env.max_episode_steps:  # Enforce max_episode_steps
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            action_probs, _ = policy_net(obs_tensor)
            action_probs = action_probs.squeeze(1)
            action = torch.argmax(action_probs, dim=-1).item()
            bid_offset, ask_offset = action_map[action]
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_pnl.append(info['pnl_history'][-1])
            episode_reward.append(reward)
            episode_inventory.append(info['inventory_history'][-1])
            episode_bid_offsets.append(bid_offset)
            episode_ask_offsets.append(ask_offset)
            obs = next_obs
            step += 1
        results['pnls'].append(episode_pnl)
        results['rewards'].append(episode_reward)
        results['inventories'].append(episode_inventory)
        results['bid_offsets'].append(episode_bid_offsets)
        results['ask_offsets'].append(episode_ask_offsets)
    return results

from functools import partial
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical as SkoptCategorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

# Define the search for hyperparameters
space = [
    Real(0.1, 0.9, name='eta'),
    Real(0.001, 0.1, name='zeta', prior='log-uniform'),
    Integer(32, 256, name='hidden_dim'),
    Real(0.0001, 0.01, name='learning_rate', prior='log-uniform'),
    Real(0.9, 0.999, name='gamma'),
    Real(0.01, 0.2, name='entropy_coef'),
    SkoptCategorical([True, False], name='use_lstm'),
    Integer(1, 3, name='lstm_layers'),
    Real(0.00001, 0.001, name='transaction_cost', prior='log-uniform'),
    Real(0.5, 1.0, name='gae_lambda')
]

# Define the objective function with all parameters
@use_named_args(space)
def objective_function(data_subset, max_episode_steps, num_episodes,
                      eta, zeta, hidden_dim, learning_rate, gamma, entropy_coef,
                      use_lstm, lstm_layers, transaction_cost, gae_lambda):
    print(f"Evaluating: eta={eta:.4f}, zeta={zeta:.4f}, hidden_dim={hidden_dim}, "
          f"lr={learning_rate:.6f}, gamma={gamma:.4f}, entropy_coef={entropy_coef:.4f}, "
          f"use_lstm={use_lstm}, lstm_layers={lstm_layers}, transaction_cost={transaction_cost:.6f}, "
          f"gae_lambda={gae_lambda:.4f}")
    
    env = EnhancedMarketMakingEnv(
        data=data_subset, 
        max_episode_steps=max_episode_steps,
        reward_config=RewardConfig(eta=eta, zeta=zeta, transaction_cost=transaction_cost),
        prob_execution=False
    )
    policy_net = (LSTMPolicyNetwork(env.observation_space.shape[0], hidden_dim, 9, lstm_layers) if use_lstm
                  else MLPPolicyNetwork(env.observation_space.shape[0], hidden_dim, 9))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    optimizer = Adam(policy_net.parameters(), lr=learning_rate)

    rewards = actor_critic_train(
        env, 
        policy_net, 
        optimizer, 
        num_episodes, 
        gamma, 
        entropy_coef,
        max_grad_norm=0.5, 
        use_gae=True, 
        gae_lambda=gae_lambda, 
        device=device
    )
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward: {avg_reward:.4f}")
    return -avg_reward  # Minimize negative reward

def run_bayesian_optimization(data_df, n_calls=10, n_initial_points=5, max_episode_steps=50, num_eval_episodes=15):
    # Use partial to fix the non-optimized parameters
    obj_func = partial(
        objective_function,
        data_subset=data_df,
        max_episode_steps=max_episode_steps,
        num_episodes=num_eval_episodes
    )
    
    print(f"Starting Bayesian Optimization with {n_calls} calls and {n_initial_points} initial points...")
    res = gp_minimize(
        obj_func,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        verbose=True,
        random_state=42
    )
    
    print(f"\nBest score: {-res.fun:.4f} (negative: {res.fun:.4f})")
    print("Best hyperparameters:")
    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    plt.figure(figsize=(8, 5))
    plot_convergence(res)
    plt.title("Bayesian Optimization Convergence")
    plt.show()
    
    return best_params

import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv('/kaggle/input/rmllll/scaled_data.csv')  
    features = ['Bid Price 1', 'Bid Size 1', 'Ask Price 1', 'Ask Size 1', 'midprice', 'spread', 'log_return', 'RV_5min', 'RSI_5min', 'OSI_10s']
    data_subset = data[features].iloc[600000:610000]  # 10,000 steps 

    # Define configurations
    config = Config(episodes=50, max_episode_steps=10000)  
    reward_config = RewardConfig(eta=0.7, zeta=0.001, transaction_cost=0.0001)  
    training_config = TrainingConfig(entropy_coef=0.05)  

    # Set up environment
    print("Setting up environment...")
    env = EnhancedMarketMakingEnv(
        data=data_subset,
        max_episode_steps=config.max_episode_steps,
        reward_config=reward_config,
        prob_execution=True,
        use_path_signatures=False
    )

    # Initialize policy networks
    state_dim = env.observation_space.shape[0]
    action_dim = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LSTM Policy
    lstm_policy = LSTMPolicyNetwork(state_dim, training_config.hidden_dim, action_dim, training_config.lstm_layers)
    lstm_policy.to(device)
    lstm_optimizer = Adam(lstm_policy.parameters(), lr=training_config.learning_rate)

    # MLP Policy
    mlp_policy = MLPPolicyNetwork(state_dim, training_config.hidden_dim, action_dim)
    mlp_policy.to(device)
    mlp_optimizer = Adam(mlp_policy.parameters(), lr=training_config.learning_rate)

    # Train LSTM policy
    print("Training LSTM policy...")
    lstm_rewards = actor_critic_train(
        env, lstm_policy, lstm_optimizer, config.episodes,
        gamma=training_config.gamma, entropy_coef=training_config.entropy_coef,
        max_grad_norm=training_config.max_gradient_norm, use_gae=training_config.use_gae,
        gae_lambda=training_config.gae_lambda, device=device
    )

    # Train MLP policy
    print("Training MLP policy...")
    mlp_rewards = actor_critic_train(
        env, mlp_policy, mlp_optimizer, config.episodes,
        gamma=training_config.gamma, entropy_coef=training_config.entropy_coef,
        max_grad_norm=training_config.max_gradient_norm, use_gae=training_config.use_gae,
        gae_lambda=training_config.gae_lambda, device=device
    )

    # Test the trained models
    print("Testing the trained LSTM model...")
    lstm_test_results = test_and_visualize(env, lstm_policy, num_episodes=config.episodes, device=device)
    print("Testing the trained MLP model...")
    mlp_test_results = test_and_visualize(env, mlp_policy, num_episodes=config.episodes, device=device)

    # Evaluate baseline agents
    print("Evaluating baseline fixed-spread agent...")
    baseline_results = baseline_fixed_spread_agent(env, fixed_bid_offset=-1.0, fixed_ask_offset=1.0, num_episodes=config.episodes)
    print("Evaluating Avellaneda-Stoikov agent...")
    av_st_results = baseline_avellaneda_stoikov_agent(env, risk_aversion=0.1, k=1.0, num_episodes=config.episodes)

    # Plotting
    # Total Reward per Episode lstm vs mlp
    plt.figure(figsize=(8,5))
    plt.plot(lstm_rewards, label='LSTM Policy (GAE)', color='blue')
    plt.plot(mlp_rewards, label='MLP Policy (GAE)', color='purple')
    plt.axhline(y=0, color='orange', linestyle='--', label='Fixed Spread')
    plt.axhline(y=0, color='green', linestyle='--', label='Avellaneda-Stoikov')
    plt.title('Total Reward per Episode', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('total_reward_per_episode.png', dpi=300)
    plt.show()

    # Total PnL per Episode 
    plt.figure(figsize=(8,5))
    plt.plot([sum(ep) for ep in lstm_test_results['pnls']], label='Optimized RL Agent (LSTM)', color='blue')
    plt.plot([sum(ep) for ep in mlp_test_results['pnls']], label='Optimized RL Agent (MLP)', color='purple')
    plt.plot([sum(ep) for ep in baseline_results['pnls']], label='Fixed Spread', color='orange')
    plt.plot([sum(ep) for ep in av_st_results['pnls']], label='Avellaneda-Stoikov', color='green')
    plt.title('Total PnL per Episode', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('PnL', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('total_pnl_per_episode.png', dpi=300)
    plt.show()

    # Final Inventory Distribution 
    plt.figure(figsize=(8,5))
    lstm_final_inventories = [ep[-1] for ep in lstm_test_results['inventories']]
    mlp_final_inventories = [ep[-1] for ep in mlp_test_results['inventories']]
    baseline_final_inventories = [ep[-1] for ep in baseline_results['inventories']]
    av_st_final_inventories = [ep[-1] for ep in av_st_results['inventories']]
    plt.hist(lstm_final_inventories, bins=range(-6, 2), alpha=0.5, label='Optimized RL (LSTM)', color='blue')
    plt.hist(mlp_final_inventories, bins=range(-6, 2), alpha=0.5, label='Optimized RL (MLP)', color='purple')
    plt.hist(baseline_final_inventories, bins=range(-6, 2), alpha=0.5, label='Fixed Spread', color='orange')
    plt.hist(av_st_final_inventories, bins=range(-6, 2), alpha=0.5, label='Avellaneda-Stoikov', color='green')
    plt.title('Final Inventory Distribution', fontsize=14)
    plt.xlabel('Final Inventory', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('final_inventory_distribution.png', dpi=300)
    plt.show()

    # Cumulative PnL Comparison -> Last Episode
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(lstm_test_results['pnls'][-1]), label='RL Agent (LSTM)', color='blue')
    plt.plot(np.cumsum(mlp_test_results['pnls'][-1]), label='RL Agent (MLP)', color='purple')
    plt.plot(np.cumsum(baseline_results['pnls'][-1]), label='Fixed Spread', color='orange')
    plt.plot(np.cumsum(av_st_results['pnls'][-1]), label='Avellaneda-Stoikov', color='green')
    plt.title('Cumulative PnL Comparison (Last Episode)', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cumulative PnL', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_pnl_comparison.png', dpi=300)
    plt.show()

    # Inventory Over Time -> last Epsiode
    plt.figure(figsize=(8,5))
    plt.plot(lstm_test_results['inventories'][-1], label='RL Agent (LSTM)', color='blue')
    plt.plot(mlp_test_results['inventories'][-1], label='RL Agent (MLP)', color='purple')
    plt.plot(baseline_results['inventories'][-1], label='Fixed Spread', color='orange')
    plt.plot(av_st_results['inventories'][-1], label='Avellaneda-Stoikov', color='green')
    plt.axhline(y=env.max_inventory, color='red', linestyle='--', label='Max Inventory')
    plt.axhline(y=-env.max_inventory, color='red', linestyle='--')
    plt.title('Inventory Over Time (Last Episode)', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Inventory', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('inventory_over_time.png', dpi=300)
    plt.show()

    # Action Distribution = 9 actions
    action_labels = ['-1.0/-1.0', '-1.0/0.0', '-1.0/1.0', '0.0/-1.0', '0.0/0.0', '0.0/1.0', '1.0/-1.0', '1.0/0.0', '1.0/1.0']
    lstm_all_actions = []
    mlp_all_actions = []
    for episode_bid_offsets, episode_ask_offsets in zip(lstm_test_results['bid_offsets'], lstm_test_results['ask_offsets']):
        actions = [f"{b}/{a}" for b, a in zip(episode_bid_offsets, episode_ask_offsets)]
        lstm_all_actions.extend(actions)
    for episode_bid_offsets, episode_ask_offsets in zip(mlp_test_results['bid_offsets'], mlp_test_results['ask_offsets']):
        actions = [f"{b}/{a}" for b, a in zip(episode_bid_offsets, episode_ask_offsets)]
        mlp_all_actions.extend(actions)
    
    lstm_action_counts = Counter(lstm_all_actions)
    mlp_action_counts = Counter(mlp_all_actions)
    
    # ensure all 9 actions are represented
    lstm_counts = [lstm_action_counts.get(label, 0) for label in action_labels]
    mlp_counts = [mlp_action_counts.get(label, 0) for label in action_labels]
    
    x = np.arange(len(action_labels))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, lstm_counts, width, label='LSTM', color='blue')
    plt.bar(x + width/2, mlp_counts, width, label='MLP', color='purple')
    plt.title('Action Distribution (Bid/Ask Offsets)', fontsize=14)
    plt.xlabel('Bid/Ask Offset Pair', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(x, action_labels, rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=300)
    plt.show()