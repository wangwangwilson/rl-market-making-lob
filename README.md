# Reinforcement Learning for Market Making in Limit Order Books  

This project explores **reinforcement learning (RL)** methods for automated market making in a **limit order book (LOB)** environment.  
The framework includes a custom simulation environment, actor–critic agents, and comparisons against established baselines such as **Avellaneda–Stoikov**.  

---

## 🔑 Key Features  

- **Market-Making Environment**
  - Inventory-aware quoting with probabilistic fills  
  - Rich LOB feature set (midprice, spread, order imbalance, volatility, RSI)  
  - Optional **path signature features** for temporal encoding  

- **Reward Design**
  - Dampened PnL term (limits overfitting to short spikes)  
  - Quadratic inventory penalty  
  - Spread penalty for competitiveness  
  - Transaction costs  

- **Agents**
  - RL actor–critic agents (MLP and LSTM)  
  - Baselines:
    - Fixed-spread quoting  
    - **Avellaneda–Stoikov model**  

- **Evaluation Metrics**
  - PnL decomposition  
  - Inventory distribution and dynamics  
  - Action distribution & trade frequency  
  - Training curves and visualisations  

---

## 📂 Repository Structure  

- `main_code.py` – environment, agents, training loop  
- `data_prep.ipynb` – preprocessing and scaling LOB data  
- `feature_engineering.ipynb` – engineered features (volatility, RSI, imbalance, path signatures)  
- `project_report.pdf` – detailed write-up of methodology, experiments, and results  

---

## 📈 Results (Highlights)  

- RL agents learned inventory-aware strategies but struggled with execution noise.  
- **Avellaneda–Stoikov** baseline remained more stable and profitable.  
- LSTM policies reduced variance compared to MLPs, showing the benefit of temporal memory.  
- Path signatures enriched features but added computational cost.  

---

## 📌 Applications  

- Educational tool for **RL in finance** and **market microstructure**  
- Research sandbox for testing reward design & quoting strategies  
- Foundation for future work on:
  - Continuous action spaces  
  - More realistic fill & latency modeling  
  - Multi-agent simulations  
  - Risk-sensitive objectives (e.g., CVaR, drawdowns)  

---

## 📝 Acknowledgement  

Developed as part of MSc research at **University College London** (2025).  
