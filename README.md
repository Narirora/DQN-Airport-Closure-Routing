# DQN-Airport-Closure-Routing
# Deep-Q-Network-Based Optimization Framework for Diversion Using Real Flight Data in Airport Closure
**空港閉鎖時における実飛行データを用いたDQNベースのダイバート（代替着陸）最適化フレームワーク**

## Project Overview / プロジェクト概要
[EN] This project presents an AI-driven decision-making framework designed to optimize aircraft diversions during sudden airport closures (e.g., severe weather or natural disasters). By utilizing Deep Q-Network (DQN) algorithms integrated with real flight trajectory data and BADA (Base of Aircraft Data) performance models, it identifies the optimal alternate airports for multiple aircraft in real-time.

[JP] 本プロジェクトは、悪天候や災害などによる突発的な空港閉鎖時に、飛行中の航空機が安全かつ効率的に代替空港へ向かう（ダイバート）ための意思決定を最適化するAIフレームワークです。実飛行データおよびBADA（航空機性能データ）を統合した環境下でDQN（深層強化学習）アルゴリズムを適用し、リアルタイムで最適な代替空港を算出します。

## Core Strategy: Business Trade-off Optimization / 核心となる戦略：ビジネス指標の最適化
[EN] The core differentiation of this project is the implementation of two distinct optimization models handling business trade-offs: Fuel Efficiency (Carbon Neutrality & Cost) vs. Passenger Convenience (Time & Customer Experience).
[JP] 本研究の最大の特徴は、単一の目標ではなく、相反する2つのビジネス指標（燃料効率/カーボンニュートラル vs 旅客の利便性/顧客体験）を基準に、それぞれ異なる報酬（Reward）関数を設計・評価した点にあります。

### 1. Fuel-Centric Model (燃料・CO2削減重視モデル)
* Goal / 目標: Minimize operational costs and achieve carbon neutrality. (運航コストの最小化およびカーボンニュートラルへの貢献)
* Reward Design / 報酬設計: Calculates real-time remaining fuel using BADA data, heavily penalizing excess fuel consumption. (BADAデータを用いてリアルタイムの残燃料を計算し、燃料消費の最小化と安全マージンの確保に最大重みを付与)
* Environment Code: envs/env_with_slot_fixed_fuelpersp.py

### 2. Passenger-Centric Model (旅客利便性・時間重視モデル)
* Goal / 目標: Maximize Customer Experience (CX) and minimize total travel delay. (顧客体験の最大化および移動遅延時間の最小化)
* Reward Design / 報酬設計: Penalizes ground transportation time, transfer counts, and additional costs from the alternate airport to the original destination. (代替空港から最終目的地までの陸上移動時間、乗り換え回数、発生費用をペナルティとして算出し最小化)
* Environment Code: envs/env_with_slot_fixed.py

## Data Pipeline & Simulation Environment / データパイプラインとシミュレーション環境
* State Space / 状態空間: Flight state (Latitude, Longitude, Altitude, Speed), Remaining Fuel, Airport Slot Congestion.
* Action Space / 行動空間: Selection among major alternate airports and the origin airport.
* Environment: Custom Python simulation environment mapping raw flight CSV logs into a Reinforcement Learning pipeline.

## Learning Curve / 学習曲線
*(Upload dqn_reward_curve_from_csv_fuel.png here / ここに学習曲線の画像をドラッグ＆ドロップしてください)*

[EN] The graph above demonstrates that as the epochs progress, the agent successfully converges to maximize the cumulative reward in the given environment.
[JP] 上記のグラフは、エポックが進むにつれてエージェントが環境下での報酬を最大化する方向へ安定して収束していることを示しています。

## Tech Stack / 技術スタック
* Language: Python 3
* AI/ML: PyTorch (DQN Implementation)
* Data Processing & Visualization: Pandas, NumPy, Matplotlib
* Domain Data: BADA (Base of Aircraft Data), Real Flight Trajectory CSV

## Repository Structure / ディレクトリ構成
```text
├── envs/                # RL Environments (Fuel-centric & Passenger-centric)
├── src/                 # DQN Training & Simulation scripts
├── models/              # Pre-trained PyTorch Model Weights (.pt)
├── data/                # Sample flight logs and BADA data
├── results/             # Simulation output CSVs & Visualization graphs
└── README.md
