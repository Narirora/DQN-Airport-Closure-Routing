import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
import ast
from env_with_slot_fixed_fuelpersp import RerouteEnv
import os

# 기존 모델 삭제 (선택사항)
if os.path.exists("dqn_model_eta_based_fuel_2.pt"):
    os.remove("dqn_model_eta_based_fuel_2.pt")

# 최대 가능한 divert 공항 수 (예: 기존 5개 + 출발공항까지)
MAX_OUTPUT_DIM = 10

# DQN 정의
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # output_dim = MAX_OUTPUT_DIM

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train():
    df = pd.read_csv("real_flight_states_with_position_final.csv")
    df['ParsedState'] = df['State'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    fuel_df = pd.read_csv("fuel_remainging_results.csv")
    bada_df = pd.read_excel("BADA_data.xlsx")
    states = df['ParsedState'].tolist()

    input_dim = len(states[0])
    hidden_dim = 128
    model = DQN(input_dim, hidden_dim, MAX_OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)
    batch_size = 32
    gamma = 0.95
    epsilon = 0.6

    # 기본 divert 공항 목록
    base_airports = {
        'NRT': (35.7635, 140.3864),
        'NGO': (34.8584, 136.8052),
        'KIX': (34.4273, 135.2441),
        'ITM': (34.7855, 135.4382),
        'SDJ': (38.1397, 140.9176)
    }
    base_capacities = {
        'NRT': 0.8, 'NGO': 0.7, 'KIX': 0.8, 'ITM': 0.6, 'SDJ': 0.4
    }

    reward_log = []
    epochs = 2000

    for epoch in range(epochs):
        total_reward = 0

        for idx, row in df.iterrows():
            # 환경 초기화
            env = RerouteEnv(
                airport_coords=base_airports.copy(),
                airport_capacities=base_capacities.copy(),
                original_destination_coord=(35.5523, 139.7798),
                current_time_min=0,
                bada_df=bada_df,
                fuel_df=fuel_df,
                slot_mode="fixed",
                max_per_slot=2
            )
            env.refresh_slot_schedule()

            state = row['ParsedState']
            origin_lat, origin_lon = row['origin_lat'], row['origin_lon']
            aircraft_type = row['aircraft_type']
            flight_id = row['flight_id']
            origin_code = row['origin_code']

            # ✅ 출발공항을 divert 후보로 등록
            env.reset(state, (origin_lat, origin_lon), aircraft_type=aircraft_type,
                      flight_id=flight_id, origin_code=origin_code)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            current_output_dim = len(env.airport_coords)

            if random.random() < epsilon:
                action = random.randint(0, current_output_dim - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values[:, :current_output_dim]).item()  # slicing 적용

            reward, valid, chosen_airport, eta, alt_score, dest_score, origin_score, penalty, fuel_eff_score = env.step(action)

            reward_log.append({
                "epoch": epoch,
                "flight_id": flight_id,
                "chosen_airport": chosen_airport,
                "reward": reward,
                "fuel_eff_score": fuel_eff_score,
                "alt_score": alt_score,
                "dest_score": dest_score,
                "origin_score": origin_score,
                "penalty": penalty
            })

            next_state = state
            done = True
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = zip(*minibatch)

                states_tensor = torch.tensor(states_mb, dtype=torch.float32)
                actions_tensor = torch.tensor(actions_mb, dtype=torch.int64).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards_mb, dtype=torch.float32).unsqueeze(1)
                next_states_tensor = torch.tensor(next_states_mb, dtype=torch.float32)
                dones_tensor = torch.tensor(dones_mb, dtype=torch.bool).unsqueeze(1)

                q_values = model(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    max_next_q = model(next_states_tensor).max(1, keepdim=True)[0]
                    target_q = rewards_tensor + gamma * max_next_q * (~dones_tensor)

                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.1, epsilon * 0.99)

        if epoch % 100 == 0:
            sample_state = torch.tensor(states[0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = model(sample_state)[0][:current_output_dim].numpy()
            q_vals_rounded = [round(v, 2) for v in q_vals.tolist()]
            print(f"[Epoch {epoch}] Total Reward: {round(total_reward, 2)}, Q-values: {q_vals_rounded}")

    pd.DataFrame(reward_log).to_csv("train_reward_component_log_fuel.csv", index=False)
    torch.save(model.state_dict(), "dqn_model_eta_based_fuel_2.pt")
    print("✅ Training complete with dynamic airport set and origin fallback.")

if __name__ == "__main__":
    train()
