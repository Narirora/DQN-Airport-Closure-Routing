import torch
import pandas as pd
import ast
import numpy as np
from env_with_slot_fixed_fuelpersp import RerouteEnv
import torch.nn as nn

MAX_OUTPUT_DIM = 10  # 최대 공항 수 (5 + 출발공항 등)

# DQN 정의
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main():
    model_path = "dqn_model_eta_based_fuel.pt"
    state_file = "real_flight_states_with_position_final.csv"
    fuel_file = "fuel_remainging_results.csv"
    bada_file = "BADA_data.xlsx"

    df = pd.read_csv(state_file)
    df['ParsedState'] = df['State'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    states = df['ParsedState'].tolist()
    bada_df = pd.read_excel(bada_file)
    fuel_df = pd.read_csv(fuel_file)

    input_dim = len(states[0])
    hidden_dim = 128
    device = torch.device("cpu")

    model = DQN(input_dim, hidden_dim, MAX_OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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

    results = []

    for idx, row in df.iterrows():
        state = row['ParsedState']
        lat, lon = state[0], state[1]
        origin_lat, origin_lon = row['origin_lat'], row['origin_lon']
        aircraft_type = row['aircraft_type']
        origin_code = row['origin_code']
        flight_id = row['flight_id']

        # ✅ 개별 환경 생성
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

        # ✅ 출발공항 포함하여 reset
        env.reset(state, (origin_lat, origin_lon), aircraft_type=aircraft_type,
                  flight_id=flight_id, origin_code=origin_code)

        current_output_dim = len(env.airport_coords)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)[:, :current_output_dim]  # slicing 적용
            action = torch.argmax(q_values, dim=1).item()

        reward, valid, chosen, eta, alt, dest, origin, penalty, fuel_eff = env.step(action)

        results.append({
            "index": idx,
            "lat": lat,
            "lon": lon,
            "chosen_airport": chosen,
            "reward": float(reward),
            "valid": valid,
            "eta_min": eta,
            "alt_score": alt,
            "dest_score": dest,
            "origin_score": origin,
            "fuel_eff_score": fuel_eff,
            "penalty": penalty,
            "origin_airport": origin_code,
            "is_origin_return": (chosen == origin_code)
        })

    pd.DataFrame(results).to_csv("simulation_results_reroutefinalfuel.csv", index=False)
    print("✅ Simulation complete: results saved to simulation_results_reroutefinalfuel.csv")

if __name__ == "__main__":
    main()
