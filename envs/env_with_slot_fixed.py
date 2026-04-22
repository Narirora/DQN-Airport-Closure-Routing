import numpy as np
from geopy.distance import geodesic

class RerouteEnv:
    def __init__(self, airport_coords, airport_capacities, original_destination_coord, bada_df, fuel_df=None,
                 current_time_min=0, slot_mode="fixed", max_per_slot=3):
        self.airport_coords = airport_coords
        self.airport_capacities = airport_capacities
        self.original_destination_coord = original_destination_coord
        self.bada_df = bada_df
        self.fuel_df = fuel_df
        self.current_time_min = current_time_min
        self.slot_mode = slot_mode
        self.max_per_slot = max_per_slot

        self.slot_schedule = self.generate_slot_schedule()
        self.reset_slot_usage()
        self.state = None
        self.origin_coord = None
        self.origin_code = None  # ✅ 출발 공항 코드
        self.aircraft_type = None
        self.flight_id = None

    def generate_slot_schedule(self):
        if self.slot_mode == "random":
            schedule = {}
            total_slots = set(np.arange(0, 300, 5))
            for airport in self.airport_coords:
                cap = self.airport_capacities.get(airport, 0.5)
                n_blocked = int(len(total_slots) * (1 - cap))
                blocked = set(np.random.choice(list(total_slots), size=n_blocked, replace=False))
                schedule[airport] = total_slots - blocked
        elif self.slot_mode == "fixed":
            schedule = {}
            for airport in self.airport_coords:
                cap = self.airport_capacities.get(airport, 0.5)
                slots = np.arange(0, 300, 5)
                limit = max(1, int(cap * self.max_per_slot))
                schedule[airport] = {int(s): limit for s in slots}
        else:
            raise ValueError("Invalid slot_mode")
        return schedule

    def refresh_slot_schedule(self):
        self.slot_schedule = self.generate_slot_schedule()
        self.reset_slot_usage()

    def reset_slot_usage(self):
        if self.slot_mode == "random":
            self.slot_usage = {a: {} for a in self.airport_coords}
        elif self.slot_mode == "fixed":
            self.slot_usage = {a: {s: 0 for s in self.slot_schedule[a]} for a in self.airport_coords}

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def reset(self, state_vec, origin_position, aircraft_type, flight_id=None, origin_code=None):
        self.state = {
            'lat': state_vec[0],
            'lon': state_vec[1],
            'dist_to_hnd': state_vec[2],
            'altitude': state_vec[3],
            'speed': state_vec[4],
            'fuel': state_vec[5],
            'origin_lat': origin_position[0],
            'origin_lon': origin_position[1]
        }
        self.origin_coord = origin_position
        self.origin_code = origin_code
        self.aircraft_type = aircraft_type
        self.flight_id = flight_id

        # ✅ 출발공항이 divert 후보에 없는 경우 등록
        if origin_code not in self.airport_coords:
            self.airport_coords[origin_code] = origin_position
            self.airport_capacities[origin_code] = 0.9
            slots = np.arange(0, 300, 5)
            limit = max(1, int(0.9 * self.max_per_slot))
            self.slot_schedule[origin_code] = {int(s): limit for s in slots}
            self.slot_usage[origin_code] = {int(s): 0 for s in slots}

        return self.state

    def slot_available_at(self, airport_code, arrival_min):
        if self.slot_mode == "random":
            valid_slots = self.slot_schedule.get(airport_code, set())
            check_slot = (arrival_min // 5) * 5
            return any((check_slot + delta) in valid_slots for delta in [-5, 0, 5])
        elif self.slot_mode == "fixed":
            slot_time = (arrival_min // 5) * 5
            return slot_time in self.slot_schedule.get(airport_code, {})

    def step(self, action):
        airport_list = list(self.airport_coords.keys())
        target_code = airport_list[action]
        target_coord = self.airport_coords[target_code]

        lat = self.state['lat']
        lon = self.state['lon']
        altitude_ft = self.state['altitude']

        dist_alt = self.haversine(lat, lon, target_coord[0], target_coord[1])
        dist_dest = self.haversine(lat, lon, self.original_destination_coord[0], self.original_destination_coord[1])
        dist_origin = self.haversine(lat, lon, self.origin_coord[0], self.origin_coord[1])

        if self.fuel_df is not None and self.flight_id is not None:
            match = self.fuel_df[
                (self.fuel_df['flight_id'] == self.flight_id) &
                (self.fuel_df['Diverting Airport'] == target_code) &
                (self.fuel_df['ACtype'] == self.aircraft_type)
            ]
        else:
            match = None

        if match is not None and not match.empty:
            eta = match.iloc[0]['flight_time_sec'] / 60.0
            fuel_needed = match.iloc[0]['Fuel_consumption_kg']
            fuel_remain = match.iloc[0]['Remaining_Fuel_kg']
            beta = 4500
            fuel_eff_score = np.exp(-fuel_needed / beta)
        elif target_code == "ORIGIN" and self.origin_fuel_needed is not None:
            fuel_needed = self.origin_fuel_needed
            fuel_remain = self.state['fuel']
            eta = fuel_needed / 100.0  # 대략적인 ETA 추정
            fuel_eff_score = np.exp(-fuel_needed / 4500)
        else:
            eta = 99
            fuel_needed = 0
            fuel_remain = 99999
            fuel_eff_score = 0.0

        arrival_min = self.current_time_min + eta
        slot_time = (arrival_min // 5) * 5

        airport_slot_dict = self.slot_usage[target_code]
        if self.slot_mode == "fixed":
            current_usage = airport_slot_dict.get(slot_time, 0)
            slot_limit = self.slot_schedule[target_code].get(slot_time, 1)
            congestion_penalty = max(0, (current_usage - slot_limit + 1)) * 10
        else:
            current_usage = airport_slot_dict.get(slot_time, 0)
            congestion_penalty = (current_usage ** 1.2) * 3

        alt_score = np.exp(-dist_alt / 250)
        dest_score = np.exp(-dist_dest / 450)
        origin_score = 1 / (1 + np.exp((dist_origin - 500) / 150))

        reward = (0.2 * alt_score + 0.4 * dest_score + 0.2 * origin_score + 0.2 * fuel_eff_score) * 200
        reward -= congestion_penalty

        # ✅ Passenger bonus
        passenger_data = {
            "SDJ": {"time": 190, "transfers": 4, "cost": 12880},
            "NRT": {"time": 102, "transfers": 3, "cost": 3120},
            "NGO": {"time": 170, "transfers": 4, "cost": 13060},
            "ITM": {"time": 217, "transfers": 3, "cost": 15650},
            "KIX": {"time": 247, "transfers": 3, "cost": 17880}
        }

        if target_code in passenger_data:
            pdata = passenger_data[target_code]
            max_time, max_transfers, max_cost = 247, 4, 17480
            w_time, w_transfers, w_cost = 0.4, 0.3, 0.3
            norm_time = pdata["time"] / max_time
            norm_transfers = pdata["transfers"] / max_transfers
            norm_cost = pdata["cost"] / max_cost
            discomfort = w_time * norm_time + w_transfers * norm_transfers + w_cost * norm_cost
            passenger_bonus_score = (1 - discomfort) * 10
            reward += passenger_bonus_score
        else:
            passenger_bonus_score = 0.0
        # ✅ Fuel infeasibility penalty
        if fuel_needed > fuel_remain:
            reward -= 100

        # ✅ Slot update
        airport_slot_dict[slot_time] = current_usage + 1

        # 슬롯 미사용 시 탈락
        if not self.slot_available_at(target_code, arrival_min):
            return -50, False, target_code, eta, alt_score, dest_score, origin_score, congestion_penalty, fuel_eff_score

        return reward, True, target_code, eta, alt_score, dest_score, origin_score, congestion_penalty, fuel_eff_score
