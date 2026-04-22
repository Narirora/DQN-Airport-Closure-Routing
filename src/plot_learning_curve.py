import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) reward component 로그 불러오기
df = pd.read_csv('train_reward_component_log_fuel.csv')

# 2) 에폭별 reward 합산
epoch_rewards = df.groupby('epoch')['reward'].sum().values

# 3) 이동평균 (예: window=50)
window = 50
smoothed = np.convolve(epoch_rewards, np.ones(window)/window, mode='valid')

# 4) 그래프 출력
plt.figure(figsize=(10, 6))
plt.plot(epoch_rewards, alpha=0.3, label='Raw Reward')
plt.plot(range(window-1, len(epoch_rewards)), smoothed,
         label=f'Smoothed (window={window})', linewidth=2)
plt.title('DQN Training Reward Curve (fuel)')
plt.xlabel('Epoch')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 5) 이미지 저장 및 출력
plt.savefig('dqn_reward_curve_from_csv_fuel.png')
plt.show()
