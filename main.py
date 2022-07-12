import gym
import json
import datetime as dt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

# df = pd.read_csv('./data/AAPL.csv')
df = pd.read_csv('./data/Gemini_BTCUSD_1h.csv')
df = df.sort_values('Date')
df = df.drop(columns=["unix", "Volume_BTC", "Symbol"]).reset_index(drop=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
count = 0
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    count += 1
    if count % 500 == 499:
        env.render()
