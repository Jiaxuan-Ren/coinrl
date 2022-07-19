import gym
import json
import datetime as dt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

# df = pd.read_csv('./data/AAPL.csv')
df = pd.read_csv('./data/BTCUSD_1h_pnl.csv')
df = df.sort_values('Date')

df = df.iloc[55000:, :].reset_index(drop=True)

split_index = int(len(df) * 0.7)

train = df.iloc[:split_index, :].reset_index(drop=True)
test = df.iloc[split_index:, :].reset_index(drop=True)
# df = df.drop(columns=["unix", "Volume_BTC", "Symbol"]).reset_index(drop=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(train)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

test_env = DummyVecEnv([lambda: StockTradingEnv(df=test, test=True)])

obs = test_env.reset()
count = 0
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    count += 1
    if count % 500 == 499:
        test_env.render()
