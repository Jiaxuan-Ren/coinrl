import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 100000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 1000000

MAX_PNL_SHIFT = 100


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, test=False):
        super(StockTradingEnv, self).__init__()
        self.test = test
        # print(df)

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, 2 + MAX_OPEN_POSITIONS), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        MAX_OPEN_POSITIONS, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        MAX_OPEN_POSITIONS, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        MAX_OPEN_POSITIONS, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        MAX_OPEN_POSITIONS, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        MAX_OPEN_POSITIONS, 'Volume'].values / MAX_NUM_SHARES,
        ])
        # print(frame)
        # print(frame.shape)
        additional_info = np.array([
            self.net_worth / MAX_ACCOUNT_BALANCE,
            0,
            self.shares_held / MAX_NUM_SHARES,
            0,
            0,
        ]).reshape(5, 1)
        # print(additional_info.shape)

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, additional_info, axis=1)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(
        #     self.df.loc[self.current_step, "Low"], self.df.loc[self.current_step, "High"])
        current_price = self.df.loc[self.current_step, "Close"]

        action_type = action[0]
        amount = action[1]
        reward = 0
        if action_type < 1:
            # Buy amount % of balance in shares
            if self.shares_held >= 0:
                total_possible = self.balance / current_price
            else:
                total_possible = self.balance / current_price
            shares_bought = total_possible * amount
            reward = shares_bought * self.df.loc[self.current_step, "pnl_1"] * 0.5 + shares_bought * \
                self.df.loc[self.current_step, "pnl_5"] * 0.25 + \
                shares_bought * self.df.loc[self.current_step, "pnl_10"] * 0.15 + \
                shares_bought * self.df.loc[self.current_step, "pnl_100"] * 0.1
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.shares_held += shares_bought

        elif action_type > 2:
            # Sell amount % of shares held
            # shares_sold = self.shares_held * amount
            # self.balance += shares_sold * current_price
            # self.shares_held -= shares_sold
            # self.total_shares_sold += shares_sold
            # self.total_sales_value += shares_sold * current_price
            if self.shares_held >= 0:
                total_possible = (self.net_worth * 2 -
                                  self.balance) / current_price
            else:
                total_possible = (
                    self.net_worth + self.shares_held * current_price) / current_price
            shares_sold = total_possible * amount
            reward = -(shares_sold * self.df.loc[self.current_step, "pnl_1"] * 0.1 + shares_sold *
                       self.df.loc[self.current_step, "pnl_5"] * 0.1 +
                       shares_sold * self.df.loc[self.current_step, "pnl_10"] * 0.3 + shares_sold * self.df.loc[self.current_step, "pnl_100"] * 0.5)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        # if reward != 0:
        #     print("reward is not 0")
        reward += self.shares_held * \
            self.df.loc[self.current_step, "pnl_10"] * 0.01
        return reward

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - (1 + MAX_OPEN_POSITIONS + MAX_PNL_SHIFT):
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        # reward = self.balance * delay_modifier
        # reward = self._get_reward()
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - (1 + MAX_OPEN_POSITIONS + MAX_PNL_SHIFT))
        if self.test:
            self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} ')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
