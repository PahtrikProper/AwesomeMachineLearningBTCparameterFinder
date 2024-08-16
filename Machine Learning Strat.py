import ccxt
import pandas as pd
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pickle
import os
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Adjustable Parameters
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 1000
INITIAL_BALANCE = 1000
NUM_EPISODES = 200
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
HIDDEN_LAYER_SIZE = 64
MIN_PROFIT_THRESHOLD = 0.004  # 0.4% profit per trade
RETRAIN_INTERVAL = 50  # Retrain model every 50 episodes

# Paths to save files
MODEL_SAVE_PATH = "q_network.pth"
REPLAY_BUFFER_SAVE_PATH = "replay_buffer.pkl"
METRICS_SAVE_PATH = "training_metrics.csv"

# Binance Data Fetching and Preprocessing
def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    dataframe = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
    dataframe.set_index('timestamp', inplace=True)
    return dataframe

def calculate_rsi(data, period):
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_willr(data, period):
    high = data['high'].rolling(window=period).max()
    low = data['low'].rolling(window=period).min()
    willr = -100 * (high - data['close']) / (high - low)
    return willr

def prepare_data(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT, rsi_period=14, willr_period=14):
    data = fetch_ohlcv(symbol, timeframe, limit)
    data['rsi'] = calculate_rsi(data, period=rsi_period)
    data['willr'] = calculate_willr(data, period=willr_period)
    data.dropna(inplace=True)
    return data

# Gym Environment
class TradingEnv(gym.Env):
    def __init__(self, dataframe, initial_balance=INITIAL_BALANCE, min_profit_threshold=MIN_PROFIT_THRESHOLD):
        self.data = dataframe
        self.initial_balance = initial_balance
        self.min_profit_threshold = min_profit_threshold
        self.current_step = 0
        self.balance = initial_balance
        self.position = None  # None means no position, 1 means long position
        self.entry_price = 0
        self.done = False
        self.max_balance = initial_balance
        self.drawdowns = []
        self.wins = 0
        self.losses = 0
        self.model = self.train_predictive_model()

    def train_predictive_model(self):
        # Prepare features and labels
        self.data['price_diff'] = self.data['close'].diff().shift(-1)
        self.data['price_movement'] = np.where(self.data['price_diff'] > 0, 1, 0)  # 1 if price will increase, else 0
        
        # Shift features to predict the next movement
        features = self.data[['rsi', 'willr', 'close']].shift(1)
        
        # Drop rows with NaN values from both features and labels to ensure alignment
        features = features.dropna()
        labels = self.data['price_movement'].loc[features.index]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train a predictive model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model

    def predict_movement(self):
        # Predict the next price movement (1 = up, 0 = down)
        current_state = self.data.iloc[self.current_step][['rsi', 'willr', 'close']].values.reshape(1, -1)
        return self.model.predict(current_state)[0]

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.done = False
        self.max_balance = self.initial_balance
        self.drawdowns = []
        self.wins = 0
        self.losses = 0
        return self._get_observation()

    def _get_observation(self):
        obs = [
            self.data.iloc[self.current_step]['rsi'],
            self.data.iloc[self.current_step]['willr'],
            self.data.iloc[self.current_step]['close'],
            1 if self.position else 0
        ]
        return np.array(obs)

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0

        if action == 1 and self.position is None:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position is not None:  # Sell
            profit = (current_price - self.entry_price) / self.entry_price
            if profit >= self.min_profit_threshold:  # Check if the trade is profitable
                self.wins += 1
                reward = profit * 100  # Reward is the profit percentage
            else:
                self.losses += 1
                reward = -1  # Penalize for losses or small gains
            self.balance += profit * self.balance
            self.position = None

        # Update max balance and drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        drawdown = (self.max_balance - self.balance) / self.max_balance
        self.drawdowns.append(drawdown)

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        pass

# DQN Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size=HIDDEN_LAYER_SIZE):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training Loop
def train_dqn(env, num_episodes=NUM_EPISODES, gamma=GAMMA, epsilon=EPSILON_START, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN, batch_size=BATCH_SIZE, retrain_interval=RETRAIN_INTERVAL):
    input_dim = env.reset().shape[0]
    output_dim = 3  # Three possible actions: Hold, Buy, Sell
    q_network = QNetwork(input_dim, output_dim)
    optimizer = optim.Adam(q_network.parameters())
    loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=2000)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(output_dim)  # Random action
            else:
                action = torch.argmax(q_network(torch.FloatTensor(state))).item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward

            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
                rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

                q_values = q_network(states).gather(1, actions)
                next_q_values = q_network(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Retrain the predictive model periodically
        if (episode + 1) % retrain_interval == 0:
            env.model = env.train_predictive_model()

    return q_network

# Simulated Live Trading with Continuous Learning
def simulated_live_trading(env, trained_q_network, replay_buffer, optimizer, loss_fn, num_steps=1000, update_interval=10):
    state = env.reset()
    total_reward = 0
    epsilon = EPSILON_MIN  # Start with a lower exploration rate during live trading
    
    for step in range(num_steps):
        # Select action using the trained Q-network
        if np.random.rand() < epsilon:
            action = np.random.choice(3)  # Random action: 0 = Hold, 1 = Buy, 2 = Sell
        else:
            action = torch.argmax(trained_q_network(torch.FloatTensor(state))).item()

        # Take the action and observe the result
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Store the experience in the replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state if not done else env.reset()

        # Fetch new live data (replace env.data with the latest market data)
        new_data = fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=1)
        env.data = pd.concat([env.data, new_data]).drop_duplicates()  # Ensure no duplicate data

        # Update the Q-network periodically
        if len(replay_buffer) >= BATCH_SIZE and step % update_interval == 0:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

            q_values = trained_q_network(states).gather(1, actions)
            next_q_values = trained_q_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reset the environment if done
        if done:
            state = env.reset()

    print(f"Simulated Live Trading Total Reward: {total_reward}")

# Evaluation Function for Bayesian Optimization
def evaluate_trading_agent(env, trained_q_network):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = torch.argmax(trained_q_network(torch.FloatTensor(state))).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    # Calculate the final metrics
    win_loss_ratio = env.wins / max(1, env.losses)
    max_drawdown = max(env.drawdowns) if env.drawdowns else 0

    print(f"Evaluation completed: Wins: {env.wins}, Losses: {env.losses}, Win/Loss Ratio: {win_loss_ratio}, Max Drawdown: {max_drawdown}")

    # The objective is to maximize reward, win/loss ratio, and minimize drawdown
    return -(total_reward - (max_drawdown * 100) + (win_loss_ratio * 10))

# Define the hyperparameter space
space  = [
    Integer(10, 20, name='rsi_period'),
    Integer(10, 20, name='willr_period'),
]

# Objective function to minimize
@use_named_args(space)
def objective(rsi_period, willr_period):
    dataframe = prepare_data(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT, rsi_period=rsi_period, willr_period=willr_period)
    env = TradingEnv(dataframe)
    trained_q_network = train_dqn(env)
    evaluation_reward = evaluate_trading_agent(env, trained_q_network)
    return evaluation_reward  # We now minimize based on the custom reward function

# Perform Bayesian Optimization
res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)

# Display the optimal parameters found
print("\nBest hyperparameters found:")
print(f"RSI_PERIOD: {res_gp.x[0]}")
print(f"WILLR_PERIOD: {res_gp.x[1]}")
print(f"Best Reward: {-res_gp.fun}")

# Function to save the model and data
def save_state(q_network, replay_buffer, metrics):
    # Save Q-network weights
    torch.save(q_network.state_dict(), MODEL_SAVE_PATH)
    # Save replay buffer
    with open(REPLAY_BUFFER_SAVE_PATH, 'wb') as f:
        pickle.dump(replay_buffer, f)
    # Save metrics (e.g., total rewards, win/loss ratios, drawdowns)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_SAVE_PATH, index=False)
    print("Model, replay buffer, and metrics saved successfully.")

# Function to load the model and data
def load_state(q_network):
    # Load Q-network weights if available
    if os.path.exists(MODEL_SAVE_PATH):
        q_network.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Q-network loaded successfully.")
    else:
        print("No saved Q-network found.")

    # Load replay buffer if available
    if os.path.exists(REPLAY_BUFFER_SAVE_PATH):
        with open(REPLAY_BUFFER_SAVE_PATH, 'rb') as f:
            replay_buffer = pickle.load(f)
        print("Replay buffer loaded successfully.")
    else:
        replay_buffer = deque(maxlen=2000)
        print("No saved replay buffer found, starting with an empty buffer.")

    # Load metrics if available
    if os.path.exists(METRICS_SAVE_PATH):
        metrics_df = pd.read_csv(METRICS_SAVE_PATH)
        metrics = metrics_df.to_dict(orient='list')
        print("Metrics loaded successfully.")
    else:
        metrics = {"total_rewards": [], "win_loss_ratios": [], "max_drawdowns": []}
        print("No saved metrics found, starting with empty metrics.")

    return q_network, replay_buffer, metrics

# Updated Main Execution Block with Saving and Loading Functionality
if __name__ == "__main__":
    # Prepare the initial data
    dataframe = prepare_data(SYMBOL, TIMEFRAME, LIMIT)
    env = TradingEnv(dataframe)
    
    # Initialize Q-network
    q_network = QNetwork(input_dim=4, output_dim=3, hidden_layer_size=HIDDEN_LAYER_SIZE)
    
    # Load previous state if available
    q_network, replay_buffer, metrics = load_state(q_network)

    # Train the Q-network with continuous learning
    optimizer = optim.Adam(q_network.parameters())
    loss_fn = nn.MSELoss()

    # Simulated live trading with continuous learning
    simulated_live_trading(env, q_network, replay_buffer, optimizer, loss_fn, num_steps=1000)

    # Save the current state
    save_state(q_network, replay_buffer, metrics)

# pip install ccxt pandas numpy gym torch scikit-optimize scikit-learn
