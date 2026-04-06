import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from cleanrl_env import DriftEnv

# ---------------------------
# DQN Model
# ---------------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# TRAINING (SAFE BLOCK)
# ---------------------------
if __name__ == "__main__":

    env = DriftEnv()
    model = DQN()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    episodes = 500
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)

        done = False
        total_reward = 0

        while not done:

            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()

            next_state, reward, done, _ = env.step(action)

            next_state = torch.tensor(next_state, dtype=torch.float32)

            # ---------------------------
            # CORRECT TRAINING BLOCK
            # ---------------------------
            target = reward + gamma * torch.max(model(next_state)).item()

            predicted = model(state)[action]

            loss = loss_fn(
                predicted,
                torch.tensor(target, dtype=torch.float32)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ---------------------------

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 50 == 0:
            print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # ---------------------------
    # SAVE MODEL
    # ---------------------------
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/dqn_model.pth")

    print("Training completed ✅ Model saved.")