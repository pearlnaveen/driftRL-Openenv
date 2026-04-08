from fastapi import FastAPI
import torch
import os

from cleanrl_env import DriftEnv
from train import DQN

app = FastAPI()

env = DriftEnv()
model = DQN()

MODEL_PATH = "model/dqn_model.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

state = None

# ROOT (both GET & POST)
@app.api_route("/", methods=["GET", "POST"])
def root():
    return {"status": "ok"}

# RESET
@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    global state
    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    return {"state": obs}

# STEP
@app.api_route("/step", methods=["GET", "POST"])
def step():
    global state

    if state is None:
        return {"error": "Call /reset first"}

    with torch.no_grad():
        action = torch.argmax(model(state)).item()

    next_state, reward, done, _ = env.step(action)
    state = torch.tensor(next_state, dtype=torch.float32)

    return {
        "action": action,
        "state": next_state,
        "reward": float(reward),
        "done": done
    }

# RUN
@app.api_route("/run", methods=["GET", "POST"])
def run():
    global state

    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)

    steps = []
    done = False

    while not done:
        with torch.no_grad():
            action = torch.argmax(model(state)).item()

        next_state, reward, done, _ = env.step(action)

        steps.append({
            "action": action,
            "state": next_state,
            "reward": float(reward)
        })

        state = torch.tensor(next_state, dtype=torch.float32)

    return {"steps": steps}

def main():
    return app
if __name__ == "__main__":
    main()
