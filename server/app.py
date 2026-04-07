from fastapi import FastAPI
import torch
import os

from cleanrl_env import DriftEnv
from train import DQN

app = FastAPI()

# ---------------------------
# LOAD ENV + MODEL
# ---------------------------
env = DriftEnv()

model = DQN()

MODEL_PATH = "model/dqn_model.pth"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Run train.py first to generate model/dqn_model.pth"
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

state = None

# ---------------------------
# ROOT
# ---------------------------
@app.get("/")
def home():
    return {"message": "DriftRL OpenEnv API running 🚀"}

# ---------------------------
# RESET ENVIRONMENT
# ---------------------------
@app.get("/reset")
def reset():
    global state
    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)

    return {
        "state": obs,
        "message": "Environment reset"
    }

# ---------------------------
# STEP (MODEL ACTION)
# ---------------------------
@app.get("/step")
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

# ---------------------------
# RUN FULL EPISODE
# ---------------------------
@app.get("/run")
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

    return {
        "initial_state": obs,
        "steps": steps,
        "message": "Cleaning completed ✅"
    }
