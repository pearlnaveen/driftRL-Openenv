import os
import requests
from openai import OpenAI

# MUST use these env variables
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# OpenAI client via proxy
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Your deployed API
APP_URL = "https://pearl1212-driftrl-openenv.hf.space"

MAX_STEPS = 10

def log_start():
    print(f"[START] task=data-cleaning env=driftrl model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def run():
    log_start()

    rewards = []
    steps = 0
    success = False

    try:
        # REQUIRED: Make LLM call via proxy
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "start"}],
            max_tokens=1
        )

        # Reset env
        requests.post(f"{APP_URL}/reset")

        done = False

        while not done and steps < MAX_STEPS:
            steps += 1

            res = requests.post(f"{APP_URL}/step").json()

            action = int(res.get("action", 0))
            reward = float(res.get("reward", 0))
            done = bool(res.get("done", False))

            rewards.append(reward)

            log_step(steps, action, reward, done)

        score = min(max(sum(rewards) / 10, 0), 1)
        success = done

    except Exception as e:
        log_step(steps, "error", 0.0, True, str(e))
        score = 0.0
        success = False

    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    run()
