# driftRL-Openenv
DriftRL OpenEnv is a reinforcement learning-based system designed to automate the process of data cleaning by modeling it as a sequential decision-making problem.

In real-world scenarios, datasets continuously evolve and degrade over time due to missing values, duplicate records, and invalid or inconsistent entries — a phenomenon commonly referred to as data drift. Traditional data cleaning approaches rely on static, rule-based methods that are often inefficient, non-adaptive, and require manual intervention.

This project addresses the problem by leveraging Reinforcement Learning (RL), where an intelligent agent learns optimal strategies for cleaning data through interaction with an environment. The dataset is treated as the environment, and its condition is represented as a state consisting of metrics such as the number of missing values, duplicates, and invalid entries.

At each step, the agent selects an action — such as removing missing values, eliminating duplicates, or correcting invalid data — and receives a reward based on the improvement in data quality. Over time, the agent learns the most efficient sequence of actions to transform a noisy dataset into a clean and usable form.

The system is built using a Deep Q-Network (DQN) implemented in PyTorch, enabling it to approximate optimal policies for different data conditions. It supports real-world datasets such as the Titanic dataset and can dynamically adapt to varying data distributions without hardcoded rules.

To ensure accessibility and scalability, the project exposes its functionality through a FastAPI-based REST API, allowing users to interact with the environment using endpoints like /reset, /step, and /run. Additionally, the entire pipeline is containerized using Docker, making it easy to deploy and reproduce across different platforms, including Hugging Face Spaces.

Key Highlights:
Reinforcement Learning-based data cleaning approach
Dynamic and adaptive decision-making (no fixed rules)
Supports real-world datasets with missing, duplicate, and invalid values
FastAPI interface for interactive usage
Dockerized for reproducibility and deployment
Scalable and extensible architecture

This project demonstrates how AI can be applied beyond prediction tasks to automate data preprocessing, which is one of the most time-consuming steps in the data science pipeline.
