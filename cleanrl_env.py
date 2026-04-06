import pandas as pd

class DriftEnv:
    def __init__(self, file_path="data/sample.csv"):
        self.file_path = file_path
        self.df = None
        self.numeric_columns = []

    def reset(self):
        self.df = pd.read_csv(self.file_path)

        self.df.columns = self.df.columns.str.lower()

        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()

        obs = self._get_obs()
        self.prev_obs = obs

        return obs

    def step(self, action):
        reward = -0.2

        if action == 0:
            self.df = self.df.dropna()

        elif action == 1:
            self.df = self.df.drop_duplicates()

        elif action == 2:
            for col in self.numeric_columns:
                self.df = self.df[pd.to_numeric(self.df[col], errors='coerce').notnull()]

        new_obs = self._get_obs()

        reward += (sum(self.prev_obs) - sum(new_obs)) * 0.5
        done = sum(new_obs) == 0

        self.prev_obs = new_obs

        return new_obs, reward, done, {}

    def _get_obs(self):
        missing = int(self.df.isnull().sum().sum())
        duplicates = int(self.df.duplicated().sum())

        invalid = 0
        for col in self.numeric_columns:
            invalid += int(pd.to_numeric(self.df[col], errors='coerce').isnull().sum())

        return [missing, duplicates, invalid]