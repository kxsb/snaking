import torch
import random
import numpy as np
from snake_env import SimpleSnakeEnv
from utils import load_model, save_model
from snake_agent import SnakeAgent

class TrainAIFast:
    def __init__(self, model_path, params):
        self.env = SimpleSnakeEnv()
        self.params = params
        self.agent = SnakeAgent(
            env_type='simple',
            gamma=self.params.get('gamma', 0.99),
            epsilon=self.params.get('epsilon', 1.0),
            epsilon_min=self.params.get('epsilon_min', 0.01),
            epsilon_decay=self.params.get('epsilon_decay', 0.995),
            learning_rate=self.params.get('learning_rate', 0.001),
            batch_size=self.params.get('batch_size', 32),
            memory_size=self.params.get('memory_size', 10000),
            fc1_units=self.params.get('fc1', 128),
            fc2_units=self.params.get('fc2', 64),
            fc3_units=self.params.get('fc3', 32),
        )
        self.model_path = model_path
        self.scores = []
        self.rewards = []
        self.penalties = []
        
        # Charger le modèle si disponible
        if model_path:
            try:
                load_model(self.agent.model, model_path)
                print(f"Modèle chargé depuis {model_path}.")
            except FileNotFoundError:
                print("Modèle non trouvé. Début de l'entraînement à partir de zéro.")

    # def train(self, episodes=None):
    #     if episodes is None:
    #         episodes = self.params.get('iterations', 500)

    #     print(f"Début de l'entraînement rapide pour {episodes} épisodes...")
    #     for e in range(episodes):
    #         state = self.agent.get_state()
    #         total_reward = 0
    #         done = False

    #         while not done:
    #             action = self.agent.act(state)
    #             next_state, reward, done = self.env.step(action)
    #             self.agent.remember(state, action, reward, next_state, done)
    #             state = next_state
    #             total_reward += reward

    #             if done:
    #                 print(f"Épisode {e + 1}/{episodes} terminé avec un score de {self.env.score}")
    #                 self.env.reset()

    #         # Entraînement du modèle après chaque épisode
    #         self.agent.replay(self.params.get('batch_size', 32))
                        
    #         print(f"An unexpected error occurred: {e}")
    #         print("Entraînement rapide terminé.")
    #         save_model(self.agent.model, self.model_path)  # Sauvegarde du modèle
    #         print(f"Modèle sauvegardé dans {self.model_path}.")
        
