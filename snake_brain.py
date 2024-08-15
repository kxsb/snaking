
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from snake_env import *  # SnakeEnv, UP, DOWN, LEFT, RIGHT, CELL_SIZE
from snake_agent import SnakeAgent
from utils import load_model, MODEL_PATH


if __name__ == "__main__":
    print("Initialisation de l'agent...")
    agent = SnakeAgent(env_type='full')  # Force l'utilisation de SnakeEnv pour l'affichage graphique
 
    print("Chargement du modèle...")
    try:
        load_model(agent.model, MODEL_PATH)
        print("Modèle chargé avec succès.")
    except FileNotFoundError:
        print("Modèle non trouvé. Début de l'entraînement à partir de zéro.")
 
    print("Début de l'entraînement...")
    agent.train(1000, 32)
 
    print("Entraînement terminé.")
