import torch
import torch.nn as nn
import torch.optim as optim
import json
from tkinter import Tk, filedialog
from snake_env import SimpleSnakeEnv, CELL_SIZE, UP, DOWN, LEFT, RIGHT
import numpy as np
import os
from create_new_model import MODEL_CONFIGS
import time

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_dims):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_size)
        self.value = nn.Linear(hidden_dims[1], 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.fc3(x)
        value = self.value(x)
        return policy, value

class TrainAIFast:  
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.env = None  # Initialiser l'environnement

    def create_model(self, grid_size):
        if grid_size in MODEL_CONFIGS:
            config = MODEL_CONFIGS[grid_size]
            model = ActorCritic(config['input_dim'], config['output_dim'], config['layers'])
            print(f"Modèle créé avec la grille de taille : {grid_size}")
            return model
        else:
            print(f"Taille de grille {grid_size} non supportée. Utilisation de la taille par défaut (20, 20).")
            return self.create_model((20, 20))  # Utilisation de la taille par défaut            
    
    def load_model_and_check_grid(self, model_path):
        model_data = torch.load(model_path)
        
        if 'grid_size' in model_data:
            model_grid_size = model_data['grid_size']
        else:
            print("Erreur : Les métadonnées du modèle ne contiennent pas 'grid_size'.")
            return False

        if model_grid_size in MODEL_CONFIGS:
            print(f"La taille de la grille {model_grid_size} est correcte pour ce modèle.")
            self.env = SimpleSnakeEnv(grid_size=model_grid_size)
            self.model = self.create_model(model_grid_size)
            return True
        else:
            print(f"Taille de grille {model_grid_size} non supportée, ajustement à (20, 20).")
            self.env = SimpleSnakeEnv(grid_size=(20, 20))
            self.model = self.create_model((20, 20))
            return True        
    
    def load_hyperparameters(self, params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        return params

    def choisir_action(self, policy, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits, _ = policy(state)
        
        # Vérifiez si des valeurs invalides existent dans les logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Logits contains NaN or inf: {logits}")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)

        probas = torch.softmax(logits, dim=1)

        # Vérifiez si des valeurs invalides existent dans les probabilités
        if torch.isnan(probas).any() or torch.isinf(probas).any() or (probas < 0).any():
            print(f"Probabilities contain NaN, inf, or < 0: {probas}")
            probas = torch.clamp(probas, min=0, max=1)
            probas = probas / probas.sum()  # Renormaliser
        
        action = torch.multinomial(probas, 1).item()
        print(f"Action choisie : {action}, probabilités : {probas}")
        return action
    
    def mise_a_jour(self, policy, optimizer, rewards, log_probs, values, gamma=0.99):
        R = 0
        policy_loss = []
        value_loss = []
        returns = []

        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.smooth_l1_loss(value, torch.tensor([[R]])))

        # Vérification des listes avant de les empiler
        if not policy_loss or not value_loss:
            raise RuntimeError("Les listes policy_loss ou value_loss sont vides. Vérifiez que les transitions sont correctement enregistrées.")

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()

    def flatten_state(self, state):
        head = state['snake'][0]
        flat_state = [
            state['direction'] == UP,
            state['direction'] == DOWN,
            state['direction'] == LEFT,
            state['direction'] == RIGHT,
            head[0] < state['food'][0],
            head[0] > state['food'][0],
            head[1] < state['food'][1],
            head[1] > state['food'][1],
            self.env.is_collision((head[0] + CELL_SIZE, head[1])),
            self.env.is_collision((head[0] - CELL_SIZE, head[1])),
            self.env.is_collision((head[0], head[1] + CELL_SIZE)),
            self.env.is_collision((head[0], head[1] - CELL_SIZE)),
        ]
        return np.array(flat_state, dtype=np.float32)

    def train_ai(self, model_path, params):
        if not self.load_model_and_check_grid(model_path):
            print("Incompatibilité entre le modèle et la grille. Arrêt de l'entraînement.")
            return
        
        optimizer = optim.Adam(self.model.parameters(), lr=params.get('learning_rate', 0.001))
        print("Modèle et grille compatibles. Début de l'entraînement...")
        
        start_time = time.time()  # Pour calculer le temps restant

        for episode in range(self.iterations):
            state = self.env.reset()
            flattened_state = self.flatten_state(state)
            rewards, log_probs, values = [], [], []

            for t in range(100):  # Limiter à 100 étapes par épisode
                action = self.choisir_action(self.model, flattened_state)
                
                output_policy = self.model(torch.from_numpy(flattened_state).float().unsqueeze(0))[0].squeeze(0)
                if action >= output_policy.size(0):
                    raise IndexError(f"Action {action} est hors des limites pour la sortie de taille {output_policy.size(0)}.")

                log_prob = torch.log_softmax(output_policy, dim=-1)[action]
                log_probs.append(log_prob)
                _, value = self.model(torch.from_numpy(flattened_state).float().unsqueeze(0))
                values.append(value)

                next_state, reward, done = self.env.step(action)
                rewards.append(reward)

                if done:
                    break

                flattened_state = self.flatten_state(next_state)

            self.mise_a_jour(self.model, optimizer, rewards, log_probs, values, gamma=params.get('gamma', 0.99))
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (episode + 1)) * (self.iterations - (episode + 1))
            print(f"Épisode {episode + 1}/{self.iterations} terminé. Score: {len(self.env.snake) - 1}")
            print(f"Temps écoulé: {elapsed_time:.2f}s, Temps restant estimé: {remaining_time:.2f}s")

        torch.save({'state_dict': self.model.state_dict(), 'grid_size': self.env.grid_size}, model_path)
        print(f"Modèle entraîné sauvegardé à {model_path}")
        
    def train_ai_iteration(self, current_iteration, model_path, params):
        if current_iteration == 0:
            if not self.load_model_and_check_grid(model_path):
                print("Incompatibilité entre le modèle et la grille. Arrêt de l'entraînement.")
                return

        optimizer = optim.Adam(self.model.parameters(), lr=params.get('learning_rate', 0.001))
        
        state = self.env.reset()
        flattened_state = self.flatten_state(state)
        rewards, log_probs, values = [], [], []

        for t in range(100):  # Limiter à 100 étapes par épisode
            action = self.choisir_action(self.model, flattened_state)
            
            output_policy = self.model(torch.from_numpy(flattened_state).float().unsqueeze(0))[0].squeeze(0)
            if action >= output_policy.size(0):
                raise IndexError(f"Action {action} est hors des limites pour la sortie de taille {output_policy.size(0)}.")

            log_prob = torch.log_softmax(output_policy, dim=-1)[action]
            log_probs.append(log_prob)
            _, value = self.model(torch.from_numpy(flattened_state).float().unsqueeze(0))
            values.append(value)

            next_state, reward, done = self.env.step(action)
            rewards.append(reward)

            if done:
                break

            flattened_state = self.flatten_state(next_state)

        self.mise_a_jour(self.model, optimizer, rewards, log_probs, values, gamma=params.get('gamma', 0.99))
        print(f"Épisode {current_iteration + 1}/{self.iterations} terminé. Score: {len(self.env.snake) - 1}")

        if current_iteration == self.iterations - 1:
            torch.save({'state_dict': self.model.state_dict(), 'grid_size': self.env.grid_size}, model_path)
            print(f"Modèle entraîné sauvegardé à {model_path}")
            
if __name__ == "__main__":
    Tk().withdraw()

    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    params_file = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

    if file_path and params_file:
        train_ai = TrainAIFast()
        params = train_ai.load_hyperparameters(params_file)
        train_ai.train_ai(iterations=1000, model_path=file_path, params=params)
    else:
        print("Veuillez sélectionner les fichiers nécessaires.")