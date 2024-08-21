# snake_train.py/ moteur d'itération pour entrainer le snake. fonctionne avec snake_env, mais distinctement de snake_brain. peut être lancer en parallèle. 

import torch
import torch.nn as nn
import torch.optim as optim
import random

import numpy as np
from collections import deque
from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT, CELL_SIZE
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

MODEL_PATH = 'snake_model.pth'

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # attention, garder les mêmes poids pour un modèle exporté en .pth
        self.fc1 = nn.Linear(input_dim, 128) # ou 64
        self.fc2 = nn.Linear(128, 64) # ou 256 ou 128 ou 64
        self.fc3 = nn.Linear(64, output_dim) # ou 128 ou 64 ou 32


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SnakeAgent:
    def __init__(self):
        # Initialisation de l'environnement du serpent
        self.env = SnakeEnv()
        
        # Initialisation de la mémoire avec une taille maximale de 10000
        self.memory = deque(maxlen=20000)
        
        
        # Facteur de discount pour le calcul des récompenses futures
        self.gamma = 0.95
        
        # Taux d'exploration initial pour l'algorithme epsilon-greedy
        self.epsilon = 0.92
        
        # Taux d'exploration minimum
        self.epsilon_min = 0.005
        
        # Taux de décroissance de l'exploration
        self.epsilon_decay = 0.995
        
        # Taux d'apprentissage pour l'optimiseur
        self.learning_rate = 0.0007
        
        # Initialisation du modèle DQN avec 12 entrées et 3 sorties (aller tout droit, tourner à gauche, tourner à droite)
        self.model = DQN(12, 3)
        
        # Initialisation de l'optimiseur Adam avec le taux d'apprentissage spécifié
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Fonction de perte utilisant l'erreur quadratique moyenne (MSE)
        self.loss_fn = nn.MSELoss()
        
    def load_model(self, model, path=MODEL_PATH):
        model.load_state_dict(torch.load(path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {path}")

    def save_model(self, model, path=MODEL_PATH):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def intense_training(self, iterations):
        print("Starting intense training...")
        for i in range(iterations):
            print(f"Starting iteration {i + 1}/{iterations}")
            state = self.get_state()
            for time in range(5000):
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                next_state = self.get_state()
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f"Intense Training - Iteration: {i + 1}, Score: {self.env.score}")
                    self.env.reset()
                    break  # Sortie de la boucle en cas de fin de partie
                self.replay(32)
            
            print(f"Completed iteration {i + 1}/{iterations}")

        print("Intense training completed.")
        self.save_model(self.model)
        print(f"Model saved to {MODEL_PATH}")    
        
    def get_state(self):
        state = self.env.get_state()
        snake_head = state['snake'][0]
        food = state['food']

        # Appel unique pour les collisions
        collisions = [
            self.env.is_collision((snake_head[0] + CELL_SIZE, snake_head[1])),
            self.env.is_collision((snake_head[0] - CELL_SIZE, snake_head[1])),
            self.env.is_collision((snake_head[0], snake_head[1] + CELL_SIZE)),
            self.env.is_collision((snake_head[0], snake_head[1] - CELL_SIZE)),
        ]

        state_vector = torch.tensor([
            state['direction'] == UP,
            state['direction'] == DOWN,
            state['direction'] == LEFT,
            state['direction'] == RIGHT,
            snake_head[0] < food[0],
            snake_head[0] > food[0],
            snake_head[1] < food[1],
            snake_head[1] > food[1],
            *collisions,
        ], dtype=torch.float32)

        return state_vector

        # return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.choice([0, 1, 2])
            # print(f"Random action: {action}")
            return action
        act_values = self.model(torch.tensor(state, dtype=torch.float32))
        action = torch.argmax(act_values).item()
        # print(f"Model action: {action}")
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Échantillonnage d'un batch de manière groupée
        minibatch = random.sample(self.memory, batch_size)
        
        # Convertir les éléments en tenseurs une seule fois, si cela n'a pas déjà été fait
        states = torch.stack([experience[0] if isinstance(experience[0], torch.Tensor) 
                            else torch.tensor(experience[0], dtype=torch.float32) 
                            for experience in minibatch])
        
        actions = torch.tensor([experience[1] for experience in minibatch], dtype=torch.long)
        rewards = torch.tensor([experience[2] for experience in minibatch], dtype=torch.float32)
        
        next_states = torch.stack([experience[3] if isinstance(experience[3], torch.Tensor) 
                                else torch.tensor(experience[3], dtype=torch.float32) 
                                for experience in minibatch])
        
        dones = torch.tensor([experience[4] for experience in minibatch], dtype=torch.float32)

        # Calcul des cibles
        target_f = self.model(states).detach()
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        target_f[range(batch_size), actions] = targets

        # Calcul et mise à jour des gradients
        predictions = self.model(states)
        loss = self.loss_fn(predictions, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                            
    def save_model(self, model, path=MODEL_PATH):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model, path=MODEL_PATH):
        model.load_state_dict(torch.load(path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {path}")

    def print_model_weights(self, model, title):
        print(title)
        for name, param in model.named_parameters():
            print(name, param.data)
            
if __name__ == "__main__":
    agent = SnakeAgent()
    
    #agent.print_model_weights(agent.model, "Model weights before loading:")
    try:
        agent.load_model(agent.model)
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")
    #agent.print_model_weights(agent.model, "Model weights after loading:")
    
    iterations = int(input("Enter the number of iterations for intense training: "))
    agent.intense_training(iterations)
    
    #agent.print_model_weights(agent.model, "Model weights after training:")
