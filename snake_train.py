# snake_train.py/ moteur d'itération pour entrainer le snake. fonctionne avec snake_env, mais distinctement de snake_brain. peut être lancer en parallèle. 
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT, CELL_SIZE
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import sys 

MODEL_PATH = 'snake_model.pth'

# Paramètres Pygame
WINDOW_SIZE = 500  # Taille de la fenêtre
GRID_SIZE = 20     # Taille de la grille (par exemple, 20x20)
FPS = 10           # Images par seconde pour la visualisation

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
        self.env = SnakeEnv()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(768, 3)  # 12 inputs, 3 outputs (go straight, turn left, turn right)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
                # Gestion des événements Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                next_state = self.get_state()
                self.remember(state, action, reward, next_state, done)
                state = next_state

                # Rendu Pygame
                self.env.render()

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
        state_vector = torch.from_numpy(self.env.grid_state()).flatten().float()
        return state_vector
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.choice([0, 1, 2])
            print(f"Random action: {action}")
            return action
        act_values = self.model(torch.tensor(state, dtype=torch.float32))
        action = torch.argmax(act_values).item()
        print(f"Model action: {action}")
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32))).item()

            current_q_values = self.model(torch.tensor(state, dtype=torch.float32))
            target_f = current_q_values.clone().detach()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, target_f)
            print(f"Loss: {loss.item()}")  # Ajout de l'impression de la perte
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