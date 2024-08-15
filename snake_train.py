# snake_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from snake_env import SimpleSnakeEnv, UP, DOWN, LEFT, RIGHT, CELL_SIZE
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from utils import load_model, save_model


MODEL_PATH = 'snake_model.pth'

# Créer la figure et les sous-graphiques une seule fois
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
scores = []
losses = []

# Initialisation pour l'affichage en temps réel
plt.ion()

def plot_scores_and_losses():
    ax1.clear()  # Efface le sous-graphe des scores
    ax2.clear()  # Efface le sous-graphe des pertes
    
    # Plot des scores
    ax1.plot(scores, label='Score')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Score')
    ax1.set_title('Scores Over Time')
    ax1.legend()

    # Plot des pertes
    ax2.plot(losses, label='Loss')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Over Time')
    ax2.legend()

    fig.canvas.draw()
    plt.pause(0.001)  # Pause pour mettre à jour l'affichage
        
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
        self.env = SimpleSnakeEnv()  # Utilisez l'environnement simplifié pour l'entraînement
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(12, 3)  # 12 inputs, 3 outputs (go straight, turn left, turn right)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
            
    def intense_training(self, iterations):
        print("Starting intense training...")
        for i in range(iterations):
            print(f"Starting iteration {i + 1}/{iterations}")
            state = self.get_state()
            total_loss = 0
            for time in range(5000):
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                next_state = self.get_state()
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f"Intense Training - Iteration: {i + 1}, Score: {self.env.score}")
                    scores.append(self.env.score)  # Ajouter le score à la liste
                    self.env.reset()
                    break  # Sortie de la boucle en cas de fin de partie
                
                loss = self.replay(32)
                total_loss += loss
            
            losses.append(total_loss)  # Ajouter la perte totale à la liste
            
            plot_scores_and_losses()

            if plt.waitforbuttonpress(0.1):  # Si une touche est pressée, interrompre l'entraînement
                print("Training interrupted by user")
                break

        print("Intense training completed.")
        save_model(self.model)
        print(f"Model saved to {MODEL_PATH}")
                    
    def get_state(self):
        state = self.env.get_state()
        snake = state['snake']
        food = state['food']
        direction = state['direction']
        head = snake[0]

        state = [
            direction == UP,
            direction == DOWN,
            direction == LEFT,
            direction == RIGHT,
            head[0] < food[0],  # food right
            head[0] > food[0],  # food left
            head[1] < food[1],  # food down
            head[1] > food[1],  # food up
            self.env.is_collision((head[0] + CELL_SIZE, head[1])),  # right collision
            self.env.is_collision((head[0] - CELL_SIZE, head[1])),  # left collision
            self.env.is_collision((head[0], head[1] + CELL_SIZE)),  # down collision
            self.env.is_collision((head[0], head[1] - CELL_SIZE)),  # up collision
        ]

        return np.array(state, dtype=int)

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
    
    def train(self, episodes, batch_size):
        rewards_history = []

        try:
            for e in range(episodes):
                state = self.get_state()
                if state is None:
                    raise ValueError("L'état est None au début de l'épisode.")
                total_reward = 0

                for time in range(35000):
                    self.env.handle_events()  # Gérer les événements Pygame
                    if not self.env.paused:
                        action = self.act(state)
                        
                        try:
                            next_state, reward, done = self.env.step(action)
                        except Exception as e:
                            raise e
                        
                        if next_state is None:
                            raise ValueError("L'état suivant est None après une action.")
                        
                        state = next_state
                        self.remember(state, action, reward, next_state, done)
                        total_reward += reward

                        if done:
                            self.env.reset()
                            break

                        self.env.render()

        except KeyboardInterrupt:
            print("Training interrupted. Saving the model...")
            save_model(self.model)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            self.env.close()  # Fermer proprement Pygame

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32))).item()

            current_q_values = self.model(torch.tensor(state, dtype=torch.float32))
            target_f = current_q_values.clone().detach()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, target_f)
            # print(f"Loss: {loss.item()}")  # Affiche la perte à chaque mini-lot
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()  # Cumul de la perte

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size  # Retourne la perte moyenne pour ce batch

    def print_model_weights(self, model, title):
        print(title)
        for name, param in model.named_parameters():
            print(name, param.data)

if __name__ == "__main__":
    agent = SnakeAgent()
    
    try:
        load_model(agent.model)  # Corrigez ceci pour appeler la fonction importée
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")
    
    iterations = int(input("Enter the number of iterations for intense training: "))
    agent.intense_training(iterations)
