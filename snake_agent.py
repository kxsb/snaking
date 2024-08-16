from snake_env import SimpleSnakeEnv, SnakeEnv, UP, DOWN, LEFT, RIGHT, CELL_SIZE  # Importer les environnements de jeu
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from utils import save_model, load_model #,  plot_scores_and_losses,
# import matplotlib
# matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'WXAgg', etc. en fonction de ton système
# import matplotlib.pyplot as plt

scores = []
losses = []

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# débuggggggggg
# print("snake_train.py is being executed")

class SnakeAgent:
    def __init__(self, env_type='simple', gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=32, memory_size=10000, fc1_units=128, fc2_units=64, fc3_units=32):
        print("Initializing SnakeAgent...")
        if env_type == 'simple':
            self.env = SimpleSnakeEnv()
        else:
            self.env = SnakeEnv()

        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = DQN(12, 3)  # Assurez-vous que le modèle est bien créé ici
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        print(f"Model initialized successfully: {self.model}")
                
    def intense_training(self, iterations, model_path):
        print("Starting intense training...")
        for i in range(iterations):
            print(f"Starting iteration {i + 1}/{iterations}")
            state = self.env.reset()  # Réinitialise l'environnement avant chaque itération
            state = self.get_state()  # Assure que state est un tableau NumPy
            total_loss = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                next_state = self.get_state()  # Assure que next_state est un tableau NumPy
                self.remember(state, action, reward, next_state, done)
                state = next_state

                loss = self.replay(32)  # Apprentissage à partir de l'expérience
                total_loss += loss

                if done:
                    print(f"Iteration: {i + 1}, Score: {self.env.score}")
                    scores.append(self.env.score)
                    losses.append(total_loss)
                    # plot_scores_and_losses()
                    break

            # Réinitialisation pour la prochaine itération
            self.env.reset()
            
            # Enregistrement du modèle après chaque itération ou toutes les X itérations
            save_model(self.model, model_path)
            print(f"Model saved to {model_path}")

        print("Intense training completed.")
    
    def get_state(self):
        state = self.env.get_state()

        # Si l'état est un dictionnaire, extraire les informations nécessaires
        if isinstance(state, dict):
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
        
        # Si l'état est déjà un tableau NumPy (comme avec SnakeEnv)
        return state
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])
        act_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        minibatch = [entry for entry in random.sample(self.memory, batch_size) if entry[0] is not None and entry[3] is not None]
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            if state is None or next_state is None:
                continue  # Ignorer les transitions avec des états invalides

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32))).item()

            current_q_values = self.model(torch.tensor(state, dtype=torch.float32))
            target_f = current_q_values.clone().detach()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, target_f)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size

    def train(self, episodes, batch_size):
        try:
            for e in range(episodes):
                state = self.get_state()
                if state is None:
                    raise ValueError("L'état est None au début de l'épisode.")
                total_reward = 0

                for time in range(35000):
                    if isinstance(self.env, SnakeEnv):
                        self.env.handle_events()

                    if not self.env.paused:
                        action = self.act(state)
                    else:
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

                    if isinstance(self.env, SnakeEnv):
                        self.env.render()

        except KeyboardInterrupt:
            print("Training interrupted. Saving the model...")
            save_model(self.model)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            self.env.close()
        
    # def print_model_weights(self, model, title):
    #         print(title)
    #         for name, param in model.named_parameters():
    #             print(name, param.data)

if __name__ == "__main__":
    agent = SnakeAgent(env_type='simple')  # Initialisation de l'agent avec le modèle

    model_path = 'snake_model.pth'

    try:
        load_model(agent.model, model_path)  # Charger le modèle
        print(f"Model loaded from {model_path}.")
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")
    
    iterations = int(input("Enter the number of iterations for intense training: "))
    agent.intense_training(iterations, model_path)