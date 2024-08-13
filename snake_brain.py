import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from snake_env import * #SnakeEnv, UP, DOWN, LEFT, RIGHT, CELL_SIZE
import matplotlib.pyplot as plt

MODEL_PATH = 'snake_model.pth' # Chemin par défaut pour sauvegarder le modèle
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Définition des couches du réseau de neurones
        self.fc1 = nn.Linear(input_dim, 128)  # Première couche entièrement connectée
        self.fc2 = nn.Linear(128, 64)  # Deuxième couche entièrement connectée
        self.fc3 = nn.Linear(64, output_dim)  # Troisième couche (sortie)
    def forward(self, x):
        # Fonction de passage avant du réseau (calcul des sorties)
        x = torch.relu(self.fc1(x))  # Activation ReLU pour la première couche
        x = torch.relu(self.fc2(x))  # Activation ReLU pour la deuxième couche
        x = self.fc3(x)  # Pas d'activation pour la couche de sortie
        return x
        
# à intégrer dans l'interace graphique ultérieurement
class SnakeAgent:
    def __init__(self):
        self.env = SnakeEnv()  # Initialisation de l'environnement
        self.memory = deque(maxlen=100000)  # Mémoire pour stocker les expériences passées
        self.gamma = 0.99  # Facteur de réduction pour les récompenses futures
        self.epsilon = 0.98  # Probabilité initiale pour l'exploration (choix aléatoire d'actions)
        self.epsilon_min = 0.02  # Probabilité minimale pour epsilon (limite inférieure d'exploration)
        self.epsilon_decay = 0.995  # Taux de décroissance pour epsilon
        self.learning_rate = 0.017  # Taux d'apprentissage pour l'optimiseur
        self.model = DQN(12, 3)  # Modèle DQN avec 12 entrées et 3 sorties (actions possibles)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Optimiseur Adam
        self.loss_fn = nn.MSELoss()  # Fonction de perte (Mean Squared Error)

    def intense_training(self, iterations):
        #débogage# print("Starting intense training...")
        for i in range(iterations):
            state = self.get_state()
            for time in range(25000):
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                next_state = self.get_state()
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    #débogage# print(f"Intense Training - Iteration: {i}, Score: {self.env.score}")
                    self.env.reset()
                    break
                self.replay(32)
        #débogage# print("Intense training completed.")
        self.save_model(self.model)
        #débogage# print(f"Model saved to {MODEL_PATH}")
        
    def get_state(self):
        state = self.env.get_state()
        if state is None:
            raise ValueError("L'état retourné est None, ce qui ne devrait pas être le cas.")
        
        snake = state['snake']
        food = state['food']
        direction = state['direction']
        head = snake[0]

        state_vector = [
            direction == UP,
            direction == DOWN,
            direction == LEFT,
            direction == RIGHT,
            head[0] < food[0],
            head[0] > food[0],
            head[1] < food[1],
            head[1] > food[1],
            self.env.is_collision((head[0] + CELL_SIZE, head[1])),
            self.env.is_collision((head[0] - CELL_SIZE, head[1])),
            self.env.is_collision((head[0], head[1] + CELL_SIZE)),
            self.env.is_collision((head[0], head[1] - CELL_SIZE)),
        ]
        
        return np.array(state_vector, dtype=float)        
    
        # Ajouter une micropunition aléatoire si le corps n'est pas aligné
        #if not aligned_x and not aligned_y:
        #    micropunition = random.uniform(-0.01, -0.211)  # Micropunition aléatoire
        #    state_vector.append(micropunition)
        #else:
        #    state_vector.append(0.0)  # Pas de punition si aligné
        
    def get_new_direction(self, action):
        current_direction = self.env.direction
        #débogage# print(f"Current direction: {current_direction}, Action taken: {action}")

        if current_direction == UP:
            if action == 0:  # Go straight
                return UP
            elif action == 1:  # Turn left
                return LEFT
            elif action == 2:  # Turn right
                return RIGHT

        elif current_direction == DOWN:
            if action == 0:  # Go straight
                return DOWN
            elif action == 1:  # Turn left
                return RIGHT
            elif action == 2:  # Turn right
                return LEFT

        elif current_direction == LEFT:
            if action == 0:  # Go straight
                return LEFT
            elif action == 1:  # Turn left
                return DOWN
            elif action == 2:  # Turn right
                return UP

        elif current_direction == RIGHT:
            if action == 0:  # Go straight
                return RIGHT
            elif action == 1:  # Turn left
                return UP
            elif action == 2:  # Turn right
                return DOWN
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if state is None:
            raise ValueError("L'état est None, impossible de continuer.")

        # S'assurer que state est un numpy array dès le départ
        if isinstance(state, dict):
            state_vector = self.get_state_vector(state)
        else:
            state_vector = state

        if np.random.rand() <= self.epsilon:
            action = random.choice([0, 1, 2])
            #débogage# print(f"Random action: {action}")
        else:
            act_values = self.model(torch.tensor(state_vector, dtype=torch.float32))
            action = torch.argmax(act_values).item()
            #débogage# print(f"Model action: {action}")
        
        new_direction = self.get_new_direction(action)
        #débogage# print(f"Action: {action}, New Direction: {new_direction}")

        return action
        
    def replay(self, batch_size):
        #débogage# print("start replay")

        if len(self.memory) < batch_size:
            #débogage# print(f"Replay aborted: Not enough memory. Memory size: {len(self.memory)}, Required: {batch_size}")
            return
        #débogage# print(f"Batch checked: Memory size is sufficient. Memory size: {len(self.memory)}, Batch size: {batch_size}")
        
        minibatch = random.sample(self.memory, batch_size)
        
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            #débogage# print(f"\nProcessing minibatch item {index + 1}/{batch_size}")
            #débogage# print(f"State: {state}")
            #débogage# print(f"Action: {action}")
            #débogage# print(f"Reward: {reward}")
            #débogage# print(f"Next State: {next_state}")
            #débogage# print(f"Done: {done}")

            target = reward
            
            if not done:
                #débogage# print("Calculating target with Q-learning formula")
                try:
                    next_state_vector = self.get_state_vector(next_state)  # Accède à l'état de manière sûre
                    next_state_tensor = torch.tensor(next_state_vector, dtype=torch.float32)
                    #débogage# print(f"Next state tensor: {next_state_tensor}")  # Ajout de vérification
                    max_future_q = torch.max(self.model(next_state_tensor)).item()
                    target = reward + self.gamma * max_future_q
                    #débogage# print(f"Target after Q-learning update: {target}")
                except Exception as e:
                    #débogage# print(f"Error when processing next_state: {e}")
                    raise e

            try:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                #débogage# print(f"State tensor: {state_tensor}")  # Ajout de vérification
                current_q_values = self.model(state_tensor) 
                #débogage# print(f"Current Q-values: {current_q_values.detach().numpy()}")
            except Exception as e:
                #débogage# print(f"Error when processing state: {e}")
                raise e

            target_f = current_q_values.clone().detach()
            target_f[0][action] = target
            #débogage# print(f"Updated target Q-values: {target_f.numpy()}")

            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, target_f)
            #débogage# print(f"Loss: {loss.item()}")
            
            loss.backward()
            self.optimizer.step()
            #débogage# print("Completed backpropagation and optimizer step")
        
        if self.epsilon > self.epsilon_min:
            old_epsilon = self.epsilon
            self.epsilon *= self.epsilon_decay
            #débogage# print(f"Epsilon decayed from {old_epsilon} to {self.epsilon}")

    def get_state_vector(self, state_dict):
        """Convert the state dictionary to a vector for processing by the model."""
        direction = state_dict['direction']
        snake_head = state_dict['snake'][0]
        food = state_dict['food']

        # Convert the relevant parts of the state into a vector
        state_vector = np.array([
            direction == UP,
            direction == DOWN,
            direction == LEFT,
            direction == RIGHT,
            snake_head[0] < food[0],
            snake_head[0] > food[0],
            snake_head[1] < food[1],
            snake_head[1] > food[1],
            self.env.is_collision((snake_head[0] + CELL_SIZE, snake_head[1])),
            self.env.is_collision((snake_head[0] - CELL_SIZE, snake_head[1])),
            self.env.is_collision((snake_head[0], snake_head[1] + CELL_SIZE)),
            self.env.is_collision((snake_head[0], snake_head[1] - CELL_SIZE)),
        ], dtype=float)

        #débogage# print(f"Generated state vector: {state_vector}")  # Ajout de vérification
        return state_vector

    def train(self, episodes, batch_size):
        plt.ion()
        fig, ax = init_plots()  # Initialiser le graphique
        rewards_history = []

        try:
            for e in range(episodes):
                #débogage# print(f"Starting episode {e+1}/{episodes}")
                state = self.get_state()
                if state is None:
                    raise ValueError("L'état est None au début de l'épisode.")
                total_reward = 0

                for time in range(35000):
                    #débogage# print(f"Time step {time+1}/35000")
                    self.env.handle_events()  # Gérer les événements Pygame
                    if not self.env.paused:
                        #débogage# print("Agent is not paused")
                        action = self.act(state)
                        #débogage# print(f"Action taken: {action}")
                        #débogage# print(f"Current State: {state}")
                        
                        try:
                            next_state, reward, done = self.env.step(action)
                        except Exception as e:
                            #débogage# print(f"Error during env.step(action): {e}")
                            raise e
                        
                        # Vérification après l'étape
                        #débogage# print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")
                        
                        if next_state is None:
                            raise ValueError("L'état suivant est None après une action.")
                        
                        state = next_state
                        self.remember(state, action, reward, next_state, done)
                        total_reward += reward

                        if done:
                            #débogage# print(f"Episode: {e}/{episodes}, Score: {self.env.score}")
                            self.env.reset()
                            break

                        self.env.render()
                        #débogage# print("Render completed")

                    update_plots(ax, e, total_reward, rewards_history)

        except KeyboardInterrupt:
            print("Training interrupted. Saving the model...")
            save_model(self.model)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            self.env.close()  # Fermer proprement Pygame
            plt.ioff()  # Désactiver le mode interactif de Matplotlib
            plt.show()  # Afficher le graphique
        
                    
def update_plots(ax, iteration, reward, rewards_history):
    rewards_history.append(reward)

    ax.plot(range(len(rewards_history)), rewards_history, 'b-')
    
    plt.pause(0.01)  # Pause pour rafraîchir l'affichage des graphiques

def init_plots():
    plt.ion()  # Activer le mode interactif de Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))  # Un seul sous-graphique
    
    # Graphique des récompenses
    ax.set_title("Récompenses par itération")
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Récompenses")
    
    return fig, ax

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path=MODEL_PATH):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")

# def print_model_weights(model, title): (pour afficher les poids du modèle au moment du chargement et de l'extinction.)
#    print(title)
#    for name, param in model.named_parameters():
#        print(name, param.data)

if __name__ == "__main__":
    agent = SnakeAgent()
    #print_model_weights(agent.model, "Model weights before loading:")
    try:
        load_model(agent.model)
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")
    #print_model_weights(agent.model, "Model weights after loading:")
    
    agent.train(1000, 32)
    
    #print_model_weights(agent.model, "Model weights after training:")
