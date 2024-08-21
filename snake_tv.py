import pygame
import matplotlib.pyplot as plt
import torch
from snake_brain import SnakeAgent, save_model, load_model
import numpy as np


class SnakeTV:
    def __init__(self, agent):
        self.agent = agent
        self.rewards_history = []
        self.fig, self.ax = self.init_plots()
        
    def init_plots(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Récompenses par épisode")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Récompenses")
        return fig, ax
    
    def update_plots(self, episode, reward):
        self.rewards_history.append(reward)
        self.ax.clear()
        
        # Affichage des récompenses
        self.ax.plot(range(len(self.rewards_history)), self.rewards_history, 'b-', label="Récompenses")
        
        # Calcul et affichage de la moyenne mobile
        if len(self.rewards_history) >= 50:
            moving_avg = np.convolve(self.rewards_history, np.ones((50,)) / 50, mode='valid')
            self.ax.plot(range(len(moving_avg)), moving_avg, 'r-', label="Moyenne mobile (50)")
        
        self.ax.legend()
        plt.pause(0.01)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Pour quitter l'application
                print("Entraînement interrompu par l'utilisateur.")
                save_model(self.agent.model)
                self.agent.env.close()
                pygame.quit()
                exit()
                    
    def run(self, episodes, batch_size):
        try:
            for e in range(episodes):
                state = self.agent.get_state()
                total_reward = 0

                for time in range(35000):
                    self.agent.env.handle_events()  # Gérer les événements Pygame
                    self.handle_events()  # Gestion des événements spécifiques à SnakeTV

                    action = self.agent.act(state)
                    next_state, reward, done = self.agent.env.step(action)
                        
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward

                    if done:
                        print(f"Épisode: {e+1}/{episodes}, Score: {self.agent.env.score}, Récompense totale: {total_reward}")
                        self.agent.env.reset()
                        break

                    self.agent.env.render()

                    self.update_plots(e, total_reward)

        except KeyboardInterrupt:
            print("Entraînement interrompu. Sauvegarde du modèle...")
            save_model(self.agent.model)
        finally:
            self.agent.env.close()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    agent = SnakeAgent()
    
    try:
        load_model(agent.model)
    except FileNotFoundError:
        print("Aucun modèle sauvegardé trouvé. Entraînement à partir de zéro.")
    
    snake_tv = SnakeTV(agent)
    snake_tv.run(episodes=1000, batch_size=32)
