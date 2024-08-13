#gère grosso modo l'environnement de jeu

import pygame
import sys
import random

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre de jeu
WIDTH, HEIGHT = 640, 480  # Largeur et hauteur de la fenêtre de jeu
CELL_SIZE = 20  # Taille d'une cellule (unité de base pour les mouvements du serpent)
BUTTON_HEIGHT = 50  # Hauteur des boutons de l'interface utilisateur

# Couleurs (définies en RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Directions pour le mouvement du serpent
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeEnv:
    def __init__(self):
        # Création de la fenêtre de jeu avec une zone pour les boutons en bas
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + BUTTON_HEIGHT))
        pygame.display.set_caption("Snake Game")  # Titre de la fenêtre
        self.clock = pygame.time.Clock()  # Initialisation de l'horloge pour contrôler la fréquence de rafraîchissement
        self.reset()  # Réinitialise l'état du jeu
        self.paused = False  # État du jeu (en pause ou non)
        self.display_stats = False  # État initial de l'affichage statistique (désactivé)
        self.create_buttons()  # Crée les boutons pour l'interface utilisateur

    def create_buttons(self):
        # Définit les zones de détection pour les boutons (rectangles)
        self.play_pause_button = pygame.Rect(10, HEIGHT + 10, 100, 30)
        self.intense_training_button = pygame.Rect(120, HEIGHT + 10, 150, 30)
        self.stats_button = pygame.Rect(280, HEIGHT + 10, 150, 30)  # Nouveau bouton pour les stats
        
    def draw_buttons(self):
        # Dessine les boutons et leur texte sur l'écran
        pygame.draw.rect(self.screen, GRAY, self.play_pause_button)
        pygame.draw.rect(self.screen, GRAY, self.intense_training_button)
        font = pygame.font.SysFont(None, 24)  # Police pour le texte des boutons
        play_pause_text = font.render('Play/Pause', True, BLACK)
        intense_training_text = font.render('Intense Training', True, BLACK)
        # Positionne le texte sur les boutons
        self.screen.blit(play_pause_text, (20, HEIGHT + 15))
        self.screen.blit(intense_training_text, (130, HEIGHT + 15))
        pygame.draw.rect(self.screen, GRAY, self.stats_button)
        font = pygame.font.SysFont(None, 24)
        stats_text = font.render('Toggle Stats', True, BLACK)
        self.screen.blit(stats_text, (290, HEIGHT + 15))  # Positionnement du texte du bouton stats
 
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.play_pause_button.collidepoint(event.pos):
                    self.paused = not self.paused
                    print("Play/Pause button clicked")
                elif self.intense_training_button.collidepoint(event.pos):
                    self.intense_training()
                    print("Intense Training button clicked")
                elif self.stats_button.collidepoint(event.pos):
                    self.display_stats = not self.display_stats  # Basculer l'affichage des stats
                    print("Toggle Stats button clicked")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                                    
    def reset(self):
        # Réinitialise l'état du jeu (position du serpent, direction, nourriture, score)
        self.snake = [(100, 100), (80, 100), (60, 100)]  # Position initiale du serpent
        self.direction = RIGHT  # Direction initiale (vers la droite)
        self.food = self.spawn_food()  # Position initiale de la nourriture
        self.score = 0  # Score initial
        return self.get_state()  # Retourne l'état initial du jeu

    def spawn_food(self):
        # Génère aléatoirement une position pour la nourriture, en s'assurant qu'elle n'apparaisse pas sur le serpent
        while True:
            x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            if (x, y) not in self.snake:
                return x, y
            
    def get_state(self):
        # Retourne l'état actuel du jeu sous forme de dictionnaire
        return {
            'snake': self.snake,
            'direction': self.direction,
            'food': self.food,
            'score': self.score
        }
        
    def close(self):
        # Ferme proprement la fenêtre de jeu
        pygame.quit()

    def step(self, action, use_ideal_path=False):
        # Réalise un pas dans le jeu en fonction de l'action donnée (0: tout droit, 1: tourner à gauche, 2: tourner à droite)
        
        # Calcul de la nouvelle direction en fonction de l'action
        new_direction = self.direction
        if action == 0:  # Continue tout droit
            new_direction = self.direction
        elif action == 1:  # Tourne à gauche
            if self.direction == UP:
                new_direction = LEFT
            elif self.direction == DOWN:
                new_direction = RIGHT
            elif self.direction == LEFT:
                new_direction = DOWN
            elif self.direction == RIGHT:
                new_direction = UP
        elif action == 2:  # Tourne à droite
            if self.direction == UP:
                new_direction = RIGHT
            elif self.direction == DOWN:
                new_direction = LEFT
            elif self.direction == LEFT:
                new_direction = UP
            elif self.direction == RIGHT:
                new_direction = DOWN

        # Met à jour la direction actuelle avec la nouvelle direction déterminée
        self.direction = new_direction
        #débogage# print(f"Action: {action}, Direction: {self.direction}")

        # Calcul de la nouvelle position de la tête du serpent
        new_head = (self.snake[0][0] + self.direction[0] * CELL_SIZE,
                    self.snake[0][1] + self.direction[1] * CELL_SIZE)

        if self.is_collision(new_head):
            return self.get_state(), -1, True  # Si collision, fin de partie avec une pénalité

        # Mise à jour de la position du serpent
        self.snake.insert(0, new_head)

        if new_head == self.food:  # Si le serpent mange la nourriture
            self.food = self.spawn_food()  # Nouvelle position pour la nourriture
            self.score += 2  # Augmentation du score
            reward = 1  # Récompense pour avoir mangé la nourriture
        else:
            self.snake.pop()  # Si pas de nourriture, la queue du serpent bouge
            reward = 0

        state = self.get_state()
        if state is None:
            raise ValueError("L'état généré après l'action est None, ce qui ne devrait pas arriver.")
        
        return state, reward, False  # Retourne l'état, la récompense et un flag indiquant si la partie est terminée

    def is_collision(self, position):
        # Vérifie si la position donnée entre en collision avec les murs ou le corps du serpent
        if (position[0] < 0 or position[0] >= WIDTH or
            position[1] < 0 or position[1] >= HEIGHT or
            position in self.snake):
            return True
        return False

    def render(self):
        # Affiche le jeu à l'écran
        self.screen.fill(BLACK)  # Fond noir
        for segment in self.snake:  # Affiche chaque segment du serpent
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0], self.food[1], CELL_SIZE, CELL_SIZE))  # Affiche la nourriture
        self.draw_buttons()  # Affiche les boutons
        pygame.display.flip()  # Met à jour l'affichage
        self.clock.tick(80)  # Contrôle la vitesse du jeu (80 FPS ici)

    def handle_events(self):
        # Gère les événements, y compris les interactions avec les boutons
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.play_pause_button.collidepoint(event.pos):  # Gestion du clic sur Play/Pause
                    self.paused = not self.paused
                    #débogage# print("Play/Pause button clicked")
                elif self.intense_training_button.collidepoint(event.pos):  # Gestion du clic sur Intense Training
                    self.intense_training()
                    #débogage# print("Intense Training button clicked")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                    
    def close(self):
        pygame.quit()
                
def intense_training(self):
    iterations = int(input("Enter the number of iterations for intense training: "))
    self.agent.intense_training(iterations)
    pygame.quit()

if __name__ == "__main__":
    env = SnakeEnv()

    while True:
        env.handle_events()
        
        if not env.paused:
            keys = pygame.key.get_pressed()
            
        if keys[pygame.K_LEFT]:
            action = 1  # Tourne à gauche
        elif keys[pygame.K_RIGHT]:
            action = 2  # Tourne à droite
        else:
            action = 0  # Continue tout droit
        
        if action is not None:
            state, reward, done = env.step(action)
            if done:
                env.reset()
        env.render()
