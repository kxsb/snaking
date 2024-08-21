import pygame
import sys
import random
import numpy as np

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
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + BUTTON_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()
        self.paused = False
        self.display_stats = False

        # Charger l'image de fond
        self.background_image = pygame.image.load('fond.jpg').convert()

        # Charger la texture pour la tête du serpent
        self.snake_head_texture = pygame.image.load('tete.png').convert_alpha()

        # Charger la texture pour le corps du serpent
        self.snake_body_texture = pygame.image.load('texture6.png').convert_alpha()

        # Redimensionner les textures pour qu'elles correspondent à la taille d'une cellule
        self.snake_head_texture = pygame.transform.scale(self.snake_head_texture, (CELL_SIZE, CELL_SIZE))
        self.snake_body_texture = pygame.transform.scale(self.snake_body_texture, (CELL_SIZE, CELL_SIZE))

        # Redimensionner l'image de fond si nécessaire pour correspondre à la taille de l'écran
        self.background_image = pygame.transform.scale(self.background_image, (WIDTH, HEIGHT + BUTTON_HEIGHT))

    def create_buttons(self):
        # Définit les zones de détection pour les boutons (rectangles)
        self.play_pause_button = pygame.Rect(10, HEIGHT + 10, 100, 30)
        self.intense_training_button = pygame.Rect(120, HEIGHT + 10, 150, 30)
        self.stats_button = pygame.Rect(280, HEIGHT + 10, 150, 30)  # Nouveau bouton pour les stats
        
    def draw_buttons(self):
        pass
    
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
            print("Collision detected")  # Debug message for collision detection
            return self.get_state(), -1, True  # If collision, end game with a penalty

        # Mise à jour de la position du serpent
        self.snake.insert(0, new_head)

        if new_head == self.food:  # Si le serpent mange la nourriture
            self.food = self.spawn_food()  # Nouvelle position pour la nourriture
            self.score += 2  # Augmentation du score
            reward = 0.9  # Récompense pour avoir mangé la nourriture
        else:
            self.snake.pop()  # Si pas de nourriture, la queue du serpent bouge
            reward = 0

        state = self.get_state()
        if state is None:
            raise ValueError("L'état généré après l'action est None, ce qui ne devrait pas arriver.")
        
        return state, reward, False  # Retourne l'état, la récompense et un flag indiquant si la partie est terminée

    def grid_state(self):
        # Dimensions de la grille
        grid_width = WIDTH // CELL_SIZE  # Par exemple 32
        grid_height = HEIGHT // CELL_SIZE  # Par exemple 24

        # Initialisation de la grille avec des zéros
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)

        # Marquer les positions du serpent dans la grille
        for segment in self.snake:
            x, y = segment
            grid[y // CELL_SIZE][x // CELL_SIZE] = 1.0  # 1.0 pour les segments du serpent

        # Marquer la position de la nourriture
        food_x, food_y = self.food
        grid[food_y // CELL_SIZE][food_x // CELL_SIZE] = -1.0  # -1.0 pour la nourriture

        return grid  # Retourner la grille 2D, qui sera aplatie dans SnakeAgent
    
    def is_collision(self, position):
        # Vérifie si la position donnée entre en collision avec les murs ou le corps du serpent
        return (position[0] < 0 or position[0] >= WIDTH or
                position[1] < 0 or position[1] >= HEIGHT or
                position in self.snake)
    
    def render(self):
        # Afficher l'image de fond
        self.screen.blit(self.background_image, (0, 0))

        # Déterminer l'angle de rotation pour chaque segment en fonction de la direction
        for i, segment in enumerate(self.snake):
            if i == 0:  # Tête du serpent
                texture = self.snake_head_texture
            else:  # Corps du serpent
                texture = self.snake_body_texture
            
            # Déterminer la direction du segment actuel
            if i == 0:
                direction = self.direction
            else:
                # Calcul de la direction entre ce segment et le segment précédent
                direction = (self.snake[i-1][0] - segment[0], self.snake[i-1][1] - segment[1])

            # Calculer l'angle en fonction de la direction
            if direction == RIGHT:
                angle = 270  # Vers la droite
            elif direction == LEFT:
                angle = 90  # Vers la gauche
            elif direction == UP:
                angle = 0  # Vers le haut
            elif direction == DOWN:
                angle = 180  # Vers le bas

            # Faire pivoter la texture selon l'angle calculé
            rotated_texture = pygame.transform.rotate(texture, angle)
            
            # Dessiner la texture pivotée sur l'écran
            self.screen.blit(rotated_texture, (segment[0], segment[1]))

        # Dessiner la nourriture (vous pouvez également utiliser une texture pour la nourriture)
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0], self.food[1], CELL_SIZE, CELL_SIZE))

        # Mettre à jour l'écran
        pygame.display.flip()
        self.clock.tick(35)  # Contrôle la vitesse du jeu (60 FPS ici)
    
def handle_events(self):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
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
