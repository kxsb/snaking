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

class SnakeMovement:
    def __init__(self, initial_direction):
        self.direction = initial_direction

    def turn_left(self):
        if self.direction == UP:
            return LEFT
        elif self.direction == DOWN:
            return RIGHT
        elif self.direction == LEFT:
            return DOWN
        elif self.direction == RIGHT:
            return UP

    def turn_right(self):
        if self.direction == UP:
            return RIGHT
        elif self.direction == DOWN:
            return LEFT
        elif self.direction == LEFT:
            return UP
        elif self.direction == RIGHT:
            return DOWN

    def move(self, action):
        if action == 1:  # Tourner à gauche
            self.direction = self.turn_left()
        elif action == 2:  # Tourner à droite
            self.direction = self.turn_right()
        return self.direction

    def get_new_head_position(self, head):
        return (head[0] + self.direction[0] * CELL_SIZE,
                head[1] + self.direction[1] * CELL_SIZE)

class SnakeEnv:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + BUTTON_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.running = True  # Initialiser 'running' à True
        self.paused = False
        self.reset()
        self.create_buttons()
        
    def create_buttons(self):
        self.play_pause_button = pygame.Rect(10, HEIGHT + 10, 100, 30)
        self.intense_training_button = pygame.Rect(120, HEIGHT + 10, 150, 30)
        self.stats_button = pygame.Rect(280, HEIGHT + 10, 150, 30)
        
    def draw_buttons(self):
        pygame.draw.rect(self.screen, GRAY, self.play_pause_button)
        pygame.draw.rect(self.screen, GRAY, self.intense_training_button)
        font = pygame.font.SysFont(None, 24)
        play_pause_text = font.render('Play/Pause', True, BLACK)
        intense_training_text = font.render('Intense Training', True, BLACK)
        self.screen.blit(play_pause_text, (20, HEIGHT + 15))
        self.screen.blit(intense_training_text, (130, HEIGHT + 15))
        pygame.draw.rect(self.screen, GRAY, self.stats_button)
        stats_text = font.render('Toggle Stats', True, BLACK)
        self.screen.blit(stats_text, (290, HEIGHT + 15))
        
    def main_loop(self):
        self.running = True  # Initialiser le flag pour le contrôle de la boucle principale

        while self.running:
            self.handle_events()  # Gérer les événements
            if not self.paused:
                self.update_game_logic()  # Votre logique de mise à jour du jeu
                self.render()  # Rendre l'affichage du jeu

        self.cleanup()  # Appeler une méthode de nettoyage à la fin

    def cleanup(self):
        pygame.quit()
        # Vous pouvez ajouter ici d'autres nettoyages nécessaires
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Définir un flag pour terminer le jeu proprement
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.play_pause_button.collidepoint(event.pos):
                    self.paused = not self.paused
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False  # Utiliser le même flag pour l'ESCAPE
                                    
    def reset(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = RIGHT
        self.food = self.spawn_food()
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            if (x, y) not in self.snake:
                return x, y
       
    def is_collision(self, position):
        if (position[0] < 0 or position[0] >= WIDTH or
            position[1] < 0 or position[1] >= HEIGHT or
            position in self.snake):
            return True
        return False       
            
    def get_state(self):
        return {
            'snake': self.snake,
            'direction': self.direction,
            'food': self.food,
            'score': self.score
        }
    
    def run(self):
        while self.running:
            self.handle_events()
            if not self.paused:
                self.step(0)  # ou toute autre logique de jeu
            self.render()
        self.close()  # Ferme proprement Pygame quand la boucle est terminée

    def close(self):
        pygame.quit()
        sys.exit()  # Appe    def close(self):

    def step(self, action):
        self.direction = SnakeMovement(self.direction).move(action)
        new_head = self.snake[0][0] + self.direction[0] * CELL_SIZE, self.snake[0][1] + self.direction[1] * CELL_SIZE

        if self.is_collision(new_head):
            return self.get_state(), -1, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.spawn_food()
            self.score += 2
            reward = 1
        else:
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, False

    def is_collision(self, position):
        if (position[0] < 0 or position[0] >= WIDTH or
            position[1] < 0 or position[1] >= HEIGHT or
            position in self.snake):
            return True
        return False

    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0], self.food[1], CELL_SIZE, CELL_SIZE))
        self.draw_buttons()
        pygame.display.flip()
        self.clock.tick(80)

class SimpleSnakeEnv:
    def __init__(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.movement = SnakeMovement(RIGHT)
        self.food = self.spawn_food()
        self.score = 0

    def reset(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.movement = SnakeMovement(RIGHT)
        self.food = self.spawn_food()
        self.score = 0
        return self.get_state()
    
    def close(self):
        pass  # Pas de Pygame à fermer ici
        
    def step(self, action):
        self.movement.move(action)
        new_head = self.movement.get_new_head_position(self.snake[0])

        if self.is_collision(new_head):
            return self.get_state(), -1, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.spawn_food()
            self.score += 2
            reward = 1
        else:
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, False

    def spawn_food(self):
        while True:
            x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            if (x, y) not in self.snake:
                return x, y

    def get_state(self):
        return {
            'snake': self.snake,
            'direction': self.movement.direction,
            'food': self.food,
            'score': self.score
        }

    def is_collision(self, position):
        if (position[0] < 0 or position[0] >= WIDTH or
            position[1] < 0 or position[1] >= HEIGHT or
            position in self.snake):
            return True
        return False
