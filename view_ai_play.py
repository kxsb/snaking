import pygame
import sys
from snake_agent import SnakeAgent
from utils import load_model, LoadModel
from snake_env import SnakeEnv
from tkinter import Tk, filedialog
import os

# Dimensions de l'interface principale
INTERFACE_WIDTH, INTERFACE_HEIGHT = 800, 600  # Fixé à 800x600
GRID_SIZES = [(6, 6), (8, 8), (10, 10), (12, 12), (15, 15), (20, 20), (25, 25)]  # Dimensions prédéfinies pour la grille

class ViewAIPlay:
    def __init__(self, model_path=None):
        self.screen = pygame.display.set_mode((INTERFACE_WIDTH, INTERFACE_HEIGHT))
        pygame.display.set_caption("Snake AI Play")

        self.model = None
        self.model_path = model_path

        if self.model_path:
            self.load_model_with_metadata(self.model_path)
        
        self.agent = SnakeAgent(env_type='full')
        self.env = SnakeEnv()
        self.clock = pygame.time.Clock()
        self.ticks_per_frame = 20
        self.grid_size = GRID_SIZES[0]
        
 
    def load_model_with_metadata(self, model_path):
        try:
            loader = LoadModel(model_path)
            self.model = loader.model
            self.grid_size = loader.grid_size
            print(f"Model loaded successfully from {model_path} for grid size {self.grid_size}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def select_model(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
        if file_path:
            self.model_name = os.path.basename(file_path)
            self.model_path = file_path
            self.load_model_with_metadata(self.model_path)

    def play_with_model(self):
        if not self.model:
            print("No model loaded. Exiting...")
            return

        while True:
            state = self.env.reset()
            done = False
            while not done:
                if not self.handle_events():
                    return

                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)

                self.draw_grid()
                self.draw_menu()
                self.draw_speed_info()

                state = next_state

                self.clock.tick(self.ticks_per_frame)       

    def draw_grid(self):
        NOIR = (0, 0, 0)
        
        # Effacer l'écran avant de dessiner
        self.screen.fill(NOIR)
        
        grid_x, grid_y = (INTERFACE_WIDTH - self.env.width) // 2, (INTERFACE_HEIGHT - self.env.height) // 2
        pygame.draw.rect(self.screen, (255, 255, 255), (grid_x - 2, grid_y - 2, self.env.width + 4, self.env.height + 4), 2)

        # Dessiner le serpent
        for segment in self.env.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(grid_x + segment[0], grid_y + segment[1], self.env.cell_size, self.env.cell_size))
        
        # Dessiner la nourriture
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(grid_x + self.env.food[0], grid_y + self.env.food[1], self.env.cell_size, self.env.cell_size))
        
        pygame.display.flip()

    def draw_speed_info(self):
        font = pygame.font.SysFont(None, 24)
        
        speed_instructions = font.render("- <- Arrow change speed -> +", True, (255, 255, 255, 128))
        self.screen.blit(speed_instructions, (20, INTERFACE_HEIGHT - 60))
        
        current_speed = font.render(f"Current speed: {self.ticks_per_frame}", True, (255, 255, 255, 128))
        self.screen.blit(current_speed, (20, INTERFACE_HEIGHT - 30))
        
        back_instruction = font.render("Press ESC to return", True, (255, 255, 255, 128))
        self.screen.blit(back_instruction, (INTERFACE_WIDTH - 200, INTERFACE_HEIGHT - 30))

        pygame.display.flip()

    def draw_menu(self):
        font = pygame.font.SysFont(None, 24)
        grid_text = font.render("Grid Size:", True, (255, 255, 255))
        self.screen.blit(grid_text, (20, 20))

        for i, size in enumerate(GRID_SIZES):
            size_text = font.render(f"{size[0]}x{size[1]}", True, (255, 255, 255))
            self.screen.blit(size_text, (20, 50 + i * 30))

        pygame.display.flip()

    def handle_menu_click(self, pos):
        for i, size in enumerate(GRID_SIZES):
            if 20 <= pos[0] <= 120 and 50 + i * 30 <= pos[1] <= 80 + i * 30:
                self.grid_size = size
                self.env.grid_width, self.env.grid_height = size
                self.env.width = self.grid_size[0] * self.env.cell_size
                self.env.height = self.grid_size[1] * self.env.cell_size
                self.env.reset()  # Réinitialise l'environnement avec la nouvelle taille de grille
                return True
        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_LEFT:
                    self.ticks_per_frame = max(10, self.ticks_per_frame - 10)
                if event.key == pygame.K_RIGHT:
                    self.ticks_per_frame = min(100, self.ticks_per_frame + 10)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.handle_menu_click(event.pos):
                    return True
        return True

if __name__ == "__main__":
    pygame.init()
    view_ai_play = ViewAIPlay()
    view_ai_play.select_model()

    if view_ai_play.model_path:
        view_ai_play.play_with_model()
    else:
        print("Aucun modèle sélectionné, arrêt du programme.")