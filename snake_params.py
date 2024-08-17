import pygame
from tkinter import Tk, filedialog

# Dimensions de la fenêtre de l'interface
INTERFACE_WIDTH, INTERFACE_HEIGHT = 800, 600
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)  # Couleur pour le curseur
YELLOW = (255, 255, 102)  # Couleur pour les bulles d'aide

class TrainingInterface:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((INTERFACE_WIDTH, INTERFACE_HEIGHT))
        pygame.display.set_caption("Snake Training Parameters Interface")
        self.font = pygame.font.SysFont(None, 24)
        self.help_font = pygame.font.SysFont(None, 18)

        # Créer des entrées interactives pour chaque paramètre
        self.inputs = {}
        self.labels = {}
        y_offset = 50
        for i, param in enumerate(self.params.keys()):
            self.inputs[param] = pygame.Rect(300, y_offset + i * 30, 140, 25)
            self.labels[param] = pygame.Rect(50, y_offset + i * 30, 200, 25)

        # Ajouter les boutons Save, Load, Back
        self.buttons = {
            'save': pygame.Rect(450, 500, 80, 40),
            # 'load': pygame.Rect(550, 500, 80, 40),
            'back': pygame.Rect(700, 500, 80, 40)
        }
        
        self.active_input = None
        self.input_text = ''
        self.prev_value = ''
        self.cursor_visible = True
        self.cursor_timer = pygame.time.get_ticks()

        self.hovered_param = None  # Conserver l'état de survol

        self.create_interface()

if __name__ == "__main__":
    interface = TrainingInterface()
    interface.run()
