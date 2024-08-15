import pygame
import sys
import subprocess
import torch
from tkinter import Tk, filedialog
from utils import load_model  # Importer la fonction load_model depuis utils.py
import os

# Dimensions de la fenêtre de l'interface
INTERFACE_WIDTH, INTERFACE_HEIGHT = 800, 600
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
HOVER_COLOR = (170, 170, 170)  # Couleur lorsque le bouton est survolé

class AIGamesInterface:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((INTERFACE_WIDTH, INTERFACE_HEIGHT))
        pygame.display.set_caption("AI Games Interface")
        self.font = pygame.font.SysFont(None, 24)
        
        # Charger l'image de fond
        self.background_image = pygame.image.load("snake_image2.jpg")
        self.background_image = pygame.transform.scale(self.background_image, (INTERFACE_WIDTH, INTERFACE_HEIGHT))

        # Variables pour le modèle et le fichier de paramètres
        self.model_name = "Empty"  # Nom du modèle chargé
        self.params_file = "Empty"  # Nom du fichier de paramètres chargé
        self.model = None  # Placeholder pour le modèle

        # Ajouter les boutons de l'interface AI games
        self.buttons = {
            'view_ai_play': pygame.Rect(100, 60, 200, 50),
            'train_ai_fast': pygame.Rect(300, 60, 200, 50),
            'train_ai_lowder': pygame.Rect(500, 60, 200, 50),
            'select_model': pygame.Rect(100, 200, 200, 50),
            'reset_model': pygame.Rect(100, 250, 200, 50),
            'create_new_model': pygame.Rect(100, 300, 200, 50),
            'load_params': pygame.Rect(100, 350, 200, 50),
            'change_params': pygame.Rect(100, 400, 200, 50),
            'back': pygame.Rect(500, 500, 250, 50)
        }
        
        self.create_interface()

    def create_interface(self):
        # Dessiner l'image de fond
        self.screen.blit(self.background_image, (0, 0))

        mouse_pos = pygame.mouse.get_pos()  # Obtenir la position actuelle de la souris

        # Dessiner les boutons
        for label, rect in self.buttons.items():
            # Changer la couleur du bouton si la souris le survole
            if rect.collidepoint(mouse_pos):
                color = HOVER_COLOR
            else:
                color = GRAY
            
            pygame.draw.rect(self.screen, color, rect)
            text_surface = self.font.render(label.replace('_', ' ').capitalize(), True, BLACK)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
        
        # Afficher les informations sur le modèle et les paramètres
        self.afficher_variables()
        
        pygame.display.flip()

    def afficher_variables(self):
        # Affiche le nom du modèle et du fichier de paramètres
        model_text = f"Model: {self.model_name}"
        params_text = f"Params File: {self.params_file}"

        model_surface = self.font.render(model_text, True, BLACK)
        params_surface = self.font.render(params_text, True, BLACK)

        self.screen.blit(model_surface, (100, 520))  # Position pour le modèle
        self.screen.blit(params_surface, (100, 550))  # Position pour le fichier de paramètres

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for label, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            print(f"Button {label} clicked")  # Placeholder for actual functionality
                            if label == 'create_new_model':
                                self.create_new_model()
                            elif label == 'select_model':
                                self.select_model()
                            elif label == 'back':
                                self.go_back()

            self.create_interface()
            pygame.display.update()

        pygame.quit()
        sys.exit()

    def select_model(self):
        Tk().withdraw()  # Cacher la fenêtre Tkinter
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])  # Sélectionner un fichier de modèle
        if file_path:
            self.model_name = os.path.basename(file_path)  # Mettre à jour le nom du modèle
            self.model = self.load_and_initialize_model(file_path)
            print(f"Model loaded: {self.model_name}")
            self.create_interface()  # Mettre à jour l'interface

    def load_and_initialize_model(self, path):
        # Initialise et charge le modèle depuis le chemin spécifié
        model = torch.nn.Module()  # Remplacez par le modèle réel que vous utilisez
        load_model(model, path)
        return model

    def go_back(self):
        pygame.quit()  # Ferme l'interface actuelle
        subprocess.run(["python", "main.py"])  # Lance le script main.py

if __name__ == "__main__":
    interface = AIGamesInterface()
    interface.run()
