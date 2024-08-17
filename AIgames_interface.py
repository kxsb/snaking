import pygame
import sys
import subprocess
import torch
import torch.nn as nn
from torch import nn, optim
from tkinter import Tk, filedialog
from utils import load_model, create_model, save_model, LoadModel #plot_scores_and_losses, scores, losses
import os
from view_ai_play import ViewAIPlay
from snake_agent import SnakeAgent
from collections import OrderedDict
import json
import time
import threading
from train_ai_fast import TrainAIFast
from create_new_model import create_and_save_model, MODEL_CONFIGS

# Dimensions de la fenêtre de l'interface
INTERFACE_WIDTH, INTERFACE_HEIGHT = 800, 600
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

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
        self.model_name = "Empty"
        self.params_file = "Empty"
        self.model = None
        self.model_path = None
        
        self.training_in_progress = False
        self.current_iteration = 0
        self.total_iterations = 0
        
        # Initialiser les paramètres par défaut
        self.params = {
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "batch_size": 32,
            "memory_size": 10000,
            "iterations": 1000,
            "tick_rate": 30,
            "evaluation_interval": 10,
            "collision_penalty": -1,
            "food_reward": 1
        }
        # Ajouter les boutons de l'interface AI games
        self.buttons = {
            'view_ai_play': pygame.Rect(100, 60, 200, 50),
            'train_ai_lowder': pygame.Rect(300, 60, 200, 50),  # Ce qui était 'train_ai_lowder'
            'train_ai_fast': pygame.Rect(500, 60, 200, 50),  
            'select_model': pygame.Rect(100, 200, 200, 50),
            'reset_model': pygame.Rect(100, 250, 200, 50),
            'create_new_model': pygame.Rect(100, 300, 200, 50),
            'load_params': pygame.Rect(100, 350, 200, 50),
            'change_params': pygame.Rect(100, 400, 200, 50),
            'back': pygame.Rect(500, 500, 250, 50)
            }        
        self.help_texts = {
            'gamma': "Le facteur d'actualisation des récompenses\nfutures. Une valeur courante est autour de 0.95 à 0.99.",
            'epsilon': "Le taux d'exploration initial de l'agent. Typiquement,\n epsilon commence à 1.0 et diminue au fil du temps.",
            'epsilon_min': "La valeur minimale de l'epsilon.\nTypiquement entre 0.01 et 0.1.",
            'epsilon_decay': "Le facteur de décroissance de l'epsilon.\nGénéralement entre 0.99 et 0.999.",
            'learning_rate': "Le taux d'apprentissage du modèle.\nDes valeurs courantes se situent entre 0.0001 et 0.01.",
            'batch_size': "La taille du mini-lot utilisé pour mettre à jour\nles poids du modèle. Typiquement entre 16 et 64.",
            'memory_size': "La capacité maximale de la mémoire d'expérience.\nTypiquement entre 10,000 et 100,000.",
            'iterations': "Le nombre total d'épisodes d'entraînement.\nTypiquement entre 1,000 et 10,000.",
            'fc1': "Le nombre de neurones dans la première couche.\nTypiquement entre 128 et 512.",
            'fc2': "Le nombre de neurones dans la deuxième couche.\nTypiquement entre 64 et 256.",
            'fc3': "Le nombre de neurones dans la troisième couche.\nTypiquement entre 32 et 128.",
            'tick_rate': "La vitesse de rafraîchissement de l'interface.\nTypiquement entre 30 et 100 FPS.",
            'evaluation_interval': "Le nombre d'épisodes après lesquels l'agent\nest évalué sans exploration. Typiquement tous les 10 à 100 épisodes.",
            'collision_penalty': "La pénalité appliquée en cas de collision.\nTypiquement autour de -1 à -10.",
            'food_reward': "La récompense obtenue lorsqu'on mange de la nourriture.\nTypiquement autour de 1 à 10.",
            'WIDTH': "La largeur de la grille du jeu en pixels.\nTypiquement entre 300 et 800.",
            'HEIGHT': "La hauteur de la grille du jeu en pixels.\nTypiquement entre 300 et 600.",
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
    
    def show_help(self, text, position): 
        lines = text.split("\n")
        width = max(self.font.size(line)[0] for line in lines) + 10
        height = len(lines) * self.font.get_height() + 10
        help_rect = pygame.Rect(position[0], position[1], width, height)
        pygame.draw.rect(self.screen, (255, 255, 102), help_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), help_rect, 1)
        for i, line in enumerate(lines):
            line_surface = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(line_surface, (position[0] + 5, position[1] + 5 + i * self.font.get_height()))
    
    def afficher_variables(self):
        # Affiche le nom du modèle et du fichier de paramètres
        model_text = f"Model: {self.model_name}"
        params_text = f"Params File: {self.params_file}"

        model_surface = self.font.render(model_text, True, BLACK)
        params_surface = self.font.render(params_text, True, BLACK)

        self.screen.blit(model_surface, (100, 520))  # Position pour le modèle
        self.screen.blit(params_surface, (100, 550))  # Position pour le fichier de paramètres
    
    def change_params(self):
        print("Affichage de l'interface de modification des paramètres...")  # Debugging
        self.run_training_interface()  # Redessiner l'interface avec les paramètres

    def save_params(self):
        Tk().withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.params, f)
            print(f"Parameters saved to {file_path}")

    def run_training_interface(self):
        running = True
        clock = pygame.time.Clock()
        active_input = None
        input_text = ''
        cursor_visible = True
        cursor_timer = pygame.time.get_ticks()
        hovered_param = None

        while running:
            self.screen.fill(WHITE)
            y_offset = 50
            inputs = {}
            labels = {}

            # Exclusion des paramètres fc1, fc2, fc3, WIDTH, HEIGHT
            params_to_display = {k: v for k, v in self.params.items() if k not in ['fc1', 'fc2', 'fc3', 'WIDTH', 'HEIGHT']}

            for i, (param, value) in enumerate(params_to_display.items()):
                label = self.font.render(f"{param}:", True, BLACK)
                self.screen.blit(label, (50, y_offset + i * 30))
                input_rect = pygame.Rect(300, y_offset + i * 30, 140, 25)
                inputs[param] = input_rect
                labels[param] = pygame.Rect(50, y_offset + i * 30, 200, 25)
                pygame.draw.rect(self.screen, GRAY, input_rect)
                display_text = input_text if active_input == param else str(value)
                text_surface = self.font.render(display_text, True, BLACK)
                self.screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

            # Ajouter les boutons Save, Load, Back
            buttons = {
                'save': pygame.Rect(450, 500, 80, 40),
                'load': pygame.Rect(550, 500, 80, 40),
                'back': pygame.Rect(700, 500, 80, 40)
            }

            for button, rect in buttons.items():
                pygame.draw.rect(self.screen, GRAY, rect)
                text_surface = self.font.render(button.capitalize(), True, BLACK)
                text_rect = text_surface.get_rect(center=rect.center)
                self.screen.blit(text_surface, text_rect)

            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif active_input is not None:
                        if event.key == pygame.K_RETURN:
                            try:
                                value = float(input_text)
                                self.params[active_input] = value
                            except ValueError:
                                pass
                            active_input = None
                            input_text = ''
                        elif event.key == pygame.K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            input_text += event.unicode

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_outside = True
                    for param, rect in inputs.items():
                        if rect.collidepoint(event.pos):
                            clicked_outside = False
                            if active_input == param:
                                continue
                            if active_input is not None:
                                input_text = ''
                            active_input = param
                            input_text = str(self.params[param])
                            break
                    if clicked_outside and active_input is not None:
                        active_input = None
                        input_text = ''

                    for button, rect in buttons.items():
                        if rect.collidepoint(event.pos):
                            if button == 'save':
                                self.save_params()
                            elif button == 'load':
                                self.load_params()
                            elif button == 'back':
                                running = False

            # Vérifie si la souris survole un paramètre
            new_hovered_param = None
            for param, rect in labels.items():
                if rect.collidepoint(mouse_pos):
                    new_hovered_param = param
                    break

            if new_hovered_param != hovered_param:
                hovered_param = new_hovered_param

            if hovered_param is not None:
                self.show_help(self.help_texts[hovered_param], (mouse_pos[0] + 20, mouse_pos[1]))

            if active_input is not None:
                current_time = pygame.time.get_ticks()
                if current_time - cursor_timer > 500:
                    cursor_visible = not cursor_visible
                    cursor_timer = current_time
                pygame.draw.rect(self.screen, BLUE, inputs[active_input], 2)
                text_surface = self.font.render(input_text + ('|' if cursor_visible else ''), True, BLACK)
                self.screen.blit(text_surface, (inputs[active_input].x + 5, inputs[active_input].y + 5))

            pygame.display.update()
            clock.tick(30)

        self.create_interface()
                
    def run_training(self, iterations):
        try:
            print(f"Début de l'entraînement rapide pour {iterations} itérations...")
            
            start_time = time.time()  # Marque le début de l'entraînement

            for i in range(iterations):
                self.handle_stop_button()  # Vérifier si le bouton "Stop" a été pressé

                self.agent.intense_training(1, self.model_path)  # Entraîner pour une itération

                # Calculer le temps écoulé et le temps restant estimé
                elapsed_time = time.time() - start_time
                time_per_iteration = elapsed_time / (i + 1)
                remaining_time = (iterations - i - 1) * time_per_iteration

                # Mettre à jour la barre de progression et afficher l'itération en cours
                self.update_progress(i + 1, iterations, remaining_time)

                # Laisser le temps à l'interface de se mettre à jour
                pygame.event.pump()  # Traitement des événements pour maintenir l'interface active
                pygame.time.delay(100)  # Pause pour que l'interface reste réactive

            print("Entraînement rapide terminé.")
            self.afficher_message("Entraînement terminé.")

        except KeyboardInterrupt:
            print("Entraînement interrompu par l'utilisateur.")

        finally:
            # Sauvegarde le modèle dans tous les cas (fin ou interruption)
            print(f"Sauvegarde du modèle dans {self.model_path}.")
            save_model(self.agent.model, self.model_path)
            print(f"Modèle sauvegardé dans {self.model_path}.")

    def train_ai_fast_interface(self):
        if self.model_path and self.params_file != "Empty":
            iterations = self.demander_iterations()  # Utilisation de la fonction pour obtenir le nombre d'itérations
            print("Début de l'entraînement rapide de l'IA...")
            train_ai = TrainAIFast(iterations=iterations)
            params = train_ai.load_hyperparameters(self.params_file)

            start_time = time.time()

            for i in range(iterations):
                # Entraîner pour une itération
                train_ai.train_ai_iteration(i, self.model_path, params)

                # Calculer le temps restant
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (i + 1)) * (iterations - (i + 1))

                # Mettre à jour la barre de progression
                self.update_progress(i + 1, iterations, remaining_time)

            print("Entraînement terminé.")
        else:
            print("Veuillez sélectionner un modèle et un fichier de paramètres.")
                    
    def train_ai_lowder(self, iterations=None):
        # Ajouter le bouton "Stop" avant d'itérer sur les autres boutons
        # self.buttons['stop'] = pygame.Rect(650, 500, 100, 50)

        # Effacer l'interface actuelle
        self.screen.fill(WHITE)
        pygame.display.update()

        # Dessiner le bouton "Stop"
        # pygame.draw.rect(self.screen, (200, 0, 0), self.buttons['stop'])  # Dessiner le bouton "Stop" en rouge
        # stop_text = self.font.render("Stop", True, WHITE)
        # self.screen.blit(stop_text, (self.buttons['stop'].x + 25, self.buttons['stop'].y + 10))  # Positionner le texte au centre du bouton

        pygame.display.update()  # Mettre à jour l'affichage pour inclure le bouton

        # Demander à l'utilisateur combien d'itérations il souhaite faire
        if iterations is None:
            iterations = self.demander_iterations()

        # Afficher un message indiquant que l'entraînement a commencé
        self.afficher_message("Entraînement en cours...")

        # Vérifier si les paramètres ont été chargés
        if self.params_file == "Empty" or not self.params:
            print("Veuillez charger les paramètres d'entraînement avant de lancer l'entraînement.")
            return

        # Initialiser l'agent pour l'entraînement
        print("Initialisation de l'agent pour un entraînement rapide...")
        self.agent = SnakeAgent(
            env_type='full',
            gamma=self.params.get('gamma', 0.99),
            epsilon=self.params.get('epsilon', 1.0),
            epsilon_min=self.params.get('epsilon_min', 0.01),
            epsilon_decay=self.params.get('epsilon_decay', 0.995),
            learning_rate=self.params.get('learning_rate', 0.001),
            batch_size=self.params.get('batch_size', 32),
            memory_size=self.params.get('memory_size', 10000),
            fc1_units=self.params.get('fc1', 128),
            fc2_units=self.params.get('fc2', 64),
            fc3_units=self.params.get('fc3', 32),
        )

        if not self.model_path:
            print("Aucun modèle chargé. Veuillez d'abord charger un modèle.")
            return

        print("Chargement du modèle...")
        try:
            load_model(self.agent.model, self.model_path)
            print("Modèle chargé avec succès.")
        except FileNotFoundError:
            print("Modèle non trouvé. Début de l'entraînement à partir de zéro.")

        # Démarrer l'entraînement dans un thread séparé
        training_thread = threading.Thread(target=self.run_training, args=(iterations,))
        training_thread.start()

        # Boucle principale pour maintenir l'interface active et réactive
        # while training_thread.is_alive():
        #     self.handle_stop_button()  # Vérifie si le bouton Stop est pressé

        #     # # Redessiner le bouton "Stop" sur chaque itération pour s'assurer qu'il reste visible
        #     # pygame.draw.rect(self.screen, (200, 0, 0), self.buttons['stop'])
        #     # self.screen.blit(stop_text, (self.buttons['stop'].x + 25, self.buttons['stop'].y + 10))

        pygame.display.flip()
        pygame.time.delay(100)  # Pause pour éviter une boucle trop rapide
                            
    def update_progress(self, current_iteration, total_iterations, remaining_time):
        self.screen.fill(WHITE)
        
        # Dessiner la barre de progression
        progress_bar_width = 600
        progress_bar_height = 40
        progress = current_iteration / total_iterations
        filled_width = int(progress * progress_bar_width)
        
        progress_bar_rect = pygame.Rect((INTERFACE_WIDTH - progress_bar_width) // 2, (INTERFACE_HEIGHT // 2) - 20, progress_bar_width, progress_bar_height)
        pygame.draw.rect(self.screen, BLACK, progress_bar_rect, 2)
        
        filled_rect = pygame.Rect(progress_bar_rect.left, progress_bar_rect.top, filled_width, progress_bar_height)
        pygame.draw.rect(self.screen, BLUE, filled_rect)
        
        # Afficher le texte du temps restant
        font = pygame.font.SysFont(None, 36)
        time_text = f"Temps restant estimé: {int(remaining_time)} secondes"
        time_surface = font.render(time_text, True, BLACK)
        time_rect = time_surface.get_rect(center=(INTERFACE_WIDTH // 2, (INTERFACE_HEIGHT // 2) + 60))
        self.screen.blit(time_surface, time_rect)

        # Afficher l'itération en cours
        iteration_text = f"Itération {current_iteration}/{total_iterations}"
        iteration_surface = font.render(iteration_text, True, BLACK)
        iteration_rect = iteration_surface.get_rect(center=(INTERFACE_WIDTH // 2, (INTERFACE_HEIGHT // 2) - 60))
        self.screen.blit(iteration_surface, iteration_rect)
        
        pygame.display.flip()


    def demander_iterations(self):
        iterations = ""
        font = pygame.font.SysFont(None, 36)
        input_active = True
        clock = pygame.time.Clock()  # Crée une horloge pour limiter le framerate

        while input_active:
            self.screen.fill(WHITE)
            self.afficher_message("Combien d'itérations souhaitez-vous faire ? Entrez un nombre :", -50)

            # Dessiner la case de saisie
            input_box = pygame.Rect(INTERFACE_WIDTH // 2 - 50, INTERFACE_HEIGHT // 2, 100, 40)
            pygame.draw.rect(self.screen, BLACK, input_box, 2)

            # Afficher le texte entré par l'utilisateur
            text_surface = font.render(iterations, True, BLACK)
            self.screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if iterations.isdigit():
                            input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        iterations = iterations[:-1]
                    else:
                        if event.unicode.isdigit():
                            iterations += event.unicode

            clock.tick(30)  # Limite le framerate à 30 FPS

        return int(iterations) if iterations.isdigit() else 1000      
    def afficher_message(self, message, y_offset=0):
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(message, True, BLACK)
        text_rect = text_surface.get_rect(center=(INTERFACE_WIDTH // 2, INTERFACE_HEIGHT // 2 + y_offset))
        self.screen.blit(text_surface, text_rect)
        pygame.display.update()                                                
    
    def view_ai_play(self):
        if self.model_name == "Empty":
            print("Aucun modèle chargé. Veuillez d'abord charger un modèle.")
        else:
            print("Lancement de la visualisation du modèle...")  # Debugging
            try:
                view_ai_play = ViewAIPlay(self.model_path)
                print("Instance de ViewAIPlay créée avec succès.")  # Debugging
                view_ai_play.play_with_model()
            except Exception as e:
                print(f"Erreur lors de l'exécution du script: {e}")            
    
    def handle_stop_button(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # elif event.type == pygame.MOUSEBUTTONDOWN:
            #     if self.buttons['stop'].collidepoint(event.pos):
            #         print("Bouton Stop pressé, interruption de l'entraînement...")
            #         raise KeyboardInterrupt()  # Lever une exception pour sortir proprement de l'entraînement            
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for label, rect in list(self.buttons.items()):
                        if rect.collidepoint(event.pos):
                            print(f"Button {label} clicked")
                            if label == 'create_new_model':
                                self.create_new_model()
                            elif label == 'select_model':
                                self.select_model()
                            elif label == 'change_params':
                                print("Bouton 'change_params' cliqué.")
                                self.change_params()
                            elif label == 'load_params':
                                print("Bouton 'load_params' cliqué.")
                                self.load_params()
                            elif label == 'view_ai_play':
                                print("Bouton 'view_ai_play' cliqué.")
                                self.view_ai_play()
                            elif label == 'train_ai_fast':
                                print("Bouton 'train_ai_fast' cliqué.")
                                self.train_ai_fast_interface()  # Appeler l'entraînement rapide
                            elif label == 'train_ai_lowder':
                                print("Bouton 'train_ai_lowder' cliqué.")
                                self.train_ai_fast_interface()  # Appeler l'entraînement rapide 
                                
                            elif label == 'back':
                                self.go_back()

            self.create_interface()
            pygame.display.update()

        pygame.quit()
        sys.exit()
        
    def create_new_model(self):
        self.ask_configuration_choice()
        input_dim = 12  # Exemple de valeur, ajustable en fonction de vos besoins
        output_dim = 3
        fc1_units = 128
        fc2_units = 64
        fc3_units = 32

        self.model = create_model(input_dim, output_dim, fc1_units, fc2_units, fc3_units)
        
        print("Nouveau modèle AI créé.")
        
        # Mettre à jour l'interface avec le nouveau modèle
        self.model_name = "Nouveau modèle"
        self.create_interface()
        
    def load_params(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                loaded_params = json.load(f)
                print(f"Params chargés: {loaded_params}")  # Ajoutez cette ligne pour vérifier les paramètres chargés
                for key in ['fc1', 'fc2', 'fc3', 'WIDTH', 'HEIGHT']:
                    loaded_params.pop(key, None)
                self.params.update(loaded_params)
                self.params_file = file_path  # Mettre à jour le chemin du fichier de paramètres utilisé
            print(f"Parameters loaded from {file_path}")
        else:
            print("No parameters file selected.")
            
    def ask_configuration_choice(self):
        # Charger l'image de fond
        background_image = pygame.image.load("snake_image3.jpg")
        background_image = pygame.transform.scale(background_image, (INTERFACE_WIDTH, INTERFACE_HEIGHT))
        
        running = True
        while running:
            # Afficher l'image de fond
            self.screen.blit(background_image, (0, 0))
            
            # Récupérer la position de la souris et l'état des clics
            souris = pygame.mouse.get_pos()
            clic = pygame.mouse.get_pressed()
            
            # Dessiner et gérer le bouton Manuel
            manual_button = pygame.Rect(100, 300, 200, 50)
            self.draw_dynamic_button(manual_button, "Manuel", souris, clic, self.manual_configuration, GRAY, (170, 170, 170))
            
            # Dessiner et gérer le bouton Pré-configuré
            pre_config_button = pygame.Rect(100, 200, 200, 50)
            self.draw_dynamic_button(pre_config_button, "Pré-configuré", souris, clic, self.pre_configuration, GRAY, (170, 170, 170))
            
            # Dessiner et gérer le bouton Back
            back_button = pygame.Rect(400, 500, 250, 50)
            self.draw_dynamic_button(back_button, "Back", souris, clic, self.go_back, GRAY, (170, 170, 170))
            
            # Dessiner et afficher le nom du modèle uploadé
            model_name_button = pygame.Rect(100, 100, 600, 50)
            self.draw_static_button(model_name_button, f"variable actuelle: {self.model_name}", souris, clic, None, GRAY, (170, 170, 170))

            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        pygame.quit()
        sys.exit()
                
    def draw_dynamic_button(self, rect, text, souris, clic, action, couleur_inactive, couleur_active):
        # Détermine la couleur du bouton en fonction de l'état de la souris
        if rect.collidepoint(souris):
            pygame.draw.rect(self.screen, couleur_active, rect)
            if clic[0] == 1:
                action()
        else:
            pygame.draw.rect(self.screen, couleur_inactive, rect)
        
        # Affiche le texte sur le bouton
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(text, True, BLACK)
        self.screen.blit(text_surface, (rect.x + 10, rect.y + 10))     
      
    def draw_static_button(self, rect, text, souris, clic, action, couleur_inactive, couleur_active):
        # Détermine la couleur du bouton en fonction de l'état de la souris
        if rect.collidepoint(souris):
            pygame.draw.rect(self.screen, couleur_active, rect)
        else:
            pygame.draw.rect(self.screen, couleur_inactive, rect)
        
        # Affiche le texte sur le bouton
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(text, True, BLACK)
        self.screen.blit(text_surface, (rect.x + 10, rect.y + 10))
        
    def manual_configuration(self):
        # Logique pour la configuration manuelle (à définir plus tard)
        pass

    def pre_configuration(self):
        # Effacer l'écran
        self.screen.fill((0, 0, 0))
        
        # Liste des tailles de grilles et boutons associés
        grid_sizes = list(MODEL_CONFIGS.keys())
        buttons = []
        y_offset = 100
        
        # Boucle pour créer les boutons de modèles
        for i, size in enumerate(grid_sizes):
            model_info = MODEL_CONFIGS[size]
            button_text = f"Modèle {size[0]}x{size[1]} | Input: {model_info['input_dim']} | Layers: {model_info['layers']} | Output: {model_info['output_dim']}"
            button = pygame.Rect(100, y_offset + i * 60, 600, 50)
            buttons.append((button, size))
        
        # Ajouter le bouton "Retour"
        return_button = pygame.Rect(100, y_offset + len(grid_sizes) * 60 + 20, 200, 50)
        
        selecting_model = True
        while selecting_model:
            # Réinitialiser l'écran
            self.screen.fill((0, 0, 0))

            # Afficher les boutons de modèles avec surbrillance
            for button, size in buttons:
                model_info = MODEL_CONFIGS[size]
                button_text = f"Modèle {size[0]}x{size[1]} | Input: {model_info['input_dim']} | Layers: {model_info['layers']} | Output: {model_info['output_dim']}"
                self.draw_dynamic_button(button, button_text, pygame.mouse.get_pos(), pygame.mouse.get_pressed(), None, GRAY, (100, 100, 250))

            # Afficher le bouton "Retour" avec surbrillance
            self.draw_dynamic_button(return_button, "Retour", pygame.mouse.get_pos(), pygame.mouse.get_pressed(), self.ask_configuration_choice, GRAY, (250, 100, 100))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for button, size in buttons:
                        if button.collidepoint(pygame.mouse.get_pos()):
                            # Demander l'emplacement de sauvegarde
                            model_path = self.ask_save_location(size)  # Passer la taille de la grille à la méthode de sauvegarde
                            if model_path:
                                print(f"Modèle {size[0]}x{size[1]} créé et sauvegardé à {model_path}")
                                selecting_model = False
                                break
                    # Gestion du bouton "Retour"
                    if return_button.collidepoint(pygame.mouse.get_pos()):
                        self.ask_configuration_choice()
                        selecting_model = False

        # Retour à l'écran précédent après sélection et sauvegarde
        self.ask_configuration_choice()
    
    def ask_save_location(self, grid_size):
        Tk().withdraw()
        save_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
        if not save_path:
            print("Aucun emplacement de sauvegarde sélectionné.")
            return None
        
        # Créer et sauvegarder le modèle avec les métadonnées
        model = create_and_save_model(grid_size, save_path)
        print(f"Model and metadata saved to {save_path}")
        
        return save_path
    
    def draw_dynamic_button(self, rect, text, souris, clic, action, couleur_inactive, couleur_active):
        # Mettre en surbrillance si la souris survole le bouton
        if rect.collidepoint(souris):
            pygame.draw.rect(self.screen, couleur_active, rect)
        else:
            pygame.draw.rect(self.screen, couleur_inactive, rect)
        
        font = pygame.font.Font(None, 24)
        text_surf = font.render(text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

        # Exécuter l'action si le bouton est cliqué
        if rect.collidepoint(souris) and clic[0] == 1 and action is not None:
            action()
    
    def select_model(self):
        Tk().withdraw()  # Cacher la fenêtre principale Tkinter
        print("Opening file dialog for model selection...")  # Debugging line
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
        
        if file_path:
            self.model_name = os.path.basename(file_path)
            self.model_path = file_path
            print(f"Model selected: {self.model_path}")  # Debugging line

            try:
                # Utiliser la classe LoadModel pour charger le modèle avec la détection automatique
                loader = LoadModel(self.model_path)
                self.model = loader.model  # Assigner le modèle chargé à l'instance
                self.grid_size = loader.grid_size  # Conserver la taille de la grille détectée
                
                print(f"Model loaded successfully from {self.model_path} for grid size {self.grid_size}")
                
                # Redessiner l'interface pour afficher le modèle sélectionné
                self.create_interface()
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
        else:
            print("No model selected.")
            
    def go_back(self):
                pygame.quit()  # Ferme l'interface actuelle
                subprocess.run(["python", "main.py"])  # Lance le script main.py
    
if __name__ == "__main__":
    interface = AIGamesInterface()
    interface.run()

    
