import pygame
import sys
import json
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

        # Paramètres par défaut
        self.params = {
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 10000,
            'iterations': 1000,
            'fc1': 128,
            'fc2': 64,
            'fc3': 32,
            'tick_rate': 80,
            'evaluation_interval': 10,
            'collision_penalty': -1,
            'food_reward': 1,
        }

        # Explications des paramètres
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
        }

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
            'load': pygame.Rect(550, 500, 80, 40),
            'back': pygame.Rect(700, 500, 80, 40)
        }
        self.active_input = None
        self.input_text = ''
        self.prev_value = ''
        self.cursor_visible = True
        self.cursor_timer = pygame.time.get_ticks()

        self.hovered_param = None  # Conserver l'état de survol

        self.create_interface()

    def create_interface(self):
        self.screen.fill(WHITE)
        y_offset = 50
        for i, (param, value) in enumerate(self.params.items()):
            label = self.font.render(f"{param}:", True, BLACK)
            self.screen.blit(label, (50, y_offset + i * 30))
            pygame.draw.rect(self.screen, GRAY, self.inputs[param])
            if self.active_input == param:
                display_text = self.input_text
            else:
                display_text = str(value)
            text_surface = self.font.render(display_text, True, BLACK)
            self.screen.blit(text_surface, (self.inputs[param].x + 5, self.inputs[param].y + 5))
        
        # Dessiner les boutons
        for button, rect in self.buttons.items():
            pygame.draw.rect(self.screen, GRAY, rect)
            text_surface = self.font.render(button.capitalize(), True, BLACK)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def show_help(self, text, position):
        lines = text.split("\n")
        width = max(self.help_font.size(line)[0] for line in lines) + 10
        height = len(lines) * self.help_font.get_height() + 10
        help_rect = pygame.Rect(position[0], position[1], width, height)
        pygame.draw.rect(self.screen, YELLOW, help_rect)
        pygame.draw.rect(self.screen, BLACK, help_rect, 1)
        for i, line in enumerate(lines):
            line_surface = self.help_font.render(line, True, BLACK)
            self.screen.blit(line_surface, (position[0] + 5, position[1] + 5 + i * self.help_font.get_height()))

    def save_params(self):
        Tk().withdraw()  # Cacher la fenêtre Tkinter
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.params, f)
            print(f"Parameters saved to {file_path}")

    def load_params(self):
        Tk().withdraw()  # Cacher la fenêtre Tkinter
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                self.params = json.load(f)
            print(f"Parameters loaded from {file_path}")
            self.create_interface()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            mouse_pos = pygame.mouse.get_pos()
            new_hovered_param = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.active_input is not None:
                        if event.key == pygame.K_RETURN:
                            try:
                                value = float(self.input_text)
                                self.params[self.active_input] = value
                            except ValueError:
                                pass  # Gérer les entrées incorrectes
                            self.active_input = None
                            self.input_text = ''
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        else:
                            self.input_text += event.unicode

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_outside = True
                    for param, rect in self.inputs.items():
                        if rect.collidepoint(event.pos):
                            clicked_outside = False
                            if self.active_input == param:
                                continue  # Si on clique sur le même champ, ne pas réinitialiser
                            if self.active_input is not None:
                                # Restaurer l'ancienne valeur si on quitte le champ sans valider
                                self.input_text = ''  # Réinitialiser le texte saisi
                            self.active_input = param
                            self.input_text = str(self.params[param])  # Charger la valeur actuelle dans la saisie
                            break
                    if clicked_outside and self.active_input is not None:
                        # Si on clique en dehors des champs actifs sans valider
                        self.active_input = None
                        self.input_text = ''

                    # Gérer les clics sur les boutons
                    for button, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            if button == 'save':
                                self.save_params()
                            elif button == 'load':
                                self.load_params()
                            elif button == 'back':
                                self.go_back()

            # Vérifie si la souris survole un paramètre
            for param, rect in self.labels.items():
                if rect.collidepoint(mouse_pos):
                    new_hovered_param = param
                    break

            # Mettre à jour le curseur
            if self.active_input is not None:
                current_time = pygame.time.get_ticks()
                if current_time - self.cursor_timer > 500:  # Clignotement toutes les 500ms
                    self.cursor_visible = not self.cursor_visible
                    self.cursor_timer = current_time

            self.create_interface()

            if self.active_input is not None:
                # Dessiner un curseur clignotant
                cursor_display = self.input_text
                if self.cursor_visible:
                    cursor_display += '|'
                pygame.draw.rect(self.screen, BLUE, self.inputs[self.active_input], 2)  # Bordure bleue pour la case active
                text_surface = self.font.render(cursor_display, True, BLACK)
                self.screen.blit(text_surface, (self.inputs[self.active_input].x + 5, self.inputs[self.active_input].y + 5))

            # Si un nouveau paramètre est survolé, mettre à jour l'état
            if new_hovered_param != self.hovered_param:
                self.hovered_param = new_hovered_param

            # Afficher la bulle d'aide si la souris survole un paramètre
            if self.hovered_param is not None:
                self.show_help(self.help_texts[self.hovered_param], (mouse_pos[0] + 20, mouse_pos[1]))

            pygame.display.update()
            clock.tick(60)  # Limite à 60 FPS pour une meilleure fluidité

        pygame.quit()
        sys.exit()

    def go_back(self):
        # Appeler le script principal
        pygame.quit()
        sys.exit()
        # À modifier pour exécuter `main.py` directement si besoin


if __name__ == "__main__":
    interface = TrainingInterface()
    interface.run()
