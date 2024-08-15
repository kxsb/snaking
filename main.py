      ## -launcher- ##

import pygame
import sys
import subprocess
import matplotlib.pyplot as plt
import json
import os
from snake import lancer_snake 
from AIgames_interface import AIGamesInterface  # Importer l'interface AI Games

pygame.init()

# Dimensions de la fenêtre
LARGEUR, HAUTEUR = 800, 600
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
BLEU = (0, 0, 255)
ROUGE = (255, 0, 0)

# Charger l'image de fond
background_image = pygame.image.load("snake_image.jpg")
background_image = pygame.transform.scale(background_image, (LARGEUR, HAUTEUR))

fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption('Interface Snake Game')

font = pygame.font.Font(None, 36)

def afficher_texte(text, x, y, couleur=NOIR):
    texte = font.render(text, True, couleur)
    fenetre.blit(texte, (x, y))

def afficher_bouton(text, x, y, w, h, couleur_active, couleur_inactive, action=None):
    souris = pygame.mouse.get_pos()
    clic = pygame.mouse.get_pressed()

    if x + w > souris[0] > x and y + h > souris[1] > y:
        pygame.draw.rect(fenetre, couleur_active, (x, y, w, h))
        if clic[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(fenetre, couleur_inactive, (x, y, w, h))

    afficher_texte(text, x + 10, y + 10)

def afficher_graphique():
    if not os.path.exists('training_scores.json'):
        print("Aucun fichier de scores trouvé.")
        return
    
    with open('training_scores.json', 'r') as f:
        scores = json.load(f)
    
    fig, ax = plt.subplots()
    ax.plot(scores, 'b-')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress')
    plt.show()

def lancer_ai_games_interface():
    pygame.quit()
    interface = AIGamesInterface()
    interface.run()

def menu_principal():
    # Dessiner l'image de fond
    fenetre.blit(background_image, (0, 0))

    afficher_texte("Welcome young snake!", 250, 50)

    afficher_bouton("Play Snake solo", 250, 150, 300, 50, BLEU, ROUGE, lancer_snake)
    afficher_bouton("Make somes AI games", 250, 350, 300, 50, BLEU, ROUGE, lancer_ai_games_interface)  # Appeler l'interface AI Games
    # afficher_bouton("play against (not implemented yet)", 300, 220, 200, 50, BLEU, ROUGE)

    pygame.display.update()

def boucle_principale():
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        menu_principal()
        clock.tick(15)

if __name__ == "__main__":
    boucle_principale()
