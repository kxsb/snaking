import pygame
import random
import sys
import pygame


TAILLE_DIS = 400  # Taille de la grille pour le jeu Snake
TAILLE_CARRE = 20
VITESSE_SNAKE = 15

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
VERT = (0, 255, 0)

pygame.font.init()
font = pygame.font.Font(None, 36)

# Variables globales pour le mode de jeu
mode = "menu"
snake_direction = (0, 0)
snake_list = []
snake_length = 1
food_x, food_y = 0, 0
score = 0
game_over = False

def lancer_snake():
    global mode, game_over, score, snake_list, snake_length, snake_direction, food_x, food_y
    mode = "jeu"
    game_over = False
    score = 0
    snake_list = []
    snake_length = 1
    snake_direction = (0, 0)
    food_x, food_y = placer_pomme(TAILLE_DIS, TAILLE_CARRE)
    game_loop()
    
def obtenir_coordonnees_centre(fenetre_principale, taille_jeu):
    largeur_fenetre, hauteur_fenetre = fenetre_principale.get_size()
    x = (largeur_fenetre - taille_jeu) // 2
    y = (hauteur_fenetre - taille_jeu) // 2
    return x, y


def game_loop():
    global snake_direction, snake_list, snake_length, food_x, food_y, score, game_over

    fenetre = pygame.display.get_surface()  # Utiliser la surface actuelle
    x1 = TAILLE_DIS // 2
    y1 = TAILLE_DIS // 2
    clock = pygame.time.Clock()

    # Obtenez les coordonnées pour centrer la surface de jeu Snake
    centre_x, centre_y = obtenir_coordonnees_centre(fenetre, TAILLE_DIS)

    while mode == "jeu":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return_to_menu()
                if event.key == pygame.K_UP and snake_direction != (0, TAILLE_CARRE):
                    snake_direction = (0, -TAILLE_CARRE)
                elif event.key == pygame.K_DOWN and snake_direction != (0, -TAILLE_CARRE):
                    snake_direction = (0, TAILLE_CARRE)
                elif event.key == pygame.K_LEFT and snake_direction != (TAILLE_CARRE, 0):
                    snake_direction = (-TAILLE_CARRE, 0)
                elif event.key == pygame.K_RIGHT and snake_direction != (-TAILLE_CARRE, 0):
                    snake_direction = (TAILLE_CARRE, 0)

        x1 += snake_direction[0]
        y1 += snake_direction[1]
        
        # Remplir la fenêtre principale en blanc
        fenetre.fill(BLANC)

        # Dessiner le cadre autour de la zone de jeu Snake
        pygame.draw.rect(fenetre, NOIR, (centre_x, centre_y, TAILLE_DIS, TAILLE_DIS), 2)  # Cadre noir avec épaisseur 2

        # Dessiner la pomme
        pygame.draw.rect(fenetre, VERT, (centre_x + food_x, centre_y + food_y, TAILLE_CARRE, TAILLE_CARRE))
        
        # Mettre à jour la position de la tête du serpent
        tete_snake = [centre_x + x1, centre_y + y1]
        snake_list.append(tete_snake)
        if len(snake_list) > snake_length:
            del snake_list[0]

        if verifier_collision_bords(x1, y1, TAILLE_DIS) or verifier_collision_serpent(tete_snake, snake_list):
            game_over = True

        # Dessiner le serpent
        for segment in snake_list:
            pygame.draw.rect(fenetre, NOIR, (segment[0], segment[1], TAILLE_CARRE, TAILLE_CARRE))
        
        # Afficher le score
        afficher_score(fenetre, score)

        pygame.display.update()

        if x1 == food_x and y1 == food_y:
            food_x, food_y = placer_pomme(TAILLE_DIS, TAILLE_CARRE)
            snake_length += 1
            score += 1

        if game_over:
            return_to_menu()

        clock.tick(VITESSE_SNAKE)

def return_to_menu():
    global mode, game_over
    mode = "menu"
    game_over = False

def afficher_pomme(fenetre, taille_carre, nourriturex, nourriturey):
    pygame.draw.rect(fenetre, VERT, (nourriturex, nourriturey, taille_carre, taille_carre))

def afficher_serpent(fenetre, taille_carre, liste_snake):
    for segment in liste_snake:
        pygame.draw.rect(fenetre, NOIR, (segment[0], segment[1], taille_carre, taille_carre))

def afficher_score(fenetre, score):
    text = font.render(f'Score: {score}', True, NOIR)
    fenetre.blit(text, [0, 0])

def verifier_collision_bords(x1, y1, taille_dis):
    return x1 < 0 or x1 >= taille_dis or y1 < 0 or y1 >= taille_dis

def verifier_collision_serpent(tete_snake, liste_snake):
    return tete_snake in liste_snake[:-1]

def placer_pomme(taille_dis, taille_carre):
    return (random.randint(0, (taille_dis - taille_carre) // taille_carre) * taille_carre,
            random.randint(0, (taille_dis - taille_carre) // taille_carre) * taille_carre)
