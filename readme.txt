snake_train.py : Contient le script d'entraînement principal pour l'agent du jeu Snake, avec l'utilisation de la bibliothèque PyTorch pour entraîner un réseau neuronal profond (DQN). Il comprend aussi la fonction d'affichage des scores et des pertes pendant l'entraînement.

main.py : Gère l'interface principale du jeu Snake avec Pygame, où tu peux choisir de jouer au jeu ou d'utiliser l'interface AI pour entraîner ou évaluer des agents.

snake.py : Contient la logique de base du jeu Snake, incluant le contrôle du serpent, la génération de nourriture, et la gestion des collisions. Ce fichier gère également l'affichage du jeu via Pygame.

snake_env.py : Définit les environnements de jeu pour le serpent. Il y a deux classes : SnakeEnv pour une version complète avec interface graphique, et SimpleSnakeEnv pour un environnement simplifié.

snake_brain.py : Contient les fonctions pour initialiser et entraîner l'agent Snake, en utilisant le modèle DQN. Ce fichier est similaire à snake_train.py mais semble être configuré pour une autre version de l'entraînement ou des tests.

snake_agent.py : Déclare la classe SnakeAgent qui encapsule la logique de l'agent Snake, avec des fonctions pour entraîner, agir, et sauvegarder l'état du modèle.

snake_params.py : Gère une interface utilisateur pour configurer les paramètres d'entraînement de l'agent Snake. Tu peux ajuster les hyperparamètres comme le taux d'apprentissage, l'epsilon pour l'exploration, etc.

AIgames_interface.py : Interface pour jouer et entraîner des agents AI dans un environnement graphique. Inclut des fonctionnalités pour charger, réinitialiser et entraîner des modèles AI.
