import torch
# import matplotlib
# matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'WXAgg', etc. en fonction de ton système
# import matplotlib.pyplot as plt


scores = []
losses = []

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# def plot_scores_and_losses():
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     ax1.plot(scores, label='Score')
#     ax1.set_xlabel('Episodes')
#     ax1.set_ylabel('Score')
#     ax1.set_title('Scores Over Time')
#     ax1.legend()

#     ax2.plot(losses, label='Loss')
#     ax2.set_xlabel('Episodes')
#     ax2.set_ylabel('Loss')
#     ax2.set_title('Loss Over Time')
#     ax2.legend()

#     plt.show()  # Afficher le graphique une seule fois à la fin

def load_model(model, path):
    """
    Charge le modèle à partir du chemin spécifié.
    
    :param model: Le modèle à charger
    :param path: Le chemin du fichier modèle (.pth)
    """
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {path}")

def save_model(model, path):
    """
    Sauvegarde le modèle au chemin spécifié.
    
    :param model: Le modèle à sauvegarder
    :param path: Le chemin du fichier où sauvegarder le modèle
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def create_model(input_dim=12, output_dim=3, fc1_units=128, fc2_units=64, fc3_units=32):
    """
    Crée un modèle DQN avec les dimensions spécifiées.
    :param input_dim: Nombre d'entrées du modèle
    :param output_dim: Nombre de sorties du modèle
    :param fc1_units: Nombre de neurones dans la première couche cachée
    :param fc2_units: Nombre de neurones dans la deuxième couche cachée
    :param fc3_units: Nombre de neurones dans la troisième couche cachée
    :return: Modèle DQN
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, fc1_units),
        torch.nn.ReLU(),
        torch.nn.Linear(fc1_units, fc2_units),
        torch.nn.ReLU(),
        torch.nn.Linear(fc2_units, fc3_units),
        torch.nn.ReLU(),
        torch.nn.Linear(fc3_units, output_dim)
    )
    print("Modèle créé avec succès.")
    return model