import torch
import torch.nn as nn
from torch import nn, optim
from create_new_model import MODEL_CONFIGS

scores = []
losses = []

class LoadModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model, self.grid_size = self.load_model()

    def load_model(self):
        # Charger les métadonnées
        checkpoint = torch.load(self.model_path)
        state_dict = checkpoint['state_dict']
        grid_size = checkpoint['grid_size']

        # Déterminer la configuration du modèle en fonction de la taille de la grille
        if grid_size == (6, 6):
            model = self._create_model_6x6()
        elif grid_size == (8, 8):
            model = self._create_model_8x8()
        elif grid_size == (10, 10):
            model = self._create_model_10x10()
        elif grid_size == (12, 12):
            model = self._create_model_12x12()
        elif grid_size == (15, 15):
            model = self._create_model_15x15()
        elif grid_size == (20, 20):
            model = self._create_model_20x20()
        elif grid_size == (25, 25):
            model = self._create_model_25x25()
        else:
            raise ValueError("Unsupported grid size.")
        
        model.load_state_dict(state_dict)
        model.eval()
        return model, grid_size

    def _create_model_6x6(self):
        return nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def _create_model_8x8(self):
        return nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def _create_model_10x10(self):
        return nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def _create_model_12x12(self):
        return nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def _create_model_15x15(self):
        return nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def _create_model_20x20(self):
        return nn.Sequential(
            nn.Linear(12, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def _create_model_25x25(self):
        return nn.Sequential(
            nn.Linear(12, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
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

def convert_weights(old_weights, new_shape):
    """
    Convertit les poids d'une couche vers une nouvelle forme, en interpolant ou en remplissant si nécessaire.
    
    Args:
    old_weights (torch.Tensor): Les poids existants de la couche à convertir.
    new_shape (tuple): La nouvelle forme que les poids doivent avoir.

    Returns:
    torch.Tensor: Un tenseur avec la nouvelle forme, avec les poids interpolés ou ajustés.
    """
    old_shape = old_weights.shape

    # Si les dimensions sont les mêmes, pas besoin de conversion
    if old_shape == new_shape:
        return old_weights

    # Création d'une nouvelle matrice de poids avec la nouvelle forme
    new_weights = torch.zeros(new_shape)

    if len(new_shape) == 1:
        # Traitement pour les biais (1D)
        scale_factor = new_shape[0] / old_shape[0]
        for i in range(new_shape[0]):
            old_idx = int(i / scale_factor)
            new_weights[i] = old_weights[old_idx]
    else:
        # Traitement pour les poids (2D)
        scale_factors = [n / o for n, o in zip(new_shape, old_shape)]
        for idx, weight in enumerate(old_weights):
            new_idx = tuple(int(i * s) for i, s in zip(idx, scale_factors))
            new_weights[new_idx] = weight

    return new_weights