import torch
import torch.nn as nn
import os
from tkinter import Tk, filedialog

# Configuration des modèles en fonction de la taille de la grille
MODEL_CONFIGS = {
    (6, 6): {'input_dim': 12, 'layers': [64, 32], 'output_dim': 3},
    (8, 8): {'input_dim': 12, 'layers': [128, 64], 'output_dim': 3},
    (10, 10): {'input_dim': 12, 'layers': [128, 64], 'output_dim': 3},
    (12, 12): {'input_dim': 12, 'layers': [256, 128, 64], 'output_dim': 3},
    (15, 15): {'input_dim': 12, 'layers': [256, 128, 64], 'output_dim': 3},
    (20, 20): {'input_dim': 12, 'layers': [512, 256, 128], 'output_dim': 3},
    (25, 25): {'input_dim': 12, 'layers': [1024, 512, 256], 'output_dim': 3},
}

def create_model(grid_size):
    config = MODEL_CONFIGS.get(grid_size)
    
    if not config:
        raise ValueError(f"Grid size {grid_size} not supported.")
    
    layers = []
    input_dim = config['input_dim']
    
    for layer_size in config['layers']:
        layers.append(nn.Linear(input_dim, layer_size))
        layers.append(nn.ReLU())
        input_dim = layer_size
    
    layers.append(nn.Linear(input_dim, config['output_dim']))
    
    model = nn.Sequential(*layers)
    return model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def create_and_save_model(grid_size, model_path):
    # Créer le modèle en fonction de la taille de la grille
    model = create_model(grid_size)  # Assurez-vous que cette fonction crée le bon modèle

    # Sauvegarder le modèle avec les métadonnées
    model_info = {
        'state_dict': model.state_dict(),
        'grid_size': grid_size
    }
    torch.save(model_info, model_path)
    print(f"Model and metadata saved to {model_path}")
    return model

def select_grid_size():
    Tk().withdraw()
    print("Select grid size:")
    grid_sizes = list(MODEL_CONFIGS.keys())
    for i, size in enumerate(grid_sizes):
        print(f"{i+1}. {size[0]}x{size[1]}")
    
    choice = int(input("Enter your choice: ")) - 1
    if choice < 0 or choice >= len(grid_sizes):
        raise ValueError("Invalid choice.")
    
    return grid_sizes[choice]

def choose_save_location():
    Tk().withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
    if not save_path:
        raise ValueError("No save location selected.")
    return save_path

def main():
    try:
        grid_size = select_grid_size()
        model_path = choose_save_location()
        create_and_save_model(grid_size, model_path)
        print(f"Model created with grid size {grid_size} and saved as {model_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
