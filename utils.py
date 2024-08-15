import torch
import matplotlib.pyplot as plt

MODEL_PATH = 'snake_model.pth'

# Initialisation pour l'affichage en temps réel
plt.ion()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
scores = []
losses = []

def plot_scores_and_losses():
    ax1.clear()  # Efface le sous-graphe des scores
    ax2.clear()  # Efface le sous-graphe des pertes
    
    # Plot des scores
    ax1.plot(scores, label='Score')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Score')
    ax1.set_title('Scores Over Time')
    ax1.legend()

    # Plot des pertes
    ax2.plot(losses, label='Loss')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Over Time')
    ax2.legend()

    fig.canvas.draw()
    plt.pause(0.001)  # Pause pour mettre à jour l'affichage

def load_model(model, path=MODEL_PATH):
    state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {path}")

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


#def print_model_weights(model, title): #(pour afficher les poids du modèle au moment du chargement et de l'extinction.)
#print(title)
#for name, param in model.named_parameters():
#print(name, param.data
