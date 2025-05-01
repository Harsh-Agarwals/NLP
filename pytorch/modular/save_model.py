import torch
import os

def save_model(model, name):
    torch.save(model.state_dict(), name)

    print(f"Model saved! Check at {name}")
