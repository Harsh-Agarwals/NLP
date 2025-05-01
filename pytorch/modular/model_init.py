
from torch import  nn
import torch 
import model_builder

def model_initialize(lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = model_builder.TinyVGG(input_shape=3, hidden_shape=10, output_shape=3).to(device=device)
    print("Model initialized")
    print(model)
    print(next(model.parameters()).device)
    print(f"Trainable params: {sum(torch.numel(i) for i in model.parameters() if i.requires_grad)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    return model, loss_fn, optimizer, device
