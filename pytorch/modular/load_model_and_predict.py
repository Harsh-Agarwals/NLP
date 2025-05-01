
import predict, model_builder
import torch

def load_model_and_predict(model_path, classes, file_path="../data/food-101/04-pizza-dad.jpeg", device="cpu"):
    model = model_builder.TinyVGG(input_shape=3, hidden_shape=10, output_shape=3).to(device=device)

    model.load_state_dict(torch.load(f=model_path))

    pred, pred_class, pred_prob = predict.predict_image(file_path=file_path, model=model, device=device, classes=classes)
    print(f"Model prediction: {pred}, Class: {pred_class}, Probability: {pred_prob}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

load_model_and_predict(model_path="../models/vgg_saved.pth", classes=["pizza", "steak", "sushi"], file_path="../data/food-101/04-pizza-dad.jpeg", device=device)