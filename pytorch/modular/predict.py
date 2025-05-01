
import torchvision
import torch
from torchvision import  transforms

def predict_image(file_path, model, device, classes):
    new_img_path = file_path
    new_img = torchvision.io.read_image(new_img_path)
    new_img = new_img.type(torch.float32)
    new_img /= 255
    custom_img_transform = transforms.Resize(size=(64, 64))

    new_img_tranformed = custom_img_transform(new_img)
    pred = torch.argmax(torch.softmax(model(new_img_tranformed.unsqueeze(dim=0).to(device=device)), dim=1), dim=1).item()
    probs = model(new_img_tranformed.unsqueeze(dim=0).to(device=device))
    probs = torch.softmax(probs, dim=1)
    pred_class = classes[pred]
    pred_prob = probs.cpu().detach().squeeze(dim=0)[pred]
    return pred, pred_class, pred_prob
