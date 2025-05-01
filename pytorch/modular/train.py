
import get_data, get_data_details, augmented_dataloader, model_init, engine, save_model
from sklearn.metrics import accuracy_score

def train():
    # get data
    get_data.get_data()
    # get data details
    get_data_details.get_data_details(folder="../data/food-101")
    # augemented dataloader
    train_dataloader_aug, test_dataloader_aug = augmented_dataloader.get_augemented_dataloader(data_path="data/food-101", img_size=64, batch_size=32)
    # initializing model and params
    model, loss_fn, optimizer, device = model_init.model_initialize(lr=1e-2)
    # Train model
    train_losses, train_accuracies, test_losses, test_accuracies = engine.train(epochs=100, model=model, train_dataloader=train_dataloader_aug, test_dataloader=test_dataloader_aug, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_score, device=device)
    # Save model
    save_model.save_model(model=model, name="models/vgg_saved.pth")

train()