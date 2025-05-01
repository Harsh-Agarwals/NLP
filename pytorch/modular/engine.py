
from sklearn.metrics import accuracy_score
from tqdm.auto import  tqdm
import torch

def train_step(model, data, loss_fn, optimizer, accuracy_fn, device):
    model.train()

    train_loss = 0
    train_acc = 0

    for batch, (X, y) in enumerate(data):
        X = X.to(device=device)
        y = y.to(device=device)

        output = model(X)
        preds = torch.argmax(torch.softmax(output, dim=1), dim=1)

        loss = loss_fn(output, y)
        train_loss += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = accuracy_fn(preds.cpu(), y.cpu())
        train_acc += accuracy

    train_loss /= len(data)
    train_acc /= len(data)

    return train_loss, train_acc

def test_step(data, loss_fn, model, accuracy_fn, device):
    model.eval()

    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for X, y in data:
            X, y = X.to(device=device), y.to(device=device)

            output = model(X)
            preds = torch.argmax(torch.softmax(output, dim=1), dim=1)

            loss = loss_fn(output, y)
            test_loss += loss

            accuracy = accuracy_fn(preds.cpu(), y.cpu())
            test_acc += accuracy

    test_loss = test_loss/len(data)
    test_acc /= len(data)

    return test_loss, test_acc

def train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device):
    epochs = epochs

    batch_train_losses = []
    batch_train_accuracies = []

    batch_test_losses = []
    batch_test_accuracies = []

    for epoch in tqdm(range(epochs)):
        epoch_train_loss, epoch_train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device=device)

        batch_train_losses.append(epoch_train_loss)
        batch_train_accuracies.append(epoch_train_acc)

        epoch_test_loss, epoch_test_acc = test_step(test_dataloader, loss_fn, model, accuracy_fn, device=device)

        batch_test_losses.append(epoch_test_loss)
        batch_test_accuracies.append(epoch_test_acc)

        if epoch%10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}, Train Accuracy: {epoch_train_acc}, Test Loss: {epoch_test_loss}, Test Accuracy: {epoch_test_acc}")

    return batch_train_losses, batch_train_accuracies, batch_test_losses, batch_test_accuracies
