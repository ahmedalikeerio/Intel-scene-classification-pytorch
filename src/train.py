import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_CLASSES,
    IMPROVED_MODEL_PATH,
    )
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.utils import plot_training_history


def train_model():
    print("Using Device :", DEVICE)

    train_loader, test_loader, class_names= get_dataloaders()

    print('Classes:', class_names)
    print("Training Batches:", len(train_loader))
    print("Testing Batches: ",len(test_loader))

    model=SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

    criterion=nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(),lr=LEARNING_RATE)
    scheduler=optim.lr_scheduler.StepLR(
        optimizer,
        step_size=4,
        gamma=0.5
    )

    train_losses=[]
    train_accuracies=[]
    test_accuracies=[]

    best_test_accuracy=0.0

    for epoch in range(EPOCHS):
        model.train()

        running_loss=0
        correct=0
        total=0

        progress_bar=tqdm(train_loader, desc=f'Epoch {epoch +1}/{EPOCHS}')

        for images,labels in progress_bar:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(images)
            loss=criterion(outputs,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted=torch.max(outputs,1)

            total += labels.size(0)

            correct+= (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': loss.item()
            })
        scheduler.step()

        epoch_loss=running_loss/len(train_loader)
        epoch_accuracy=100*correct/total
        test_accuracy=evaluate_during_training(model,test_loader)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f'Epoch [{epoch +1}/{EPOCHS}]'
            f'Loss: {epoch_loss:.4f}'
            f'Train Acc: {epoch_accuracy:.2f}%'
            f'Test Acc: {test_accuracy:.2f}%'
        )
        
        if test_accuracy>best_test_accuracy:
            best_test_accuracy=test_accuracy
            torch.save(model.state_dict(),IMPROVED_MODEL_PATH)
            print(f"Best improved model saved: {IMPROVED_MODEL_PATH}")


    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")


    plot_training_history(train_losses,test_accuracies, train_accuracies, model_name='improved_cnn')

def evaluate_during_training(model, test_loader):
    model.eval()

    correct=0
    total=0

    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(images)
            _, predicted=torch.max(outputs,1)

            total+=labels.size(0)
            correct+= (predicted == labels).sum().item()

    model.train()

    return 100*correct/total

if __name__=='__main__':
    train_model()
