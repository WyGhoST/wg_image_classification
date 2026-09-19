import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from train import *

def main():
    torch.manual_seed(42)

    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SoftmaxClassifier(
        input_size=28 * 28,
        num_classes=10
    )

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            LEARNING_RATE
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()