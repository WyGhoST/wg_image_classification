import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

generator = torch.Generator().manual_seed(42)

train_dataset, valid_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
import torch
import torch.nn as nn

from cnn_model import CNNClassifier
from train import train_one_epoch, evaluate


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Thiết bị:", device)

model = CNNClassifier().to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

number_of_epochs = 10
best_valid_loss = float("inf")

for epoch in range(number_of_epochs):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        loss_function,
        optimizer,
        device
    )

    valid_loss, valid_accuracy = evaluate(
        model,
        valid_loader,
        loss_function,
        device
    )

    print(
        f"Epoch {epoch + 1:02d}/{number_of_epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy * 100:.2f}% | "
        f"Valid loss: {valid_loss:.4f} | "
        f"Valid acc: {valid_accuracy * 100:.2f}%"
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(
            model.state_dict(),
            "best_cnn_model.pth"
        )
        
model.load_state_dict(
    torch.load(
        "best_cnn_model.pth",
        map_location=device,
        weights_only=True
    )
)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    loss_function,
    device
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")