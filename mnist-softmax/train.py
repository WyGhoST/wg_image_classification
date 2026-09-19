import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from softmax_classifier import *


BATCH_SIZE = 64
LEARNING_RATE = 0.1
NUM_EPOCHS = 10


def cross_entropy_with_gradient(logits, labels):
    batch_size = logits.shape[0]

    sample_indices = torch.arange(
        batch_size,
        device=logits.device
    )

    # 1. Tính xác suất Softmax
    probabilities = softmax(logits, dim=1)

    # 2. Lấy xác suất của đúng lớp cho từng ảnh
    correct_probabilities = probabilities[
        sample_indices,
        labels
    ]

    # Tránh trường hợp log(0)
    correct_probabilities = correct_probabilities.clamp_min(1e-12)

    # Loss riêng của từng ảnh dùng Cross-Entropy
    sample_losses = -torch.log(correct_probabilities)

    # Loss trung bình của batch
    loss = sample_losses.mean()

    # 3. gradient dL/dlogits = (P-Y)/B
    grad_logits = probabilities.clone()

    grad_logits[
        sample_indices,
        labels
    ] -= 1.0

    grad_logits /= batch_size

    return loss, grad_logits


def backprobagation(images, grad_logits):
    """
    Tính:
        grad_W = dL/dW
        grad_b = dL/db
    """

    batch_size = images.shape[0]

    x = images.reshape(batch_size, -1)

    grad_W = x.T @ grad_logits

    grad_b = grad_logits.sum(dim=0)

    return grad_W, grad_b


def sgd_step(model, grad_W, grad_b, learning_rate):
    """
    Cập nhật tham số bằng phương pháp SGD.
    """

    model.W -= learning_rate * grad_W
    model.b -= learning_rate * grad_b


def train_one_epoch(model, train_loader, learning_rate):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        # 1. Forward
        logits = model(images)

        # 2. Tính loss và gradient theo logits
        loss, grad_logits = cross_entropy_with_gradient(
            logits,
            labels
        )

        # 3. Backpropagation
        grad_W, grad_b = backprobagation(
            images,
            grad_logits
        )

        # 4. Cập nhật W và b
        sgd_step(
            model,
            grad_W,
            grad_b,
            learning_rate
        )

        # 5. Tính accuracy
        predictions = torch.argmax(logits, dim=1)

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def evaluate(model, test_loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        logits = model(images)

        loss, _ = cross_entropy_with_gradient(
            logits,
            labels
        )

        predictions = torch.argmax(logits, dim=1)

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


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