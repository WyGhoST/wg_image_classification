"""Softmax: gradient thủ công. CNN: autograd trong hàm riêng."""
import torch

BATCH_SIZE = 64
LEARNING_RATE = 0.1
NUM_EPOCHS = 10


def cross_entropy_with_gradient(logits, labels):
    """Loss ổn định số học và dL/dlogits = (P - Y) / B."""
    batch_size = logits.shape[0]
    indices = torch.arange(batch_size, device=logits.device)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -log_probabilities[indices, labels].mean()
    grad_logits = log_probabilities.exp()
    grad_logits[indices, labels] -= 1.0
    grad_logits /= batch_size
    return loss, grad_logits


def backpropagation(images, grad_logits):
    x = images.reshape(images.shape[0], -1)
    return x.T @ grad_logits, grad_logits.sum(dim=0)


# Giữ tên cũ để code đang import vẫn dùng được.
backprobagation = backpropagation


@torch.no_grad()
def sgd_step(model, grad_W, grad_b, learning_rate):
    model.W -= learning_rate * grad_W
    model.b -= learning_rate * grad_b


@torch.no_grad()
def train_one_epoch(model, train_loader, learning_rate, device="cpu"):
    # Không dùng backward(): toàn bộ gradient Softmax được tính thủ công.
    if hasattr(model, "train"):
        model.train()
    total_loss = total_correct = total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss, grad_logits = cross_entropy_with_gradient(logits, labels)
        grad_W, grad_b = backpropagation(images, grad_logits)
        sgd_step(model, grad_W, grad_b, learning_rate)
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += n
    return total_loss / total_samples, total_correct / total_samples


def train_cnn_one_epoch(model, train_loader, optimizer, device="cpu"):
    model.train()
    total_loss = total_correct = total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += n
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, data_loader, device="cpu", return_predictions=False):
    if hasattr(model, "eval"):
        model.eval()
    total_loss = total_correct = total_samples = 0
    all_true, all_pred, all_confidence = [], [], []
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (predictions == labels).sum().item()
        total_samples += n
        if return_predictions:
            all_true.append(labels.cpu())
            all_pred.append(predictions.cpu())
            all_confidence.append(torch.softmax(logits, dim=1).amax(dim=1).cpu())
    result = (total_loss / total_samples, total_correct / total_samples)
    if return_predictions:
        return (*result, torch.cat(all_true).numpy(), torch.cat(all_pred).numpy(),
                torch.cat(all_confidence).numpy())
    return result
