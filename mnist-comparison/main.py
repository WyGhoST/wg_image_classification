"""Chạy: python main.py --model softmax  hoặc  python main.py --model cnn."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from softmax_classifier import SoftmaxClassifier
from train import (BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, evaluate,
                   train_one_epoch, train_cnn_one_epoch)
from visualize import save_run_report


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST: train, valid, test và biểu đồ")
    parser.add_argument("--model", choices=["softmax", "cnn"], default="softmax")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    # CPU mặc định để dễ chạy và đo trên cùng thiết bị.
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.lr is not None and args.lr <= 0):
        parser.error("epochs, batch-size và lr phải lớn hơn 0")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng; hãy dùng --device cpu")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    output = Path(args.output_dir) / args.model
    output.mkdir(parents=True, exist_ok=True)

    # Chỉ scale pixel về [0, 1] bằng ToTensor, giống code gốc.
    full_train = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    train_set, valid_set = random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(args.seed)
    )
    # Lưu chính xác chỉ số để tái tạo/kiểm tra phép chia giữa hai model.
    train_indices = np.asarray(train_set.indices, dtype=np.int64)
    valid_indices = np.asarray(valid_set.indices, dtype=np.int64)
    np.savez_compressed(output / "split_indices.npz", train=train_indices, valid=valid_indices)
    split_hash = hashlib.sha256(train_indices.tobytes() + valid_indices.tobytes()).hexdigest()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    learning_rate = args.lr if args.lr is not None else (LEARNING_RATE if args.model == "softmax" else 0.001)

    if args.model == "softmax":
        model = SoftmaxClassifier(input_size=784, num_classes=10)
        # Hỗ trợ cả class thường và nn.Module có W, b như code gốc.
        if isinstance(model, torch.nn.Module):
            model.to(device)
        else:
            model.W, model.b = model.W.to(device), model.b.to(device)
        if tuple(model.W.shape) != (784, 10) or tuple(model.b.shape) != (10,):
            raise ValueError("Softmax cần W.shape=(784,10), b.shape=(10,) và trả logits")
        num_parameters = model.W.numel() + model.b.numel()
        optimizer = None
    else:
        from cnn_model import CNNClassifier
        model = CNNClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        num_parameters = sum(p.numel() for p in model.parameters())

    history = {key: [] for key in ("train_loss", "valid_loss", "train_accuracy", "valid_accuracy")}
    best_valid_loss, best_epoch = float("inf"), 0
    training_seconds = validation_seconds = 0.0
    checkpoint_path = output / "best_model.pth"

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for epoch in range(1, args.epochs + 1):
        synchronize()
        start = time.perf_counter()
        if args.model == "softmax":
            train_loss, train_accuracy = train_one_epoch(model, train_loader, learning_rate, device)
        else:
            train_loss, train_accuracy = train_cnn_one_epoch(model, train_loader, optimizer, device)
        synchronize()
        training_seconds += time.perf_counter() - start
        start = time.perf_counter()
        valid_loss, valid_accuracy = evaluate(model, valid_loader, device)
        synchronize()
        validation_seconds += time.perf_counter() - start
        for key, value in zip(history, (train_loss, valid_loss, train_accuracy, valid_accuracy)):
            history[key].append(value)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if valid_loss < best_valid_loss:
            best_valid_loss, best_epoch = valid_loss, epoch
            state = ({"W": model.W.detach().cpu().clone(), "b": model.b.detach().cpu().clone()}
                     if args.model == "softmax" else
                     {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            torch.save({"model": args.model, "epoch": epoch, "valid_loss": valid_loss,
                        "state": state, "seed": args.seed, "split_hash": split_hash,
                        "preprocessing": "ToTensor:[0,1]"}, checkpoint_path)
        print(f"Epoch {epoch:02d}/{args.epochs} | Train loss {train_loss:.4f}, acc {train_accuracy:.2%}"
              f" | Valid loss {valid_loss:.4f}, acc {valid_accuracy:.2%}")

    # Chỉ sử dụng test sau khi chọn checkpoint theo VALIDATION LOSS.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if args.model == "softmax":
        with torch.no_grad():
            model.W.copy_(checkpoint["state"]["W"])
            model.b.copy_(checkpoint["state"]["b"])
    else:
        model.load_state_dict(checkpoint["state"])
    test_set = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loss, test_accuracy, y_true, y_pred, confidence = evaluate(
        model, test_loader, device, return_predictions=True
    )
    # raw uint8 phục vụ hiển thị ảnh; không lưu ảnh đã normalize.
    images = test_set.data.numpy()
    sample_ids = np.arange(len(test_set))
    np.savez_compressed(output / "predictions.npz", sample_id=sample_ids, y_true=y_true,
                        y_pred=y_pred, confidence=confidence, images=images)
    config = {
        "model": args.model, "dataset": "MNIST", "train_size": len(train_set),
        "valid_size": len(valid_set), "test_size": len(test_set), "seed": args.seed,
        "split_hash": split_hash,
        "test_hash": hashlib.sha256(images.tobytes() + y_true.tobytes()).hexdigest(),
        "preprocessing": "ToTensor:[0,1]", "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": learning_rate,
        "optimizer": "manual SGD" if args.model == "softmax" else "Adam",
        "device": str(device), "torch_version": str(torch.__version__),
        "best_epoch": best_epoch, "best_valid_loss": best_valid_loss,
        "num_parameters": num_parameters, "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "test_loss": test_loss, "test_accuracy": test_accuracy,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_run_report(output)
    print(f"Best epoch: {best_epoch} | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2%}")
    print(f"Đã lưu model, số liệu và biểu đồ tại: {output.resolve()}")


if __name__ == "__main__":
    main()
