"""Vẽ kết quả đã lưu, không cần train lại.
python visualize.py --run results/softmax
python visualize.py --softmax results/softmax --cnn results/cnn
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Lưu PNG, không chặn chương trình bằng cửa sổ đồ thị.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = list(range(10))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def learning_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for color, (name, history) in zip(["tab:blue", "tab:orange"], runs):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            for split, style in [("train", "-"), ("valid", "--")]:
                values = np.asarray(history[f"{split}_{metric}"])
                if metric == "accuracy":
                    values = values * 100
                ax.plot(epochs, values, linestyle=style, color=color, label=f"{name} {split}")
            ax.set(xlabel="Epoch", ylabel="Loss" if metric == "loss" else "Accuracy (%)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    save_figure(fig, path)


def draw_matrix(ax, matrix, name, vmax):
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=vmax)
    for i in CLASSES:
        for j in CLASSES:
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if normalized[i, j] > 0.5 else "black")
    ax.set(xticks=CLASSES, yticks=CLASSES, xlabel="Predicted label", ylabel="True label", title=name)
    return im


def confusion_plot(items, path):
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 5), squeeze=False)
    for ax, (name, matrix) in zip(axes[0], items):
        im = draw_matrix(ax, matrix, name, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction within true class")
    save_figure(fig, path)


def image_grid(images, indices, title, label_function, path):
    # Chọn theo thứ tự test, không chọn thủ công những trường hợp đẹp nhất.
    chosen = np.asarray(indices)[:12]
    if not len(chosen):
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No matching samples", ha="center", va="center")
    else:
        rows = (len(chosen) + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(10, 2.8 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, chosen):
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(label_function(i), fontsize=9)
    fig.suptitle(title)
    save_figure(fig, path)


def load_run(directory):
    directory = Path(directory)
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    return read_json(directory / "config.json"), read_json(directory / "history.json"), predictions


def report_for(predictions):
    return classification_report(predictions["y_true"], predictions["y_pred"],
                                 labels=CLASSES, output_dict=True, zero_division=0)


def summary_row(config, report):
    return {"model": config["model"], "test_loss": config["test_loss"],
            "test_accuracy": config["test_accuracy"], "macro_f1": report["macro avg"]["f1-score"],
            "num_parameters": config["num_parameters"], "best_epoch": config["best_epoch"],
            "training_seconds": config["training_seconds"],
            "validation_seconds": config["validation_seconds"], "device": config["device"],
            "optimizer": config["optimizer"], "learning_rate": config["learning_rate"]}


def save_run_report(directory):
    directory = Path(directory)
    config, history, p = load_run(directory)
    name = config["model"].upper()
    report = report_for(p)
    (directory / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_rows = [{"class": key, **report[key]} for key in map(str, CLASSES)]
    write_csv(directory / "per_class_metrics.csv", ["class", "precision", "recall", "f1-score", "support"], report_rows)
    row = summary_row(config, report)
    write_csv(directory / "summary.csv", list(row), [row])
    write_csv(directory / "history.csv", ["epoch", *history],
              ({"epoch": i + 1, **{key: values[i] for key, values in history.items()}}
               for i in range(len(history["train_loss"]))))
    write_csv(directory / "predictions.csv", ["sample_id", "true_label", "predicted_label", "confidence"],
              ({"sample_id": int(i), "true_label": int(t), "predicted_label": int(y), "confidence": float(c)}
               for i, t, y, c in zip(p["sample_id"], p["y_true"], p["y_pred"], p["confidence"])))
    matrix = confusion_matrix(p["y_true"], p["y_pred"], labels=CLASSES)
    np.savetxt(directory / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    learning_curves([(name, history)], directory / "learning_curves.png")
    confusion_plot([(name, matrix)], directory / "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    for offset, metric in zip([-0.25, 0, 0.25], ["precision", "recall", "f1-score"]):
        ax.bar(x + offset, [report[str(i)][metric] for i in CLASSES], width=0.25, label=metric)
    ax.set(xticks=x, xlabel="Digit", ylabel="Score", ylim=(0, 1.05), title=name)
    ax.legend()
    save_figure(fig, directory / "per_class_metrics.png")
    for correct, filename in [(True, "correct_examples.png"), (False, "wrong_examples.png")]:
        indices = np.flatnonzero((p["y_true"] == p["y_pred"]) == correct)
        image_grid(p["images"], indices, f"{name}: {'correct' if correct else 'wrong'} predictions",
                   lambda i: f"ID {p['sample_id'][i]} | True {p['y_true'][i]}, pred {p['y_pred'][i]}\n"
                             f"Softmax score: {p['confidence'][i]:.1%}", directory / filename)


def compare_runs(softmax_dir, cnn_dir, output):
    sc, sh, sp = load_run(softmax_dir)
    cc, ch, cp = load_run(cnn_dir)
    if sc["model"] != "softmax" or cc["model"] != "cnn":
        raise ValueError("Chọn đúng thư mục Softmax và CNN.")
    for key in ["dataset", "split_hash", "test_hash", "preprocessing"]:
        if sc[key] != cc[key]:
            raise ValueError(f"Không thể so sánh: hai lần chạy khác {key}.")
    if not np.array_equal(sp["sample_id"], cp["sample_id"]) or not np.array_equal(sp["y_true"], cp["y_true"]):
        raise ValueError("Dự đoán không tương ứng cùng thứ tự ảnh test.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    learning_curves([("Softmax", sh), ("CNN", ch)], output / "learning_curves.png")
    confusion_plot([(name, confusion_matrix(p["y_true"], p["y_pred"], labels=CLASSES))
                    for name, p in [("Softmax", sp), ("CNN", cp)]], output / "confusion_matrices.png")
    sr, cr = report_for(sp), report_for(cp)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    ax.bar(x - 0.2, [sr[str(i)]["f1-score"] for i in CLASSES], width=0.4, label="Softmax")
    ax.bar(x + 0.2, [cr[str(i)]["f1-score"] for i in CLASSES], width=0.4, label="CNN")
    ax.set(xticks=x, xlabel="Digit", ylabel="F1-score", ylim=(0, 1.05))
    ax.legend()
    save_figure(fig, output / "f1_comparison.png")
    rows = [summary_row(sc, sr), summary_row(cc, cr)]
    write_csv(output / "summary.csv", list(rows[0]), rows)
    fig, ax = plt.subplots(figsize=(11, 2.7))
    ax.axis("off")
    table = ax.table(cellText=[
        [r["model"], f"{r['test_accuracy']:.2%}", f"{r['macro_f1']:.4f}",
         f"{r['num_parameters']:,}", str(r["best_epoch"]), f"{r['training_seconds']:.1f}"] for r in rows],
        colLabels=["Model", "Test accuracy", "Macro F1", "Parameters", "Best epoch", "Train seconds"],
        cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title("Measured results (see config.json for training settings)")
    save_figure(fig, output / "summary.png")
    s_ok, c_ok = sp["y_pred"] == sp["y_true"], cp["y_pred"] == cp["y_true"]
    groups = [(~s_ok & c_ok, "softmax_wrong_cnn_correct"),
              (s_ok & ~c_ok, "softmax_correct_cnn_wrong"), (~s_ok & ~c_ok, "both_wrong")]
    for mask, name in groups:
        image_grid(sp["images"], np.flatnonzero(mask), f"{name} (total: {mask.sum()})",
                   lambda i: f"ID {sp['sample_id'][i]} | True: {sp['y_true'][i]}\n"
                             f"Softmax: {sp['y_pred'][i]} | CNN: {cp['y_pred'][i]}", output / f"{name}.png")
    notes = ["Accuracy in CSV uses [0,1]. Train seconds excludes validation, test, plotting, and checkpoint writes.",
             "Different optimizers/lr are recorded: results compare these configurations, not architecture alone.",
             "Training metrics are measured during updates (CNN dropout enabled); validation uses evaluation mode."]
    for key in ["device", "epochs", "batch_size"]:
        if sc[key] != cc[key]:
            notes.append(f"NOTE: {key} differs: Softmax={sc[key]}, CNN={cc[key]}.")
    (output / "notes.txt").write_text("\n".join(notes), encoding="utf-8")
    print(f"Đã lưu so sánh: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Vẽ lại một model từ số liệu đã lưu")
    parser.add_argument("--softmax")
    parser.add_argument("--cnn")
    parser.add_argument("--output", default="results/comparison")
    args = parser.parse_args()
    if args.run and not (args.softmax or args.cnn):
        save_run_report(Path(args.run))
    elif args.softmax and args.cnn and not args.run:
        compare_runs(args.softmax, args.cnn, args.output)
    else:
        parser.error("Dùng --run THU_MUC hoặc --softmax THU_MUC --cnn THU_MUC")
