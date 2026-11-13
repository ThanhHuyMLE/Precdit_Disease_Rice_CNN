"""
Module tinh toan va truc quan hoa cac chi so danh gia mo hinh (Evaluation Metrics).
Cac ham bao gom:
- calculate_metrics(): Tinh Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
- compute_confusion_matrix(): Tinh ma tran nham lan
- plot_confusion_matrix(): Ve Heatmap ma tran nham lan
- get_classification_report(): Xuat bao cao phan loai chi tiet
- evaluate_dataloader(): Chay danh gia tren tap DataLoader
- plot_learning_curves(): Ve do thi loss va accuracy qua cac epoch
- print_metrics_summary(): In bang tong ket ra console

Luu y: Module nay chi cung cap cac ham tinh toan va truc quan hoa, khong chua du lieu gia lap.
Ket qua danh gia chi duoc sinh ra khi chay qua main.py.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch

# Cau hinh font chu tieng Viet cho do thi
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_CLASS_NAMES = [
    'Bạc lá lúa',
    'Đạo ôn lá',
    'Đốm nâu',
    'Lá khỏe mạnh'
]


def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Tinh toan day du cac chi so danh gia chinh:
    - Accuracy (Do chinh xac tong the)
    - Precision (Macro & Weighted)
    - Recall (Macro & Weighted)
    - F1-Score (Macro & Weighted)
    - Chi tiet tung lop (Per-class Precision, Recall, F1, Support)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = class_names or DEFAULT_CLASS_NAMES

    # 1. Cac chi so tong the
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 2. Chi so chi tiet cho tung lop
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_df = pd.DataFrame({
        'Tên lớp': [class_names[i] if i < len(class_names) else f"Lớp {i}" for i in range(len(prec_per_class))],
        'Precision': np.round(prec_per_class, 4),
        'Recall': np.round(rec_per_class, 4),
        'F1-Score': np.round(f1_per_class, 4),
        'Số lượng mẫu (Support)': [(y_true == i).sum() for i in range(len(prec_per_class))]
    })

    return {
        'accuracy': round(acc, 4),
        'precision_macro': round(prec_macro, 4),
        'precision_weighted': round(prec_weighted, 4),
        'recall_macro': round(rec_macro, 4),
        'recall_weighted': round(rec_weighted, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'per_class_table': per_class_df
    }


def compute_confusion_matrix(y_true, y_pred):
    """
    Tinh ma tran nham lan dang mang 2D (NumPy Array).
    """
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Ma trận nhầm lẫn (Confusion Matrix)', save_path=None):
    """
    Ve va truc quan hoa ma tran nham lan dang Heatmap bang Seaborn.
    Hien thi dong thoi ca so luong mau va ty le phan tram (%).
    """
    class_names = class_names or DEFAULT_CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j] * 100
            annot[i, j] = f"{c}\n({p:.1f}%)"

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        annot_kws={'size': 11}
    )

    plt.title(title, fontsize=13, weight='bold', pad=15)
    plt.xlabel('Nhãn dự đoán (Predicted)', fontsize=11, labelpad=10)
    plt.ylabel('Nhãn thực tế (Ground Truth)', fontsize=11, labelpad=10)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def get_classification_report(y_true, y_pred, class_names=None):
    """
    Tra ve bao cao phan loai dang chuoi (Classification Report).
    """
    class_names = class_names or DEFAULT_CLASS_NAMES
    return classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)


def evaluate_dataloader(model, data_loader, device='cpu', criterion=None):
    """
    Chay danh gia mo hinh tren toan bo DataLoader:
    Tra ve: loss_trung_binh, y_true (numpy array), y_pred (numpy array), y_probs (numpy array)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            total_samples += targets.size(0)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = (total_loss / total_samples) if total_samples > 0 and criterion else None
    return avg_loss, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def plot_learning_curves(history, save_path=None):
    """
    Ve do thi ham hoc tap (Learning Curves): Loss va Accuracy qua cac epoch.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Do thi Ham mat mat (Loss)
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=1.8, markersize=4)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(epochs, history['val_loss'], 'r--s', label='Validation Loss', linewidth=1.8, markersize=4)
    axes[0].set_title('Đồ thị hàm mất mát (Loss Curve)', fontsize=12, weight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Do thi Do chinh xac (Accuracy)
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy', linewidth=1.8, markersize=4)
    if 'val_acc' in history and len(history['val_acc']) > 0:
        axes[1].plot(epochs, [a * 100 for a in history['val_acc']], 'g--s', label='Validation Accuracy', linewidth=1.8, markersize=4)
    axes[1].set_title('Đồ thị độ chính xác (Accuracy Curve)', fontsize=12, weight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Độ chính xác (%)', fontsize=10)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def print_metrics_summary(metrics):
    """
    In bang tong ket cac chi so danh gia ra console bang tieng Viet chuan.
    """
    print("=" * 70)
    print("BẢNG TỔNG HỢP CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH (METRICS SUMMARY)")
    print("=" * 70)
    print(f"Độ chính xác tổng thể (Accuracy) : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (Macro Average)        : {metrics['precision_macro'] * 100:.2f}%")
    print(f"Precision (Weighted Average)     : {metrics['precision_weighted'] * 100:.2f}%")
    print(f"Recall (Macro Average)           : {metrics['recall_macro'] * 100:.2f}%")
    print(f"Recall (Weighted Average)        : {metrics['recall_weighted'] * 100:.2f}%")
    print(f"F1-Score (Macro Average)         : {metrics['f1_macro'] * 100:.2f}%")
    print(f"F1-Score (Weighted Average)      : {metrics['f1_weighted'] * 100:.2f}%")
    print("-" * 70)
    print("CHI TIẾT THEO TỪNG LỚP BỆNH:")
    print(metrics['per_class_table'].to_string(index=False))
    print("=" * 70)
