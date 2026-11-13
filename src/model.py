"""
Module dinh nghia kien truc mang CNN va lop quan ly mo hinh RiceDiseaseClassifier.
Form chuan: __init__, build(), fit(), predict().
Luu y: Module nay chi khai bao kien truc va phuong thuc, khong khoi chay danh gia.
Tat ca sieu tham so (epochs, learning_rate, batch_size...) se duoc truyen tu main.py.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import bo bien doi mac dinh
try:
    from src.reprocess import RiceLeafTransform
except ImportError:
    from reprocess import RiceLeafTransform


# =========================================================================
# 1. KIEN TRUC MANG CNN (PyTorch Module)
# =========================================================================
class RiceLeafCNN(nn.Module):
    """
    Kien truc mang CNN gom 4 khoi Convolutional tich hop BatchNorm va MaxPool
    chuyen dung trich xuat dac trung ton thuong tren phien la lua.
    """
    def __init__(self, num_classes=4):
        super(RiceLeafCNN, self).__init__()

        # Khoi trich xuat dac trung (Feature Extractor)
        self.features = nn.Sequential(
            # Block 1: (3, 224, 224) -> (32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (32, 112, 112) -> (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (64, 56, 56) -> (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: (128, 28, 28) -> (256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Khoi phan loai (Classifier)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


# =========================================================================
# 2. CLASS QUAN LY MO HINH (Wrapper theo form: __init__, build, fit, predict)
# =========================================================================
class RiceDiseaseClassifier:
    """
    Class quan ly mo hinh CNN.
    Cac sieu tham so duoc cau hinh dong va truyen tu main.py.
    """
    def __init__(self, num_classes=4, device=None, class_names=None):
        """
        Khoi tao cau truc co ban cua mo hinh.
        """
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names or {
            0: 'Bạc lá lúa',
            1: 'Đạo ôn lá',
            2: 'Đốm nâu',
            3: 'Lá khỏe mạnh'
        }

        # Cac thanh phan se duoc khoi tao qua ham build()
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.transform_fn = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def build(self, learning_rate=1e-3, weight_decay=1e-4, scheduler_patience=3):
        """
        Xay dung mang CNN, gan ham mat mat va bo toi uu hoa dua tren
        cac sieu tham so duoc truyen vao tu main.py.
        """
        self.model = RiceLeafCNN(num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=scheduler_patience
        )
        self.transform_fn = RiceLeafTransform(is_train=False, image_size=(224, 224))
        return self

    def fit(self, train_loader, val_loader=None, epochs=20, save_path=None, verbose=True, logger=None):
        """
        Thuc hien huan luyen mo hinh qua so luong epoch duoc truyen tu main.py.
        Ho tro ghi nhat ky qua doi tuong logger (neu co).
        """
        if self.model is None:
            raise RuntimeError("Mo hinh chua duoc build. Vui long goi ham build() truoc khi fit()!")

        log_fn = logger.info if (logger and hasattr(logger, 'info')) else print
        best_val_loss = float('inf')
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            # 1. Huan luyen (Training Phase)
            self.model.train()
            total_train_loss = 0.0
            train_correct = 0
            train_samples = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_samples += labels.size(0)

            epoch_train_loss = total_train_loss / train_samples
            epoch_train_acc = train_correct / train_samples

            # 2. Tham dinh (Validation Phase)
            epoch_val_loss, epoch_val_acc = 0.0, 0.0
            if val_loader:
                self.model.eval()
                total_val_loss = 0.0
                val_correct = 0
                val_samples = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)

                        total_val_loss += loss.item() * images.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += (preds == labels).sum().item()
                        val_samples += labels.size(0)

                epoch_val_loss = total_val_loss / val_samples
                epoch_val_acc = val_correct / val_samples
                self.scheduler.step(epoch_val_loss)

                # Luu checkpoint neu dat loss thap nhat
                if epoch_val_loss < best_val_loss and save_path:
                    best_val_loss = epoch_val_loss
                    torch.save(self.model.state_dict(), save_path)
                    if verbose:
                        log_fn(f"  --> [Checkpoint] Da luu model tot nhat tai Epoch {epoch:02d} (Val Loss: {best_val_loss:.4f})")

            # Luu lich su
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            if val_loader:
                self.history['val_loss'].append(epoch_val_loss)
                self.history['val_acc'].append(epoch_val_acc)

            if verbose:
                val_info = f" | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%" if val_loader else ""
                log_fn(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}%{val_info}")

        return self.history

    def predict(self, image_input):
        """
        Nhan vao anh (duong dan, PIL.Image hoac Tensor) va tra ve ket qua suy luan.
        """
        if self.model is None:
            raise RuntimeError("Mo hinh chua duoc build hoac load trong so!")

        self.model.eval()

        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                tensor = self.transform_fn(img)
            tensor = tensor.unsqueeze(0)
        elif isinstance(image_input, Image.Image):
            tensor = self.transform_fn(image_input).unsqueeze(0)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input.unsqueeze(0) if image_input.dim() == 3 else image_input
        else:
            raise TypeError("image_input phai la duong dan tep, PIL.Image hoac torch.Tensor!")

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            pred_class_idx = pred_idx.item()
            confidence = round(conf.item() * 100, 2)
            all_probs = {self.class_names[i]: round(probs[0, i].item() * 100, 2) for i in range(self.num_classes)}

        return {
            'class_idx': pred_class_idx,
            'class_name': self.class_names.get(pred_class_idx, 'Khong xac dinh'),
            'confidence': confidence,
            'all_probabilities': all_probs
        }

    def save(self, filepath):
        """Luu trong so mo hinh."""
        if self.model is not None:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """Tai trong so mo hinh tu file checkpoint."""
        if self.model is None:
            self.build()
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
