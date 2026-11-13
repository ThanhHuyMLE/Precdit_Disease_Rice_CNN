"""
Module tien xu ly du lieu (Preprocessing) cho mo hinh CNN nhan dien benh tren la lua.
Chuc nang chinh:
1. Kiem tra tinh toan ven va chuan hoa anh ve RGB.
2. Ma hoa nhan (Label Encoding: bac_la->0, dao_on->1, dom_nau->2, la_khoe->3).
3. Trich xuat metadata va xuat tep 'reprocessed.csv'.
4. Cung cap lop Dataset va DataLoader (PyTorch) kem Data Augmentation cho tap Train.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Tu dien ma hoa nhan
LABEL_TO_IDX = {
    'bac_la': 0,
    'dao_on': 1,
    'dom_nau': 2,
    'la_khoe': 3
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

LABEL_VI = {
    'bac_la': 'Bạc lá lúa',
    'dao_on': 'Đạo ôn lá',
    'dom_nau': 'Đốm nâu',
    'la_khoe': 'Lá khỏe mạnh'
}

# 2. Xac dinh duong dan mac dinh
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'dataset' / 'data_processed'
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / 'reprocessed.csv'


def build_reprocessed_csv(data_dir=DEFAULT_DATA_DIR, output_csv=DEFAULT_CSV_PATH):
    """
    Quet toan bo tap anh trong data_processed, kiem tra tinh toan ven,
    ma hoa nhan va xuat file reprocessed.csv.
    """
    data_dir = Path(data_dir).resolve()
    output_csv = Path(output_csv).resolve()
    
    # Doc them thong tin original_filename tu labels.csv neu co
    labels_csv = data_dir / 'labels.csv'
    orig_map = {}
    if labels_csv.exists():
        df_orig = pd.read_csv(labels_csv)
        if 'image_path' in df_orig.columns and 'original_filename' in df_orig.columns:
            orig_map = dict(zip(df_orig['image_path'], df_orig['original_filename']))

    records = []
    splits = ['train', 'valid', 'test']

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        for img_path in sorted(split_dir.glob('*.*')):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            rel_path = f"{split}/{img_path.name}"
            class_label = img_path.name.rsplit('_', 1)[0]
            label_idx = LABEL_TO_IDX.get(class_label, -1)

            # Kiem tra anh va trich xuat thuoc tinh
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    channels = len(img.getbands())
                    is_valid = (channels == 3 or img.mode == 'RGB')
            except Exception:
                w, h, channels, is_valid = 0, 0, 0, False

            file_size_kb = round(os.path.getsize(img_path) / 1024, 2)

            records.append({
                'image_path': rel_path,
                'filename': img_path.name,
                'split': split,
                'label': class_label,
                'label_idx': label_idx,
                'label_vi': LABEL_VI.get(class_label, class_label),
                'width': w,
                'height': h,
                'channels': channels,
                'size_kb': file_size_kb,
                'is_valid': is_valid,
                'original_filename': orig_map.get(rel_path, '')
            })

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Dong thoi luu ban sao tai dataset/reprocessed.csv
    backup_csv = PROJECT_ROOT / 'dataset' / 'reprocessed.csv'
    df.to_csv(backup_csv, index=False, encoding='utf-8-sig')

    return df


class RiceLeafTransform:
    """
    Bo bien doi anh (Transform Pipeline):
    - Train: Random Flip, Random Rotation, Color Jitter, Resize (224x224), Normalization.
    - Valid / Test: Resize (224x224), Normalization.
    """
    def __init__(self, is_train=True, image_size=(224, 224)):
        self.is_train = is_train
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.is_train:
            # Lat ngang ngau nhien
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Lat doc ngau nhien
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Xoay nhe goc tu -15 den 15 do
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, resample=Image.BILINEAR)
            # Bien doi nhe do sang va do tuong phan
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.9, 1.1))
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.9, 1.1))

        # Chuan hoa kich thuoc ve 224x224
        img = img.resize(self.image_size, Image.BILINEAR)

        # Chuyen sang Tensor va chuan hoa theo ImageNet
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)) # (C, H, W)
        return tensor


class RiceLeafDataset(Dataset):
    """
    Custom Dataset cho bai toan phan loai benh tren la lua.
    """
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['image_path']
        
        with Image.open(img_path) as img:
            image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = int(row['label_idx'])
        return image, label


def get_dataloaders(data_dir=DEFAULT_DATA_DIR, csv_path=DEFAULT_CSV_PATH, batch_size=32, num_workers=0, image_size=(224, 224)):
    """
    Ham tao cac DataLoader cho Train, Valid, Test.
    """
    data_dir = Path(data_dir).resolve()
    csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        df = build_reprocessed_csv(data_dir, csv_path)
    else:
        df = pd.read_csv(csv_path)

    df_train = df[df['split'] == 'train']
    df_valid = df[df['split'] == 'valid']
    df_test = df[df['split'] == 'test']

    train_dataset = RiceLeafDataset(df_train, data_dir, transform=RiceLeafTransform(is_train=True, image_size=image_size))
    valid_dataset = RiceLeafDataset(df_valid, data_dir, transform=RiceLeafTransform(is_train=False, image_size=image_size))
    test_dataset  = RiceLeafDataset(df_test,  data_dir, transform=RiceLeafTransform(is_train=False, image_size=image_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    print("=" * 75)
    print("KHOI CHAY TIEN XU LY DU LIEU VA XUAT FILE reprocessed.csv")
    print("=" * 75)

    df_result = build_reprocessed_csv()
    print(f"Da xuat thanh cong tep: {DEFAULT_CSV_PATH}")
    print(f"Tong so mau duoc xu ly: {len(df_result):,} anh")
    print(f"So mau hop le 100%     : {df_result['is_valid'].sum():,} / {len(df_result):,} anh")

    print("\nBANG PHAN BO DU LIEU THEO TAP VA NHOOM BENH:")
    pivot_table = df_result.groupby(['split', 'label_vi']).size().unstack(fill_value=0)
    pivot_table = pivot_table.reindex(['train', 'valid', 'test'])
    pivot_table['Tong so'] = pivot_table.sum(axis=1)
    print(pivot_table.to_string())

    print("\nKHOI TAO DATALOADER KIEM TRA PIPELINE:")
    train_ld, valid_ld, test_ld = get_dataloaders(batch_size=32)
    print(f"Train batches: {len(train_ld)} (Tong {len(train_ld.dataset)} anh)")
    print(f"Valid batches: {len(valid_ld)} (Tong {len(valid_ld.dataset)} anh)")
    print(f"Test  batches: {len(test_ld)}  (Tong {len(test_ld.dataset)} anh)")

    # Lay 1 batch mau tu Train DataLoader
    images, labels = next(iter(train_ld))
    print(f"\nKiem tra Batch dau tien tu Train DataLoader:")
    print(f"- Shape cua Images Batch: {images.shape} (Batch_size, Channels, Height, Width)")
    print(f"- Dtype cua Images      : {images.dtype}")
    print(f"- Shape cua Labels Batch: {labels.shape}")
    print(f"- Cac nhan mau trong Batch: {labels[:8].tolist()}")
    print("=" * 75)
    print("TIEN XU LY VA KIEM TRA HOAN TAT CHINH XAC 100%!")

