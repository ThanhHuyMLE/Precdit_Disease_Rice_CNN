"""
File thực thi trung tâm (main.py):
Quản lý toàn diện quy trình huấn luyện và đánh giá mô hình CNN nhận diện bệnh trên lá lúa.
Tích hợp hệ thống ghi nhật ký (Logging) xuyên suốt 4 giai đoạn chuẩn:
- Phase 1: LOAD_DATA  - Nạp và tiền xử lý dữ liệu (Train / Valid / Test)
- Phase 2: TRAINING   - Huấn luyện mô hình CNN trên tập Train và theo dõi qua các Epoch
- Phase 3: EVALUATION - Đánh giá hiệu năng chi tiết trên tập Validation (Val Set)
- Phase 4: PREDICT    - Thực hiện dự đoán trên tập Test và xuất toàn bộ chỉ số đánh giá thực tế
"""

import os
import sys
import time
import logging
from pathlib import Path
import torch

# Đảm bảo đường dẫn gốc được nhận diện trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reprocess import get_dataloaders
from src.model import RiceDiseaseClassifier
from src.metrics import (
    calculate_metrics,
    evaluate_dataloader,
    plot_confusion_matrix,
    plot_learning_curves,
    print_metrics_summary
)


# =========================================================================
# 1. CẤU HÌNH SIÊU THAM SỐ VÀ ĐƯỜNG DẪN (CONFIGURATIONS)
# =========================================================================
IMAGE_SIZE = (224, 224)                     # Kích thước ảnh chuẩn hóa (Chiều cao x Chiều rộng)
BATCH_SIZE = 32                            # Kích thước batch nạp dữ liệu
NUM_WORKERS = 0                            # Số luồng nạp dữ liệu song song (0 an toàn trên Windows)
NUM_CLASSES = 4                            # 4 lớp bệnh: Bạc lá, Đạo ôn, Đốm nâu, Lá khỏe

EPOCHS = 25                                # Số lượng Epoch huấn luyện lặp qua toàn bộ tập ảnh Train
LEARNING_RATE = 1e-3                       # Tốc độ học khởi tạo của bộ tối ưu Adam
WEIGHT_DECAY = 1e-4                        # Hệ số phạt L2 Regularization chống Overfitting
SCHEDULER_PATIENCE = 3                     # Số Epoch kiên nhẫn trước khi giảm Learning Rate

# Thư mục lưu trữ kết quả và tệp Log
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoint'
LOG_FILE_PATH = CHECKPOINT_DIR / 'training.log'
MODEL_SAVE_PATH = CHECKPOINT_DIR / 'best_rice_cnn.pth'
CURVES_SAVE_PATH = CHECKPOINT_DIR / 'learning_curves.png'

# Tệp kết quả cho Phase 3 (Validation Evaluation)
VAL_CONFUSION_MATRIX_PATH = CHECKPOINT_DIR / 'val_confusion_matrix.png'
VAL_REPORT_SAVE_PATH = CHECKPOINT_DIR / 'val_evaluation_report.txt'

# Tệp kết quả cho Phase 4 (Test Predict & Evaluation)
TEST_CONFUSION_MATRIX_PATH = CHECKPOINT_DIR / 'test_confusion_matrix.png'
TEST_REPORT_SAVE_PATH = CHECKPOINT_DIR / 'test_evaluation_report.txt'


# =========================================================================
# 2. HÀM THIẾT LẬP NHẬT KÝ HỆ THỐNG (LOGGER SETUP)
# =========================================================================
def setup_logger(log_file=LOG_FILE_PATH):
    """
    Khởi tạo hệ thống logging ghi đồng thời ra màn hình Console và tệp tin trên đĩa.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("RiceLeafCNN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Ghi ra Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Ghi ra tệp tin log trên đĩa
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_phase_header(logger, phase_num, phase_name, description=""):
    """Ghi tiêu đề phân đoạn cho từng giai đoạn."""
    logger.info("=" * 85)
    logger.info(f"PHASE {phase_num}: {phase_name.upper()} - {description}")
    logger.info("=" * 85)


# =========================================================================
# 3. QUY TRÌNH THỰC THI CHÍNH (MAIN FUNCTION)
# =========================================================================
def main():
    logger = setup_logger(LOG_FILE_PATH)
    start_total_time = time.time()

    logger.info("DỰ ÁN PHÂN LOẠI BỆNH TRÊN LÁ LÚA BẰNG MÔ HÌNH CNN (RICE LEAF DISEASE CLASSIFICATION)")
    logger.info(f"Tệp nhật ký đầy đủ được ghi tại: {LOG_FILE_PATH}")

    # Kiểm tra thiết bị phần cứng
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Thiết bị phần cứng phát hiện: {device.upper()}")
    if device == 'cuda':
        logger.info(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Dung lượng VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        logger.info("Chế độ chạy: CPU (Khuyến nghị dùng GPU CUDA để tăng tốc độ huấn luyện).")

    # =====================================================================
    # PHASE 1: LOAD_DATA
    # =====================================================================
    log_phase_header(logger, 1, "LOAD_DATA", "Nạp và tiền xử lý tập dữ liệu (Train / Valid / Test)")
    logger.info(f"Cấu hình dữ liệu:")
    logger.info(f"  + Kích thước ảnh chuẩn hóa (Image Size): {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3")
    logger.info(f"  + Kích thước Batch (Batch Size)     : {BATCH_SIZE}")
    logger.info(f"  + Số lượng lớp bệnh (Classes)       : {NUM_CLASSES}")

    phase1_start = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )

    logger.info(f"-> Tập Huấn luyện (Train): {len(train_loader.dataset):,} ảnh ({len(train_loader)} batches)")
    logger.info(f"-> Tập Thẩm định  (Valid): {len(val_loader.dataset):,} ảnh ({len(val_loader)} batches)")
    logger.info(f"-> Tập Kiểm thử   (Test) : {len(test_loader.dataset):,} ảnh ({len(test_loader)} batches)")
    logger.info(f"-> Tổng cộng             : {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):,} ảnh")
    logger.info(f"Thời gian hoàn tất Phase 1: {time.time() - phase1_start:.2f} giây")

    # =====================================================================
    # PHASE 2: TRAINING
    # =====================================================================
    log_phase_header(logger, 2, "TRAINING", "Khởi tạo kiến trúc và huấn luyện mô hình CNN")
    logger.info("Cấu hình huấn luyện:")
    logger.info(f"  + Số lượng Epochs       : {EPOCHS}")
    logger.info(f"  + Tốc độ học khởi tạo   : {LEARNING_RATE}")
    logger.info(f"  + Hệ số phạt L2 Regular : {WEIGHT_DECAY}")
    logger.info(f"  + Thuật toán tối ưu     : Adam")
    logger.info(f"  + Hàm mất mát           : CrossEntropyLoss")
    logger.info(f"  + Nơi lưu Checkpoint    : {MODEL_SAVE_PATH}")

    classifier = RiceDiseaseClassifier(num_classes=NUM_CLASSES, device=device)
    classifier.build(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        scheduler_patience=SCHEDULER_PATIENCE
    )
    total_params = sum(p.numel() for p in classifier.model.parameters())
    logger.info(f"Tổng số tham số của mô hình RiceLeafCNN: {total_params:,}")

    logger.info("Bắt đầu quá trình huấn luyện qua từng Epoch:")
    phase2_start = time.time()
    history = classifier.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path=str(MODEL_SAVE_PATH),
        verbose=True,
        logger=logger
    )
    phase2_duration = time.time() - phase2_start
    logger.info(f"Hoàn thành {EPOCHS} Epochs trong: {phase2_duration/60:.2f} phút ({phase2_duration:.2f} giây)")
    logger.info(f"Trọng số mô hình có Validation Loss tốt nhất đã lưu tại: {MODEL_SAVE_PATH}")

    # Xuất đồ thị Learning Curves
    logger.info("Đang vẽ và lưu biểu đồ đường cong học tập (Learning Curves)...")
    plot_learning_curves(history, save_path=str(CURVES_SAVE_PATH))
    logger.info(f"Biểu đồ đường cong học tập đã lưu tại: {CURVES_SAVE_PATH}")

    # =====================================================================
    # PHASE 3: EVALUATION (ĐÁNH GIÁ TRÊN TẬP VALIDATION)
    # =====================================================================
    log_phase_header(logger, 3, "EVALUATION", f"Đánh giá mô hình tốt nhất trên tập Validation ({len(val_loader.dataset)} ảnh)")
    logger.info(f"Tải lại trọng số tốt nhất từ Checkpoint: {MODEL_SAVE_PATH}")
    classifier.load(str(MODEL_SAVE_PATH))

    phase3_start = time.time()
    val_loss, y_true_val, y_pred_val, y_probs_val = evaluate_dataloader(
        model=classifier.model,
        data_loader=val_loader,
        device=classifier.device,
        criterion=classifier.criterion
    )

    class_names_list = list(classifier.class_names.values())
    val_metrics = calculate_metrics(y_true_val, y_pred_val, class_names=class_names_list)

    logger.info("-" * 85)
    logger.info("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION (VAL SET):")
    logger.info(f"  + Val Loss (Hàm mất mát)       : {val_loss:.4f}")
    logger.info(f"  + Độ chính xác (Accuracy)      : {val_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  + Precision (Macro Average)    : {val_metrics['precision_macro'] * 100:.2f}%")
    logger.info(f"  + Precision (Weighted Average) : {val_metrics['precision_weighted'] * 100:.2f}%")
    logger.info(f"  + Recall (Macro Average)       : {val_metrics['recall_macro'] * 100:.2f}%")
    logger.info(f"  + Recall (Weighted Average)    : {val_metrics['recall_weighted'] * 100:.2f}%")
    logger.info(f"  + F1-Score (Macro Average)     : {val_metrics['f1_macro'] * 100:.2f}%")
    logger.info(f"  + F1-Score (Weighted Average)  : {val_metrics['f1_weighted'] * 100:.2f}%")
    logger.info("-" * 85)
    logger.info("CHI TIẾT THEO TỪNG LỚP BỆNH (VALIDATION):\n" + val_metrics['per_class_table'].to_string(index=False))

    # Vẽ và lưu Confusion Matrix tập Validation
    logger.info("Đang xuất biểu đồ Ma trận nhầm lẫn cho tập Validation...")
    plot_confusion_matrix(
        y_true_val,
        y_pred_val,
        class_names=class_names_list,
        title='Ma trận nhầm lẫn trên tập Validation (Rice Leaf Disease)',
        save_path=str(VAL_CONFUSION_MATRIX_PATH)
    )
    logger.info(f"Ma trận nhầm lẫn tập Validation đã lưu tại: {VAL_CONFUSION_MATRIX_PATH}")

    # Ghi báo cáo Validation ra file text
    with open(VAL_REPORT_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION (PHASE 3: EVALUATION)\n")
        f.write("=" * 85 + "\n")
        f.write(f"Validation Loss              : {val_loss:.4f}\n")
        f.write(f"Độ chính xác (Accuracy)      : {val_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision (Macro Average)    : {val_metrics['precision_macro'] * 100:.2f}%\n")
        f.write(f"Precision (Weighted Average) : {val_metrics['precision_weighted'] * 100:.2f}%\n")
        f.write(f"Recall (Macro Average)       : {val_metrics['recall_macro'] * 100:.2f}%\n")
        f.write(f"Recall (Weighted Average)    : {val_metrics['recall_weighted'] * 100:.2f}%\n")
        f.write(f"F1-Score (Macro Average)     : {val_metrics['f1_macro'] * 100:.2f}%\n")
        f.write(f"F1-Score (Weighted Average)  : {val_metrics['f1_weighted'] * 100:.2f}%\n")
        f.write("-" * 85 + "\n")
        f.write("CHI TIẾT THEO TỪNG LỚP:\n")
        f.write(val_metrics['per_class_table'].to_string(index=False) + "\n")
        f.write("=" * 85 + "\n")
    logger.info(f"Tệp báo cáo Validation đã lưu tại: {VAL_REPORT_SAVE_PATH}")
    logger.info(f"Thời gian hoàn tất Phase 3: {time.time() - phase3_start:.2f} giây")

    # =====================================================================
    # PHASE 4: PREDICT (DỰ ĐOÁN VÀ ĐÁNH GIÁ TOÀN DIỆN TRÊN TẬP TEST)
    # =====================================================================
    log_phase_header(logger, 4, "PREDICT", f"Dự đoán và xuất đánh giá toàn diện trên tập Test ({len(test_loader.dataset)} ảnh)")
    phase4_start = time.time()

    # 1. Thực hiện dự đoán trên toàn bộ tập Test và đánh giá hiệu năng
    logger.info(f"1. Tiến hành dự đoán và chấm điểm trên toàn bộ {len(test_loader.dataset)} ảnh tập Test...")
    test_loss, y_true_test, y_pred_test, y_probs_test = evaluate_dataloader(
        model=classifier.model,
        data_loader=test_loader,
        device=classifier.device,
        criterion=classifier.criterion
    )

    test_metrics = calculate_metrics(y_true_test, y_pred_test, class_names=class_names_list)

    logger.info("-" * 85)
    logger.info("KẾT QUẢ DỰ ĐOÁN VÀ ĐÁNH GIÁ THỰC TẾ TRÊN TẬP TEST (TEST SET):")
    logger.info(f"  + Test Loss (Hàm mất mát)      : {test_loss:.4f}")
    logger.info(f"  + Độ chính xác dự đoán (Acc)   : {test_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  + Precision (Macro Average)    : {test_metrics['precision_macro'] * 100:.2f}%")
    logger.info(f"  + Precision (Weighted Average) : {test_metrics['precision_weighted'] * 100:.2f}%")
    logger.info(f"  + Recall (Macro Average)       : {test_metrics['recall_macro'] * 100:.2f}%")
    logger.info(f"  + Recall (Weighted Average)    : {test_metrics['recall_weighted'] * 100:.2f}%")
    logger.info(f"  + F1-Score (Macro Average)     : {test_metrics['f1_macro'] * 100:.2f}%")
    logger.info(f"  + F1-Score (Weighted Average)  : {test_metrics['f1_weighted'] * 100:.2f}%")
    logger.info("-" * 85)
    logger.info("CHI TIẾT DỰ ĐOÁN TỪNG LỚP BỆNH TRÊN TẬP TEST:\n" + test_metrics['per_class_table'].to_string(index=False))

    # Vẽ và lưu biểu đồ Confusion Matrix tập Test
    logger.info("Đang xuất biểu đồ Ma trận nhầm lẫn cho tập Test...")
    plot_confusion_matrix(
        y_true_test,
        y_pred_test,
        class_names=class_names_list,
        title='Ma trận nhầm lẫn trên tập Test (Rice Leaf Disease)',
        save_path=str(TEST_CONFUSION_MATRIX_PATH)
    )
    logger.info(f"Ma trận nhầm lẫn tập Test đã lưu tại: {TEST_CONFUSION_MATRIX_PATH}")

    # Ghi báo cáo đánh giá Test ra file text
    with open(TEST_REPORT_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO DỰ ĐOÁN VÀ ĐÁNH GIÁ TRÊN TẬP TEST (PHASE 4: PREDICT & TEST EVALUATION)\n")
        f.write("=" * 85 + "\n")
        f.write(f"Test Loss                    : {test_loss:.4f}\n")
        f.write(f"Độ chính xác (Accuracy)      : {test_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision (Macro Average)    : {test_metrics['precision_macro'] * 100:.2f}%\n")
        f.write(f"Precision (Weighted Average) : {test_metrics['precision_weighted'] * 100:.2f}%\n")
        f.write(f"Recall (Macro Average)       : {test_metrics['recall_macro'] * 100:.2f}%\n")
        f.write(f"Recall (Weighted Average)    : {test_metrics['recall_weighted'] * 100:.2f}%\n")
        f.write(f"F1-Score (Macro Average)     : {test_metrics['f1_macro'] * 100:.2f}%\n")
        f.write(f"F1-Score (Weighted Average)  : {test_metrics['f1_weighted'] * 100:.2f}%\n")
        f.write("-" * 85 + "\n")
        f.write("CHI TIẾT THEO TỪNG LỚP:\n")
        f.write(test_metrics['per_class_table'].to_string(index=False) + "\n")
        f.write("=" * 85 + "\n")
    logger.info(f"Tệp báo cáo Test đã lưu tại: {TEST_REPORT_SAVE_PATH}")

    # 2. Suy luận thử nghiệm trên các ảnh mẫu đơn lẻ từ thư mục test
    test_dir = PROJECT_ROOT / 'dataset' / 'data_processed' / 'test'
    sample_prefixes = ['bac_la', 'dao_on', 'dom_nau', 'la_khoe']
    logger.info("\n2. Suy luận mẫu trực tiếp trên 4 ảnh thực tế từ 4 lớp trong thư mục test/:")

    for idx, prefix in enumerate(sample_prefixes):
        matching_files = sorted(list(test_dir.glob(f"{prefix}_*.jpg")))
        if matching_files:
            sample_img_path = matching_files[0]
            pred_res = classifier.predict(sample_img_path)
            ground_truth_name = classifier.class_names[idx]
            is_correct = (pred_res['class_idx'] == idx)

            logger.info(f"\n[Ảnh mẫu {idx+1}]: {sample_img_path.name}")
            logger.info(f"  - Nhãn thực tế (Ground Truth) : {ground_truth_name}")
            logger.info(f"  - Nhãn dự đoán (Predicted)    : {pred_res['class_name']} ({'ĐÚNG' if is_correct else 'SAI'})")
            logger.info(f"  - Độ tin cậy (Confidence)     : {pred_res['confidence']}%")
            logger.info(f"  - Phân bố xác suất 4 lớp     :")
            for name, prob in pred_res['all_probabilities'].items():
                logger.info(f"      + {name:<15}: {prob}%")

    logger.info(f"\nThời gian hoàn tất Phase 4: {time.time() - phase4_start:.2f} giây")

    # =====================================================================
    # TỔNG KẾT TOÀN BỘ QUY TRÌNH
    # =====================================================================
    total_time = time.time() - start_total_time
    logger.info("=" * 85)
    logger.info("TOÀN BỘ QUY TRÌNH 4 GIAI ĐOẠN ĐÃ ĐƯỢC THỰC THI HOÀN TẤT!")
    logger.info(f"Tổng thời gian toàn bộ tiến trình: {total_time/60:.2f} phút ({total_time:.2f} giây)")
    logger.info("Tóm tắt các tệp kết quả đã được xuất:")
    logger.info(f"  1. Tệp Log tiến trình    : {LOG_FILE_PATH}")
    logger.info(f"  2. Trọng số mô hình (.pth): {MODEL_SAVE_PATH}")
    logger.info(f"  3. Biểu đồ học tập       : {CURVES_SAVE_PATH}")
    logger.info(f"  4. Ma trận nhầm lẫn Val  : {VAL_CONFUSION_MATRIX_PATH}")
    logger.info(f"  5. Báo cáo đánh giá Val  : {VAL_REPORT_SAVE_PATH}")
    logger.info(f"  6. Ma trận nhầm lẫn Test : {TEST_CONFUSION_MATRIX_PATH}")
    logger.info(f"  7. Báo cáo dự đoán Test  : {TEST_REPORT_SAVE_PATH}")
    logger.info("=" * 85)


if __name__ == '__main__':
    main()
