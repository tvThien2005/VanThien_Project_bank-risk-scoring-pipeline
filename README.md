# Credit Fraud ML Pipeline

Pipeline Python để dự đoán phân hạng rủi ro tín dụng và điểm rủi ro gian lận ở cấp khách hàng.

> **Lưu ý:** toàn bộ dữ liệu của dự án là dữ liệu **synthetic**. Dự án không phải hệ thống phát hiện gian lận thực tế và không nên dùng cho quyết định tín dụng hay gian lận trong sản xuất.

## Yêu cầu

- Python 3.13

## Mục lục

- [Yêu cầu](#yêu-cầu)
- [Bài toán](#bài-toán)
- [Quy trình dữ liệu](#quy-trình-dữ-liệu)
- [Lệnh thường dùng](#lệnh-thường-dùng)

## Bài toán

- `credit_risk`: phân loại bốn lớp có thứ tự (0–3). Chỉ số chính: QWK và macro-F1.
- `fraud_risk_score`: hồi quy điểm rủi ro. Chỉ số chính: RMSE và MAE.

Quy trình sẽ dùng một lần chia train/validation/test 70/15/15, với phân tầng theo `credit_risk`. Train dùng để học; validation dùng để chọn cấu hình; test chỉ đánh giá khi chốt mô hình.

## Quy trình dữ liệu

```text
raw CSV → validate/clean → shared split 70/15/15 → feature engineering
        → train/validation → versioned model bundle → raw-batch inference
```

Model bundle gồm model tree, preprocessor đã fit trên train, feature list, metrics và metadata. Batch inference luôn bắt đầu từ dữ liệu raw, nên dùng cùng cleaning và feature logic với lúc train.

## Lệnh thường dùng

```powershell
# Cài môi trường
python -m pip install -r requirements.txt

# Kiểm tra code
ruff check src/ scripts/ tests/ api/
python -m pytest tests -v

# Chuẩn bị dữ liệu và feature artifact
python -m scripts.run_data_quality
python -m scripts.run_feature_engineering

# Train cả hai model từ raw data; không dùng test
python -m scripts.run_training

# Đánh giá test đúng một lần sau khi chốt model
python -m scripts.run_training --final

# Governance reports
python -m scripts.run_threshold_report
python -m scripts.run_fairness_report
python -m scripts.run_explainability

# Inference và EDA
python -m scripts.run_batch_inference
python -m scripts.run_eda

# Theo dõi MLflow và chạy API cục bộ (cần có models/*/v1.0.0)
mlflow ui --backend-store-uri ./mlruns
uvicorn api.app:app --reload

# Chạy API bằng Docker; model được mount read-only khi container khởi động
docker compose -f docker/docker-compose.yml up --build
```

`run_training --final` và `run_threshold_report --final` có guard chống đọc test lặp lại. Chỉ dùng `--force` khi sửa bug pipeline và có lý do ghi nhận rõ ràng.

Batch inference mặc định đọc `data/raw/synthetic_finance_20250901.csv` và ghi `data/predictions/batch_predictions.csv`. Có thể chỉ định input/output khác:

```powershell
python -m scripts.run_batch_inference --input path/to/customers.csv --output data/predictions/customers.csv --task both
```

API dùng bundle `v1.0.0` trong `models/` theo mặc định. Khi model chưa được mount hoặc thiếu policy fraud đã chọn trên validation, `GET /health` trả `model_not_loaded` và các endpoint dự đoán trả `503` thay vì dự đoán không đầy đủ. Các endpoint gồm `POST /predict/credit`, `POST /predict/fraud`, và `POST /predict/both`; schema tương tác có tại `http://127.0.0.1:8000/docs` sau khi chạy API.

Sau khi cài/xác minh môi trường mới, có thể pin lại dependencies từ venv đã activate bằng lệnh phù hợp với shell đang dùng. Với PowerShell, nên ưu tiên cập nhật thủ công `requirements.txt` để không vô tình thay đổi toàn bộ dependency tree.

## Công bằng và biến nhạy cảm

Mô hình credit dùng `age`, `age_bin` và `marital_status` trong phiên bản mặc định. Đây chỉ là dữ liệu synthetic và không phải khuyến nghị sử dụng các biến này để ra quyết định tín dụng thực tế. Chạy `python -m scripts.run_fairness_report` để so sánh validation với biến thể `no_sensitive`, vốn loại ba cột này và dùng nguyên hyperparameter đã chọn, không tune lại.
