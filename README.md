# Credit Fraud ML Pipeline

Pipeline Python để dự đoán phân hạng rủi ro tín dụng và điểm rủi ro gian lận ở cấp khách hàng.

> **Lưu ý:** toàn bộ dữ liệu của dự án là dữ liệu **synthetic**. Dự án không phải hệ thống phát hiện gian lận thực tế và không nên dùng cho quyết định tín dụng hay gian lận trong sản xuất.

## Yêu cầu

- Python 3.13

## Mục lục

- [Yêu cầu](#yêu-cầu)
- [Bài toán](#bài-toán)
- [Lệnh thường dùng](#lệnh-thường-dùng)

## Bài toán

- `credit_risk`: phân loại bốn lớp có thứ tự (0–3). Chỉ số chính: QWK và macro-F1.
- `fraud_risk_score`: hồi quy điểm rủi ro. Chỉ số chính: RMSE và MAE.

Quy trình sẽ dùng một lần chia train/validation/test 70/15/15, với phân tầng theo `credit_risk`. Train dùng để học; validation dùng để chọn cấu hình; test chỉ đánh giá khi chốt mô hình.

## Lệnh thường dùng

```powershell
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m pytest tests -v
ruff check src/ scripts/ tests/
```

Các lệnh training, đánh giá và API sẽ được bổ sung ở các giai đoạn sau.
