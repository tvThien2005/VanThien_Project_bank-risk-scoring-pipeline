# Model card — Fraud Risk Score LightGBM v1.0.0

> **Dữ liệu synthetic.** Model này dùng dữ liệu tổng hợp ở cấp khách hàng, không có chuỗi giao dịch thực. Nó không phải fraud detection system và không được dùng để điều tra, khóa tài khoản hoặc ra quyết định bất lợi cho khách hàng.

## Mục đích

Model ước lượng `fraud_risk_score`, một score liên tục trong dữ liệu synthetic. Mục tiêu là minh họa ranking hồ sơ để ưu tiên review trong môi trường học tập, không phải dự báo xác suất gian lận thật.

## Dữ liệu và split

- Nguồn: 750.000 hồ sơ **synthetic**; mô tả cột ở [finance_dictionary.md](finance_dictionary.md).
- Dùng cùng split khách hàng 70/15/15 với credit model, phân tầng theo `credit_risk` để giữ hai task đồng bộ.
- Train fit toàn bộ transform có trạng thái; validation dùng để tune, early stop và chọn fraud top-5% cutoff; test chỉ dùng một lần sau khi policy đóng băng.
- Không đưa `customer_id`, cả hai target, hoặc ngày gốc vào ma trận feature. Clean, feature engineering và fitted preprocessor được đóng gói trong bundle.

## Thiết kế model và review policy

- LightGBM regression với objective Huber; deterministic CPU training.
- Optuna tối đa 50 trial hoặc 1.800 giây, tối thiểu hóa RMSE validation; không k-fold.
- Bundle: `models/fraud_risk/v1.0.0/`.
- Policy review: chọn top 5% score trên validation, threshold cố định `0.693236`. `precision_at_k`/`recall_at_k` định nghĩa positive thực là top 5% `fraud_risk_score` thật trong split đánh giá.

## Hiệu năng

| Split | RMSE | MAE | R² | Spearman | Precision@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.1094 | 0.0835 | 0.5294 | 0.7000 | 0.4779 |
| Test, frozen policy | 0.1088 | 0.0831 | 0.5320 | 0.7025 | 0.4750 |

Kết quả cho thấy khả năng ranking vừa phải trên target synthetic. Precision@5% khoảng 0.475 không đồng nghĩa 47.5% cảnh báo là fraud thật trong môi trường thực.

## Fairness, explainability và giới hạn

- Không có fairness conclusion cho fraud task. Bất kỳ so sánh nhóm nào trên dữ liệu synthetic cũng không đủ cho compliance hoặc adverse-action analysis.
- SHAP TreeExplainer xuất global importance và local case trên tối đa 10.000 dòng validation; đây không phải bằng chứng nhân quả.
- Không có transaction sequence, device, merchant, chargeback, nhãn điều tra, temporal backtest, delayed label, prevalence thật hay chi phí false positive/negative.
- Không dùng score để tự động khóa tài khoản hoặc cáo buộc khách hàng. Cần human review, calibration, monitoring drift, investigation feedback và privacy/legal review trước mọi triển khai thật.

## Tái lập và vận hành

- Bundle `v1.0.0` lưu config hash, git commit và timestamp.
- Chỉ dùng `predict_batch` hoặc API để bảo toàn preprocessing lúc train.
- Theo dõi RMSE, MAE, Spearman, precision/recall@k, score distribution, top-k volume và drift đầu vào trên dữ liệu mới.
- Không tune lại theo test đã đánh giá; re-train cần split/holdout mới và policy review mới.
