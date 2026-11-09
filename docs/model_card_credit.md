# Model card — Credit Risk LightGBM v1.0.0

> **Dữ liệu synthetic.** Model này được huấn luyện và đánh giá hoàn toàn trên dữ liệu tổng hợp. Không dùng cho phê duyệt tín dụng, định giá, từ chối dịch vụ, hay bất kỳ quyết định thực tế nào về khách hàng.

## Mục đích

Model dự đoán `credit_risk`, lớp rủi ro có thứ tự từ 0 đến 3, nhằm minh họa pipeline ML có kiểm soát leakage, governance và serving. Nó chỉ hỗ trợ sắp xếp hồ sơ trong bài tập mô phỏng, không phải công cụ ra quyết định tự động.

## Dữ liệu và split

- Nguồn: 750.000 hồ sơ khách hàng **synthetic**; schema ở [finance_dictionary.md](finance_dictionary.md).
- Một split duy nhất, phân tầng theo `credit_risk`: train/validation/test = 70/15/15.
- Train fit imputer, encoder, scaler và model. Validation chọn hyperparameter, early stopping và ordered threshold. Test chỉ được đánh giá một lần sau khi chốt.
- Không dùng `customer_id`, `credit_risk`, `fraud_risk_score` hoặc ngày thô làm feature. Feature phái sinh gồm tuổi/recency tài khoản, tỷ lệ chi tiêu–thu nhập, đầu tư–thu nhập, khoản vay–thẻ và các bin cấu hình.

## Thiết kế model

- LightGBM multiclass, `class_weight=balanced`, deterministic CPU training.
- Optuna tối đa 50 trial hoặc 1.800 giây, tối đa hóa QWK validation; không k-fold.
- Bundle: `models/credit_risk/v1.0.0/`, gồm tree model, fitted preprocessor, feature list, metadata và metrics.
- Decision policy: expected class score từ xác suất bốn lớp, với ngưỡng validation `0.824218`, `1.625558`, `2.383162`.

## Hiệu năng

| Split / policy | QWK | Macro-F1 |
| --- | ---: | ---: |
| Validation, argmax | 0.7793 | 0.5920 |
| Validation, threshold ordinal | 0.7881 | — |
| Test, frozen threshold policy | 0.7888 | 0.6014 |

Test per-class F1 cho lớp 0–3 là 0.8279, 0.4609, 0.4733 và 0.6433. Lớp 1 và 2 vẫn khó phân biệt; metric tổng hợp không nên che khuất hạn chế này.

## Fairness và explainability

- Báo cáo validation so sánh `age_bin`, `marital_status`, `region`, `education_level` bằng selection rate, true/false-positive rate và macro-F1 theo nhóm.
- Biến thể `no_sensitive` loại `age`, `age_bin`, `marital_status`, không tune lại: QWK validation 0.7793 và macro-F1 0.5921, gần model mặc định (0.7793 và 0.5920).
- SHAP TreeExplainer cung cấp importance toàn cục và local case trên tối đa 10.000 dòng. Đây là mô tả hành vi model trên dữ liệu synthetic, không chứng minh quan hệ nhân quả.

## Giới hạn và rủi ro

- Dữ liệu/target synthetic không phản ánh default, affordability, discrimination, quy định hay hành vi khách hàng thực.
- Không có kiểm tra drift theo thời gian, calibration trên outcome thật, utility/cost matrix hay human-review workflow.
- Các biến nhân khẩu học và proxy có thể gây tác động không mong muốn. Không suy luận fairness pháp lý từ báo cáo này.
- Không dùng prediction như quyết định cuối cùng; nếu áp dụng vào bối cảnh thật cần validation độc lập, legal/compliance review, monitoring và cơ chế khiếu nại/override.

## Tái lập và vận hành

- Bundle `v1.0.0` lưu config hash, git commit và timestamp.
- Inference phải đi qua `predict_batch` hoặc API để dùng cùng clean, feature logic và preprocessor đã fit.
- Re-train chỉ trên dữ liệu mới; không tối ưu lại trên test đã khóa. Theo dõi QWK, macro-F1, per-class recall, missingness, drift feature và chênh lệch fairness theo nhóm.
