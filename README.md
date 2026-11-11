# Credit–Fraud ML Pipeline

Pipeline dự đoán `credit_risk` và `fraud_risk_score` ở cấp khách hàng, từ dữ liệu CSV thô đến train, governance, batch inference và API.

> **Dữ liệu synthetic.** Toàn bộ 750.000 hồ sơ là dữ liệu tổng hợp; kết quả chỉ phục vụ học tập và thử nghiệm kỹ thuật. Không dùng mô hình này cho quyết định tín dụng, điều tra gian lận, hay bất kỳ quyết định thực tế nào về con người.

## Mục lục

- [Bài toán và dữ liệu](#bài-toán-và-dữ-liệu)
- [Insight từ EDA](#insight-từ-eda)
- [Pipeline và chống rò rỉ](#pipeline-và-chống-rò-rỉ)
- [Luồng hoạt động và công nghệ](#luồng-hoạt-động-và-công-nghệ)
- [Kết quả model v1.0.0](#kết-quả-model-v100)
- [Công bằng, giải thích và giới hạn](#công-bằng-giải-thích-và-giới-hạn)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Cấu trúc repository](#cấu-trúc-repository)

## Bài toán và dữ liệu

| Task | Target | Loại bài toán | Metric chính |
| --- | --- | --- | --- |
| Credit risk | `credit_risk` (0–3, có thứ tự) | Phân loại đa lớp | QWK, macro-F1 |
| Fraud risk | `fraud_risk_score` | Hồi quy | RMSE, MAE |

Dữ liệu gồm hồ sơ khách hàng, hành vi tín dụng, thông tin tài khoản và ngày giao dịch. Chi tiết cột có ở [finance_dictionary.md](docs/finance_dictionary.md). Đây là điểm rủi ro cấp khách hàng, không phải dữ liệu giao dịch và **không phải** hệ thống chống gian lận thực.

## Insight từ EDA

Các quan sát dưới đây lấy từ **toàn bộ 750.000 dòng raw**, trước cleaning. Có thể tái tạo bằng `python -m scripts.run_eda`; đặt `eda.sample_size` thành một số nguyên trong `configs/config.yaml` nếu chỉ muốn chạy trên mẫu.

- **Credit risk mất cân bằng:** lớp 0 chiếm 50,7% (380.490 dòng), lớp 1 là 23,3%, lớp 2 là 17,9% và lớp 3 là 8,1%. Vì lớp có thứ tự và không cân bằng, model dùng class weight; QWK và macro-F1 được báo cáo cùng nhau thay vì chỉ accuracy.
- **Fraud score tập trung quanh vùng giữa:** median là 0,447; phân vị 5% và 95% lần lượt là 0,272 và 0,774. Vì mục tiêu vận hành là ưu tiên review có giới hạn, pipeline dùng ranking/precision@5% bên cạnh RMSE và MAE.
- **Missingness tập trung ở feature hành vi:** `savings_rate` 0,77%, `monthly_spending` 0,70%, `credit_utilization` 0,70% và `debt_to_income` 0,69%; `income` và `investment_value` thấp hơn 0,1%. Pipeline giữ missingness flag và chỉ fit imputation trên train để không làm mất tín hiệu hoặc rò rỉ dữ liệu.
- **Nhiều numeric feature lệch phải và có tail dài:** income có median khoảng 36,3k nhưng max 4,98M; monthly spending median 474 nhưng max 250,0k; investment value median 22,3k nhưng max 2,97M. Vì vậy preprocessor xác định log transform từ skew trên train, còn cleaning chỉ gắn cờ outlier thay vì âm thầm thay đổi giá trị.
- **Có entry error/outlier rõ ràng trong raw:** ví dụ credit score thấp tới 32, employment years và late-payment months tới 99, trong khi account/card behavior thông thường tập trung thấp hơn nhiều. Data-quality rules chuyển lỗi nhập xác định được thành missing và thêm cờ cho outlier theo rule cấu hình.
- **Categorical feature có cardinality thấp:** các biến employment, education, region có 5 mức; marital status và home ownership có 4 mức. Điều này phù hợp với encoding phân loại trong fitted preprocessor; `customer_id` có cardinality bằng số dòng nên luôn bị loại khỏi feature matrix.

### Insight theo góc nhìn nghiệp vụ ngân hàng

Các bảng phân khúc đã được thêm vào EDA report. Chúng mô tả association của target synthetic sau khi loại lỗi nhập rõ ràng; không phải bằng chứng nhân quả và không được dùng một mình làm chính sách tín dụng/fraud.

| Phân khúc | Tỷ lệ `credit_risk` cao (lớp 2–3) | Mean fraud score | Diễn giải cho mô phỏng |
| --- | ---: | ---: | --- |
| Income dưới 50k | 34,1% | 0,502 | Cao hơn rõ rệt nhóm 50–100k (6,3% / 0,413); nhóm trên 250k chỉ có 699 dòng và có outlier, nên không suy diễn xu hướng thu nhập đơn giản. |
| Credit score 300–579 | 80,6% | 0,630 | Quan hệ mạnh nhất: score 740–850 chỉ có 0,9% high risk và fraud score 0,379. Đây là tín hiệu cần được kiểm tra về chất lượng/độ ổn định trước khi dùng thực tế. |
| Credit utilization ≥60% | 74,0% | 0,696 | So với utilization dưới 30% (4,4% / 0,370), mức sử dụng hạn mức cao là một tín hiệu ưu tiên review trong dữ liệu mô phỏng. |
| Late payment 3–24 tháng | 53,9% | 0,601 | Cao hơn nhóm không late payment (10,2% / 0,400), gợi ý lịch sử thanh toán là feature có ý nghĩa cho triage mô phỏng. |
| Unemployed | 73,0% | 0,607 | Chỉ là association trên synthetic data; không được dùng độc lập cho adverse action. Cần fairness review và human review trong bất kỳ bối cảnh thật nào. |

Tương quan Spearman cũng cho thấy credit score có quan hệ âm mạnh với credit risk (-0,711) và fraud score (-0,535), trong khi credit utilization có quan hệ dương mạnh (0,573 và 0,640). Savings rate và investment value có quan hệ âm đáng kể với cả hai target. Các quan hệ này có thể do cơ chế sinh dữ liệu, nên chỉ dùng để định hướng feature review và kiểm tra drift.

Đây là các pattern của dữ liệu **synthetic**, không phải insight về khách hàng ngân hàng thật và không được diễn giải như quan hệ nhân quả.

## Pipeline và chống rò rỉ

```text
Raw CSV
  → schema + data-quality rules
  → clean không trạng thái, cờ missing/outlier
  → một split chung 70/15/15 (stratify credit_risk)
  → feature engineering + preprocessor fit trên train
  → LightGBM tune trên train, chọn bằng validation
  → bundle phiên bản hóa → batch/API inference
```

- **Train 70%**: học preprocessor và model.
- **Validation 15%**: Optuna, early stopping, threshold và fairness diagnostic.
- **Test 15%**: chỉ đánh giá một lần sau khi chốt; script từ chối chạy lại nếu không có `--force`.
- Không dùng `customer_id`, target, hay cột ngày gốc làm feature. Feature ngày/tỷ lệ tính theo từng dòng; imputer, encoder, scaler và danh sách cột chỉ fit trên train.
- Cả hai task dùng cùng split customer ID để so sánh nhất quán.

## Luồng hoạt động và công nghệ

```text
CSV raw
  → Pandas: schema, kiểu dữ liệu và data-quality rules
  → Pandas/PyArrow: dataset đã clean và feature artifact dạng Parquet
  → scikit-learn: split, preprocessing fit trên train
  → LightGBM + Optuna: tune và train credit/fraud models
  → MLflow + SHAP + Fairlearn: tracking, giải thích và governance
  → model bundle (LightGBM + Joblib + JSON/YAML)
  → Pandas batch inference hoặc FastAPI/Uvicorn API
  → Docker Compose: API container + models mount read-only
```

| Chặng | Công nghệ | Hoạt động |
| --- | --- | --- |
| 1. Ingestion | Python 3.13, Pandas | Đọc CSV raw, kiểm tra đúng schema 23 cột, ép kiểu số/ngày/category và lưu bản trung gian. |
| 2. Data quality | Pandas, PyYAML | Đọc rules YAML; entry error được đổi thành missing, còn outlier và missingness được giữ lại qua các cờ feature. |
| 3. Chia dữ liệu | scikit-learn, NumPy | Tạo hoặc nạp lại split 70/15/15 stratify theo `credit_risk`; lưu customer ID để hai task dùng cùng tập dữ liệu. |
| 4. Features | Pandas | Tạo recency theo ngày, các tỷ lệ tài chính, age/income bins và các feature cờ chất lượng. |
| 5. Preprocessing | scikit-learn `ColumnTransformer` | Chỉ fit trên train: log transform theo skew, median imputation, scaling, ordinal encoding và one-hot encoding. |
| 6. Train/tune | LightGBM, Optuna | Train LightGBM multiclass cho credit và Huber regression cho fraud; Optuna chọn hyperparameter theo validation và early stopping. |
| 7. Đánh giá/governance | scikit-learn, SciPy, Fairlearn, SHAP | Tính QWK/F1/RMSE/MAE/top-k, chọn policy trên validation, kiểm tra fairness và tạo SHAP report. |
| 8. Tracking/artifact | MLflow, PyArrow, Joblib, JSON/YAML | Log experiment cục bộ; lưu model, preprocessor, feature list, metrics và inference metadata thành bundle versioned. |
| 9. Serving | Pandas, FastAPI, Pydantic, Uvicorn | Batch inference hoặc REST API; input API được validate trước khi đi qua cùng cleaning/feature/preprocessing flow. |
| 10. Container | Docker, Docker Compose | Multi-stage image `python:3.13-slim`; container chỉ chứa runtime/code, bundle được mount từ `models/` ở chế độ read-only. |

### Luồng inference sau khi đã train

Batch và API đều nhận **dữ liệu raw**, không nhận feature matrix đã biến đổi. Bundle đã lưu chứa preprocessor, feature list và metadata của data-quality/feature settings, nên inference tái tạo đúng chuỗi transform đã dùng khi train:

```text
Raw customer record
  → validate input (API) / kiểm tra customer_id (batch)
  → cleaning theo rules đã đóng gói
  → stateless feature engineering
  → preprocessor đã fit trên train
  → chọn đúng feature list của bundle
  → credit class + probabilities và/hoặc fraud score
```

Docker không train model trong lúc build. Trước khi chạy Compose, cần có các bundle `models/credit_risk/v1.0.0` và `models/fraud_risk/v1.0.0` trên máy host để mount vào `/app/models`; nếu không, `GET /health` trả `model_not_loaded`.

## Kết quả model v1.0.0

Các kết quả lấy từ artifact test đã khóa (`threshold_test_metrics.json`). Credit dùng threshold chọn trên validation; fraud dùng policy top 5% chọn trên validation.

| Model chính | Test metric | Giá trị |
| --- | --- | ---: |
| Credit LightGBM | QWK | 0.7888 |
| Credit LightGBM | Macro-F1 | 0.6014 |
| Fraud LightGBM (Huber) | RMSE | 0.1088 |
| Fraud LightGBM (Huber) | MAE | 0.0831 |
| Fraud LightGBM (Huber) | R² | 0.5320 |
| Fraud LightGBM (Huber) | Spearman | 0.7025 |
| Fraud LightGBM (Huber) | Precision@5% | 0.4750 |

Credit validation QWK tăng từ 0.7793 với `argmax` lên 0.7881 sau khi dùng ba ngưỡng ordinal chọn **chỉ trên validation**. Fraud review policy chọn top 5% score, với cutoff validation `0.693236`.

Model card chi tiết: [credit](docs/model_card_credit.md) và [fraud](docs/model_card_fraud.md).

## Công bằng, giải thích và giới hạn

- SHAP TreeExplainer tạo global importance và local case trên tối đa 10.000 dòng validation.
- Fairness diagnostic credit so sánh `age_bin`, `marital_status`, `region`, `education_level`. Biến thể `no_sensitive` bỏ `age`, `age_bin`, `marital_status` nhưng giữ hyperparameter; QWK validation vẫn là 0.7793. Đây chỉ là diagnostic trên dữ liệu synthetic, không chứng minh fairness pháp lý hay an toàn production.
- Chất lượng model chưa đủ để triển khai thực tế: không có outcome thật, drift, dữ liệu giao dịch chi tiết, chi phí sai lầm, human-review process hay đánh giá bias trên dân số thực.
- Trước mọi ứng dụng thật, cần validation theo thời gian, review pháp lý/compliance, calibration, monitoring, audit fairness, chính sách xử lý ngoại lệ và human-in-the-loop.

## Cài đặt và chạy

Mọi lệnh chạy tại gốc repository bằng Python 3.13.

```powershell
# Cài dependency và kiểm tra code
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ruff check src/ scripts/ tests/ api/
.\.venv\Scripts\python.exe -m pytest tests -v

# Chuẩn bị dữ liệu, feature và train lại trên dữ liệu mới
.\.venv\Scripts\python.exe -m scripts.run_data_quality
.\.venv\Scripts\python.exe -m scripts.run_feature_engineering
.\.venv\Scripts\python.exe -m scripts.run_training

# Chỉ dùng một lần để đánh giá test sau khi chốt model
.\.venv\Scripts\python.exe -m scripts.run_training --final

# Governance trên validation; không dùng --final nếu test đã khóa
.\.venv\Scripts\python.exe -m scripts.run_threshold_report
.\.venv\Scripts\python.exe -m scripts.run_fairness_report
.\.venv\Scripts\python.exe -m scripts.run_explainability
```

### Batch prediction

```powershell
.\.venv\Scripts\python.exe -m scripts.run_batch_inference `
  --input tests/data/sample.csv `
  --output data/predictions/sample_predictions.csv `
  --task both
```

Kết quả có `credit_risk_prediction`, xác suất từng lớp và `fraud_risk_score_prediction`. Input phải có `customer_id` và các cột raw đã train; pipeline tự áp dụng clean, feature engineering và preprocessor đã lưu.

### API và Docker

```powershell
# API local; Swagger: http://127.0.0.1:8000/docs
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload

# API Docker; models được mount read-only, không nằm trong image
docker compose -f docker/docker-compose.yml up --build
```

`GET /health` trả `model_not_loaded` nếu chưa có bundle hoặc fraud threshold policy. Các endpoint `POST /predict/credit`, `/predict/fraud`, `/predict/both` trả 422 cho input không hợp lệ và 503 khi bundle chưa sẵn sàng.

MLflow: `mlflow ui --backend-store-uri ./mlruns`.

## Cấu trúc repository

```text
configs/      cấu hình dữ liệu, feature và model
src/          pipeline, model, evaluation, governance
scripts/      CLI cho data quality, train, report và inference
api/          FastAPI schemas và endpoints
docker/       Dockerfile API và Compose
tests/        unit + integration tests trên fixture 50 dòng
docs/         dictionary, roadmap và model cards
```

`data/`, `models/`, `mlruns/` và HTML reports là artifact cục bộ, không commit vào Git.
