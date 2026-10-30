# Roadmap dự án credit-fraud-ml-pipeline (bản giao cho Codex)

## Bối cảnh

- Dataset: bảng khách hàng ngân hàng **tổng hợp (synthetic)**, 750.000 dòng, 23 cột (xem `docs/data_dictionary.md`).
- **Task A**: `credit_risk`, phân loại 4 lớp có thứ tự (0 đến 3). Metric chính: QWK, macro-F1.
- **Task B**: `fraud_risk_score`, hồi quy. Metric chính: RMSE, MAE.
- Môi trường: **Python 3.13**.
- Đây là điểm rủi ro cấp khách hàng trên dữ liệu tổng hợp, **không phải hệ thống phát hiện gian lận thực** (không có dữ liệu giao dịch chi tiết).

## Quy tắc làm việc cho Codex

1. **Không tự commit.** Sau mỗi checkpoint (cuối mỗi giai đoạn), người dùng review rồi tự commit với message rõ ràng.
2. **Không tự sinh `.ipynb`.** Chỉ viết `.py`. Người dùng tự mở notebook, import từ `src/`, chạy và lưu.
3. **Đọc trước khi viết.** Phải đọc `data/raw/` và `configs/config.yaml` hiện có trước khi viết code, không được giả định schema. Nếu `configs/config.yaml` chưa tồn tại thì tạo bản tối thiểu và liệt kê các key để người dùng duyệt.
4. **Mơ hồ thì dừng và hỏi.** Ví dụ chưa rõ dtype cột X, chưa rõ ý nghĩa một cột, thì hỏi chứ không tự bịa.
5. **Nhắc "dữ liệu là synthetic" ở 3 chỗ**: README, model card, và đầu mỗi report HTML. Kết quả đẹp có thể gây hiểu lầm.
6. **Chống rò rỉ dữ liệu.** Mọi phép biến đổi dựa trên thống kê (median, ngưỡng winsorize, scaler, encoder, danh sách cột cần log) chỉ fit trên train. Hàm không có trạng thái (tính tỷ lệ, tính ngày) thì để ở `build_features.py`; bước có trạng thái thì để trong preprocessor.
7. **Không bao giờ dùng làm feature**: `credit_risk`, `fraud_risk_score`, `customer_id`, và các cột ngày thô (chỉ dùng feature phái sinh từ ngày).
8. **Không hard-code đường dẫn, seed, ngưỡng.** Tất cả lấy từ `configs/`.
9. **Không dùng k-fold ở bất kỳ bước nào.** Chỉ có một lần chia train/valid/test duy nhất (70/15/15, phân tầng theo `credit_risk`). Train dùng để học, valid dùng để chọn tham số, early stopping và chọn ngưỡng, test chỉ dùng một lần ở cuối.
10. **Cách chạy code (không có `pyproject.toml`, không có Makefile, không có `pytest.ini`, không có `ruff.toml`):** mọi lệnh chạy từ **thư mục gốc repo**; script chạy dạng `python -m scripts.<tên_script>`; test chạy bằng `python -m pytest tests -v` (phải có `python -m` để `from src...` tìm được module); ruff dùng cấu hình mặc định. Mọi package (`src/`, các thư mục con của `src/`, `scripts/`, `api/`, `tests/`) đều có `__init__.py`.
11. **Mỗi giai đoạn kết thúc bằng**: `ruff check` và `python -m pytest tests -v` đều xanh, rồi báo cáo checkpoint gồm: danh sách file đã tạo/sửa, lệnh để kiểm tra, các giả định đã dùng, câu hỏi còn mở.
12. Mỗi hàm có type hint và docstring ngắn. Dùng `logger`, không dùng `print`.

## Cấu trúc thư mục đích

```
credit-fraud-ml-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── .dockerignore
├── .github/workflows/ci.yml
├── configs/        config.yaml, data_quality.yaml, model_credit.yaml, model_fraud.yaml
├── data/           raw/, interim/, processed/, features/   (không commit)
├── docs/           data_dictionary.md, model_card_credit.md, model_card_fraud.md
├── notebooks/      do người dùng viết
├── src/            utils/, data/, features/, models/, evaluation/, explainability/, pipelines/
├── scripts/
├── tests/          unit/, integration/, data/sample.csv
├── models/         credit_risk/v1.0.0/, fraud_risk/v1.0.0/   (không commit)
├── reports/        figures/ và các file HTML                  (không commit)
├── api/            app.py, schemas.py
├── docker/         Dockerfile.api, docker-compose.yml
└── mlruns/                                                    (không commit)
```

---

## Giai đoạn 0: Nền móng (0,5 đến 1 ngày)

### Code cần viết

- `.gitignore`: dùng đường dẫn **neo từ gốc repo** để không vô tình ignore code:
  `/data/`, `/models/`, `/mlruns/`, `/reports/*.html`, `/reports/figures/`, `.env`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `*.joblib`.
  Lý do: `models/` hoặc `data/` không có dấu `/` đầu sẽ ignore luôn `src/models/` và `tests/data/`.
- `__init__.py` cho `src/` (và các thư mục con khi tạo), `scripts/`, `tests/` (và `tests/unit/`, `tests/integration/`).
- `requirements.txt` (Python 3.13):
  - **Không dùng danh sách pin cũ** (numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, scipy 1.14.0, matplotlib 3.9.1, pyyaml 6.0.1, shap 0.46.0 không có bản dựng sẵn cho Python 3.13, `pip` sẽ cố biên dịch từ mã nguồn và dễ lỗi).
  - Khai báo gói với cận dưới tương thích 3.13: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `lightgbm`, `optuna`, `mlflow`, `shap`, `fairlearn`, `pyyaml`, `joblib`, `matplotlib`, `seaborn`, `pytest`, `httpx`, `ruff`, `fastapi`, `uvicorn`, `pydantic`.
  - Cài trong venv Python 3.13, chạy test, rồi **pin lại đúng các phiên bản đã cài được và test xanh** (`pip freeze` lọc theo các gói trực tiếp). Báo cáo gói nào gặp vấn đề (đặc biệt `shap`, `numba`, `lightgbm`).
- `scripts/make_sample_data.py`: sinh `tests/data/sample.csv` (~50 dòng) có đủ 23 cột đúng dtype theo `docs/data_dictionary.md` (và đối chiếu với `data/raw/` nếu đã có). Cố ý chèn:
  - 2 dòng thiếu giá trị ở cột nullable (ghi rõ cột đã chọn trong docstring).
  - 1 dòng `credit_score` = 999.
  - 1 dòng `age` = 200.
  - 1 dòng một cột tỷ lệ > 1 (ví dụ `credit_utilization`).
  - 1 dòng `last_transaction_date` trước `account_creation_date`.
  - 1 dòng `income` âm.
  - Dùng seed cố định. File sample **được commit** (không bị `.gitignore`).
- `.github/workflows/ci.yml`: Python 3.13, `pip install -r requirements.txt`, `ruff check src/ scripts/ tests/`, `python -m pytest tests -v` (integration test chạy trên 50 dòng nên nhanh). Không build Docker trong CI.
- `README.md` khung: tên dự án, 1 câu mô tả, yêu cầu Python 3.13, mục lục, ghi chú "data là synthetic", và mục "Lệnh thường dùng" (để trống, hoàn thiện dần theo từng giai đoạn).

### Điều kiện hoàn thành

- `python --version` trong venv là 3.13.x.
- Từ gốc repo chạy được: `pip install -r requirements.txt`, `ruff check src/ scripts/ tests/`, `python -m pytest tests -v`, `python -m scripts.make_sample_data`.
- CI xanh trên Python 3.13.
- `git check-ignore -v src/models tests/data/sample.csv` không trả về gì (hai đường dẫn này không bị ignore).
- Không có data hoặc secret trong git.
- `requirements.txt` đã pin theo phiên bản cài thực tế.

---

## Giai đoạn 1: Hiểu dữ liệu (1 đến 2 ngày)

### Code cần viết

- `src/utils/config.py`: đọc YAML, merge với env var override, validate key bắt buộc.
- `src/utils/logger.py`: structured logging (JSON).
- `src/utils/seeds.py`: set seed cho numpy, random, `PYTHONHASHSEED`.
- `src/utils/io.py`: `read_csv_safe`, `read_parquet_safe`, `write_parquet_atomic`.
- `src/data/load.py`: `load_raw(cfg) -> pd.DataFrame`; ép dtype (cột ngày thành datetime, cột phân loại thành category) dựa trên schema thật; lưu `data/interim/customers.parquet`.
- `scripts/download_data.py`: kiểm tra `data/raw/` đã có file chưa; nếu chưa thì báo lỗi kèm hướng dẫn (không tự tải).
- Notebook do người dùng viết (Codex không đụng): `01_data_understanding.ipynb`, `02_eda.ipynb`.

### Điều kiện hoàn thành

- File Parquet đọc lại không lỗi, dtype đúng.
- README ghi rõ 2 bài toán, target, metric.
- Hình EDA được lưu trong `reports/figures/`.

---

## Giai đoạn 2: Chất lượng dữ liệu (2 đến 3 ngày)

### Code cần viết

- `configs/data_quality.yaml`: mỗi luật gồm `column`, `rule` (`in_range` / `regex` / `relation`), `action` (`to_nan` / `flag`). Codex đọc schema thật từ `data/raw/` để viết luật, miền hợp lệ lấy từ `docs/data_dictionary.md`.
- `src/data/validate.py`: `run_rules(df, cfg) -> dict[str, int]` trả về số vi phạm của mỗi luật (kèm tỷ lệ).
- `src/utils/report.py`: hàm dùng chung để dựng file HTML, **tự chèn banner "Dữ liệu synthetic" ở đầu** mọi report.
- `scripts/run_data_quality.py`: xuất `reports/data_quality_report.html` (tự viết từ `pandas.to_html()` và vài hình matplotlib; không dùng ydata-profiling trên 750k dòng).
- `src/data/clean.py` gồm 3 hàm **không có trạng thái** (chỉ dùng luật cố định, không dùng thống kê của dữ liệu):
  - `apply_entry_error_rules`: lỗi nhập rõ ràng thì đặt NaN.
  - `handle_outliers`: chỉ **gắn cờ** outlier (cột `*_outlier_flag`), không sửa giá trị.
  - `add_missing_flags`: thêm cột chỉ báo thiếu cho các cột có thiếu.
  - Việc impute và winsorize (cần thống kê) **không làm ở đây**, sẽ nằm trong preprocessor ở giai đoạn 4 và chỉ fit trên train.
- `tests/unit/test_validate.py`, `tests/unit/test_clean.py`: dùng `sample.csv` với các lỗi cố ý.

### Điều kiện hoàn thành

- Báo cáo liệt kê số vi phạm theo từng luật.
- Đã ghi nhận quy luật thiếu không ngẫu nhiên (kiểm tra bằng `groupby` theo `employment_status`, `income_bin`, target).
- `clean` chạy toàn bộ dữ liệu, không còn giá trị ngoài miền hợp lệ.
- Test xanh.

---

## Giai đoạn 3: Chia dữ liệu (0,5 ngày)

### Code cần viết

- `src/data/split.py`: `make_splits(df, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`.
  - **Một lần chia duy nhất** thành train/valid/test theo tỷ lệ 70/15/15 (lấy từ config), **phân tầng theo `credit_risk`** để tỷ lệ các lớp `credit_risk` giống nhau ở cả 3 tập.
  - Seed lấy từ config. Một lần chia dùng chung cho cả hai task.
  - Lưu `train_ids.npy`, `valid_ids.npy`, `test_ids.npy` vào `data/processed/`.
- `tests/unit/test_split.py`:
  - Không trùng `customer_id` giữa các tập, và 3 tập gộp lại đủ toàn bộ dòng.
  - Kích thước 3 tập xấp xỉ 70/15/15.
  - Tỷ lệ mỗi lớp `credit_risk` giữa các tập lệch nhau dưới 1 điểm phần trăm.
  - Chạy lại cùng seed cho cùng kết quả.

### Điều kiện hoàn thành

- Các file id đã lưu.
- Test xanh.

---

## Giai đoạn 4: Feature engineering (2 đến 3 ngày)

### Code cần viết

- `src/features/build_features.py` (các hàm **không có trạng thái**):
  - `add_date_features(df, reference_date)`: `account_age_days`, `days_since_last_txn`. `reference_date` lấy từ config (cố định), không dùng ngày hôm nay.
  - `add_ratio_features(df)`: `spending_to_income`, `investment_to_income`, `loans_per_card` (chia cho 0 thì ra NaN).
  - `add_bins(df, age_bins, income_bins)`: biên các bin lấy từ config.
- `src/features/preprocess.py`:
  - `build_preprocessor(cfg) -> ColumnTransformer` (các bước **có trạng thái**): impute (median cho số, mode cho phân loại), log cho cột lệch (danh sách cột có `|skew| > skew_threshold` được xác định **trên train** và lưu trong preprocessor), winsorize nếu dùng (ngưỡng tính trên train), mã hóa (`education_level` thứ tự; các cột phân loại còn lại one-hot), scale nếu model cần.
  - Loại bỏ `customer_id`, cột ngày thô, hai target khỏi ma trận feature. Hỗ trợ `features.exclude` trong config để bỏ một số feature (dùng cho thí nghiệm fairness ở giai đoạn 7).
  - `fit_preprocessor(preprocessor, X_train)`, `save_preprocessor` / `load_preprocessor` bằng joblib.
- `src/features/feature_selection.py`: permutation importance và correlation filter, **chỉ tính trên train**.
- `scripts/run_feature_engineering.py`: sinh `data/features/{X,y_credit,y_fraud}_{train,valid,test}.parquet`.
- `tests/unit/test_features.py`.

### Điều kiện hoàn thành

- Preprocessor fit và transform đúng (transform lại cho kết quả giống nhau); lưu rồi nạp lại cho kết quả giống hệt.
- Không dùng target của task này làm feature của task kia.
- Đã kiểm tra feature nào giải thích gần hết target và ghi lại kết quả.
- Test xanh.

---

## Giai đoạn 5: Baseline (2 ngày)

### Code cần viết

- `src/evaluation/metrics.py`:
  - Credit: `macro_f1`, `qwk`, `per_class_prf`, `confusion_matrix`.
  - Fraud: `rmse`, `mae`, `r2`, `spearman`, `precision_recall_at_k`. Với `precision_recall_at_k`, "dương tính thật" được định nghĩa là các khách hàng có `fraud_risk_score` thật nằm trong top k% (ghi rõ trong docstring).
- `tests/unit/test_metrics.py`: ví dụ nhỏ có đáp án biết trước.
- `configs/model_credit.yaml`, `configs/model_fraud.yaml`: task, metric chính, model, khối optuna.
- `src/models/train_credit.py`: Dummy (`most_frequent`) và Logistic Regression (multinomial, `class_weight='balanced'`, `max_iter` đủ lớn); huấn luyện trên train, đánh giá trên valid; log MLflow.
- `src/models/train_fraud.py`: Dummy (mean) và Ridge; huấn luyện trên train, đánh giá trên valid; log MLflow.
- `src/models/evaluate.py`: `evaluate_classification`, `evaluate_regression` trả về dict metric và xuất HTML report (có banner synthetic).

### Điều kiện hoàn thành

- Có bảng kết quả baseline trên valid.
- MLflow đã ghi log.
- Test metric xanh.
- Chạy lại cho cùng kết quả.

---

## Giai đoạn 6: Mô hình chính (3 đến 5 ngày)

### Code cần viết

Không dùng k-fold. Huấn luyện trên **train**, chọn tham số và early stopping trên **valid**.

- Mở rộng `train_credit.py`: LightGBM multiclass, `class_weight='balanced'`. Optuna `n_trials=50` có seed, `timeout=1800s` làm chốt an toàn; mỗi trial huấn luyện trên train và chấm điểm trên valid (tối đa hóa QWK hoặc macro-F1 theo `model_credit.yaml`); early stopping dùng valid.
- Mở rộng `train_fraud.py`: LightGBM `objective='huber'`. Optuna `n_trials=50` có seed, `timeout=1800s`; mỗi trial huấn luyện trên train, chấm điểm trên valid (tối thiểu hóa RMSE hoặc MAE theo `model_fraud.yaml`); early stopping dùng valid.
- Model cuối cùng là model huấn luyện trên train với tham số tốt nhất. **Không gộp valid vào train** để valid và test giữ nguyên vai trò.
- Config có `tune_sample_size` (tùy chọn, mặc định `null` nghĩa là dùng toàn bộ train): nếu mỗi trial quá chậm thì lấy mẫu train có phân tầng để tìm tham số, sau đó huấn luyện lại trên toàn bộ train.
- LightGBM đặt `seed`, `deterministic=True`, `n_jobs` cố định để tái lập.
- `src/models/predict.py`: `load_model_bundle(version, task)`, `predict_batch(X_raw, bundle)` (đi qua clean, build_features, preprocess, model).
- Lưu model bundle vào `models/credit_risk/v1.0.0/` và `models/fraud_risk/v1.0.0/` gồm: `model.txt`, `preprocessor.joblib`, `feature_list.json`, `metrics.json`, `metadata.yaml` (git hash, config hash, timestamp; nếu không có git thì ghi `unknown`).
- Đánh giá theo phân khúc (`employment_status`, `age_bin`, `income_bin`) trên valid.
- Đánh giá trên test **đúng một lần** khi chốt model: lệnh `python -m scripts.run_training --final` ghi `test_metrics.json`; nếu file đã tồn tại thì từ chối chạy lại trừ khi có `--force`.

### Điều kiện hoàn thành

- Model chính thắng baseline có ý nghĩa trên valid.
- Có bảng metric theo từng phân khúc.
- Model bundle nạp lại cho dự đoán giống hệt.
- Nếu điểm cao bất thường: đã kiểm tra feature importance, thử bỏ nhóm feature chính, và ghi chú "data synthetic" vào README.

---

## Giai đoạn 7: Ngưỡng, giải thích, fairness (3 đến 4 ngày)

### Code cần viết

- `src/evaluation/threshold.py`:
  - Credit: `find_qwk_thresholds(y_true, y_proba, n_classes)`: tính điểm kỳ vọng `score = Σ k · p_k` từ xác suất dự đoán, rồi tìm `n_classes - 1` ngưỡng cắt tối đa QWK trên valid; so sánh với cách lấy `argmax`.
  - Fraud: `top_k_threshold(scores, k=0.05)`: ngưỡng điểm tại phân vị 95% của điểm dự đoán trên valid, kèm precision/recall tại top-k theo định nghĩa ở giai đoạn 5.
- `src/explainability/shap_explain.py`: SHAP `TreeExplainer`, summary toàn cục và waterfall cho vài ca; **tính trên mẫu tối đa 10.000 dòng** để chạy nhanh.
- `src/evaluation/fairness.py`: `compute_fairness_metrics(y_true, y_pred, sensitive_cols)` dùng `fairlearn.MetricFrame`; nhóm theo `age_bin`, `marital_status`, `region`, `education_level`; so sánh model huấn luyện **có và không có** tuổi và hôn nhân (huấn luyện lại bằng `features.exclude`, lưu vào `models/<task>/experiments/no_sensitive/`).
- `scripts/run_fairness_report.py`: xuất `reports/fairness_report.html` (có banner synthetic).

### Điều kiện hoàn thành

- Ngưỡng chọn trên valid, đánh giá lại trên test (không tune lại trên test), có ghi lý do.
- Có biểu đồ SHAP.
- Có bảng fairness.
- README có kết luận về việc dùng các biến nhạy cảm.

---

## Giai đoạn 8: Đóng gói pipeline (3 đến 4 ngày)

### Code cần viết

- `src/pipelines/training_pipeline.py`: nối load, validate, clean, split, build_features, preprocess, train, evaluate, save.
- `src/pipelines/inference_pipeline.py`: nạp model bundle, dự đoán từ dữ liệu thô (dùng đúng clean, build_features, preprocess đã fit).
- `scripts/run_training.py` (chạy toàn bộ training pipeline cho cả hai task, có cờ `--final` và `--force`), `scripts/run_batch_inference.py`, `scripts/run_eda.py` (xuất `reports/eda_report.html` từ mẫu 50k dòng, không dùng ydata-profiling trên toàn bộ).
  Các script `run_data_quality.py` và `run_feature_engineering.py` đã có từ giai đoạn 2 và 4, không viết lại.
- Hoàn thiện mục "Lệnh thường dùng" trong README (thay cho Makefile), gồm tối thiểu:
  - `pip install -r requirements.txt`
  - `ruff check src/ scripts/ tests/ api/` và `python -m pytest tests -v`
  - `python -m scripts.run_data_quality`
  - `python -m scripts.run_feature_engineering`
  - `python -m scripts.run_training` rồi `python -m scripts.run_training --final`
  - `python -m scripts.run_fairness_report`
  - `python -m scripts.run_batch_inference`
  - `python -m scripts.run_eda`
  - `uvicorn api.app:app --reload`
- `tests/integration/test_training_pipeline.py`, `tests/integration/test_inference_pipeline.py`: chạy trên `sample.csv`.

### Điều kiện hoàn thành

- Clone mới, cài `requirements.txt`, chạy lần lượt các lệnh trong README ra số khớp với số trong README (cùng seed, cùng phiên bản thư viện đã pin; sai lệch metric rất nhỏ là chấp nhận được).
- Integration test xanh.
- Batch inference chạy được trên dữ liệu thô mới.

---

## Giai đoạn 9: API và Docker (2 đến 3 ngày)

### Code cần viết

- `api/__init__.py`, `api/schemas.py`: `CustomerFeatures` (Pydantic, các cột thô cần thiết; miền hợp lệ lấy từ cùng nguồn với `data_quality.yaml`), `CreditPrediction` (lớp và xác suất), `FraudPrediction` (điểm và cờ).
- `api/app.py`: FastAPI với `/health`, `/predict/credit`, `/predict/fraud`, `/predict/both`; nạp model bundle một lần khi khởi động từ thư mục chỉ định bởi biến môi trường `MODEL_DIR` (mặc định `models/`). Nếu chưa có model thì khởi động được nhưng `/health` báo `model_not_loaded` và `/predict/*` trả lỗi rõ ràng.
- `.dockerignore` (ở gốc repo): loại `data/`, `models/`, `mlruns/`, `reports/`, `notebooks/`, `tests/`, `.git/`, `.venv/`.
- `docker/Dockerfile.api`: multi-stage, `python:3.13-slim`, cài `libgomp1` (LightGBM cần). **Image chỉ chứa code chính**: `src/`, `api/`, `configs/` và `requirements.txt`. **Không chứa model, data, báo cáo hay notebook.**
- `docker/docker-compose.yml`: service `api`, build context là gốc repo; model được đưa vào **lúc chạy** bằng volume chỉ đọc (`../models:/app/models:ro`) hoặc trỏ `MODEL_DIR`, không nằm trong image. Phải chạy `python -m scripts.run_training` trước để có model.
- Thêm `api/` vào lệnh `ruff check` ở CI.
- Test API: gọi `/predict/*` với 1 dòng từ sample bằng `TestClient`.

### Điều kiện hoàn thành

- `docker build` thành công và `docker run --rm <image> ls /app` không thấy `models/` hay `data/`.
- Chạy `docker compose -f docker/docker-compose.yml up` (từ gốc repo, đã có model) thì `/health` trả OK.
- `/predict` trả đúng định dạng và khớp với kết quả từ inference pipeline.
- Đầu vào sai thì trả 422 kèm thông báo rõ ràng.

---

## Giai đoạn 10: Hoàn thiện tài liệu (2 đến 3 ngày)

### File cần viết

- `README.md` đầy đủ: business context, dataset (nói rõ synthetic), bảng metric (lấy số từ `metrics.json`), sơ đồ pipeline, cách chạy, bài học, hạn chế. Ghi rõ cách chia 70/15/15 và vai trò của train/valid/test.
- `docs/model_card_credit.md`, `docs/model_card_fraud.md`: mục đích, dữ liệu huấn luyện, metric, fairness, hạn chế, **ghi chú dữ liệu synthetic**.
- Dọn repo: xóa thư mục rỗng, file không dùng.

### Điều kiện hoàn thành

- Người chưa biết dự án đọc README 5 phút hiểu bài toán và kết quả.
- Làm theo hướng dẫn ra đúng số.
- Model card đầy đủ.
- "Synthetic" xuất hiện đủ ở README, hai model card, và đầu mọi report HTML.
