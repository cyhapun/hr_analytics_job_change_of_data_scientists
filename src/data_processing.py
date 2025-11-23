import numpy as np
from typing import Dict, List, Tuple

MISSING_TOKENS = {
    "",
    "na", "n/a", "nan", "<na>",
    "null", "none", "missing", "?",
}


def load_hr_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc file CSV HR Analytics thành (header, data_raw).

    - header: mảng 1D các tên cột (dtype=str)
    - data_raw: mảng 2D (dtype=str), chưa xử lý missing
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")
    header = np.array(first_line.split(","), dtype=str)

    data_raw = np.genfromtxt(
        path,
        delimiter=",",
        dtype=str,
        encoding="utf-8",
        skip_header=1,
    )
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape(1, -1)
    return header, data_raw


def normalize_missing_values(data: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa các giá trị missing về chuỗi rỗng ''.

    Input: data dtype=str (đọc từ CSV)
    Output: bản copy đã thay missing → ''
    """
    out = data.astype(str).copy()
    n_rows, n_cols = out.shape

    for i in range(n_rows):
        for j in range(n_cols):
            val = out[i, j]
            v = val.strip()
            if v == "" or v.lower() in MISSING_TOKENS:
                out[i, j] = ""
    return out


def is_missing_str(x: str) -> bool:
    if not isinstance(x, str):
        x = str(x)
    v = x.strip()
    return (v == "") or (v.lower() in MISSING_TOKENS)

def to_float_array(col: np.ndarray) -> np.ndarray:
    """
    Chuyển 1 cột string sang float, giá trị không hợp lệ → np.nan.
    """
    out = np.full(col.shape[0], np.nan, dtype=float)
    for i, v in enumerate(col):
        if is_missing_str(v):
            continue
        try:
            out[i] = float(v)
        except ValueError:
            # giá trị không parse được ⇒ xem như missing
            continue
    return out


def parse_experience_column(col: np.ndarray) -> np.ndarray:
    """
    Encode cột 'experience' dạng text:
        '<1'  → 0
        '1'..'20' → số năm tương ứng
        '>20' → 21
    Không parse được / missing → np.nan
    """
    out = np.full(col.shape[0], np.nan, dtype=float)
    for i, raw in enumerate(col):
        if is_missing_str(raw):
            continue
        s = raw.strip()
        if s.startswith("<"):
            out[i] = 0.0
        elif s.startswith(">"):
            out[i] = 21.0
        else:
            try:
                out[i] = float(s)
            except ValueError:
                # giá trị lạ → bỏ qua
                continue
    return out


def parse_last_new_job_column(col: np.ndarray) -> np.ndarray:
    """
    Encode cột 'last_new_job':
        'never' → 0
        '1','2','3','4' → 1..4
        '>4' → 5
    Các giá trị khác / missing → np.nan
    """
    mapping = {
        "never": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        ">4": 5,
    }
    out = np.full(col.shape[0], np.nan, dtype=float)
    for i, raw in enumerate(col):
        if is_missing_str(raw):
            continue
        key = raw.strip()
        if key in mapping:
            out[i] = float(mapping[key])
    return out


def encode_company_size_ordinal(col: np.ndarray) -> np.ndarray:
    """
    Encode 'company_size' theo thứ tự tăng dần quy mô.
    """
    order = [
        "<10",
        "10/49",
        "50-99",
        "100-500",
        "500-999",
        "1000-4999",
        "5000-9999",
        "10000+",
    ]
    index = {val: i + 1 for i, val in enumerate(order)}  # 1..8
    out = np.full(col.shape[0], np.nan, dtype=float)
    for i, raw in enumerate(col):
        if is_missing_str(raw):
            continue
        key = raw.strip()
        if key in index:
            out[i] = float(index[key])
    return out

def impute_numeric_median(col: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Điền missing cho biến số bằng median.
    Trả về: (col_đã_điền, median)
    """
    x = col.astype(float)
    mask_valid = ~np.isnan(x)
    if not np.any(mask_valid):
        # toàn missing → trả về 0
        return np.zeros_like(x), 0.0
    median_val = float(np.median(x[mask_valid]))
    x[~mask_valid] = median_val
    return x, median_val


def impute_categorical_mode(col: np.ndarray, unknown_token: str = "Unknown") -> Tuple[np.ndarray, str]:
    """
    Điền missing cho biến categorical bằng mode.
    Nếu không có giá trị nào → dùng 'Unknown'.
    """
    x = col.astype(str).copy()
    valid = np.array([v for v in x if not is_missing_str(v)], dtype=str)
    if valid.size == 0:
        fill = unknown_token
    else:
        uniq, counts = np.unique(valid, return_counts=True)
        fill = uniq[np.argmax(counts)]
    for i, v in enumerate(x):
        if is_missing_str(v):
            x[i] = fill
    return x, fill

def zscore_scale(col: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Z-score: (x - mean) / std.
    Trả về (col_scaled, mean, std).
    """
    x = col.astype(float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std == 0.0:
        return np.zeros_like(x), mean, std
    return (x - mean) / std, mean, std


def minmax_scale(col: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Min-Max: (x - min) / (max - min) trong [0, 1].
    """
    x = col.astype(float)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx == mn:
        return np.zeros_like(x), mn, mx
    return (x - mn) / (mx - mn), mn, mx

def one_hot_encode_strings(col: np.ndarray, prefix: str = "") -> Tuple[np.ndarray, List[str]]:
    """
    One-hot encode 1 cột string.
    Trả về:
        - encoded: shape (n_samples, n_unique)
        - feature_names: list tên feature sau khi encode
    """
    x = col.astype(str)
    categories = np.unique(x)
    n = x.shape[0]
    k = categories.shape[0]

    encoded = np.zeros((n, k), dtype=float)
    for j, cat in enumerate(categories):
        encoded[:, j] = (x == cat).astype(float)

    feature_names = [
        f"{prefix}{cat}" if prefix else str(cat)
        for cat in categories
    ]
    return encoded, feature_names

def build_hr_feature_matrix(
    header: np.ndarray,
    data_norm: np.ndarray,
    scale_numeric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Xây dựng ma trận đặc trưng X và vector nhãn y cho dataset HR.

    Các bước chính:
        - Chọn/encode các cột numeric & ordinal
        - Điền missing
        - Chuẩn hóa (Z-score) cho numeric
        - One-hot encode một số cột categorical
        - Ghép tất cả lại thành 1 ma trận NumPy

    Returns:
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,)
        feature_names: list[str]
        meta: dict – lưu thông tin median, mean, std, mappings,...
    """
    col_index = {name: i for i, name in enumerate(header)}

    # --- Lấy cột target ---
    y_raw = data_norm[:, col_index["target"]]
    y = np.array([int(v) for v in y_raw], dtype=float)

    # --- Numeric / ordinal gốc (chưa impute) ---
    cdi = to_float_array(data_norm[:, col_index["city_development_index"]])
    training = to_float_array(data_norm[:, col_index["training_hours"]])
    exp_years = parse_experience_column(data_norm[:, col_index["experience"]])
    last_nj = parse_last_new_job_column(data_norm[:, col_index["last_new_job"]])
    comp_size_ord = encode_company_size_ordinal(data_norm[:, col_index["company_size"]])

    numeric_dict = {
        "city_development_index": cdi,
        "training_hours": training,
        "experience_years": exp_years,
        "last_new_job_num": last_nj,
        "company_size_ord": comp_size_ord,
    }

    numeric_imputed = {}
    numeric_stats = {}

    for name, col in numeric_dict.items():
        filled, median_val = impute_numeric_median(col)
        if scale_numeric:
            scaled, mean_val, std_val = zscore_scale(filled)
            numeric_imputed[name] = scaled
            numeric_stats[name] = {
                "median": median_val,
                "mean": mean_val,
                "std": std_val,
                "scaled": True,
            }
        else:
            numeric_imputed[name] = filled
            numeric_stats[name] = {
                "median": median_val,
                "scaled": False,
            }

    # --- Categorical: fill missing + one-hot ---
    cat_cols = [
        "gender",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "company_type",
        "city",
    ]

    cat_encoded_blocks = []
    cat_feature_names: List[str] = []
    cat_impute_values = {}

    for col_name in cat_cols:
        raw_col = data_norm[:, col_index[col_name]]
        filled_col, fill_val = impute_categorical_mode(raw_col)
        cat_impute_values[col_name] = fill_val

        block, names_block = one_hot_encode_strings(
            filled_col,
            prefix=f"{col_name}=",
        )
        cat_encoded_blocks.append(block)
        cat_feature_names.extend(names_block)

    # --- Ghép tất cả special numeric + categorical ---
    X_numeric = np.column_stack([numeric_imputed[k] for k in numeric_imputed.keys()])
    numeric_feature_names = list(numeric_imputed.keys())

    if cat_encoded_blocks:
        X_categorical = np.column_stack(cat_encoded_blocks)
        X = np.column_stack([X_numeric, X_categorical])
        feature_names = numeric_feature_names + cat_feature_names
    else:
        X = X_numeric
        feature_names = numeric_feature_names

    meta = {
        "numeric_features": numeric_feature_names,
        "numeric_stats": numeric_stats,
        "categorical_features": cat_cols,
        "categorical_fill_values": cat_impute_values,
        "feature_names": feature_names,
    }

    return X, y, feature_names, meta

def preprocess_hr_dataset(
    csv_path: str,
    scale_numeric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Pipeline tiện dụng:
        - Đọc CSV
        - Chuẩn hóa missing
        - Build ma trận X, y

    Dùng trong notebook:
        header, raw = load_hr_csv(...)
        data_norm = normalize_missing_values(raw)
        X, y, feature_names, meta = build_hr_feature_matrix(header, data_norm)
    Hoặc đơn giản hơn:
        X, y, feature_names, meta = preprocess_hr_dataset(csv_path)
    """
    header, raw = load_hr_csv(csv_path)
    data_norm = normalize_missing_values(raw)
    X, y, feature_names, meta = build_hr_feature_matrix(
        header,
        data_norm,
        scale_numeric=scale_numeric,
    )
    return X, y, feature_names, meta
