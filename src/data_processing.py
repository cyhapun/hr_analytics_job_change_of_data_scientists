import numpy as np

# Các hằng số & thứ tự ordinal cho một số cột quan trọng
MISSING_TOKENS = (
    "",
    " ",
    "nan",
    "NaN",
    "NA",
    "N/A",
    "<NA>",
    "null",
    "None",
    "none",
    "missing",
    "?",
)

EXPERIENCE_ORDER = [
    "<1",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16",
    "17", "18", "19", "20",
    ">20",
]

COMPANY_SIZE_ORDER = [
    "<10",
    "10-49",
    "50-99",
    "100-500",
    "500-999",
    "1000-4999",
    "5000-9999",
    "10000+",
]

LAST_NEW_JOB_ORDER = [
    "never",
    "1",
    "2",
    "3",
    "4",
    ">4",
]

EDUCATION_LEVEL_ORDER = [
    "Primary School",
    "High School",
    "Graduate",
    "Masters",
    "Phd",
]

ENROLLED_UNI_ORDER = [
    "no_enrollment",
    "Part time course",
    "Full time course",
]


# ==============================
# Các hàm tiện ích chung
def load_csv_as_str(path, delimiter=",", has_header=True, encoding="utf-8"):
    """
    Đọc CSV → (header, data) với data dạng np.ndarray[str].
    Dùng genfromtxt cho tốc độ, vẫn hỗ trợ có/không header.
    """
    if has_header:
        with open(path, "r", encoding=encoding) as f:
            header_line = f.readline().strip()
        header = [h.strip() for h in header_line.split(delimiter)]
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=str,
            autostrip=True,
            skip_header=1,
        )
    else:
        header = None
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=str,
            autostrip=True,
        )

    # Trường hợp chỉ 1 dòng → genfromtxt trả 1D, reshape lại về (n_rows, n_cols)
    if data.ndim == 1:
        data = data.reshape(-1, data.shape[0])

    return header, data.astype(str)

def _normalize_str_array(arr):
    """
    Chuẩn hoá mảng str:
    - Ép sang str
    - strip khoảng trắng
    - lower-case
    → thuận tiện cho so sánh & nhận diện missing.
    """
    arr = arr.astype(str)
    return np.char.strip(np.char.lower(arr))

def build_missing_mask(
    data,
    missing_tokens=MISSING_TOKENS,
    extra_tokens=None,
):
    """
    Xây mask missing cho mảng data (có thể 1D hoặc 2D) dựa trên:
    - MISSING_TOKENS mặc định
    - extra_tokens bổ sung (nếu có)
    Dùng np.isin để vectorized (KHÔNG loop từng phần tử data).
    """
    data = np.asarray(data, dtype=str)
    normalized = _normalize_str_array(data)

    tokens = set(missing_tokens)
    if extra_tokens is not None:
        tokens.update(extra_tokens)

    tokens_norm = [tk.strip().lower() for tk in tokens]
    tokens_norm = np.array(tokens_norm, dtype=str)

    mask = np.isin(normalized, tokens_norm)
    return mask

def summarize_missing(data, header=None, extra_tokens=None):
    """
    In thống kê số lượng & tỷ lệ missing theo từng cột.
    Trả về (counts, ratios).
    """
    miss_mask = build_missing_mask(data, extra_tokens=extra_tokens)
    counts = miss_mask.sum(axis=0)
    ratios = counts / data.shape[0]

    if header is not None:
        print("Missing summary theo column:")
        for name, c, r in zip(header, counts, ratios):
            print(f"{name:25s} | {int(c):6d} missing ({r:6.2%})")

    return counts, ratios

# ==============================
# Impute & chuyển đổi numeric/categorical
def string_column_to_float(col, missing_tokens=MISSING_TOKENS):
    """
    Chuyển một cột str → float, các giá trị thuộc missing_tokens → np.nan.
    Dùng build_missing_mask để vectorized.
    """
    col = np.asarray(col, dtype=str)
    mask_missing = build_missing_mask(col, missing_tokens=missing_tokens)

    tmp = col.copy()
    tmp[mask_missing] = "nan"
    # np.array(["nan", ...]).astype(float) → np.nan
    return tmp.astype(float)

def impute_numeric(col, strategy="constant", missing_tokens=MISSING_TOKENS):
    """
    Impute cho cột numeric (đang ở dạng string):
    - strategy='mean'   → điền mean
    - strategy='median' → điền median
    - strategy='constant' → điền 0.0
    Trả về (x_imputed, fill_value).
    """
    x = string_column_to_float(col, missing_tokens=missing_tokens)
    mask_nan = np.isnan(x)

    if strategy == "mean":
        fill_value = np.nanmean(x)
    elif strategy == "median":
        fill_value = np.nanmedian(x)
    elif strategy == "constant":
        fill_value = 0.0
    else:
        raise ValueError("strategy must be 'mean', 'median' or 'constant'")

    x[mask_nan] = fill_value
    return x, float(fill_value)

def impute_categorical(
    col,
    strategy="mode",
    constant_value="Unknown",
    missing_tokens=MISSING_TOKENS,
):
    """
    Impute cho cột categorical:
    - strategy='mode'    → điền giá trị xuất hiện nhiều nhất (mode)
    - strategy='constant'→ điền constant_value
    """
    col = np.asarray(col, dtype=str)
    norm = _normalize_str_array(col)

    mask_missing = build_missing_mask(norm, missing_tokens=missing_tokens)

    if np.all(mask_missing):
        fill = constant_value
    else:
        if strategy == "mode":
            valid = col[~mask_missing]
            values, counts = np.unique(valid, return_counts=True)
            fill = values[np.argmax(counts)]
        elif strategy == "constant":
            fill = constant_value
        else:
            raise ValueError("strategy must be 'mode' hoặc 'constant'")

    col_filled = col.copy()
    col_filled[mask_missing] = fill
    return col_filled, fill

def knn_impute_categorical(
    target_col,
    feature_matrix,
    k=5,
    missing_tokens=MISSING_TOKENS,
):
    """
    KNN-impute cho cột categorical:
    - Chuẩn hoá feature_matrix (z-score).
    - Với mỗi dòng missing trong target_col, tìm k hàng gần nhất (theo L2)
      rồi chọn mode của target_col các hàng này.
    Lưu ý: ở đây vẫn phải loop trên từng missing-row (bản chất thuật toán KNN).
    """
    target_col = np.asarray(target_col, dtype=str)
    norm = _normalize_str_array(target_col)

    miss_mask = build_missing_mask(norm, missing_tokens=missing_tokens)

    if not np.any(miss_mask):
        return target_col.copy()

    X = np.asarray(feature_matrix, dtype=float)
    # Chuẩn hoá z-score cho mỗi feature
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_std = (X - mean) / std

    col_filled = target_col.copy()
    idx_missing = np.where(miss_mask)[0]

    for idx in idx_missing:
        diff = X_std - X_std[idx]          # (n_features,)
        # Dùng np.einsum cho tính khoảng cách L2
        dist = np.einsum("ij,ij->i", diff, diff)
        dist[idx] = np.inf  # bỏ chính nó

        neighbor_idx = np.argpartition(dist, k)[:k]
        neighbor_vals = target_col[neighbor_idx]

        # Bỏ các hàng hàng xóm rỗng/missing
        valid_neighbors = neighbor_vals[
            ~build_missing_mask(neighbor_vals, missing_tokens=missing_tokens)
        ]

        if valid_neighbors.size == 0:
            col_filled[idx] = "Unknown"
        else:
            vals, counts = np.unique(valid_neighbors, return_counts=True)
            col_filled[idx] = vals[np.argmax(counts)]

    return col_filled

def kmeans_impute_city_training(
    data,
    col_idx_city,
    col_idx_train,
    k=3,
    max_iter=50,
    tol=1e-4,
    random_state=23120329,
):
    """
    K-means impute đồng thời 2 cột numeric: city_development_index & training_hours.
    - Chạy K-means trên các dòng đầy đủ 2 cột này.
    - Dùng centroid của cluster để điền cho các dòng có missing:
      + missing cả 2: dùng centroid của cluster đông nhất.
      + missing 1: gán cluster theo chiều còn lại, lấy giá trị từ centroid.
    Trả về (city_filled, training_filled, centroids).
    """
    rng = np.random.default_rng(random_state)

    city = string_column_to_float(data[:, col_idx_city])
    train = string_column_to_float(data[:, col_idx_train])
    n = city.shape[0]

    X = np.stack([city, train], axis=1)

    mask_city_nan = np.isnan(city)
    mask_train_nan = np.isnan(train)
    mask_any_nan = mask_city_nan | mask_train_nan
    mask_full = ~mask_any_nan

    # Nếu dữ liệu đầy đủ quá ít thì fallback về impute đơn giản
    if mask_full.sum() < max(k, 3):
        city_imp, _ = impute_numeric(data[:, col_idx_city], strategy="constant")
        train_imp, _ = impute_numeric(data[:, col_idx_train], strategy="constant")
        mean_city = float(np.mean(city_imp))
        mean_train = float(np.mean(train_imp))
        centroids = np.array([[mean_city, mean_train]])
        return city_imp, train_imp, centroids

    X_full = X[mask_full]
    m = X_full.shape[0]

    if k > m:
        k = m
    init_idx = rng.choice(m, size=k, replace=False)
    centroids = X_full[init_idx].copy()

    # Vòng lặp K-means (vectorized theo toàn bộ X_full)
    for _ in range(max_iter):
        # diff: (m, k, 2), dists: (m, k)
        diff = X_full[:, None, :] - centroids[None, :, :]
        dists = np.einsum("ijk,ijk->ij", diff, diff)
        labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            members = X_full[labels == j]
            if members.shape[0] > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                # Nếu cluster rỗng, random 1 điểm làm centroid mới
                ridx = rng.integers(0, m)
                new_centroids[j] = X_full[ridx]

        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids
        if shift < tol:
            break

    counts = np.bincount(labels, minlength=k)
    major_cluster = int(np.argmax(counts))

    city_filled = city.copy()
    training_filled = train.copy()

    # Điền lại cho tất cả các hàng có missing (loop theo bản ghi)
    for i in range(n):
        c_nan = mask_city_nan[i]
        t_nan = mask_train_nan[i]

        if not c_nan and not t_nan:
            continue

        if c_nan and t_nan:
            # Thiếu cả 2 → gán cluster đông nhất
            cluster = major_cluster
            city_filled[i] = centroids[cluster, 0]
            training_filled[i] = centroids[cluster, 1]
        else:
            if c_nan and not t_nan:
                known_val = training_filled[i]
                d1 = (centroids[:, 1] - known_val) ** 2
                cluster = int(np.argmin(d1))
                city_filled[i] = centroids[cluster, 0]
            elif t_nan and not c_nan:
                known_val = city_filled[i]
                d0 = (centroids[:, 0] - known_val) ** 2
                cluster = int(np.argmin(d0))
                training_filled[i] = centroids[cluster, 1]

    return city_filled, training_filled, centroids


# ==============================
# Xử lý outlier & scale / transform numeric
def iqr_bounds(x, whisker=1.5):
    """
    Tính ngưỡng dưới/trên dựa trên IQR (Q1, Q3).
    """
    x = np.asarray(x, dtype=float)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    return lower, upper

def detect_outliers_iqr(x, whisker=1.5):
    """
    Trả về mask outlier theo IQR.
    """
    x = np.asarray(x, dtype=float)
    low, up = iqr_bounds(x, whisker=whisker)
    return (x < low) | (x > up)

def clip_outliers_iqr(x, whisker=1.5):
    """
    Cắt (clip) các giá trị nằm ngoài [low, up] (IQR).
    """
    x = np.asarray(x, dtype=float)
    low, up = iqr_bounds(x, whisker=whisker)
    return np.clip(x, low, up)

def clip_outliers_zscore(x, threshold=3.0):
    """
    Cắt outlier theo z-score (|x - mean| > threshold * std).
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std()
    if std == 0:
        return x.copy()
    low = mean - threshold * std
    up = mean + threshold * std
    return np.clip(x, low, up)

def min_max_scale(x, feature_range=(0.0, 1.0)):
    """
    Min-Max scaling: map x → [a, b].
    Trả về (x_scaled, min_val, max_val).
    """
    x = np.asarray(x, dtype=float)
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.zeros_like(x), float(min_val), float(max_val)
    a, b = feature_range
    scaled = (x - min_val) / (max_val - min_val)
    scaled = a + scaled * (b - a)
    return scaled, float(min_val), float(max_val)

def log_transform(x, eps=1e-8):
    """
    Biến đổi log(x), với bảo vệ x<=0 bằng cách thay bằng eps.
    """
    x = np.asarray(x, dtype=float)
    x = x.copy()
    x[x <= 0] = eps
    return np.log(x)

def decimal_scaling(x):
    """
    Decimal scaling: chia cho 10^j sao cho |x_scaled| < 1.
    Trả về (x_scaled, j).
    """
    x = np.asarray(x, dtype=float)
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x.copy(), 0
    j = int(np.ceil(np.log10(max_abs)))
    scaled = x / (10 ** j)
    return scaled, j

def zscore_standardize(x):
    """
    Chuẩn hoá z-score: (x - mean) / std.
    Trả về (x_std, mean, std).
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std()
    if std == 0:
        return np.zeros_like(x), float(mean), float(std)
    return (x - mean) / std, float(mean), float(std)


# ==============================
# Encoding categorical
def ordinal_encode(col, ordered_values):
    """
    Ordinal encoding vectorized:
    - Mỗi category trong ordered_values được gán index 0..len-1.
    - Category không nằm trong ordered_values → -1.
    Trả về (encoded, lookup_dict).
    """
    col = np.asarray(col, dtype=str)
    norm = _normalize_str_array(col)

    ordered_values = list(ordered_values)
    cats_norm = _normalize_str_array(np.array(ordered_values, dtype=str))

    # Broadcast so sánh toàn bộ: (n,1) == (1,m) → (n,m)
    norm_expanded = norm.reshape(-1, 1)
    cats_expanded = cats_norm.reshape(1, -1)
    match = (norm_expanded == cats_expanded)

    # Index của category: nếu không có match → -1
    has_match = match.any(axis=1)
    indices = np.full(norm.shape[0], -1, dtype=float)
    indices[has_match] = match.argmax(axis=1)[has_match].astype(float)

    lookup = {v: i for i, v in enumerate(ordered_values)}
    return indices, lookup

def one_hot_encode(col, categories=None, drop_first=False):
    """
    One-hot encode vectorized cho 1 cột categorical:
    - Nếu categories=None, dùng tất cả unique.
    - drop_first=True: bỏ category đầu làm baseline.
    Trả về (encoded_matrix, used_categories).
    """
    col = np.asarray(col, dtype=str)

    if categories is None:
        categories = np.unique(col)
    categories = list(categories)

    if drop_first and len(categories) > 1:
        baseline = categories[0]
        used_categories = categories[1:]
    else:
        baseline = None  # chỉ để tham khảo, không dùng
        used_categories = categories

    if len(used_categories) == 0:
        encoded = np.zeros((col.shape[0], 0), dtype=float)
        return encoded, used_categories

    # Broadcast so sánh: (n,1) == (1,k) → (n,k)
    col_expanded = col.reshape(-1, 1)
    cats_arr = np.array(used_categories).reshape(1, -1)
    encoded = (col_expanded == cats_arr).astype(float)

    return encoded, used_categories

def experience_to_numeric(col):
    """
    Clean chuỗi 'experience' → numeric index dựa trên EXPERIENCE_ORDER.
    - Dùng ordinal_encode vectorized.
    - Giá trị không hợp lệ hoặc missing → np.nan.
    """
    col = np.asarray(col, dtype=str)
    norm = _normalize_str_array(col)

    # Missing → np.nan
    mask_missing = build_missing_mask(norm, missing_tokens=MISSING_TOKENS)

    encoded, _ = ordinal_encode(col, EXPERIENCE_ORDER)
    # encoded unknown = -1 → np.nan
    encoded[(encoded < 0) | mask_missing] = np.nan
    return encoded

def last_new_job_to_numeric(col):
    """
    Clean chuỗi 'last_new_job' → numeric index dựa trên LAST_NEW_JOB_ORDER.
    - Dùng ordinal_encode vectorized.
    - Giá trị không hợp lệ hoặc missing → np.nan.
    """
    col = np.asarray(col, dtype=str)
    norm = _normalize_str_array(col)

    mask_missing = build_missing_mask(norm, missing_tokens=MISSING_TOKENS)

    encoded, _ = ordinal_encode(col, LAST_NEW_JOB_ORDER)
    encoded[(encoded < 0) | mask_missing] = np.nan
    return encoded

def frequency_encode(col):
    """
    Frequency encoding cho 1 cột categorical:
    - encoded[i] = tần suất xuất hiện của col[i] trong toàn bộ cột.
    Trả về (encoded, freq_map).
    """
    col = np.asarray(col, dtype=str)
    values, inv = np.unique(col, return_inverse=True)
    counts = np.bincount(inv)
    freqs = counts.astype(float) / col.shape[0]

    encoded = freqs[inv]
    freq_map = {v: f for v, f in zip(values, freqs)}
    return encoded, freq_map
