import numpy as np

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

def load_csv_as_str(path, delimiter=",", has_header=True, encoding="utf-8"):
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

    if data.ndim == 1:
        data = data.reshape(-1, data.shape[0])

    return header, data.astype(str)

def _normalize_str_array(arr):
    arr = arr.astype(str)
    return np.char.strip(np.char.lower(arr))

def build_missing_mask(data, extra_tokens=None):
    tokens = set(MISSING_TOKENS)
    if extra_tokens is not None:
        tokens.update(extra_tokens)

    normalized = _normalize_str_array(data)
    mask = np.zeros_like(normalized, dtype=bool)
    for tk in tokens:
        tk_norm = tk.strip().lower()
        mask |= (normalized == tk_norm)
    return mask

def summarize_missing(data, header=None, extra_tokens=None):
    miss_mask = build_missing_mask(data, extra_tokens=extra_tokens)
    counts = miss_mask.sum(axis=0)
    ratios = counts / data.shape[0]

    if header is not None:
        print("Missing summary theo column:")
        for name, c, r in zip(header, counts, ratios):
            print(f"{name:25s} | {c:6d} missing ({r:6.2%})")

def string_column_to_float(col, missing_tokens=MISSING_TOKENS):
    col = col.astype(str)
    normalized = _normalize_str_array(col)

    mask_missing = np.zeros_like(normalized, dtype=bool)
    for tk in missing_tokens:
        tk_norm = tk.strip().lower()
        mask_missing |= (normalized == tk_norm)

    tmp = col.copy()
    tmp[mask_missing] = "nan"
    return tmp.astype(float)

def impute_numeric(col, strategy="constant", missing_tokens=MISSING_TOKENS):
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

def impute_categorical(col, strategy="mode", constant_value="Unknown",
                       missing_tokens=MISSING_TOKENS):
    col = col.astype(str)
    normalized = _normalize_str_array(col)

    mask_missing = np.zeros_like(normalized, dtype=bool)
    for tk in missing_tokens:
        tk_norm = tk.strip().lower()
        mask_missing |= (normalized == tk_norm)

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

def knn_impute_categorical(target_col, feature_matrix, k=5,
                           missing_tokens=MISSING_TOKENS):
    target_col = target_col.astype(str)
    norm = _normalize_str_array(target_col)

    miss_mask = np.zeros_like(norm, dtype=bool)
    for tk in missing_tokens:
        miss_mask |= (norm == tk.strip().lower())

    if not np.any(miss_mask):
        return target_col.copy()

    X = np.asarray(feature_matrix, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_std = (X - mean) / std

    col_filled = target_col.copy()
    idx_missing = np.where(miss_mask)[0]

    for idx in idx_missing:
        diff = X_std - X_std[idx]
        dist = np.sum(diff * diff, axis=1)
        dist[idx] = np.inf

        neighbor_idx = np.argpartition(dist, k)[:k]
        neighbor_vals = target_col[neighbor_idx]

        valid_neighbors = neighbor_vals[
            _normalize_str_array(neighbor_vals) != ""
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
    rng = np.random.default_rng(random_state)

    city = string_column_to_float(data[:, col_idx_city])
    train = string_column_to_float(data[:, col_idx_train])
    n = city.shape[0]

    X = np.stack([city, train], axis=1)

    mask_city_nan = np.isnan(city)
    mask_train_nan = np.isnan(train)
    mask_any_nan = mask_city_nan | mask_train_nan
    mask_full = ~mask_any_nan

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

    for _ in range(max_iter):
        diff = X_full[:, None, :] - centroids[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            members = X_full[labels == j]
            if members.shape[0] > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
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

    for i in range(n):
        c_nan = mask_city_nan[i]
        t_nan = mask_train_nan[i]

        if not c_nan and not t_nan:
            continue

        if c_nan and t_nan:
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

def iqr_bounds(x, whisker=1.5):
    x = np.asarray(x, dtype=float)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    return lower, upper

def detect_outliers_iqr(x, whisker=1.5):
    x = np.asarray(x, dtype=float)
    low, up = iqr_bounds(x, whisker=whisker)
    return (x < low) | (x > up)

def clip_outliers_iqr(x, whisker=1.5):
    x = np.asarray(x, dtype=float)
    low, up = iqr_bounds(x, whisker=whisker)
    return np.clip(x, low, up)

def clip_outliers_zscore(x, threshold=3.0):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std()
    if std == 0:
        return x.copy()
    low = mean - threshold * std
    up = mean + threshold * std
    return np.clip(x, low, up)

def min_max_scale(x, feature_range=(0.0, 1.0)):
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
    x = np.asarray(x, dtype=float)
    x = x.copy()
    x[x <= 0] = eps
    return np.log(x)

def decimal_scaling(x):
    x = np.asarray(x, dtype=float)
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x.copy(), 0
    j = int(np.ceil(np.log10(max_abs)))
    scaled = x / (10 ** j)
    return scaled, j

def zscore_standardize(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std()
    if std == 0:
        return np.zeros_like(x), float(mean), float(std)
    return (x - mean) / std, float(mean), float(std)

def ordinal_encode(col, ordered_values):
    col = col.astype(str)
    lookup = {v: i for i, v in enumerate(ordered_values)}
    encoded = np.full(col.shape, -1, dtype=float)
    for i, v in enumerate(col):
        encoded[i] = lookup.get(v, -1)
    return encoded, lookup

def one_hot_encode(col, categories=None):
    col = col.astype(str)
    if categories is None:
        categories = np.unique(col)
    categories = list(categories)

    col_expanded = col.reshape(-1, 1)
    cats_arr = np.array(categories).reshape(1, -1)
    encoded = (col_expanded == cats_arr).astype(float)
    return encoded, categories