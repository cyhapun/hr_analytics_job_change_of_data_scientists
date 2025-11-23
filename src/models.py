# src/models.py
# ============================================================
# MÔ HÌNH & HÀM ĐÁNH GIÁ THUẦN NUMPY CHO BÀI TOÁN PHÂN LOẠI
# - Logistic Regression (binary) cài từ đầu bằng NumPy
# - Hàm mất mát (binary cross-entropy) + gradient descent
# - Các độ đo: accuracy, precision, recall, F1, confusion matrix
# - train/val split + K-fold cross-validation
#
# Ghi chú:
# - Tuy đề cho phép dùng scikit-learn cho modeling,
#   nhưng ở đây ta tự cài bằng NumPy để được bonus điểm. :contentReference[oaicite:1]{index=1}
# - Notebook chỉ gọi các hàm bên dưới, KHÔNG cài thuật toán trong notebook.
# ============================================================

import os
import numpy as np

# Nếu muốn load trực tiếp từ CSV sạch:
# (Giả sử em đã có load_csv_as_str trong src/data_processing.py)
try:
    from src.data_processing import load_csv_as_str
except ImportError:
    load_csv_as_str = None  # để tránh lỗi nếu chưa dùng đến

RANDOM_STATE = 23120329

# ============================================================
# 0. Hàm tiện ích: load X, y từ CSV sạch (tùy chọn)
#    Dùng khi dữ liệu đã encode/scale, chỉ còn numeric + 1 cột target.
#    Nếu em đã có build_hr_feature_matrix rồi thì có thể KHÔNG dùng phần này.
# ============================================================

def load_xy_from_clean_csv(csv_path: str, target_col: str = "target"):
    """
    Load dữ liệu từ một file CSV đã được xử lý sạch, tất cả các cột
    (trừ target) đều là numeric.

    Parameters
    ----------
    csv_path : str
        Đường dẫn tới file CSV sạch.
    target_col : str
        Tên cột chứa nhãn (mặc định "target").

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    feature_names : list[str]
        Tên các feature tương ứng với từng cột của X.

    Ghi chú
    -------
    - Hàm này CHỈ là tiện ích: nếu em đã có X, y từ chỗ khác
      (ví dụ dùng build_hr_feature_matrix) thì notebook có thể
      bỏ qua hàm này.
    """
    if load_csv_as_str is None:
        raise ImportError(
            "Không tìm thấy hàm load_csv_as_str trong src.data_processing. "
            "Nếu muốn dùng load_xy_from_clean_csv, hãy đảm bảo đã cài hàm đó."
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    header, data_str = load_csv_as_str(
        csv_path,
        delimiter=",",
        has_header=True,
        encoding="utf-8",
    )

    if target_col not in header:
        raise ValueError(f"Không tìm thấy cột target '{target_col}' trong header CSV.")
    target_idx = header.index(target_col)

    # y: cột target → float
    y_raw = data_str[:, target_idx]
    y = np.array([float(v) for v in y_raw], dtype=float)

    # X: toàn bộ cột trừ target → float
    feat_indices = [i for i in range(len(header)) if i != target_idx]
    X_raw = data_str[:, feat_indices]

    n_samples, n_features = X_raw.shape
    X = np.empty((n_samples, n_features), dtype=float)
    for j in range(n_features):
        X[:, j] = np.array([float(v) for v in X_raw[:, j]], dtype=float)

    feature_names = [header[i] for i in feat_indices]
    return X, y, feature_names


# ============================================================
# 1. CHIA TRAIN / VALIDATION
# ============================================================

def train_val_split(X, y, val_ratio=0.2, shuffle=True, random_state=RANDOM_STATE):
    """
    Chia dữ liệu thành train và validation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Ma trận đặc trưng.
    y : np.ndarray, shape (n_samples,)
        Vector nhãn (0/1).
    val_ratio : float
        Tỷ lệ mẫu dành cho validation.
    shuffle : bool
        Có xáo trộn dữ liệu trước khi chia hay không.
    random_state : int
        Seed để tái lập kết quả.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    assert X.shape[0] == y.shape[0], "Số dòng của X và y phải trùng nhau."

    n_samples = X.shape[0]
    n_val = int(np.round(n_samples * val_ratio))

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, X_val, y_train, y_val


# ============================================================
# 2. HÀM ACTIVATION & LOSS CHO LOGISTIC REGRESSION
# ============================================================

def sigmoid(z):
    """
    Hàm sigmoid ổn định số học: sigmoid(z) = 1 / (1 + exp(-z))
    - Có clip z để tránh overflow khi |z| quá lớn.
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob, l2_term=0.0):
    """
    Hàm mất mát binary cross-entropy cho phân loại nhị phân.

    Parameters
    ----------
    y_true : (n_samples,)
    y_prob : (n_samples,) xác suất dự đoán P(y=1)
    l2_term : float
        Thành phần regularization (đã chia batch nếu cần).

    Returns
    -------
    loss : float
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    ce = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    loss = np.mean(ce) + l2_term
    return loss


# ============================================================
# 3. LOGISTIC REGRESSION NHỊ PHÂN (NUMPY THUẦN)
# ============================================================

class LogisticRegressionBinary:
    """
    Logistic Regression cho bài toán phân loại nhị phân (0/1).

    - Dùng gradient descent full-batch.
    - Hỗ trợ L2 regularization.
    - Thuần NumPy, không dùng scikit-learn.
    - Phù hợp với bài toán HR Analytics: Job Change of Data Scientists
      và các bài toán binary khác trong đề. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        lr=0.1,
        n_epochs=1000,
        l2=0.0,
        fit_intercept=True,
        verbose=False,
        random_state=RANDOM_STATE,
    ):
        """
        Parameters
        ----------
        lr : float
            Learning rate cho gradient descent.
        n_epochs : int
            Số vòng lặp huấn luyện.
        l2 : float
            Hệ số L2 regularization (lambda).
        fit_intercept : bool
            Có học thêm bias (intercept) hay không.
        verbose : bool
            Nếu True, in loss theo từng epoch (hoặc mỗi vài epoch).
        random_state : int
            Seed khởi tạo trọng số ban đầu.
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state

        self.W = None              # vector trọng số (kể cả bias nếu có)
        self.loss_history_ = []    # để vẽ đường hội tụ trong notebook

    # ----------------------------
    # Hàm nội bộ
    # ----------------------------
    def _add_intercept(self, X):
        """
        Thêm cột 1 ở đầu ma trận X nếu fit_intercept=True.
        """
        if not self.fit_intercept:
            return X

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1), dtype=float)
        return np.concatenate([ones, X], axis=1)

    def _init_weights(self, n_features):
        """
        Khởi tạo trọng số W ngẫu nhiên nhỏ.
        """
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(loc=0.0, scale=0.01, size=(n_features,))

    # ----------------------------
    # API chính: fit / predict
    # ----------------------------
    def fit(self, X, y):
        """
        Huấn luyện mô hình Logistic Regression trên dữ liệu (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Thêm intercept nếu cần
        X_ext = self._add_intercept(X)
        n_samples, n_features_ext = X_ext.shape

        # Khởi tạo trọng số
        self._init_weights(n_features_ext)

        for epoch in range(self.n_epochs):
            # 1. Linear combination: z = X_ext @ W
            z = X_ext @ self.W              # shape (n_samples,)
            # 2. Xác suất dự đoán
            y_prob = sigmoid(z)             # shape (n_samples,)

            # 3. Gradient của loss (không tính regularization)
            # error = y_hat - y
            error = y_prob - y              # shape (n_samples,)
            grad = (X_ext.T @ error) / n_samples

            # 4. Thêm thành phần L2 (không regularize bias)
            if self.l2 > 0:
                if self.fit_intercept:
                    w_reg = self.W.copy()
                    w_reg[0] = 0.0  # không phạt bias
                else:
                    w_reg = self.W
                grad += (self.l2 / n_samples) * w_reg

            # 5. Cập nhật trọng số
            self.W -= self.lr * grad

            # 6. Lưu loss (cho notebook vẽ)
            if self.verbose or epoch == self.n_epochs - 1:
                reg_term = 0.0
                if self.l2 > 0:
                    if self.fit_intercept:
                        w_reg_loss = self.W[1:]  # bỏ bias
                    else:
                        w_reg_loss = self.W
                    reg_term = (self.l2 / (2 * n_samples)) * np.sum(w_reg_loss ** 2)

                loss = binary_cross_entropy(y, y_prob, l2_term=reg_term)
                self.loss_history_.append(loss)

                if self.verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                    print(f"[Epoch {epoch:4d}] Loss = {loss:.6f}")

        return self

    def predict_proba(self, X):
        """
        Trả về xác suất P(y=1 | x) cho từng mẫu.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_prob : np.ndarray, shape (n_samples,)
        """
        if self.W is None:
            raise ValueError("Model chưa được fit. Hãy gọi .fit(X, y) trước.")

        X = np.asarray(X, dtype=float)
        X_ext = self._add_intercept(X)
        z = X_ext @ self.W
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Dự đoán nhãn 0/1 dựa trên threshold.
        """
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)


# ============================================================
# 4. METRICS & CONFUSION MATRIX
# ============================================================

def accuracy_score(y_true, y_pred):
    """
    Accuracy = số mẫu dự đoán đúng / tổng số mẫu.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    assert y_true.shape == y_pred.shape
    return np.mean(y_true == y_pred)


def confusion_matrix_binary(y_true, y_pred, positive_label=1):
    """
    Confusion matrix cho bài toán nhị phân:

        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

    pos = positive_label
    neg = 1 - pos

    TP = np.sum((y_true == pos) & (y_pred == pos))
    TN = np.sum((y_true == neg) & (y_pred == neg))
    FP = np.sum((y_true == neg) & (y_pred == pos))
    FN = np.sum((y_true == pos) & (y_pred == neg))

    return np.array([[TN, FP],
                     [FN, TP]], dtype=int)


def precision_recall_f1(y_true, y_pred, positive_label=1):
    """
    Tính precision, recall, F1 cho lớp positive_label.
    """
    cm = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
    TN, FP, FN, TP = cm.ravel()

    precision = 0.0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0.0 if (TP + FN) == 0 else TP / (TP + FN)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_binary_classification(y_true, y_prob, threshold=0.5, positive_label=1):
    """
    Hàm "tổng hợp" cho phân loại nhị phân:

    - Chuyển xác suất thành nhãn với threshold.
    - Tính Accuracy, Precision, Recall, F1, Confusion matrix.

    Trả về dict:
        {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "confusion_matrix": np.array([[TN, FP],[FN, TP]]),
            "threshold": threshold,
        }
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred, positive_label=positive_label)
    cm = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
    }
    return metrics


# ============================================================
# 5. K-FOLD CROSS-VALIDATION CHO LOGISTIC REGRESSION
# ============================================================

def k_fold_indices(n_samples, k=5, shuffle=True, random_state=RANDOM_STATE):
    """
    Tạo index cho K-fold cross-validation.

    Returns
    -------
    folds : list of (train_idx, val_idx)
    """
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1  # chia đều phần dư

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop

    return folds


def cross_val_logistic(
    X,
    y,
    k=5,
    lr=0.1,
    n_epochs=1000,
    l2=0.0,
    fit_intercept=True,
    verbose=False,
    random_state=RANDOM_STATE,
    threshold=0.5,
):
    """
    Thực hiện K-fold cross-validation cho LogisticRegressionBinary.

    Parameters
    ----------
    X, y : dữ liệu đầy đủ
    k : số folds
    Các tham số còn lại truyền cho LogisticRegressionBinary.

    Returns
    -------
    results : dict
        - "accuracy", "precision", "recall", "f1": trung bình trên các folds
        - "fold_metrics": danh sách metric từng fold
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_samples = X.shape[0]
    folds = k_fold_indices(n_samples, k=k, shuffle=True, random_state=random_state)

    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = LogisticRegressionBinary(
            lr=lr,
            n_epochs=n_epochs,
            l2=l2,
            fit_intercept=fit_intercept,
            verbose=verbose,
            random_state=random_state + fold_i,  # đổi seed nhẹ
        )
        clf.fit(X_train, y_train)
        y_val_prob = clf.predict_proba(X_val)
        metrics = evaluate_binary_classification(
            y_true=y_val,
            y_prob=y_val_prob,
            threshold=threshold,
            positive_label=1,
        )
        fold_metrics.append(metrics)

    accs = [m["accuracy"] for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs = [m["recall"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]

    results = {
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
        "fold_metrics": fold_metrics,
    }
    return results
