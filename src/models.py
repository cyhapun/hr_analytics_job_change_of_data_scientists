# src/models.py
# ============================================================
# Các mô hình và hàm đánh giá thuần NumPy
# Dùng cho bài toán phân loại nhị phân (0 / 1), ví dụ:
#   - HR Analytics: Job Change of Data Scientists
# ============================================================

import numpy as np

RANDOM_STATE = 23120329


# ------------------------------------------------------------
# 1. Tiện ích chia train / val
# ------------------------------------------------------------
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
        Tỷ lệ dành cho validation (ví dụ 0.2 = 20%).
    shuffle : bool
        Có xáo trộn dữ liệu trước khi chia hay không.
    random_state : int
        Seed cho bộ sinh số ngẫu nhiên (để tái lập kết quả).

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


# ------------------------------------------------------------
# 2. Hàm activation & loss cho Logistic Regression
# ------------------------------------------------------------
def sigmoid(z):
    """
    Hàm sigmoid ổn định số học.

    sigmoid(z) = 1 / (1 + exp(-z))
    """
    z = np.asarray(z, dtype=float)
    # Tránh overflow: clip giá trị z
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob, l2_term=0.0):
    """
    Hàm mất mát cross-entropy cho phân loại nhị phân.

    Parameters
    ----------
    y_true : (n_samples,)
    y_prob : (n_samples,) xác suất dự đoán P(y=1)
    l2_term : float
        Thành phần regularization (không chia cho batch ở đây, chỉ cộng trực tiếp).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # - [ y log(p) + (1-y) log(1-p) ]
    ce = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    loss = np.mean(ce) + l2_term
    return loss


# ------------------------------------------------------------
# 3. Logistic Regression nhị phân thuần NumPy
# ------------------------------------------------------------
class LogisticRegressionBinary:
    """
    Logistic Regression cho bài toán phân loại nhị phân.

    - Sử dụng gradient descent full-batch.
    - Hỗ trợ L2 regularization.
    - Thuần NumPy, không dùng scikit-learn.
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
            Nếu True, mỗi một vài epoch sẽ in ra loss để theo dõi.
        random_state : int
            Seed để khởi tạo trọng số ban đầu.
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state

        # Trọng số sau khi fit
        self.W = None  # shape (n_features,) hoặc (n_features+1,) nếu có bias

        # Lịch sử loss (để plot nếu muốn)
        self.loss_history_ = []

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
        # Khởi tạo nhỏ để tránh saturate sigmoid ngay từ đầu
        self.W = rng.normal(loc=0.0, scale=0.01, size=(n_features,))

    def fit(self, X, y):
        """
        Huấn luyện mô hình.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Nhãn nhị phân 0/1.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Thêm intercept nếu cần
        X_ext = self._add_intercept(X)  # shape (n_samples, n_features_ext)
        n_samples, n_features_ext = X_ext.shape

        # Khởi tạo W
        self._init_weights(n_features_ext)

        for epoch in range(self.n_epochs):
            # 1. Tính linear combination
            z = X_ext @ self.W  # shape (n_samples,)
            # 2. Tính xác suất
            y_prob = sigmoid(z)  # shape (n_samples,)

            # 3. Tính gradient
            # dL/dW = (1/n) * X^T (y_hat - y) + (l2/n) * W_no_bias
            error = y_prob - y  # shape (n_samples,)

            grad = (X_ext.T @ error) / n_samples  # shape (n_features_ext,)

            # Regularization: không regularize bias (phần tử W[0]) nếu có bias
            if self.l2 > 0:
                if self.fit_intercept:
                    w_reg = self.W.copy()
                    w_reg[0] = 0.0  # không phạt bias
                else:
                    w_reg = self.W
                grad += (self.l2 / n_samples) * w_reg

            # 4. Cập nhật W
            self.W -= self.lr * grad

            # 5. Lưu loss cho việc theo dõi
            if self.verbose or epoch == self.n_epochs - 1:
                reg_term = 0.0
                if self.l2 > 0:
                    if self.fit_intercept:
                        w_reg = self.W[1:]  # bỏ bias
                    else:
                        w_reg = self.W
                    reg_term = (self.l2 / (2 * n_samples)) * np.sum(w_reg ** 2)

                loss = binary_cross_entropy(y, y_prob, l2_term=reg_term)
                self.loss_history_.append(loss)

                if self.verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                    print(f"[Epoch {epoch:4d}] Loss = {loss:.6f}")

        return self

    def predict_proba(self, X):
        """
        Trả về xác suất P(y=1 | x).

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
        Dự đoán nhãn 0/1 dựa trên threshold (mặc định 0.5).
        """
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)


# ------------------------------------------------------------
# 4. Các hàm đánh giá mô hình
# ------------------------------------------------------------
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
    Confusion matrix cho nhị phân: [[TN, FP], [FN, TP]]
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

    # precision = TP / (TP + FP)
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    # recall = TP / (TP + FN)
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    # F1 = 2 * P * R / (P + R)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_binary_classification(y_true, y_prob, threshold=0.5, positive_label=1):
    """
    Convenience function:
    - Chuyển xác suất thành nhãn
    - Tính accuracy, precision, recall, F1 và confusion matrix
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


# ------------------------------------------------------------
# 5. K-fold Cross-validation đơn giản
# ------------------------------------------------------------
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
    fold_sizes[: n_samples % k] += 1  # phân bổ phần dư

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
    Các tham số còn lại truyền vào LogisticRegressionBinary

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
            random_state=random_state + fold_i,  # đổi seed chút xíu
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

    # Trung bình trên các folds
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
