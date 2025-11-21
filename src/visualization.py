# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def plot_missing_bar(feature_names, missing_counts, title="Missing value cho từng feature"):
    feature_names = np.array(feature_names)
    missing_counts = np.array(missing_counts)

    plt.figure(figsize=(14, 5))
    plt.bar(feature_names, missing_counts)
    
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Số lượng missing")
    plt.tight_layout()
    plt.show()


def plot_target_distribution(target_array, title="Phân phối biến mục tiêu (target)"):
    target_array = np.asarray(target_array, dtype=float)
    target_clean = target_array[~np.isnan(target_array)]
    values, counts = np.unique(target_clean, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(values.astype(str), counts)
    plt.title(title)
    plt.xlabel("Giá trị target")
    plt.ylabel("Số lượng mẫu")
    plt.tight_layout()
    plt.show()

    total = counts.sum()
    print("=== Phân phối target ===")
    for v, c in zip(values, counts):
        print(f"target={int(v)} : {c} mẫu ({c/total:.2%})")


def plot_numeric_distribution(x, name, bins=30, show_kde=True, log_scale=False):
    x = np.asarray(x, dtype=float)
    x_clean = x[~np.isnan(x)]
    if x_clean.size == 0:
        print(f"Không có dữ liệu hợp lệ cho {name} để vẽ.")
        return

    if log_scale:
        x_clean = x_clean[x_clean > 0]
        if x_clean.size == 0:
            print(f"{name}: không có giá trị dương để vẽ log-scale.")
            return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax0 = axes[0]
    ax0.hist(x_clean, bins=bins, density=False, alpha=0.7)
    if show_kde:
        sns.kdeplot(x_clean, ax=ax0)
    ax0.set_title(f"Histogram of {name}" + (" (log scale)" if log_scale else ""))
    ax0.set_xlabel(name)
    ax0.set_ylabel("Count")
    if log_scale:
        ax0.set_xscale("log")

    ax1 = axes[1]
    ax1.boxplot(x_clean, vert=True)
    ax1.set_title(f"Boxplot of {name}")
    ax1.set_ylabel(name)
    if log_scale:
        ax1.set_yscale("log")

    plt.tight_layout()
    plt.show()


def plot_ecdf(x, name):
    x = np.asarray(x, dtype=float)
    x_clean = np.sort(x[~np.isnan(x)])
    if x_clean.size == 0:
        print(f"Không có dữ liệu hợp lệ cho {name} để vẽ ECDF.")
        return

    n = x_clean.size
    y = np.arange(1, n + 1) / n

    plt.figure(figsize=(6, 4))
    plt.step(x_clean, y, where="post")
    plt.title(f"ECDF of {name}")
    plt.xlabel(name)
    plt.ylabel("F(x)")
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(col_data, col_name, top_k=None):
    col_data = np.asarray(col_data, dtype=str)
    values, counts = np.unique(col_data, return_counts=True)

    idx_sorted = np.argsort(-counts)
    values = values[idx_sorted]
    counts = counts[idx_sorted]

    if top_k is not None:
        values = values[:top_k]
        counts = counts[:top_k]

    plt.figure(figsize=(8, 4))
    plt.bar(values, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Phân phối biến categorical: {col_name}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print(f"=== Thống kê cho {col_name} ===")
    for v, c in zip(values, counts):
        print(f"{v:20s} : {c}")
    print()


def plot_pie(col_data, col_name, top_k=None):
    col_data = np.asarray(col_data, dtype=str)
    values, counts = np.unique(col_data, return_counts=True)

    idx_sorted = np.argsort(-counts)
    values = values[idx_sorted]
    counts = counts[idx_sorted]

    if top_k is not None:
        values = values[:top_k]
        counts = counts[:top_k]

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=values, autopct="%1.1f%%")
    plt.title(f"Pie chart: {col_name}")
    plt.tight_layout()
    plt.show()


def plot_target_rate_by_category(col_data, target_array, col_name):
    col_data = np.asarray(col_data, dtype=str)
    target_array = np.asarray(target_array, dtype=float)

    uniq_vals = np.unique(col_data)
    rates = []
    counts = []

    for v in uniq_vals:
        mask = (col_data == v)
        y = target_array[mask]
        y_clean = y[~np.isnan(y)]
        if len(y_clean) == 0:
            rates.append(np.nan)
            counts.append(0)
        else:
            rates.append(np.mean(y_clean))
            counts.append(len(y_clean))

    rates = np.array(rates)
    counts = np.array(counts)
    uniq_vals = np.array(uniq_vals)

    valid_mask = ~np.isnan(rates)
    rates = rates[valid_mask]
    counts = counts[valid_mask]
    uniq_vals = uniq_vals[valid_mask]

    idx_sorted = np.argsort(-rates)
    rates = rates[idx_sorted]
    counts = counts[idx_sorted]
    uniq_vals = uniq_vals[idx_sorted]

    plt.figure(figsize=(8, 4))
    plt.bar(uniq_vals, rates)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Tỷ lệ target = 1")
    plt.title(f"Tỷ lệ đổi việc theo {col_name}")
    plt.tight_layout()
    plt.show()

    print(f"=== Tỷ lệ target=1 theo {col_name} ===")
    for v, r, c in zip(uniq_vals, rates, counts):
        print(f"{v:20s} | n={c:4d} | rate={r:.3f}")
    print()


def boxplot_numeric_by_target(x, target_array, x_name):
    x = np.asarray(x, dtype=float)
    target_array = np.asarray(target_array, dtype=float)

    mask0 = (target_array == 0)
    mask1 = (target_array == 1)

    x0 = x[mask0]
    x1 = x[mask1]

    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]

    if x0.size == 0 or x1.size == 0:
        print(f"Không đủ dữ liệu để vẽ boxplot {x_name} theo target.")
        return

    plt.figure(figsize=(6, 4))
    plt.boxplot([x0, x1], labels=["target=0", "target=1"])
    plt.title(f"{x_name} theo target")
    plt.ylabel(x_name)
    plt.tight_layout()
    plt.show()

    print(f"=== Thống kê {x_name} theo target ===")
    print("target=0:",
          "mean =", np.mean(x0),
          "| median =", np.median(x0),
          "| n =", len(x0))
    print("target=1:",
          "mean =", np.mean(x1),
          "| median =", np.median(x1),
          "| n =", len(x1))
    print()


def plot_hist_overlay_by_target(x, target_array, x_name, bins=30):
    x = np.asarray(x, dtype=float)
    target_array = np.asarray(target_array, dtype=float)

    mask0 = (target_array == 0)
    mask1 = (target_array == 1)

    x0 = x[mask0]
    x1 = x[mask1]

    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]

    if x0.size == 0 or x1.size == 0:
        print(f"Không đủ dữ liệu để vẽ histogram overlay cho {x_name}.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(x0, bins=bins, alpha=0.5, label="target=0", density=True)
    plt.hist(x1, bins=bins, alpha=0.5, label="target=1", density=True)
    plt.title(f"Histogram overlay: {x_name} theo target")
    plt.xlabel(x_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter(x, y, x_name, y_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if x_clean.size == 0:
        print(f"Không đủ dữ liệu để vẽ scatter {x_name} vs {y_name}.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(x_clean, y_clean, alpha=0.5)
    plt.title(f"{y_name} vs {x_name}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(X, feature_names):
    X = np.asarray(X, dtype=float)
    mask_valid = ~np.isnan(X).any(axis=1)
    X_clean = X[mask_valid]

    k = X_clean.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(3 * k, 3 * k))

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(X_clean[:, j], bins=30)
            else:
                ax.scatter(X_clean[:, j], X_clean[:, i], alpha=0.5, s=5)
            if i == k - 1:
                ax.set_xlabel(feature_names[j])
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(feature_names[i])
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix, feature_names, title="Correlation heatmap"):
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        corr_matrix,
        annot=True,
        xticklabels=feature_names,
        yticklabels=feature_names,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_line(x_values, y_values, x_label, y_label, title):
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def plot_outliers(x, mask_out, name):
    """
    Vẽ biểu đồ outlier cho một biến numeric.
    Tham số:
        x        : array-like numeric (có thể chứa np.nan)
        mask_out : mảng bool cùng chiều với x, True nếu là outlier (theo IQR)
        name     : tên biến, dùng cho tiêu đề/trục

    Biểu đồ:
        - Bên trái: boxplot để thấy median, IQR và các điểm outlier.
        - Bên phải: scatter index vs value, tô khác màu cho inlier và outlier.
    """
    x = np.asarray(x, dtype=float)
    mask_out = np.asarray(mask_out, dtype=bool)

    idx = np.arange(len(x))

    # Lọc giá trị hợp lệ (không NaN)
    valid_mask = ~np.isnan(x)
    x_valid = x[valid_mask]
    idx_valid = idx[valid_mask]
    out_valid = mask_out[valid_mask]

    if x_valid.size == 0:
        print(f"Không có dữ liệu hợp lệ cho {name} để vẽ outlier.")
        return

    in_valid = ~out_valid

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Boxplot bên trái
    axes[0].boxplot(x_valid, vert=True)
    axes[0].set_title(f"Boxplot – {name}")
    axes[0].set_ylabel(name)

    # Scatter index vs value bên phải
    axes[1].scatter(idx_valid[in_valid], x_valid[in_valid], alpha=0.5, label="Inlier")
    if np.any(out_valid):
        axes[1].scatter(idx_valid[out_valid], x_valid[out_valid], alpha=0.9, marker="x", label="Outlier")
    axes[1].set_title(f"Outliers theo index – {name}")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel(name)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
