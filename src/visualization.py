# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

def _clean_categorical(col_data):
    """
    Chuẩn hoá giá trị categorical (vectorized, không dùng for-loop trên từng phần tử)
    """
    arr = np.asarray(col_data, dtype=object)

    # None
    mask_none = (arr == None)  # noqa: E711

    # np.nan (trong trường hợp phần tử là float và là nan)
    mask_nan = np.frompyfunc(
        lambda x: isinstance(x, float) and np.isnan(x), 1, 1
    )(arr).astype(bool)

    # Chuẩn hoá sang str, strip khoảng trắng
    str_arr = np.char.strip(arr.astype(str))

    # Rỗng
    mask_empty = (str_arr == "")

    # Các token đại diện missing
    lower_arr = np.char.lower(str_arr)
    missing_tokens = np.array(
        ["nan", "na", "n/a", "<na>", "null", "none", "missing", "?"], dtype=str
    )
    mask_special = np.isin(lower_arr, missing_tokens)

    is_missing = mask_none | mask_nan | mask_empty | mask_special

    cleaned = np.where(is_missing, "MISSING", str_arr)
    return cleaned.astype(str)

def plot_missing_bar(feature_names, missing_counts, title="Missing value cho từng feature"):
    feature_names = np.array(feature_names)
    missing_counts = np.array(missing_counts)
    total_missing = missing_counts.sum()

    print("\nThống kê missing values:")
    for name, cnt in zip(feature_names, missing_counts):
        print(f"{str(name):25s} : {int(cnt):6d}")
    print(f"Tổng số missing: {int(total_missing)}\n")

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
    total = counts.sum()

    print("Phân phối target:")
    for v, c in zip(values, counts):
        print(f"target={int(v)} : {c} mẫu ({c/total:.2%})")
    print()

    plt.figure(figsize=(6, 4))
    plt.bar(values.astype(str), counts)
    plt.title(title)
    plt.xlabel("Giá trị target")
    plt.ylabel("Số lượng mẫu")
    plt.tight_layout()
    plt.show()

def plot_numeric_distribution(x, name, bins=30, show_kde=True, log_scale=False, ax=None):
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram
    ax.hist(
        x_clean,
        bins=bins,
        density=False,
        alpha=0.6,
        label="Histogram"
    )

    # KDE trên trục phụ
    if show_kde and np.unique(x_clean).size > 1:
        ax2 = ax.twinx()
        sns.kdeplot(
            x=x_clean,
            fill=False,
            bw_adjust=1.0,
            linewidth=2,
            color="orange",
            label="KDE",
            ax=ax2,
        )
        ax2.set_ylabel("KDE Density")
        ax2.legend(loc="upper right")
    else:
        print(f"{name}: bỏ KDE do dữ liệu quá ít hoặc trùng lặp.")

    ax.set_title(f"Histogram of {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")

def plot_ecdf(x, name, ax=None):
    x = np.asarray(x, dtype=float)
    x_clean = x[~np.isnan(x)]
    if x_clean.size == 0:
        print(f"Không có dữ liệu hợp lệ cho {name} để vẽ ECDF.")
        return

    x_sorted = np.sort(x_clean)
    n = x_sorted.size
    y = np.arange(1, n + 1) / n

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.step(x_sorted, y, where="post")
    ax.set_title(f"ECDF of {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("F(x)")

def plot_categorical_distribution(col_data, col_name, top_k=None, ax=None, missing_label="Missing"):
    col_data = np.array(col_data, dtype=str)

    col_norm = np.char.strip(col_data)
    col_norm[col_norm == ""] = missing_label

    values, counts = np.unique(col_norm, return_counts=True)
    if top_k is not None and values.size > top_k:
        idx = np.argsort(-counts)[:top_k]
        values = values[idx]
        counts = counts[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(values, counts)
    ax.set_title(col_name)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

def plot_pie(col_data, col_name, top_k=5, ax=None, missing_label="Missing"):
    col_data = np.array(col_data, dtype=str)

    col_norm = np.char.strip(col_data)
    col_norm[col_norm == ""] = missing_label

    values, counts = np.unique(col_norm, return_counts=True)
    if values.size > top_k:
        idx = np.argsort(-counts)[:top_k]
        values = values[idx]
        counts = counts[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.8,
    )

    ax.set_title(f"{col_name} (top {top_k})")

    ax.legend(
        wedges,
        values,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=8,
        title=col_name,
    )

def plot_target_rate_by_category(col_data, target_array, col_name):
    col_data = _clean_categorical(col_data)
    y = np.asarray(target_array, dtype=float)

    # Chỉ lấy những mẫu có target hợp lệ
    mask_valid = ~np.isnan(y)
    col_valid = col_data[mask_valid]
    y_valid = y[mask_valid]

    if col_valid.size == 0:
        print(f"Không có dữ liệu hợp lệ cho {col_name}.")
        return

    uniq_vals, inv = np.unique(col_valid, return_inverse=True)

    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=y_valid)
    rates = sums / counts

    # Loại bỏ NaN (nếu có)
    valid_mask = ~np.isnan(rates)
    rates = rates[valid_mask]
    counts = counts[valid_mask]
    uniq_vals = uniq_vals[valid_mask]

    # Sort theo rate giảm dần
    idx_sorted = np.argsort(-rates)
    rates = rates[idx_sorted]
    counts = counts[idx_sorted]
    uniq_vals = uniq_vals[idx_sorted]

    max_cat_show = 15
    k = min(max_cat_show, len(uniq_vals))
    max_cat_len = 28

    def fmt_cat(val):
        s = str(val)
        if len(s) > max_cat_len:
            return s[: max_cat_len - 3] + "..."
        return s

    print("\n" + "=" * 80)
    print(f"\t    Tỷ lệ target = 1 theo '{col_name}':")
    print("-" * 80)
    header = (
        f"{'Category':<{max_cat_len}} | "
        f"{'n':>6} | "
        f"{'rate':>6} | "
        f"{'% target=1':>11}"
    )
    print(header)
    print("-" * 80)

    for v, r, c in zip(uniq_vals[:k], rates[:k], counts[:k]):
        cat_str = fmt_cat(v)
        print(
            f"{cat_str:<{max_cat_len}} | "
            f"{c:6d} | "
            f"{r:6.3f} | "
            f"{r * 100:10.2f}%"
        )

    if len(uniq_vals) > k:
        print("-" * 80)
        print(f"(Đã ẩn bớt {len(uniq_vals) - k} category ít quan trọng hơn để bảng gọn hơn.)")
    print("=" * 80)

    plt.figure(figsize=(9, 4))
    plt.bar(uniq_vals[:k], rates[:k])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Tỷ lệ target = 1")
    plt.title(f"Tỷ lệ đổi việc theo {col_name} (top {k} category theo rate)")
    plt.tight_layout()
    plt.show()

def boxplot_numeric_by_target(x, target, name, ax=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(target, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(t)
    x_clean = x[mask]
    t_clean = t[mask]

    data0 = x_clean[t_clean == 0]
    data1 = x_clean[t_clean == 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot([data0, data1], labels=["target=0", "target=1"])
    ax.set_title(f"{name} vs target")
    ax.set_ylabel(name)

def plot_hist_overlay_by_target(x, target, name, bins=30, ax=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(target, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(t)
    x_clean = x[mask]
    t_clean = t[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    x0 = x_clean[t_clean == 0]
    x1 = x_clean[t_clean == 1]

    ax.hist(x0, bins=bins, alpha=0.6, density=True, label="target=0")
    ax.hist(x1, bins=bins, alpha=0.6, density=True, label="target=1")
    ax.set_title(f"Histogram of {name} by target")
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.legend()

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

def plot_correlation_heatmap_from_data(X, feature_names, title="Correlation heatmap (from data)"):
    """
    Tính ma trận tương quan từ dữ liệu X bằng np.einsum (đáp ứng yêu cầu kỹ thuật NumPy 2.2)
    và vẽ heatmap.

    X: array shape (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    if X_clean.shape[0] == 0:
        print("Không có dữ liệu hợp lệ để tính correlation.")
        return

    # Center dữ liệu để tính covariance ổn định hơn
    X_centered = X_clean - X_clean.mean(axis=0, keepdims=True)
    n = X_centered.shape[0]

    if n <= 1:
        print("Không đủ số mẫu để tính correlation.")
        return

    # Covariance: (X^T X) / (n-1) với np.einsum
    cov = np.einsum("ni,nj->ij", X_centered, X_centered) / max(n - 1, 1)

    # Chuyển sang correlation
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    denom[denom == 0] = np.nan  # tránh chia cho 0
    corr_matrix = cov / denom

    plot_correlation_heatmap(corr_matrix, feature_names, title=title)

def plot_outliers(x, mask_out, name):
    x = np.asarray(x, dtype=float)
    mask_out = np.asarray(mask_out, dtype=bool)

    idx = np.arange(len(x))
    valid_mask = ~np.isnan(x)
    x_valid = x[valid_mask]
    idx_valid = idx[valid_mask]
    out_valid = mask_out[valid_mask]

    if x_valid.size == 0:
        print(f"Không có dữ liệu hợp lệ cho {name} để vẽ outlier.")
        return

    in_valid = ~out_valid

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot(x_valid, vert=True)
    axes[0].set_title(f"Boxplot – {name}")
    axes[0].set_ylabel(name)

    axes[1].scatter(idx_valid[in_valid], x_valid[in_valid], alpha=0.5, label="Inlier")
    if np.any(out_valid):
        axes[1].scatter(idx_valid[out_valid], x_valid[out_valid], alpha=0.9, marker="x", label="Outlier")
    axes[1].set_title(f"Outliers theo index – {name}")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel(name)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def boxplot_numeric_by_category(x, cat, num_name, cat_name, ax=None, top_k=None):
    x = np.asarray(x, dtype=float)
    cat = _clean_categorical(cat)

    mask_valid = ~np.isnan(x)
    x = x[mask_valid]
    cat = cat[mask_valid]

    values_all, counts_all = np.unique(cat, return_counts=True)
    if top_k is not None and values_all.size > top_k:
        idx = np.argsort(-counts_all)[:top_k]
        keep_vals = values_all[idx]
        keep_mask = np.isin(cat, keep_vals)
        x = x[keep_mask]
        cat = cat[keep_mask]

    values = np.unique(cat)
    data = [x[cat == v] for v in values]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.boxplot(data, labels=values)
    ax.set_title(f"{num_name} by {cat_name}")
    ax.set_xlabel(cat_name)
    ax.set_ylabel(num_name)
    ax.tick_params(axis="x", rotation=45)

def plot_mean_numeric_by_category(x, cat, num_name, cat_name, ax=None, top_k=None):
    x = np.asarray(x, dtype=float)
    cat = _clean_categorical(cat)

    mask_valid = ~np.isnan(x)
    x = x[mask_valid]
    cat = cat[mask_valid]

    values_all, counts_all = np.unique(cat, return_counts=True)
    if top_k is not None and values_all.size > top_k:
        idx = np.argsort(-counts_all)[:top_k]
        keep_vals = values_all[idx]
        keep_mask = np.isin(cat, keep_vals)
        x = x[keep_mask]
        cat = cat[keep_mask]

    uniq, inv = np.unique(cat, return_inverse=True)
    sum_x = np.bincount(inv, weights=x)
    cnt_x = np.bincount(inv)
    means = sum_x / cnt_x

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(uniq, means)
    ax.set_title(f"Mean {num_name} by {cat_name}")
    ax.set_xlabel(cat_name)
    ax.set_ylabel(f"Mean {num_name}")
    ax.tick_params(axis="x", rotation=45)

def plot_crosstab_heatmap(cat1, cat2, name1, name2, ax=None, top_k1=None, top_k2=None, normalize=False):
    c1 = _clean_categorical(cat1)
    c2 = _clean_categorical(cat2)

    vals1_all, counts1_all = np.unique(c1, return_counts=True)
    vals2_all, counts2_all = np.unique(c2, return_counts=True)

    if top_k1 is not None and vals1_all.size > top_k1:
        idx1 = np.argsort(-counts1_all)[:top_k1]
        keep1 = set(vals1_all[idx1])
    else:
        keep1 = set(vals1_all)

    if top_k2 is not None and vals2_all.size > top_k2:
        idx2 = np.argsort(-counts2_all)[:top_k2]
        keep2 = set(vals2_all[idx2])
    else:
        keep2 = set(vals2_all)

    mask1 = np.isin(c1, list(keep1))
    mask2 = np.isin(c2, list(keep2))
    mask = mask1 & mask2

    c1 = c1[mask]
    c2 = c2[mask]

    vals1, inv1 = np.unique(c1, return_inverse=True)
    vals2, inv2 = np.unique(c2, return_inverse=True)

    n1 = vals1.size
    n2 = vals2.size

    index = inv1 * n2 + inv2
    mat = np.bincount(index, minlength=n1 * n2).reshape(n1, n2).astype(float)

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        mat,
        annot=True,
        fmt=".0f" if not normalize else ".2f",
        xticklabels=vals2,
        yticklabels=vals1,
        ax=ax,
    )
    ax.set_xlabel(name2)
    ax.set_ylabel(name1)
    ax.set_title(f"{name1} vs {name2} ({'row-normalized' if normalize else 'counts'})")
    ax.tick_params(axis="x", rotation=45)

def visualize_q1_risk_profiles(experience, last_new_job, training_hours, city_development_index, target,
                               top_n=6, min_count=50):
    exp = np.asarray(experience, dtype=float)
    lnj = np.asarray(last_new_job, dtype=float)
    trh = np.asarray(training_hours, dtype=float)
    cdi = np.asarray(city_development_index, dtype=float)
    t = np.asarray(target, dtype=float)

    mask_valid = (
        ~np.isnan(exp)
        & ~np.isnan(lnj)
        & ~np.isnan(trh)
        & ~np.isnan(cdi)
        & ~np.isnan(t)
    )
    exp = exp[mask_valid]
    lnj = lnj[mask_valid]
    trh = trh[mask_valid]
    cdi = cdi[mask_valid]
    t = t[mask_valid]

    if exp.size == 0:
        print("Q1: Không có dữ liệu hợp lệ để phân tích bộ feature.")
        return

    exp_bins = np.array([0, 1, 3, 7, 50])
    exp_labels = np.array(["≤1 năm", "1–3 năm", "3–7 năm", ">7 năm"], dtype=str)

    lnj_bins = np.array([0, 1, 2, 4, 50])
    lnj_labels = np.array(["≤1 năm", "1–2 năm", "2–4 năm", ">4 năm"], dtype=str)

    trh_quantiles = np.quantile(trh, [0.25, 0.5, 0.75])
    trh_bins = np.concatenate([[0], trh_quantiles, [np.max(trh) + 1]])
    trh_labels = np.array(["Thấp", "Trung bình", "Cao", "Rất cao"], dtype=str)

    cdi_bins = np.array([0.0, 0.6, 0.8, 1.0])
    cdi_labels = np.array(["CDI thấp", "CDI trung bình", "CDI cao"], dtype=str)

    def bin_with_labels(x, bins, labels, right=True):
        idx = np.digitize(x, bins, right=right) - 1
        idx[idx < 0] = 0
        idx[idx >= len(labels)] = len(labels) - 1
        return idx, labels

    exp_idx, exp_labels = bin_with_labels(exp, exp_bins, exp_labels, right=True)
    lnj_idx, lnj_labels = bin_with_labels(lnj, lnj_bins, lnj_labels, right=True)
    trh_idx, trh_labels = bin_with_labels(trh, trh_bins, trh_labels, right=True)
    cdi_idx, cdi_labels = bin_with_labels(cdi, cdi_bins, cdi_labels, right=True)

    nE = len(exp_labels)
    nL = len(lnj_labels)
    nT = len(trh_labels)
    nC = len(cdi_labels)

    # Mã hoá 4 chiều về 1 chiều để group bằng bincount
    code = (((exp_idx * nL + lnj_idx) * nT + trh_idx) * nC + cdi_idx)

    sum_y = np.bincount(code, weights=t, minlength=nE * nL * nT * nC)
    cnt_y = np.bincount(code, minlength=nE * nL * nT * nC)

    valid_mask = cnt_y >= min_count
    if not np.any(valid_mask):
        print("Q1: Không có profile nào đủ số lượng (min_count) để phân tích.")
        return

    rates_all = np.zeros_like(sum_y, dtype=float)
    rates_all[valid_mask] = sum_y[valid_mask] / cnt_y[valid_mask]

    valid_indices = np.nonzero(valid_mask)[0]

    profiles = []
    for idx in valid_indices:
        tmp = idx
        ic = tmp % nC
        tmp //= nC
        it = tmp % nT
        tmp //= nT
        il = tmp % nL
        ie = tmp // nL

        profiles.append(
            {
                "exp_label": exp_labels[ie],
                "lnj_label": lnj_labels[il],
                "trh_label": trh_labels[it],
                "cdi_label": cdi_labels[ic],
                "count": int(cnt_y[idx]),
                "rate": float(rates_all[idx]),
            }
        )

    profiles_sorted = sorted(profiles, key=lambda d: d["rate"], reverse=True)
    top_profiles = profiles_sorted[:top_n]

    line_width = 110
    print(
        f"\tQ1: Top {len(top_profiles)} bộ feature có tỷ lệ target=1 cao nhất "
        f"(chỉ xét bộ feature có ít nhất {min_count} mẫu)"
    )
    print("=" * line_width)
    header = (
        f"{'ID':<4} "
        f"{'Kinh nghiệm':<15} "
        f"{'Last_new_job':<15} "
        f"{'Training':<12} "
        f"{'CDI':<14} "
        f"{'Số mẫu':>8} "
        f"{'Tỷ lệ target=1':>18}"
    )
    print(header)
    print("-" * line_width)

    profile_names = []
    rates = []

    for i, p in enumerate(top_profiles, 1):
        pid = f"P{i:02d}"
        rate_pct = p["rate"] * 100.0
        row = (
            f"{pid:<4} "
            f"{p['exp_label']:<15} "
            f"{p['lnj_label']:<15} "
            f"{p['trh_label']:<12} "
            f"{p['cdi_label']:<14} "
            f"{p['count']:>8d} "
            f"{rate_pct:>17.1f}%"
        )
        print(row)

        profile_names.append(pid)
        rates.append(p["rate"])

    print("=" * line_width)

    plt.figure(figsize=(7, 4))
    x = np.arange(len(profile_names))
    rates = np.array(rates, dtype=float)
    plt.bar(x, rates)
    for i, r in enumerate(rates):
        plt.text(i, r, f"{r*100:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, profile_names)
    plt.ylabel("Tỷ lệ target=1")
    plt.xlabel("Bộ feature (P01, P02, ...)")
    plt.title("Q1: Top bộ feature theo tỷ lệ muốn đổi job")
    plt.tight_layout()
    plt.show()

def visualize_q2_churn_last_new_job_by_company_type(
    company_type_col, last_new_job_col, target_array,
    type_a="Funded Startup", type_b="Pvt Ltd"
):
    """
    So sánh churn theo last_new_job cho 2 loại company_type (mặc định:
    Funded Startup vs Pvt Ltd).
    """
    company_type_clean = _clean_categorical(company_type_col)
    last_new_job_clean = _clean_categorical(last_new_job_col)
    y = np.asarray(target_array, dtype=float)

    def _churn_by_last_new_job_for_type(company_type_name):
        mask = (company_type_clean == company_type_name)
        lnj = last_new_job_clean[mask]
        y_sub = y[mask]

        # Chỉ lấy target hợp lệ
        mask_valid = ~np.isnan(y_sub)
        lnj = lnj[mask_valid]
        y_sub = y_sub[mask_valid]

        if lnj.size == 0:
            return np.array([], dtype=str), np.array([], dtype=int), np.array([], dtype=float)

        vals, inv = np.unique(lnj, return_inverse=True)
        counts = np.bincount(inv)
        sums = np.bincount(inv, weights=y_sub)
        rates = sums / counts
        return vals, counts, rates

    vals_a, cnt_a, rate_a = _churn_by_last_new_job_for_type(type_a)
    vals_b, cnt_b, rate_b = _churn_by_last_new_job_for_type(type_b)

    preferred_order = ["never", "<1", "1", "2", "3", "4", ">4", "MISSING"]
    all_levels = sorted(
        set(vals_a) | set(vals_b),
        key=lambda x: preferred_order.index(x) if x in preferred_order else 999
    )

    def _align(vals, arr, levels):
        if vals.size == 0:
            return np.full(len(levels), np.nan, dtype=float)
        out = []
        for lv in levels:
            idx = np.where(vals == lv)[0]
            if idx.size == 0:
                out.append(np.nan)
            else:
                out.append(arr[idx[0]])
        return np.array(out, dtype=float)

    rate_a_aligned = _align(vals_a, rate_a, all_levels)
    rate_b_aligned = _align(vals_b, rate_b, all_levels)
    cnt_a_aligned = _align(vals_a, cnt_a.astype(float), all_levels)
    cnt_b_aligned = _align(vals_b, cnt_b.astype(float), all_levels)

    print(f"\nCross-tab '{type_a}' × last_new_job")
    print(f"{'last_new_job':12s} | {'count':>6s} | {'tỷ lệ rời':>10s}")
    print("-" * 34)
    for lv, c, r in zip(all_levels, cnt_a_aligned, rate_a_aligned):
        if np.isnan(c) or np.isnan(r):
            continue
        print(f"{lv:12s} | {int(c):6d} | {r:10.3f}")

    print(f"\nCross-tab '{type_b}' × last_new_job")
    print(f"{'last_new_job':12s} | {'count':>6s} | {'tỷ lệ rời':>10s}")
    print("-" * 34)
    for lv, c, r in zip(all_levels, cnt_b_aligned, rate_b_aligned):
        if np.isnan(c) or np.isnan(r):
            continue
        print(f"{lv:12s} | {int(c):6d} | {r:10.3f}")

    x = np.arange(len(all_levels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Churn theo last_new_job: "
        f"{type_a} vs {type_b}", fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    ax.bar(all_levels, rate_a_aligned)
    ax.set_title(f"{type_a}: churn theo last_new_job")
    ax.set_xlabel("last_new_job")
    ax.set_ylabel("Tỷ lệ target = 1 (churn)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width / 2, rate_a_aligned, width, label=type_a)
    ax.bar(x + width / 2, rate_b_aligned, width, label=type_b)
    ax.set_xticks(x)
    ax.set_xticklabels(all_levels, rotation=45)
    ax.set_xlabel("last_new_job")
    ax.set_ylabel("Tỷ lệ target = 1 (churn)")
    ax.set_title(f"So sánh churn: {type_a} vs {type_b}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_q3_training_hours_by_experience_groups(
    exp_encoded, training_hours, target_array,
    junior_max_code=1, senior_min_code=11
):
    """
    Phân tích training_hours theo 3 nhóm kinh nghiệm:
      - Junior   : code <= junior_max_code
      - Mid level: junior_max_code+1 .. senior_min_code-1
      - Senior   : code >= senior_min_code
    """
    exp = np.asarray(exp_encoded, dtype=float)
    trh = np.asarray(training_hours, dtype=float)
    y = np.asarray(target_array, dtype=float)

    mask_valid = (~np.isnan(exp) & (exp >= 0) & ~np.isnan(trh) & ~np.isnan(y))
    exp = exp[mask_valid]
    trh = trh[mask_valid]
    y = y[mask_valid]

    groups = [
        (f"Junior (năm <= {junior_max_code})",
         (exp <= junior_max_code)),
        (f"Mid-level (năm: {junior_max_code + 1}..{senior_min_code - 1})",
         ((exp >= junior_max_code + 1) & (exp <= senior_min_code - 1))),
        (f"Senior (năm >= {senior_min_code})",
         (exp >= senior_min_code)),
    ]

    print("  Thống kê training_hours theo nhóm kinh nghiệm\n")

    for name, mask in groups:
        trh_g = trh[mask]
        y_g = y[mask]

        print(f"{name}:")
        n = trh_g.shape[0]
        print(f"Số mẫu tổng : {n}")
        if n == 0:
            print("Không có dữ liệu cho nhóm này.\n")
            continue

        header = f"{'Target':>6s} | {'N':>6s} | {'Mean':>10s} | {'Median':>11s}"
        print(header)
        print("-" * len(header))

        for t_val in [0, 1]:
            mask_t = (y_g == t_val)
            n_t = int(np.sum(mask_t))
            if n_t == 0:
                print(f"{t_val:6d} | {n_t:6d} | {'-':>10s} | {'-':>11s}")
            else:
                mean_val = float(np.nanmean(trh_g[mask_t]))
                median_val = float(np.nanmedian(trh_g[mask_t]))
                print(f"{t_val:6d} | {n_t:6d} | {mean_val:10.2f} | {median_val:11.2f}")
        print()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training_hours vs Target theo nhóm kinh nghiệm", fontsize=14, fontweight="bold")

    for idx, (name, mask) in enumerate(groups):
        trh_g = trh[mask]
        y_g = y[mask]

        ax_box = axes[0, idx]
        ax_hist = axes[1, idx]

        if trh_g.size == 0:
            ax_box.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
            ax_box.set_axis_off()
            ax_hist.set_axis_off()
            continue

        data_box = [trh_g[y_g == 0], trh_g[y_g == 1]]
        labels_box = ["0", "1"]

        ax_box.boxplot(data_box, labels=labels_box, showfliers=False)
        ax_box.set_title(name)
        ax_box.set_xlabel("Target")
        ax_box.set_ylabel("Training_hours")

        trh_0 = trh_g[y_g == 0]
        trh_1 = trh_g[y_g == 1]

        if trh_0.size > 0:
            ax_hist.hist(trh_0, bins=30, alpha=0.6, label="target=0", density=True)
        if trh_1.size > 0:
            ax_hist.hist(trh_1, bins=30, alpha=0.6, label="target=1", density=True)

        ax_hist.set_xlabel("Training_hours")
        ax_hist.set_ylabel("Mật độ (density)")
        ax_hist.set_title(name + " – phân phối theo target")
        ax_hist.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_q4_leave_city(
    exp_encoded, city_development_index, target_array,
    junior_max_code=1, senior_min_code=11, low_cdi=0.7, high_cdi=0.9
):
    """
    Câu 4: Ai là người rời bỏ các thành phố kém phát triển?
    """
    exp = np.asarray(exp_encoded, dtype=float)
    cdi = np.asarray(city_development_index, dtype=float)
    y = np.asarray(target_array, dtype=float)

    mask_valid = (~np.isnan(exp) & (exp >= 0) & ~np.isnan(cdi) & ~np.isnan(y))
    exp = exp[mask_valid]
    cdi = cdi[mask_valid]
    y = y[mask_valid]

    is_junior = (exp <= junior_max_code)
    is_senior = (exp >= senior_min_code)
    is_city_low = (cdi < low_cdi)
    is_city_high = (cdi > high_cdi)

    groups = {
        "Junior + City Low":   is_junior & is_city_low,
        "Junior + City High":  is_junior & is_city_high,
        "Senior + City Low":   is_senior & is_city_low,
        "Senior + City High":  is_senior & is_city_high,
    }

    result_names = []
    counts = []
    churn_rates = []

    print("Tỷ lệ rời đi theo kinh nghiệm & mức độ phát triển thành phố\n")

    header = f"{'Group':30s} | {'N':>6s} | {'target=1 (%)':>12s}"
    print(header)
    print("-" * len(header))

    for name, mask in groups.items():
        y_g = y[mask]
        n = y_g.size
        if n == 0:
            churn = np.nan
        else:
            churn = np.mean(y_g) * 100

        result_names.append(name)
        counts.append(n)
        churn_rates.append(churn)

        print(f"{name:30s} | {n:6d} | {churn:12.2f}")

    heat_data = np.array([
        [churn_rates[0], churn_rates[1]],
        [churn_rates[2], churn_rates[3]],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Brain Drain Analysis – Tỷ lệ muốn rời đi (target=1)", fontsize=15, fontweight="bold")

    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=["City Low", "City High"],
        yticklabels=["Junior", "Senior"],
        ax=axes[0]
    )
    axes[0].set_title("Heatmap: Churn Rate (%)")

    x = np.arange(len(result_names))
    axes[1].bar(x, churn_rates, color=["#c62828", "#ef5350", "#ad1457", "#ff8a80"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(result_names, rotation=30, ha="right")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_title("So sánh churn giữa 4 nhóm")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_q5_enrollment_experience_interaction(
    exp_encoded, enrolled_university_raw, target_array,
    junior_max_code=1, senior_min_code=11
):
    """
    Câu 5: Interaction giữa enrolled_university và experience.
    """

    exp = np.asarray(exp_encoded, dtype=float)
    enr_raw = np.asarray(enrolled_university_raw, dtype=str)
    y = np.asarray(target_array, dtype=float)

    def clean_enrollment(col):
        """
        Vectorized cleaning cho enrolled_university (không for-loop trên từng phần tử).
        Chuẩn hoá về 3 nhóm chính: Full time, Part time, None.
        """
        col = np.asarray(col, dtype=str)
        stripped = np.char.strip(col)
        lower = np.char.lower(stripped)

        full_tokens = np.array(
            ["full time course", "full time", "full_time_course"], dtype=str
        )
        part_tokens = np.array(
            ["part time course", "part time", "part_time_course"], dtype=str
        )
        none_tokens = np.array(
            ["no_enrollment", "no enrollment", "no course", "none", "", "nan"], dtype=str
        )

        full_mask = np.isin(lower, full_tokens)
        part_mask = np.isin(lower, part_tokens)
        none_mask = np.isin(lower, none_tokens)

        result = stripped.copy()
        result[full_mask] = "Full time"
        result[part_mask] = "Part time"
        result[none_mask] = "None"

        return result

    enr = clean_enrollment(enr_raw)

    mask_valid = (
        ~np.isnan(exp)
        & (exp >= 0)
        & ~np.isnan(y)
        & ((y == 0) | (y == 1))
    )
    exp = exp[mask_valid]
    enr = enr[mask_valid]
    y = y[mask_valid]

    is_junior = (exp <= junior_max_code)
    is_senior = (exp >= senior_min_code)

    is_full = (enr == "Full time")
    is_part = (enr == "Part time")
    is_none = (enr == "None")

    groups = {
        "Junior + Full time":  is_junior & is_full,
        "Junior + Part time":  is_junior & is_part,
        "Junior + None":       is_junior & is_none,
        "Senior + Full time":  is_senior & is_full,
        "Senior + Part time":  is_senior & is_part,
        "Senior + None":       is_senior & is_none,
    }

    print("Phân tích Interaction: Enrollment × Experience")
    header = f"{'Group':28s} | {'N':>6s} | {'Churn (%)':>10s}"
    print(header)
    print("-" * len(header))

    group_names = []
    counts = []
    churns = []

    for name, mask in groups.items():
        y_g = y[mask]
        n = y_g.size
        churn = np.mean(y_g) * 100 if n > 0 else np.nan

        group_names.append(name)
        counts.append(n)
        churns.append(churn)

        print(f"{name:28s} | {n:6d} | {churn:10.2f}")

    heat_data = np.array([
        [churns[0], churns[1], churns[2]],
        [churns[3], churns[4], churns[5]],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Enrollment × Experience – Churn Rate", fontsize=15, fontweight="bold")

    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=["Full time", "Part time", "None"],
        yticklabels=["Junior", "Senior"],
        ax=axes[0]
    )
    axes[0].set_title("Heatmap: Churn (%)")

    x = np.arange(len(group_names))
    axes[1].bar(x, churns, color=[
        "#c62828", "#ef5350", "#ff8a80",
        "#6a1b9a", "#ab47bc", "#ce93d8"
    ])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(group_names, rotation=25, ha="right")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_title("So sánh 6 nhóm Enrollment × Experience")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
