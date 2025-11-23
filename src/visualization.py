# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

def _clean_categorical(col_data):
    cleaned = []
    for v in col_data:
        if v is None:
            cleaned.append("MISSING")
        else:
            s = str(v).strip()
            if s == "":
                cleaned.append("MISSING")
            else:
                cleaned.append(s)
    return np.array(cleaned, dtype=str)

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

    ax.hist(x_clean, bins=bins, density=False, alpha=0.6, label="Histogram")

    if show_kde and np.unique(x_clean).size > 1:
        sns.kdeplot(x=x_clean, fill=False, bw_adjust=1.0, linewidth=2, label="KDE", ax=ax)
    else:
        print(f"{name}: bỏ KDE do dữ liệu quá ít hoặc trùng lặp.")

    ax.set_title(f"Histogram of {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.legend()

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

    # chuẩn hóa & thay rỗng thành missing_label
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
    """
    Vẽ và in bảng tỷ lệ target=1 theo từng category của biến col_name.
    - col_data: mảng category (string, object, ...)
    - target_array: mảng target (0/1 hoặc có NaN)
    - col_name: tên biến (string)
    """
    col_data = _clean_categorical(col_data)
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

    rates = np.array(rates, dtype=float)
    counts = np.array(counts, dtype=int)
    uniq_vals = np.array(uniq_vals)

    # Loại bỏ category không tính được rate
    valid_mask = ~np.isnan(rates)
    rates = rates[valid_mask]
    counts = counts[valid_mask]
    uniq_vals = uniq_vals[valid_mask]

    # Sắp xếp giảm dần theo rate (tỷ lệ target=1)
    idx_sorted = np.argsort(-rates)
    rates = rates[idx_sorted]
    counts = counts[idx_sorted]
    uniq_vals = uniq_vals[idx_sorted]

    # Giới hạn số category hiển thị cho đẹp
    max_cat_show = 15
    k = min(max_cat_show, len(uniq_vals))

    # Hàm format tên category (cắt bớt nếu quá dài)
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

    # Vẽ bar chart cho top k category
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

def visualize_q1_risk_profiles(experience, last_new_job, training_hours, city_development_index, target, top_n=6, min_count=50):
    exp = np.asarray(experience, dtype=float)
    lnj = np.asarray(last_new_job, dtype=float)
    trh = np.asarray(training_hours, dtype=float)
    cdi = np.asarray(city_development_index, dtype=float)
    t = np.asarray(target, dtype=float)

    mask_valid = ~np.isnan(exp) & ~np.isnan(lnj) & ~np.isnan(trh) & ~np.isnan(cdi) & ~np.isnan(t)
    exp = exp[mask_valid]
    lnj = lnj[mask_valid]
    trh = trh[mask_valid]
    cdi = cdi[mask_valid]
    t = t[mask_valid]

    if exp.size == 0:
        print("Q1: Không có dữ liệu hợp lệ để phân tích bộ feature.")
        return

    exp_bins = np.array([0, 1, 3, 7, 50])
    exp_labels = ["≤1 năm", "1–3 năm", "3–7 năm", ">7 năm"]

    lnj_bins = np.array([0, 1, 2, 4, 50])
    lnj_labels = ["≤1 năm", "1–2 năm", "2–4 năm", ">4 năm"]

    trh_quantiles = np.quantile(trh[~np.isnan(trh)], [0.25, 0.5, 0.75])
    trh_bins = np.concatenate([[0], trh_quantiles, [np.max(trh) + 1]])
    trh_labels = ["Thấp", "Trung bình", "Cao", "Rất cao"]

    cdi_bins = np.array([0.0, 0.6, 0.8, 1.0])
    cdi_labels = ["CDI thấp", "CDI trung bình", "CDI cao"]

    def bin_with_labels(x, bins, labels, right=True):
        idx = np.digitize(x, bins, right=right) - 1
        idx[idx < 0] = 0
        idx[idx >= len(labels)] = len(labels) - 1
        return idx, labels

    exp_idx, exp_labels = bin_with_labels(exp, exp_bins, exp_labels, right=True)
    lnj_idx, lnj_labels = bin_with_labels(lnj, lnj_bins, lnj_labels, right=True)
    trh_idx, trh_labels = bin_with_labels(trh, trh_bins, trh_labels, right=True)
    cdi_idx, cdi_labels = bin_with_labels(cdi, cdi_bins, cdi_labels, right=True)

    profiles = []

    for ie in range(len(exp_labels)):
        for il in range(len(lnj_labels)):
            for it in range(len(trh_labels)):
                for ic in range(len(cdi_labels)):
                    mask = (
                        (exp_idx == ie)
                        & (lnj_idx == il)
                        & (trh_idx == it)
                        & (cdi_idx == ic)
                    )
                    n = np.sum(mask)
                    if n < min_count:
                        continue
                    rate = np.mean(t[mask])
                    profiles.append(
                        {
                            "exp_label": exp_labels[ie],
                            "lnj_label": lnj_labels[il],
                            "trh_label": trh_labels[it],
                            "cdi_label": cdi_labels[ic],
                            "count": int(n),
                            "rate": float(rate),
                        }
                    )

    if not profiles:
        print("Q1: Không có profile nào đủ số lượng (min_count) để phân tích.")
        return

    profiles_sorted = sorted(profiles, key=lambda d: d["rate"], reverse=True)
    top_profiles = profiles_sorted[:top_n]

    line_width = 110
    print(
        f"Q1: Top {len(top_profiles)} bộ feature có tỷ lệ target=1 cao nhất "
        f"(chỉ xét profile có ít nhất {min_count} mẫu)"
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
    plt.bar(x, rates)
    for i, r in enumerate(rates):
        plt.text(i, r, f"{r*100:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, profile_names)
    plt.ylabel("Tỷ lệ target=1")
    plt.xlabel("Risk profile (P01, P02, ...)")
    plt.title("Q1: Top bộ feature theo tỷ lệ muốn đổi job")
    plt.tight_layout()
    plt.show()

def visualize_q2_training_hours_effect(training_hours, target, bin_edges=None,):
    x = np.asarray(training_hours, dtype=float)
    t = np.asarray(target, dtype=float)

    mask_valid = ~np.isnan(x) & ~np.isnan(t)
    x_valid = x[mask_valid]
    t_valid = t[mask_valid]

    if x_valid.size == 0:
        print("Q2: Không có dữ liệu hợp lệ cho training_hours.")
        return

    if bin_edges is None:
        bin_edges = np.array([0, 10, 20, 40, 80, 160, 500])

    bin_indices = np.digitize(x_valid, bin_edges, right=False)

    labels = []
    rates = []
    counts = []

    for b in range(1, len(bin_edges)):
        mask = bin_indices == b
        n = np.sum(mask)
        if n == 0:
            rates.append(np.nan)
            counts.append(0)
        else:
            rates.append(np.mean(t_valid[mask]))
            counts.append(n)
        labels.append(f"[{bin_edges[b-1]}, {bin_edges[b]})")

    rates = np.array(rates, dtype=float)
    counts = np.array(counts, dtype=int)

    print("Q2: Tỷ lệ target=1 theo các khoảng training_hours:")
    for lab, r, c in zip(labels, rates, counts):
        if np.isnan(r):
            print(f"{lab:15s}: n = {c:5d}, rate = N/A")
        else:
            print(f"{lab:15s}: n = {c:5d}, rate = {r:.3f}")

    plt.figure(figsize=(8, 4))
    plt.plot(labels, rates, marker="o")
    for i, r in enumerate(rates):
        if not np.isnan(r):
            plt.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Tỷ lệ target=1")
    plt.xlabel("Khoảng training_hours")
    plt.title("Q2: Tỷ lệ đổi job theo mức training_hours (binned)")
    plt.tight_layout()
    plt.show()

    boxplot_numeric_by_target(training_hours, target, "training_hours")
    plot_hist_overlay_by_target(training_hours, target, "training_hours", bins=30)

def visualize_q3_exp_cdi_interaction(
    experience,
    city_development_index,
    target,
):
    exp = np.asarray(experience, dtype=float)
    cdi = np.asarray(city_development_index, dtype=float)
    t = np.asarray(target, dtype=float)

    mask_valid = ~np.isnan(exp) & ~np.isnan(cdi) & ~np.isnan(t)
    exp_valid = exp[mask_valid]
    cdi_valid = cdi[mask_valid]
    t_valid = t[mask_valid]

    if exp_valid.size == 0:
        print("Q3: Không có dữ liệu hợp lệ để phân tích.")
        return

    exp_bins = np.array([0, 1, 5, 10, 50])
    exp_labels = ["0-1", "1-5", "5-10", "10+"]

    exp_idx = np.digitize(exp_valid, exp_bins, right=True) - 1
    exp_idx[exp_idx < 0] = 0
    exp_idx[exp_idx >= len(exp_labels)] = len(exp_labels) - 1

    cdi_bins = np.array([0.0, 0.6, 0.8, 1.0])
    cdi_labels = ["Low", "Medium", "High"]

    cdi_idx = np.digitize(cdi_valid, cdi_bins, right=True) - 1
    cdi_idx[cdi_idx < 0] = 0
    cdi_idx[cdi_idx >= len(cdi_labels)] = len(cdi_labels) - 1

    heat = np.full((len(exp_labels), len(cdi_labels)), np.nan)

    for i in range(len(exp_labels)):
        for j in range(len(cdi_labels)):
            m = (exp_idx == i) & (cdi_idx == j)
            n = np.sum(m)
            if n > 0:
                heat[i, j] = np.mean(t_valid[m])

    print("Q3: Ma trận tỷ lệ target=1 (experience_group x cdi_group):")
    print(heat)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(heat, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(im, label="Tỷ lệ target=1")
    plt.xticks(np.arange(len(cdi_labels)), cdi_labels)
    plt.yticks(np.arange(len(exp_labels)), exp_labels)
    plt.xlabel("city_development_index group")
    plt.ylabel("experience group")
    plt.title("Q3: Tỷ lệ đổi job theo (experience, CDI)")
    for i in range(len(exp_labels)):
        for j in range(len(cdi_labels)):
            if not np.isnan(heat[i, j]):
                plt.text(
                    j,
                    i,
                    f"{heat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    plt.tight_layout()
    plt.show()


def visualize_q4_career_trajectories(
    experience,
    last_new_job,
    target,
):
    exp = np.asarray(experience, dtype=float)
    lnj = np.asarray(last_new_job, dtype=float)
    t = np.asarray(target, dtype=float)

    def classify_trajectory(e, l):
        if np.isnan(e) or np.isnan(l):
            return "Unknown"
        if e <= 1:
            return "Newcomer"
        if l <= 1:
            return "Frequent mover"
        if l >= 4:
            return "Stable"
        return "Intermediate"

    categories = []
    t_valid = []
    for e, l, y in zip(exp, lnj, t):
        if np.isnan(y):
            continue
        categories.append(classify_trajectory(e, l))
        t_valid.append(y)

    if len(t_valid) == 0:
        print("Q4: Không có dữ liệu hợp lệ để phân tích.")
        return

    categories = np.array(categories, dtype=object)
    t_valid = np.array(t_valid, dtype=float)

    types, counts = np.unique(categories, return_counts=True)
    print("Q4: Các loại trajectory và số lượng:")
    for typ, c in zip(types, counts):
        print(f"{typ:15s}: {c}")

    rates = []
    for typ in types:
        m = categories == typ
        rates.append(np.mean(t_valid[m]))
    rates = np.array(rates, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.bar(types, rates)
    for i, r in enumerate(rates):
        plt.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Tỷ lệ target=1")
    plt.xlabel("Career trajectory")
    plt.title("Q4: Tỷ lệ đổi job theo kiểu hành trình nghề nghiệp")
    plt.tight_layout()
    plt.show()


def visualize_q5_missing_patterns(
    missing_company_info,
    missing_edu_info,
    target,
    label_company_present="company info present",
    label_company_missing="company info missing",
    label_edu_present="edu info present",
    label_edu_missing="edu info missing",
):
    t = np.asarray(target, dtype=float)
    m_comp = np.asarray(missing_company_info, dtype=int)
    m_edu = np.asarray(missing_edu_info, dtype=int)

    mask_valid = ~np.isnan(t)

    print("Q5: Tỷ lệ target=1 theo missing_company_info:")
    for v, lab in [(0, label_company_present), (1, label_company_missing)]:
        m = (m_comp == v) & mask_valid
        n = np.sum(m)
        if n == 0:
            continue
        rate = np.mean(t[m])
        print(f"{lab:22s}: n = {n:5d}, rate = {rate:.3f}")

    print("\nQ5: Tỷ lệ target=1 theo missing_edu_info:")
    for v, lab in [(0, label_edu_present), (1, label_edu_missing)]:
        m = (m_edu == v) & mask_valid
        n = np.sum(m)
        if n == 0:
            continue
        rate = np.mean(t[m])
        print(f"{lab:20s}: n = {n:5d}, rate = {rate:.3f}")

    labels_comp = [label_company_present, label_company_missing]
    rates_comp = []
    for v in [0, 1]:
        m = (m_comp == v) & mask_valid
        if np.sum(m) == 0:
            rates_comp.append(np.nan)
        else:
            rates_comp.append(np.mean(t[m]))

    plt.figure(figsize=(5, 4))
    plt.bar(labels_comp, rates_comp)
    for i, r in enumerate(rates_comp):
        if not np.isnan(r):
            plt.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Tỷ lệ target=1")
    plt.title("Q5: Tỷ lệ đổi job theo việc thiếu thông tin công ty")
    plt.tight_layout()
    plt.show()

    labels_edu = [label_edu_present, label_edu_missing]
    rates_edu = []
    for v in [0, 1]:
        m = (m_edu == v) & mask_valid
        if np.sum(m) == 0:
            rates_edu.append(np.nan)
        else:
            rates_edu.append(np.mean(t[m]))

    plt.figure(figsize=(5, 4))
    plt.bar(labels_edu, rates_edu)
    for i, r in enumerate(rates_edu):
        if not np.isnan(r):
            plt.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Tỷ lệ target=1")
    plt.title("Q5: Tỷ lệ đổi job theo việc thiếu thông tin học vấn")
    plt.tight_layout()
    plt.show()


def visualize_q6_risk_buckets(p_hat, target):
    p = np.asarray(p_hat, dtype=float)
    t = np.asarray(target, dtype=float)

    mask_valid = ~np.isnan(p) & ~np.isnan(t)
    p_valid = p[mask_valid]
    t_valid = t[mask_valid]

    if p_valid.size == 0:
        print("Q6: Không có dữ liệu hợp lệ (p_hat, target).")
        return

    risk_levels = np.empty_like(p_valid, dtype=object)
    risk_levels[p_valid <= 0.33] = "Low"
    risk_levels[(p_valid > 0.33) & (p_valid <= 0.66)] = "Medium"
    risk_levels[p_valid > 0.66] = "High"

    types, counts = np.unique(risk_levels, return_counts=True)
    print("Q6: Số lượng ứng viên theo risk bucket:")
    for typ, c in zip(types, counts):
        print(f"{typ:6s}: {c}")

    rates = []
    for typ in types:
        m = risk_levels == typ
        rates.append(np.mean(t_valid[m]))
    rates = np.array(rates, dtype=float)

    plt.figure(figsize=(5, 4))
    plt.bar(types, rates)
    for i, r in enumerate(rates):
        plt.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Tỷ lệ target=1 (thực tế)")
    plt.xlabel("Risk bucket (dự đoán)")
    plt.title("Q6: Hiệu quả phân loại mức độ rủi ro đổi job")
    plt.tight_layout()
    plt.show()