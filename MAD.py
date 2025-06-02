print("\n🔍 Benford's Law Analysis:")
for col in features:
    print(f"\n📊 Checking '{col}':")
    col_data = df[col]
    actual_rel, actual_abs, valid_count = leading_digit_distribution(col_data)
    actual_rel_full = actual_rel.reindex(benford_dist.index, fill_value=0)
    actual_abs_full = actual_abs.reindex(benford_dist.index, fill_value=0)
    expected_abs = (benford_dist * valid_count).round().astype(int)

    # ---- MAD and interpretation ----
    mad = np.mean(np.abs(actual_rel_full - benford_dist))
    if mad < 0.006:
        conformity = "✔️ Close conformity (likely Benford-compliant)"
        color = 'green'
    elif mad < 0.012:
        conformity = "⚠️ Acceptable conformity (some deviation)"
        color = 'orange'
    elif mad < 0.015:
        conformity = "❗Marginal conformity (possible issue)"
        color = 'darkorange'
    else:
        conformity = "❌ Non-conforming (does not follow Benford's Law)"
        color = 'red'

    print(f"  ➤ Mean Absolute Deviation (MAD): {mad:.4f} → {conformity}")

    # ---- Plot comparison ----
    comparison_df = pd.DataFrame({
        'Benford': benford_dist,
        'Actual': actual_rel_full
    })

    ax = comparison_df.plot(kind='bar', figsize=(8, 4), title=f"Benford's Law vs Actual for {col}", color=['gray', color])
    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Proportion")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Chi-square test ----
    if expected_abs.sum() == actual_abs_full.sum():
        chi_stat, p_value = chisquare(actual_abs_full, f_exp=expected_abs)
        print(f"  ➤ Chi-Square Statistic: {chi_stat:.2f}, p-value: {p_value:.4f}")
    else:
        print(f"  ⚠️ Chi-square skipped: count mismatch (expected {expected_abs.sum()}, actual {actual_abs_full.sum()})")
