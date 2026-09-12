
"""
Lab-4: Applied Statistical Modeling & Interactive Web Dashboard
Dataset: Restaurant Tipping Behavior
Run with: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
import streamlit as st

st.set_page_config(page_title="Tipping Behavior Analytics", layout="wide", page_icon="🍽️")


@st.cache_data
def load_data():
    df = pd.read_csv("data/tips.csv")
    df["tip_pct"] = df["tip"] / df["total_bill"] * 100
    return df


df_raw = load_data()


@st.cache_resource
def fit_model(df):
    m = df.copy()
    m["smoker_bin"] = (m.smoker == "Yes").astype(int)
    m["sex_bin"] = (m.sex == "Male").astype(int)
    m["time_bin"] = (m.time == "Dinner").astype(int)
    m = pd.get_dummies(m, columns=["day"], drop_first=True)
    day_cols = [c for c in m.columns if c.startswith("day_")]
    feature_cols = ["total_bill", "size", "smoker_bin", "sex_bin", "time_bin"] + day_cols
    X = sm.add_constant(m[feature_cols].astype(float))
    y = m["tip"].astype(float)
    model = sm.OLS(y, X).fit()
    return model, feature_cols, day_cols


model, feature_cols, day_cols = fit_model(df_raw)
all_days = list(df_raw["day"].unique())

st.title("🍽️ Restaurant Tipping Behavior — Statistical Dashboard")
st.caption("M.Sc. Data Science · Lab-4 · EDA → Hypothesis Testing → OLS Regression → Interactive Dashboard")

st.sidebar.header("🔎 Filters — Data Exploration")
bill_range = st.sidebar.slider(
    "Total bill range ($)", float(df_raw.total_bill.min()), float(df_raw.total_bill.max()),
    (float(df_raw.total_bill.min()), float(df_raw.total_bill.max()))
)
size_sel = st.sidebar.multiselect("Party size", sorted(df_raw["size"].unique()), default=sorted(df_raw["size"].unique()))
day_sel = st.sidebar.multiselect("Day", all_days, default=all_days)
smoker_sel = st.sidebar.multiselect("Smoker", ["Yes", "No"], default=["Yes", "No"])

df = df_raw[
    (df_raw.total_bill.between(*bill_range)) &
    (df_raw["size"].isin(size_sel)) &
    (df_raw.day.isin(day_sel)) &
    (df_raw.smoker.isin(smoker_sel))
]

tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"])

with tab1:
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (filtered)", f"{len(df):,}")
    c2.metric("Avg. Tip", f"${df.tip.mean():.2f}" if len(df) else "—")
    c3.metric("Avg. Tip %", f"{df.tip_pct.mean():.1f}%" if len(df) else "—")
    c4.metric("Avg. Bill", f"${df.total_bill.mean():.2f}" if len(df) else "—")

    if df.empty:
        st.warning("No rows match the current filters.")
    else:
        st.dataframe(df.describe().T.style.format("{:.2f}"), width='stretch')
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            num_col = st.selectbox("Feature to visualize distribution", ["total_bill", "tip", "tip_pct", "size"], index=1)
            fig_hist = px.histogram(df, x=num_col, color="smoker", marginal="box", nbins=30, opacity=0.75,
                                     title=f"Distribution of {num_col} (by smoker status)")
            st.plotly_chart(fig_hist, width='stretch')
        with col_b:
            fig_scatter = px.scatter(df, x="total_bill", y="tip", color="day", size="size",
                                      hover_data=["sex", "time"], title="Tip vs Total Bill", opacity=0.7)
            st.plotly_chart(fig_scatter, width='stretch')
        st.markdown("---")
        st.markdown("**Correlation matrix**")
        corr = df[["total_bill", "tip", "size", "tip_pct"]].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, width='stretch')

with tab2:
    st.subheader("Interactive Hypothesis Testing")
    cat_options = ["smoker", "sex", "day", "time"]
    num_options = ["tip", "tip_pct", "total_bill", "size"]

    colx, coly = st.columns(2)
    with colx:
        cat_factor = st.selectbox("Categorical factor", cat_options, index=0)
    with coly:
        num_metric = st.selectbox("Numerical metric", num_options, index=1)

    groups_dict = {lvl: df_raw.loc[df_raw[cat_factor] == lvl, num_metric] for lvl in df_raw[cat_factor].unique()}
    n_groups = len(groups_dict)
    st.markdown(f"**Groups in `{cat_factor}`:** {list(groups_dict.keys())} ({n_groups} groups)")

    st.markdown("#### Step 1 — Normality Check (Shapiro-Wilk)")
    norm_rows, all_normal = [], True
    for lvl, vals in groups_dict.items():
        sw = stats.shapiro(vals) if len(vals) >= 3 else None
        p = sw.pvalue if sw else np.nan
        norm_rows.append({"Group": lvl, "n": len(vals), "Shapiro p-value": p, "Normal (p>0.05)?": p > 0.05 if not np.isnan(p) else "N/A"})
        if np.isnan(p) or p <= 0.05:
            all_normal = False
    st.dataframe(pd.DataFrame(norm_rows), width='stretch')

    if n_groups == 2:
        st.markdown("#### Step 2 — Equal Variance (Levene's Test)")
        vals_list = list(groups_dict.values())
        lev = stats.levene(*vals_list)
        st.write(f"Levene's statistic = **{lev.statistic:.4f}**, p-value = **{lev.pvalue:.4e}**")
        equal_var = lev.pvalue > 0.05

        st.markdown("#### Step 3 — Test Result")
        if all_normal:
            test_res = stats.ttest_ind(*vals_list, equal_var=equal_var)
            test_name = "Two-Sample t-test" + (" (equal var)" if equal_var else " (Welch's)")
        else:
            test_res = stats.mannwhitneyu(*vals_list, alternative="two-sided")
            test_name = "Mann-Whitney U test (non-parametric)"

        r1, r2, r3 = st.columns(3)
        r1.metric("Test used", test_name)
        r2.metric("Statistic", f"{test_res.statistic:.3f}")
        r3.metric("p-value", f"{test_res.pvalue:.4e}")

        verdict = "🔴 Reject H0" if test_res.pvalue < 0.05 else "🟢 Fail to Reject H0"
        st.markdown(f"**Conclusion at α = 0.05:** {verdict}")
        st.plotly_chart(px.box(df_raw, x=cat_factor, y=num_metric, color=cat_factor, points="all"), width='stretch')

    elif n_groups > 2:
        st.markdown("#### Step 2 — One-Way ANOVA")
        vals_list = list(groups_dict.values())
        anova = stats.f_oneway(*vals_list)
        r1, r2 = st.columns(2)
        r1.metric("F-statistic", f"{anova.statistic:.4f}")
        r2.metric("p-value", f"{anova.pvalue:.4e}")
        verdict = "🔴 Reject H0" if anova.pvalue < 0.05 else "🟢 Fail to Reject H0"
        st.markdown(f"**Conclusion at α = 0.05:** {verdict}")
        st.plotly_chart(px.box(df_raw, x=cat_factor, y=num_metric, color=cat_factor, points="all"), width='stretch')

    st.markdown("---")
    st.markdown("#### Bonus — Chi-Square Test of Independence")
    cc1, cc2 = st.columns(2)
    with cc1:
        cat1 = st.selectbox("Categorical variable 1", cat_options, index=0, key="chi1")
    with cc2:
        cat2 = st.selectbox("Categorical variable 2", cat_options, index=2, key="chi2")
    if cat1 != cat2:
        ct = pd.crosstab(df_raw[cat1], df_raw[cat2])
        chi2, chip, dof, exp = stats.chi2_contingency(ct)
        st.write(f"Chi² = **{chi2:.4f}**, p-value = **{chip:.4e}**, dof = **{dof}**")
        verdict = "🔴 Reject H0 — associated" if chip < 0.05 else "🟢 Fail to Reject H0 — no association"
        st.markdown(f"**Conclusion at α = 0.05:** {verdict}")
        st.dataframe(ct, width='stretch')
    else:
        st.info("Pick two different categorical variables.")

with tab3:
    st.subheader("Model: Multiple OLS Regression — Predicting Tip Amount")
    st.latex(r"Tip = \beta_0 + \beta_1 TotalBill + \beta_2 Size + \beta_3 Smoker + \beta_4 Sex + \beta_5 Time + \sum \beta_i Day_i + \varepsilon")

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{model.rsquared:.4f}")
    c2.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
    c3.metric("F-statistic p-value", f"{model.f_pvalue:.2e}")

    with st.expander("Full statsmodels OLS summary"):
        st.text(model.summary().as_text())

    st.markdown("---")
    st.markdown("### 🔮 Predict a New Bill's Tip")
    p1, p2, p3 = st.columns(3)
    with p1:
        in_bill = st.slider("Total bill ($)", 3.0, 55.0, 20.0, step=0.5)
        in_size = st.slider("Party size", 1, 6, 2)
    with p2:
        in_smoker = st.radio("Smoker", ["Yes", "No"], horizontal=True)
        in_sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    with p3:
        in_time = st.radio("Time", ["Dinner", "Lunch"], horizontal=True)
        in_day = st.selectbox("Day", all_days)

    row = {c: 0.0 for c in feature_cols}
    row["total_bill"] = in_bill
    row["size"] = in_size
    row["smoker_bin"] = 1.0 if in_smoker == "Yes" else 0.0
    row["sex_bin"] = 1.0 if in_sex == "Male" else 0.0
    row["time_bin"] = 1.0 if in_time == "Dinner" else 0.0
    day_col = f"day_{in_day}"
    if day_col in row:
        row[day_col] = 1.0
    X_new = pd.DataFrame([row])[feature_cols]
    X_new = sm.add_constant(X_new, has_constant="add")
    X_new = X_new[model.params.index]

    pred = model.get_prediction(X_new)
    pred_summary = pred.summary_frame(alpha=0.05)
    point_est = pred_summary["mean"].iloc[0]
    ci_low, ci_high = pred_summary["mean_ci_lower"].iloc[0], pred_summary["mean_ci_upper"].iloc[0]
    pi_low, pi_high = pred_summary["obs_ci_lower"].iloc[0], pred_summary["obs_ci_upper"].iloc[0]

    st.markdown("#### Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Tip", f"${point_est:.2f}")
    r2.metric("95% CI (mean)", f"${ci_low:.2f} – ${ci_high:.2f}")
    r3.metric("95% PI (individual)", f"${pi_low:.2f} – ${pi_high:.2f}")

    st.markdown("---")
    st.markdown("### 📉 Residual Diagnostics (Gauss–Markov Assumptions)")
    resid = model.resid
    fitted_vals = model.fittedvalues

    d1, d2 = st.columns(2)
    with d1:
        fig_rf = px.scatter(x=fitted_vals, y=resid, opacity=0.6,
                             labels={"x": "Fitted values", "y": "Residuals"},
                             title="Residuals vs. Fitted")
        fig_rf.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rf, width='stretch')
    with d2:
        qq = sm.qqplot(resid, line="45", fit=True)
        qq_ax = qq.axes[0]
        x_qq, y_qq = qq_ax.get_lines()[0].get_data()
        ref_x, ref_y = qq_ax.get_lines()[1].get_data()
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=x_qq, y=y_qq, mode="markers", name="Residuals"))
        fig_qq.add_trace(go.Scatter(x=ref_x, y=ref_y, mode="lines", name="45° reference", line=dict(color="red")))
        fig_qq.update_layout(title="Normal Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, width='stretch')

    jb_stat, jb_p, _, _ = sm.stats.stattools.jarque_bera(resid)
    st.markdown(f"**Jarque-Bera test:** stat = {jb_stat:.2f}, p = {jb_p:.2e} "
                f"({'residuals deviate from normality' if jb_p < 0.05 else 'residuals approx. normal'})")

    st.markdown("#### Multicollinearity — VIF")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_feats = ["total_bill", "size"]
    Xvif = sm.add_constant(df_raw[vif_feats].astype(float))
    vif_data = pd.DataFrame({
        "Feature": ["const"] + vif_feats,
        "VIF": [variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1])]
    })
    st.dataframe(vif_data, width='stretch')
    st.caption("VIF < 5 for all continuous predictors ⇒ no multicollinearity concern.")