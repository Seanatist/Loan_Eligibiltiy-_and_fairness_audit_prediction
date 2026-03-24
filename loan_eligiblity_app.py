import streamlit as st

from data_prep import (
    basic_eda_summary,
    clean_loan_data,
    generate_synthetic_loan_data,
    train_test_splits,
)
from fairness_audit import fairness_summary
from model_training import train_and_evaluate


st.set_page_config(page_title="Loan Eligibility & Fairness Audit", layout="wide")


@st.cache_data
def load_data():
    df_raw = generate_synthetic_loan_data()
    df_clean = clean_loan_data(df_raw)
    return df_raw, df_clean


@st.cache_resource
def train_model(df_clean):
    X_train, X_test, y_train, y_test = train_test_splits(df_clean)
    model, test_metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    return model, X_test, y_test, test_metrics


def render_data_section(df_raw, df_clean):
    st.subheader("1. Data Cleaning & Analysis")

    tabs = st.tabs(["Raw vs Cleaned", "Summary Stats", "Group Distributions"])

    with tabs[0]:
        st.write("**Raw data (sample)**")
        st.dataframe(df_raw.head())
        st.write("**Cleaned data (sample)**")
        st.dataframe(df_clean.head())

    summary = basic_eda_summary(df_clean)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape**")
            st.json({"rows": summary["shape"][0], "columns": summary["shape"][1]})
            st.write("**Class balance (approved)**")
            st.json(summary["class_balance"])
        with col2:
            st.write("**Numerical features summary**")
            st.json(summary["numerical_summary"])

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Gender counts**")
            st.json(summary["group_counts_gender"])
        with c2:
            st.write("**Race counts**")
            st.json(summary["group_counts_race"])


def render_model_section(df_clean):
    st.subheader("2. Model Training & Evaluation")

    model, X_test, y_test, test_metrics = train_model(df_clean)
    st.write("**Test metrics**")
    st.json(test_metrics)

    return model, X_test, y_test


def render_fairness_section(model, X_test, y_test):
    st.subheader("3. Fairness Audit")

    df_test = X_test.copy()
    y_pred = model.predict(df_test)

    sensitive_feature = st.selectbox("Sensitive attribute for audit", ["gender", "race"])

    group_metrics, fairness = fairness_summary(
        df_test=df_test,
        y_true=y_test.values,
        y_pred=y_pred,
        sensitive_column=sensitive_feature,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Group-wise performance (precision & recall)**")
        st.json(group_metrics)
    with c2:
        st.write("**Fairness summary metrics**")
        st.json(fairness)

    st.caption(
        "Lower absolute values of demographic parity and equalized odds differences "
        "indicate smaller gaps between groups."
    )


def main():
    st.title("Loan Eligibility & Fairness Audit")
    st.markdown(
        "This demo generates synthetic loan data, trains a model to predict loan approval, "
        "and then audits the model for fairness across selected sensitive attributes."
    )

    df_raw, df_clean = load_data()

    render_data_section(df_raw, df_clean)
    model, X_test, y_test = render_model_section(df_clean)
    render_fairness_section(model, X_test, y_test)


if __name__ == "__main__":
    main()

