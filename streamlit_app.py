# OptimaLife churn prediction dashboard using pre-computed predictions
# Co-authored with CoCo
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="OptimaLife Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("OptimaLife Product Portfolio — Churn Risk Dashboard")

st.caption(
    "Interactive dashboard for evaluating customer churn risk, "
    "model performance, revenue trends, and retention metrics "
    "across the OptimaLife product portfolio."
)

# --- Dashboard notes ---
st.subheader("Dashboard Notes")

st.info(
    '''
**This dashboard has been optimized for demonstration on Streamlit Community Cloud.**

- Customer-level data displayed in this dashboard represents a **representative sample of up to 10,000 customers per product** to optimize application performance.

- Aggregate visualizations, model comparisons, and performance metrics reflect the underlying analytical results.

- Customer-level tables, the **Highest Risk Customers** list, and the **Customer Lookup** feature are limited to the sampled records.

- If a customer cannot be found using the Customer Lookup tool, they may not be included in the demonstration sample.
'''
)

# --- Load data ---
@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent

    predictions = pd.read_csv(base_path / "churn_predictions.csv")
    comparison = pd.read_csv(base_path / "model_comparison_results.csv")
    best = pd.read_csv(base_path / "best_model_per_product.csv")
    arr = pd.read_csv(base_path / "arr_by_product.csv")
    renewal = pd.read_csv(base_path / "renewal_by_product.csv")
    cltr = pd.read_csv(base_path / "cltr_by_income.csv")

    # Renewal probability is the complement of churn probability.
    if "RENEWAL_PROBABILITY" not in predictions.columns:
        predictions["RENEWAL_PROBABILITY"] = (
            1 - predictions["CHURN_PROBABILITY"]
        )

    return predictions, comparison, best, arr, renewal, cltr


predictions, comparison, best_df, arr_df, renewal_df, cltr_df = load_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")

all_products = sorted(
    predictions["PRODUCT"].dropna().unique()
)

selected_products = st.sidebar.multiselect(
    "Products",
    options=all_products,
    default=all_products
)

filtered = predictions[
    predictions["PRODUCT"].isin(selected_products)
]

if filtered.empty:
    st.warning(
        "Select at least one product in the sidebar "
        "to view the dashboard."
    )
    st.stop()

# --- Tab layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Churn Predictions",
    "Model Comparison",
    "Risk Distribution",
    "Revenue & Retention",
    "Customer Lookup"
])

# --- Tab 1: Churn Predictions ---
with tab1:
    st.header("Churn Predictions by Product")

    predict_product = st.selectbox(
        "Select Product",
        options=sorted(filtered["PRODUCT"].unique()),
        key="pred_product"
    )

    product_preds = filtered[
        filtered["PRODUCT"] == predict_product
    ]

    # Model info
    model_name = product_preds["BEST_MODEL"].iloc[0]
    model_auc = product_preds["MODEL_AUC"].iloc[0]

    st.info(
        f"**Model:** {model_name}  |  "
        f"**ROC AUC:** {model_auc:.3f}"
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers in Demo Sample",
        f"{len(product_preds):,}"
    )

    c2.metric(
        "Avg Churn Prob",
        f"{product_preds['CHURN_PROBABILITY'].mean():.1%}"
    )

    c3.metric(
        "Actual Churn Rate",
        f"{(1 - product_preds['RENEWED'].mean()):.1%}"
    )

    c4.metric(
        "High Risk (Q4)",
        f"{(
            product_preds['RISK_QUARTILE']
            == 'Q4 (High Risk)'
        ).sum():,}"
    )

    # Quartile breakdown
    st.subheader("Risk Quartile Summary")

    q_summary = (
        product_preds
        .groupby(
            "RISK_QUARTILE",
            observed=True
        )
        .agg(
            CUSTOMERS=(
                "CUSTOMER_ID",
                "count"
            ),
            AVG_CHURN_PROB=(
                "CHURN_PROBABILITY",
                "mean"
            ),
            ACTUAL_CHURN_RATE=(
                "RENEWED",
                lambda x: 1 - x.mean()
            )
        )
        .reset_index()
    )

    fig_q = px.bar(
        q_summary,
        x="RISK_QUARTILE",
        y="ACTUAL_CHURN_RATE",
        text_auto=".1%",
        title=(
            f"{predict_product} — "
            "Actual Churn by Risk Quartile"
        )
    )

    fig_q.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig_q,
        width="stretch"
    )

    # Top at-risk customers
    st.subheader("Highest Risk Customers")

    st.caption(
        "This table includes only customers contained "
        "in the demonstration sample."
    )

    highest_risk = (
        product_preds
        .nlargest(
            100,
            "CHURN_PROBABILITY"
        )[
            [
                "CUSTOMER_ID",
                "CHURN_PROBABILITY",
                "RENEWAL_PROBABILITY",
                "RISK_QUARTILE",
                "RENEWED"
            ]
        ]
    )

    st.dataframe(
        highest_risk,
        width="stretch",
        hide_index=True,
        column_config={
            "CUSTOMER_ID": st.column_config.NumberColumn(
                "Customer ID",
                format="%d"
            ),
            "CHURN_PROBABILITY": st.column_config.ProgressColumn(
                "Churn Probability",
                min_value=0,
                max_value=1,
                format="%.1%%"
            ),
            "RENEWAL_PROBABILITY": st.column_config.ProgressColumn(
                "Renewal Probability",
                min_value=0,
                max_value=1,
                format="%.1%%"
            ),
            "RISK_QUARTILE": "Risk Quartile",
            "RENEWED": st.column_config.NumberColumn(
                "Renewed",
                format="%d"
            )
        }
    )

# --- Tab 2: Model Comparison ---
with tab2:
    st.header("Model Performance by Product")

    comparison_filtered = comparison[
        comparison["PRODUCT"].isin(selected_products)
    ]

    fig_comp = px.bar(
        comparison_filtered,
        x="PRODUCT",
        y="ROC_AUC",
        color="MODEL",
        barmode="group",
        title="ROC AUC by Model and Product",
        text_auto=".3f"
    )

    fig_comp.update_layout(
        yaxis_range=[0.5, 1.0]
    )

    st.plotly_chart(
        fig_comp,
        width="stretch"
    )

    st.subheader("Best Model per Product")

    st.dataframe(
        best_df[
            best_df["PRODUCT"].isin(selected_products)
        ],
        width="stretch",
        hide_index=True
    )

# --- Tab 3: Risk Distribution ---
with tab3:
    st.header("Churn Probability Distribution")

    fig_hist = px.histogram(
        filtered,
        x="CHURN_PROBABILITY",
        color="PRODUCT",
        nbins=50,
        title="Distribution of Predicted Churn Probabilities",
        opacity=0.7
    )

    fig_hist.update_xaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig_hist,
        width="stretch"
    )

    # Cross-product quartile comparison
    st.subheader(
        "Actual Churn Rate by Risk Quartile "
        "(All Selected Products)"
    )

    cross_q = (
        filtered
        .groupby(
            ["PRODUCT", "RISK_QUARTILE"],
            observed=True
        )
        .agg(
            ACTUAL_CHURN_RATE=(
                "RENEWED",
                lambda x: 1 - x.mean()
            )
        )
        .reset_index()
    )

    fig_cross = px.bar(
        cross_q,
        x="RISK_QUARTILE",
        y="ACTUAL_CHURN_RATE",
        color="PRODUCT",
        barmode="group",
        text_auto=".1%"
    )

    fig_cross.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig_cross,
        width="stretch"
    )

# --- Tab 4: Revenue & Retention ---
with tab4:
    st.header("ARR & Renewal Trends")

    col1, col2 = st.columns(2)

    with col1:
        arr_df["YEAR_DATE"] = pd.to_datetime(
            arr_df["YEAR_DATE"],
            errors="coerce"
        )

        arr_filtered = arr_df[
            arr_df["PRODUCT"].isin(selected_products)
        ]

        fig_arr = px.line(
            arr_filtered,
            x="YEAR_DATE",
            y="ARR",
            color="PRODUCT",
            title="Annual Recurring Revenue by Product",
            markers=True
        )

        fig_arr.update_yaxes(
            tickprefix="$",
            tickformat=",.0f"
        )

        st.plotly_chart(
            fig_arr,
            width="stretch"
        )

    with col2:
        renewal_df["YEAR_DATE"] = pd.to_datetime(
            renewal_df["YEAR_DATE"],
            errors="coerce"
        )

        renewal_filtered = renewal_df[
            renewal_df["PRODUCT"].isin(selected_products)
        ]

        fig_ren = px.line(
            renewal_filtered,
            x="YEAR_DATE",
            y="RENEWAL_RATE",
            color="PRODUCT",
            title="Customer Renewal Rate by Product",
            markers=True
        )

        fig_ren.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig_ren,
            width="stretch"
        )

    st.subheader("Average CLTR by Income Level")

    fig_cltr = px.bar(
        cltr_df,
        x="INCOME_LEVEL",
        y="AVERAGE_CLTR",
        title="Average Customer Lifetime Revenue by Income Level",
        text_auto="$,.0f"
    )

    fig_cltr.update_yaxes(
        tickprefix="$",
        tickformat=",.0f"
    )

    st.plotly_chart(
        fig_cltr,
        width="stretch"
    )

# --- Tab 5: Customer Lookup ---
with tab5:
    st.header("Customer Lookup")

    st.info(
        "Customer lookup is limited to the demonstration "
        "sample of up to 10,000 customers per product. "
        "A valid customer may not appear if they were not "
        "selected for the sample."
    )

    search_id = st.number_input(
        "Enter Customer ID",
        min_value=0,
        step=1
    )

    if search_id > 0:
        customer_rows = predictions[
            predictions["CUSTOMER_ID"] == search_id
        ]

        if customer_rows.empty:
            st.warning(
                "Customer ID not found in the demonstration "
                "sample. The customer may exist in the full "
                "dataset but may not have been selected for "
                "this application."
            )
        else:
            st.success(
                f"Found {len(customer_rows):,} matching "
                "record(s) in the demonstration sample."
            )

            st.dataframe(
                customer_rows,
                width="stretch",
                hide_index=True
            )
