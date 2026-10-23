
import os
from datetime import date

import requests
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "PREDICTION_API_URL",
    "https://superstore-api.omar6325.blitz.cloud/predict",
)

st.set_page_config(
    page_title="Superstore | Profit Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #132744;
        --navy-light: #203D62;
        --teal: #21C7B7;
        --background: #F3F6FA;
        --text: #1B2A40;
        --muted: #77869A;
        --border: #E2E8F0;
        --white: #FFFFFF;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding: 2rem 2.5rem 3rem;
    }

    h1, h2, h3, p, label {
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1.2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 28px;
    }

    .brand-icon {
        background: var(--navy);
        color: var(--teal);
        border-radius: 12px;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
    }

    .brand-name {
        color: var(--navy);
        font-size: 17px;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 11px;
        margin-top: 2px;
    }

    .sidebar-heading {
        color: var(--navy);
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin: 8px 0 4px;
    }

    .sidebar-description {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    /* Main header */
    .topline {
        color: #159E92;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .page-title {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(30px, 3.5vw, 43px);
        font-weight: 800;
        letter-spacing: -1.6px;
        line-height: 1.15;
        margin-bottom: 12px;
    }

    .page-description {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.8;
        max-width: 720px;
        margin-bottom: 25px;
    }

    .hero {
        background: linear-gradient(
            115deg,
            #132744 0%,
            #203D62 68%,
            #176C75 100%
        );
        border-radius: 20px;
        padding: 26px 30px;
        color: white;
        margin: 10px 0 24px;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -55px;
        top: -95px;
        width: 220px;
        height: 220px;
        border: 35px solid rgba(255,255,255,0.07);
        border-radius: 50%;
    }

    .hero-label {
        color: #8DEBE1;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.8px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #D3DFEB;
        font-size: 13px;
        line-height: 1.7;
        max-width: 650px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 105px;
        box-shadow: 0 4px 14px rgba(19,39,68,0.025);
    }

    .metric-label {
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
    }

    /* Prediction result */
    .result-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 26px;
        margin-top: 24px;
        box-shadow: 0 8px 25px rgba(19,39,68,0.04);
    }

    .result-label {
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.8px;
        margin-bottom: 15px;
    }

    .result-status {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(28px, 4vw, 38px);
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 10px;
    }

    .result-description {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.8;
    }

    .empty-card {
        background: white;
        border: 1px dashed #C9D4E2;
        border-radius: 18px;
        padding: 38px 25px;
        text-align: center;
        margin-top: 24px;
    }

    .empty-icon {
        font-size: 30px;
        margin-bottom: 12px;
    }

    .empty-title {
        color: var(--navy);
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .empty-text {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }

    /* ========================================================
       INPUT VISIBILITY FIX
       ======================================================== */

    /* Number inputs, text inputs and date inputs */
    .stApp input,
    .stApp textarea,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #1B2A40 !important;
        -webkit-text-fill-color: #1B2A40 !important;
        background-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* Input backgrounds and borders */
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="select"] > div,
    .stApp [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        border-radius: 9px !important;
    }

    /* Selected dropdown values */
    .stApp [data-baseweb="select"] div,
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] input {
        color: #1B2A40 !important;
        -webkit-text-fill-color: #1B2A40 !important;
    }

    /* Dropdown menu options */
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li,
    [role="option"] {
        color: #1B2A40 !important;
        background-color: #FFFFFF !important;
    }

    /* Number input increment/decrement controls */
    .stApp [data-testid="stNumberInput"] button {
        color: #1B2A40 !important;
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }

    /* Date input */
    .stApp [data-testid="stDateInput"] input {
        color: #1B2A40 !important;
        -webkit-text-fill-color: #1B2A40 !important;
        background-color: #FFFFFF !important;
    }

    /* Input labels */
    .stApp [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #43546B !important;
        font-weight: 600;
        font-size: 12px;
    }

    /* Submit button */
    .stFormSubmitButton > button {
        background: var(--navy) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--navy) !important;
        border-radius: 10px;
        min-height: 47px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stFormSubmitButton > button:hover {
        background: #203D62 !important;
        color: #FFFFFF !important;
        border-color: #203D62 !important;
    }

    /* Section headings */
    .section-title {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: 19px;
        font-weight: 800;
        margin: 28px 0 5px;
    }

    .section-description {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 16px;
    }

    .footer {
        color: #8A97A8;
        font-size: 11px;
        text-align: center;
        border-top: 1px solid var(--border);
        padding-top: 20px;
        margin-top: 40px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        .hero {
            padding: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR: ORDER INPUTS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">📦</div>
            <div>
                <div class="brand-name">Superstore</div>
                <div class="brand-caption">Profit Intelligence</div>
            </div>
        </div>

        <div class="sidebar-heading">Order configuration</div>
        <div class="sidebar-description">
            Enter order details to estimate its profit status.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("**01 — Financial details**")

        sales = st.number_input(
            "Sales ($)",
            min_value=0.0,
            value=23.076,
            step=1.0,
            format="%.3f",
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=3,
            step=1,
        )

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            format="%.2f",
        )

        st.markdown("**02 — Product details**")

        category = st.selectbox(
            "Category",
            ["Furniture", "Office Supplies", "Technology"],
        )

        sub_category = st.selectbox(
            "Sub-category",
            [
                "Bookcases",
                "Chairs",
                "Furnishings",
                "Tables",
                "Appliances",
                "Art",
                "Binders",
                "Envelopes",
                "Fasteners",
                "Labels",
                "Paper",
                "Storage",
                "Supplies",
                "Accessories",
                "Copiers",
                "Machines",
                "Phones",
            ],
        )

        st.markdown("**03 — Customer and delivery**")

        segment = st.selectbox(
            "Customer segment",
            ["Consumer", "Corporate", "Home Office"],
        )

        region = st.selectbox(
            "Region",
            ["Central", "East", "South", "West"],
        )

        ship_mode = st.selectbox(
            "Shipping mode",
            [
                "Standard Class",
                "Second Class",
                "First Class",
                "Same Day",
            ],
        )

        order_date = st.date_input(
            "Order date",
            value=date(2025, 1, 15),
            min_value=date(2000, 1, 1),
            max_value=date(2100, 12, 31),
        )

        submitted = st.form_submit_button(
            "Generate prediction  →",
            use_container_width=True,
        )

# ============================================================
# DERIVED FEATURES
# ============================================================

order_year = order_date.year
order_month = order_date.month
order_quarter = (order_month - 1) // 3 + 1
order_day = order_date.day
order_day_of_week = order_date.weekday()
sales_per_unit = sales / quantity if quantity else 0.0

# ============================================================
# MAIN DASHBOARD
# ============================================================

st.markdown(
    """
    <div class="topline">Machine learning · Decision support</div>
    <div class="page-title">Profit Intelligence</div>
    <div class="page-description">
        Understand the potential profitability of an order using your
        trained machine learning model, served through an API.
    </div>

    <div class="hero">
        <div class="hero-label">SUPERSTORE ANALYTICS PLATFORM</div>
        <div class="hero-title">Turn order data into insight.</div>
        <div class="hero-text">
            Configure an order in the sidebar and generate a prediction.
            The model classifies whether the order is expected to produce
            positive profit or zero/negative profit.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4, gap="medium")

metrics = [
    (col1, "ORDER VALUE", f"${sales:,.2f}"),
    (col2, "QUANTITY", f"{quantity:,}"),
    (col3, "DISCOUNT", f"{discount:.0%}"),
    (col4, "SALES PER UNIT", f"${sales_per_unit:,.2f}"),
]

for column, label, value in metrics:
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="section-title">Prediction overview</div>
    <div class="section-description">
        Your model's result for the configured order.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PREDICTION
# ============================================================

if submitted:
    payload = {
        "features": {
            "Sales": float(sales),
            "Quantity": int(quantity),
            "Discount": float(discount),
            "Category": category,
            "Sub-Category": sub_category,
            "Segment": segment,
            "Region": region,
            "Ship Mode": ship_mode,
            "Order Year": int(order_year),
            "Order Month": int(order_month),
            "Order Quarter": int(order_quarter),
            "Order Day": int(order_day),
            "Order Day of Week": int(order_day_of_week),
            "Sales Per Unit": float(sales_per_unit),
        }
    }

    try:
        with st.spinner("Running prediction..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

        if response.ok:
            result = int(response.json()["prediction"][0])

            if result == 1:
                result_label = "Profitable"
                result_color = "#0C9B83"
                result_description = (
                    "The model predicts this order will generate "
                    "positive profit."
                )
                icon = "↗"
            else:
                result_label = "Unprofitable"
                result_color = "#D6535A"
                result_description = (
                    "The model predicts this order will generate "
                    "zero or negative profit."
                )
                icon = "↘"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        MODEL PREDICTION
                    </div>
                    <div class="result-status"
                         style="color: {result_color};">
                        {icon} &nbsp; {result_label}
                    </div>
                    <div class="result-description">
                        {result_description}
                        <br><br>
                        This is a prediction, not a guarantee of
                        the order's actual financial outcome.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View submitted order features"):
                st.json(payload["features"])

        else:
            st.error(
                f"API error ({response.status_code}): {response.text}"
            )

    except requests.RequestException as exc:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure FastAPI is running and API_URL is correct."
        )
        st.caption(str(exc))

else:
    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-icon">◎</div>
            <div class="empty-title">Ready to analyze your order?</div>
            <div class="empty-text">
                Set your order details in the sidebar, then select
                <b>Generate prediction</b> to see the model's result here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SUPERSTORE PROFIT INTELLIGENCE
        &nbsp; · &nbsp; Scikit-learn
        &nbsp; · &nbsp; MLflow
        &nbsp; · &nbsp; FastAPI
        &nbsp; · &nbsp; Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

