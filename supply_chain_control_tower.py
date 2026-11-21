from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NEXUS | Global Supply Chain Control Tower",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Light visual system ----------
NAVY = "#F4F7FA"
PANEL = "#FFFFFF"
PANEL_2 = "#FFFFFF"
TEAL = "#247C73"
BLUE = "#416A8A"
AMBER = "#A66A12"
RED = "#B64C5A"
MUTED = "#5B6B7C"
TEXT = "#1C2C3D"

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      html, body { background: #F4F7FA; color: #1C2C3D; }
      .stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF3F7 100%); color: #1C2C3D; }
      [data-testid="stAppViewContainer"] { background: transparent; }
      [data-testid="stHeader"] { background: rgba(248,250,252,.94); }
      [data-testid="stSidebar"] { background: #EAF0F5; border-right: 1px solid #D6E0E9; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] label { color: #26384B !important; }
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] small { color: #536579; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] input { background: #FFFFFF; color: #1C2C3D; border-color: #C9D4DF; }
      [data-testid="stFileUploaderDropzone"] { background: #FFFFFF; border: 1px dashed #AAB9C8; }
      [data-testid="stFileUploaderDropzone"] * { color: #33475B !important; }
      .block-container { padding-top: 1.3rem; padding-bottom: 2.4rem; max-width: 1600px; }
      h1, h2, h3 { color: #1C2C3D; letter-spacing: -0.035em; }
      [data-testid="stMarkdownContainer"] { color: #1C2C3D; }
      .hero { padding: 1.25rem 1.45rem; border: 1px solid #D9E2EA; border-radius: 18px;
        background: linear-gradient(110deg, #FFFFFF, #EDF3F7); box-shadow: 0 6px 20px rgba(27,48,69,.05);
        margin-bottom: 1rem; }
      .eyebrow { color: #247C73; font-size: .72rem; letter-spacing: .16em; font-weight: 800; text-transform: uppercase; }
      .hero-title { font-size: clamp(1.55rem, 3vw, 2.35rem); line-height: 1.08; font-weight: 800; color: #1C2C3D; margin: .35rem 0; }
      .hero-sub { color: #526579; font-size: .95rem; margin: 0; }
      .kpi { background: linear-gradient(145deg, #FFFFFF, #F8FAFC); border: 1px solid #DCE5ED;
        border-radius: 16px; padding: 1rem 1.05rem; min-height: 116px; box-shadow: 0 4px 14px rgba(27,48,69,.045); }
      .kpi-label { color: #5B6B7C; font-size: .74rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      .kpi-value { color: #1C2C3D; font-size: 1.6rem; font-weight: 800; line-height: 1.2; margin-top: .42rem; }
      .kpi-note { color: #526579; font-size: .77rem; margin-top: .34rem; }
      .section-label { color: #1C2C3D; font-weight: 750; font-size: 1.04rem; margin: .25rem 0 .6rem; }
      .small-muted { color: #5B6B7C; font-size: .82rem; }
      div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #DCE5ED; border-radius: 14px; overflow: hidden; }
      div[data-testid="stMetric"] { background: transparent; }
      .stDownloadButton button { border-radius: 10px; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, Arial, sans-serif"),
    margin=dict(l=18, r=18, t=42, b=18),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(color=TEXT),
    ),
    xaxis=dict(gridcolor="#E4EAF0", zerolinecolor="#C9D4DF", title=None),
    yaxis=dict(gridcolor="#E4EAF0", zerolinecolor="#C9D4DF", title=None),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#D5DEE7", font_color=TEXT),
)

# ---------- Safe, deterministic sample data ----------
@st.cache_data(show_spinner=False)
def make_demo_data(rows: int = 2400, seed: int = 29) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = {
        "North America": ["United States", "Canada", "Mexico"],
        "Latin America": ["Brazil", "Chile", "Colombia", "Argentina"],
        "Europe": ["Germany", "United Kingdom", "France", "Poland", "Spain"],
        "South Asia": ["India", "Bangladesh", "Sri Lanka"],
        "East Asia": ["China", "Japan", "South Korea"],
        "Southeast Asia": ["Thailand", "Vietnam", "Malaysia", "Indonesia"],
        "Middle East & Africa": ["United Arab Emirates", "South Africa", "Kenya", "Egypt"],
        "Oceania": ["Australia", "New Zealand"],
    }
    region_list = list(regions)
    region = rng.choice(
        region_list, rows, p=[.20, .10, .18, .13, .12, .12, .09, .06]
    )
    country = [rng.choice(regions[r]) for r in region]
    product_categories = [
        "Food & Beverage",
        "Personal Care",
        "Home Care",
        "Ingredients",
        "Packaging",
    ]
    cat = rng.choice(product_categories, rows, p=[.34, .18, .17, .17, .14])
    suppliers = [f"Supplier {x:02d}" for x in range(1, 19)]
    plants = [f"Plant {x:02d}" for x in range(1, 13)]
    dcs = [f"Distribution Hub {x:02d}" for x in range(1, 17)]
    modes = rng.choice(["Ocean", "Road", "Rail", "Air"], rows, p=[.43, .35, .16, .06])
    ordered = np.maximum(12, rng.lognormal(mean=5.1, sigma=.72, size=rows)).astype(int)
    shipped_ratio = np.clip(rng.normal(.956, .07, rows), .55, 1.12)
    shipped = np.minimum(ordered, np.maximum(0, (ordered * shipped_ratio).astype(int)))
    promised = rng.integers(4, 25, rows)
    mode_delay = np.select(
        [modes == "Ocean", modes == "Air", modes == "Rail"], [4, -2, 1], default=0
    )
    actual = np.maximum(
        1, promised + mode_delay + rng.normal(0, 4.2, rows).round().astype(int)
    )
    unit_price = rng.uniform(8, 145, rows) * np.select(
        [cat == "Ingredients", cat == "Packaging"], [.72, .45], default=1
    )
    inv_days = np.clip(rng.normal(34, 12, rows), 4, 95)
    demand = np.maximum(1, (ordered * rng.uniform(.85, 1.15, rows)).astype(int))
    forecast = np.maximum(1, (demand * rng.normal(1.015, .16, rows)).astype(int))
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 690, rows), unit="D"
    )
    currency_fx = rng.choice([.74, 1.0, 1.08, 1.34], rows, p=[.18, .52, .18, .12])
    return pd.DataFrame({
        "date": dates,
        "region": region,
        "country": country,
        "product_category": cat,
        "sku": [f"SKU-{x:05d}" for x in rng.integers(1, 451, rows)],
        "supplier": rng.choice(suppliers, rows),
        "plant": rng.choice(plants, rows),
        "distribution_center": rng.choice(dcs, rows),
        "channel": rng.choice(
            ["Retail", "E-commerce", "Wholesale", "Food Service"],
            rows,
            p=[.42, .22, .25, .11],
        ),
        "transport_mode": modes,
        "units_ordered": ordered,
        "units_shipped": shipped,
        "demand_forecast": forecast,
        "revenue": np.round(shipped * unit_price * currency_fx, 2),
        "inventory_on_hand": np.maximum(0, (ordered * inv_days / 30).astype(int)),
        "lead_time_days": np.maximum(
            1, actual + rng.normal(0, 2.5, rows)
        ).round(1),
        "promised_delivery_days": promised,
        "actual_delivery_days": actual,
        "carbon_kg": np.round(ordered * rng.uniform(.015, .22, rows), 2),
        "order_status": np.where(
            shipped >= ordered,
            "Complete",
            np.where(shipped > 0, "Partial", "At Risk"),
        ),
    })

ALIASES = {
    "date": ["date", "order date", "shipment date", "delivery date", "week", "month", "period"],
    "region": ["region", "market", "geography", "geo", "sales region"],
    "country": ["country", "ship to country", "destination country", "market country"],
    "product_category": ["product category", "category", "product family", "family"],
    "sku": ["sku", "item", "item number", "material", "product id", "product"],
    "supplier": ["supplier", "vendor", "supplier name", "vendor name"],
    "plant": ["plant", "factory", "manufacturing site", "source plant"],
    "distribution_center": ["distribution center", "distribution centre", "dc", "warehouse", "hub"],
    "channel": ["channel", "sales channel", "customer channel"],
    "transport_mode": ["transport mode", "mode", "shipping mode", "carrier mode"],
    "units_ordered": ["units ordered", "ordered qty", "order quantity", "quantity ordered", "orders", "demand"],
    "units_shipped": ["units shipped", "shipped qty", "quantity shipped", "delivered qty", "actual quantity"],
    "demand_forecast": ["demand forecast", "forecast", "forecast qty", "planned demand"],
    "revenue": ["revenue", "sales", "sales amount", "net sales", "order value", "value"],
    "inventory_on_hand": ["inventory on hand", "on hand", "inventory", "stock", "stock on hand", "ending inventory"],
    "lead_time_days": ["lead time days", "lead time", "supplier lead time", "actual lead time"],
    "promised_delivery_days": ["promised delivery days", "promised days", "target lead time", "sla days", "planned delivery days"],
    "actual_delivery_days": ["actual delivery days", "delivery days", "actual days", "transit days"],
    "carbon_kg": ["carbon kg", "co2 kg", "emissions", "carbon emissions", "co2e"],
    "order_status": ["order status", "status", "shipment status"],
}

def norm_key(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())

def normalize_workbook(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {norm_key(c): c for c in df.columns}
    rename = {}

    for canonical, names in ALIASES.items():
        for name in [canonical.replace("_", " "), *names]:
            if norm_key(name) in lookup:
                rename[lookup[norm_key(name)]] = canonical
                break

    df = df.rename(columns=rename)
    notes: list[str] = []

    if df.empty:
        return df, ["The selected sheet has no data rows."]

    if "date" not in df:
        df["date"] = pd.Timestamp.today().normalize()
        notes.append(
            "No recognizable date column: the file is shown as one current-period snapshot."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].fillna(pd.Timestamp.today().normalize())

    if "region" not in df:
        df["region"] = "Unspecified"
        notes.append("No region column found; grouped under Unspecified.")
    if "country" not in df:
        df["country"] = df["region"]
    if "product_category" not in df:
        df["product_category"] = "Unspecified"
    if "sku" not in df:
        df["sku"] = [f"Row {i + 1}" for i in range(len(df))]

    for col in [
        "supplier",
        "plant",
        "distribution_center",
        "channel",
        "transport_mode",
        "order_status",
    ]:
        if col not in df:
            df[col] = "Unspecified"

    if "units_ordered" not in df:
        df["units_ordered"] = 1
        notes.append("No ordered-quantity field found; order volume defaults to 1 per row.")
    df["units_ordered"] = (
        pd.to_numeric(df["units_ordered"], errors="coerce").fillna(0).clip(lower=0)
    )

    if "units_shipped" not in df:
        df["units_shipped"] = df["units_ordered"]
        notes.append(
            "No shipped-quantity field found; fill rate uses ordered quantity as a proxy."
        )
    df["units_shipped"] = (
        pd.to_numeric(df["units_shipped"], errors="coerce").fillna(0).clip(lower=0)
    )

    if "revenue" not in df:
        df["revenue"] = np.nan
        notes.append("No revenue/sales field found; revenue KPI is unavailable for this workbook.")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    if "inventory_on_hand" not in df:
        df["inventory_on_hand"] = np.nan
        notes.append("No inventory field found; inventory KPI is unavailable for this workbook.")
    df["inventory_on_hand"] = pd.to_numeric(df["inventory_on_hand"], errors="coerce")

    if "demand_forecast" not in df:
        df["demand_forecast"] = np.nan
        notes.append("No forecast field found; forecast accuracy is unavailable for this workbook.")
    df["demand_forecast"] = pd.to_numeric(df["demand_forecast"], errors="coerce")

    for col in [
        "lead_time_days",
        "promised_delivery_days",
        "actual_delivery_days",
        "carbon_kg",
    ]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["actual_delivery_days"].isna().all() and df["lead_time_days"].notna().any():
        df["actual_delivery_days"] = df["lead_time_days"]
        notes.append("Actual delivery time is approximated from lead time.")

    if (
        df["promised_delivery_days"].isna().all()
        and df["actual_delivery_days"].notna().any()
    ):
        df["promised_delivery_days"] = df["actual_delivery_days"].median()
        notes.append(
            "Promised delivery time is approximated by the median actual delivery time."
        )

    for col in [
        "region",
        "country",
        "product_category",
        "sku",
        "supplier",
        "plant",
        "distribution_center",
        "channel",
        "transport_mode",
        "order_status",
    ]:
        df[col] = df[col].fillna("Unspecified").astype(str)

    return df, notes

def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 100_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.0f}"

def kpi_card(label: str, value: str, note: str, color: str = TEAL) -> None:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

def style_fig(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig

# ---------- Header and data source ----------
st.markdown(
    '<div class="hero"><div class="eyebrow">NEXUS · GLOBAL OPERATIONS INTELLIGENCE</div>'
    '<div class="hero-title">Supply Chain Control Tower</div>'
    '<p class="hero-sub">A connected view from demand and sourcing through production, inventory, and delivery.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🌐 NEXUS")
    st.caption("GLOBAL SUPPLY CHAIN · CONTROL TOWER")
    st.markdown("### Data source")
    uploaded = st.file_uploader(
        "Upload an Excel workbook",
        type=["xlsx", "xls", "csv"],
        help="Choose a workbook or CSV. Data is processed in this app session and is not saved by this dashboard.",
    )
    raw_df = None
    source_label = "Synthetic global demo data"

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded)
                sheet_name = "CSV"
            else:
                xls = pd.ExcelFile(uploaded)
                sheet_name = st.selectbox("Workbook sheet", xls.sheet_names)
                raw_df = pd.read_excel(xls, sheet_name=sheet_name)
            source_label = f"Uploaded workbook · {sheet_name}"
        except Exception as exc:
            st.error(f"Could not read the selected file: {exc}")

    st.markdown("---")
    st.caption("Demo data is randomly generated and contains no company data.")
    st.caption("For a public deployment, do not upload confidential or sensitive workbooks.")

if raw_df is None:
    df = make_demo_data()
    data_notes: list[str] = []
else:
    df, data_notes = normalize_workbook(raw_df)

if df.empty:
    st.error("No usable rows were found. Check that the selected sheet contains a header row and data.")
    st.stop()

# ---------- Filters ----------
st.sidebar.markdown("### Explore the network")
date_min = pd.to_datetime(df["date"].min()).date()
date_max = pd.to_datetime(df["date"].max()).date()

if date_min == date_max:
    date_range = (date_min, date_max)
else:
    date_range = st.sidebar.date_input(
        "Order / shipment date",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

filtered = df[
    (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
].copy()

for col, title in [
    ("region", "Region"),
    ("product_category", "Product category"),
    ("supplier", "Supplier"),
    ("channel", "Channel"),
    ("transport_mode", "Transport mode"),
]:
    choices = sorted(filtered[col].dropna().astype(str).unique().tolist())
    chosen = st.sidebar.multiselect(
        title, choices, default=choices, key=f"filter_{col}"
    )
    if chosen:
        filtered = filtered[filtered[col].astype(str).isin(chosen)]

if filtered.empty:
    st.warning("No rows match the current filters. Widen the date range or select more filter values.")
    st.stop()

st.caption(
    f"**Source:** {source_label}  ·  **Rows in view:** {len(filtered):,} / {len(df):,}  ·  "
    f"**Date range:** {start_date:%d %b %Y} – {end_date:%d %b %Y}"
)

for note in data_notes:
    st.info(note)

# ---------- Metrics ----------
ordered_total = filtered["units_ordered"].sum()
shipped_total = filtered["units_shipped"].sum()
fill_rate = shipped_total / ordered_total if ordered_total else np.nan

valid_ontime = (
    filtered["actual_delivery_days"].notna()
    & filtered["promised_delivery_days"].notna()
)
otif_rate = (
    (
        filtered.loc[valid_ontime, "actual_delivery_days"]
        <= filtered.loc[valid_ontime, "promised_delivery_days"]
    ).mean()
    if valid_ontime.any()
    else np.nan
)

forecast_mask = (
    filtered["demand_forecast"].notna() & (filtered["units_ordered"] > 0)
)
wape = (
    filtered.loc[forecast_mask, "demand_forecast"]
    .sub(filtered.loc[forecast_mask, "units_ordered"])
    .abs()
    .sum()
    / filtered.loc[forecast_mask, "units_ordered"].sum()
    if forecast_mask.any()
    else np.nan
)

revenue_total = filtered["revenue"].sum(min_count=1)
inventory_total = filtered["inventory_on_hand"].sum(min_count=1)

cols = st.columns(5, gap="small")
with cols[0]:
    kpi_card("Net revenue", money(revenue_total), "Filtered order / shipment value")
with cols[1]:
    kpi_card(
        "Fill rate",
        f"{fill_rate:.1%}" if pd.notna(fill_rate) else "—",
        f"{shipped_total:,.0f} of {ordered_total:,.0f} units",
        BLUE,
    )
with cols[2]:
    kpi_card(
        "On-time delivery",
        f"{otif_rate:.1%}" if pd.notna(otif_rate) else "—",
        "Orders delivered on or before promise",
        TEAL,
    )
with cols[3]:
    kpi_card(
        "Inventory on hand",
        f"{inventory_total:,.0f}" if pd.notna(inventory_total) else "—",
        "Units across the filtered network",
        AMBER,
    )
with cols[4]:
    kpi_card(
        "Forecast WAPE",
        f"{wape:.1%}" if pd.notna(wape) else "—",
        "Lower is better · vs. ordered quantity",
        RED,
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ---------- Demand and service ----------
left, right = st.columns([1.55, 1], gap="medium")

with left:
    st.markdown(
        '<div class="section-label">Demand & fulfillment trajectory</div>',
        unsafe_allow_html=True,
    )
    trend = (
        filtered.assign(
            period=filtered["date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby("period", as_index=False)
        .agg(
            Ordered=("units_ordered", "sum"),
            Shipped=("units_shipped", "sum"),
            Revenue=("revenue", "sum"),
        )
    )
    trend_long = trend.melt(
        id_vars="period",
        value_vars=["Ordered", "Shipped"],
        var_name="Measure",
        value_name="Units",
    )
    fig = px.line(
        trend_long,
        x="period",
        y="Units",
        color="Measure",
        markers=True,
        color_discrete_map={"Ordered": BLUE, "Shipped": TEAL},
        template="plotly_white",
    )
    fig.update_traces(
        line_width=3,
        marker_size=6,
        hovertemplate="%{x|%b %Y}<br>%{fullData.name}: %{y:,.0f}<extra></extra>",
    )
    fig.update_layout(hovermode="x unified", xaxis_tickformat="%b<br>%Y")
    st.plotly_chart(
        style_fig(fig, 350),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    st.markdown(
        '<div class="section-label">Regional service performance</div>',
        unsafe_allow_html=True,
    )
    reg = filtered.groupby("region", as_index=False).agg(
        Ordered=("units_ordered", "sum"),
        Shipped=("units_shipped", "sum"),
    )
    reg["Fill rate"] = np.where(
        reg["Ordered"] > 0, reg["Shipped"] / reg["Ordered"] * 100, 0
    )
    reg = reg.sort_values("Fill rate", ascending=True)
    fig = px.bar(
        reg,
        x="Fill rate",
        y="region",
        orientation="h",
        color="Fill rate",
        color_continuous_scale=["#B64C5A", "#C6973E", "#247C73"],
        hover_data={
            "Ordered": ":,.0f",
            "Shipped": ":,.0f",
            "Fill rate": ":.1f",
        },
        template="plotly_white",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_ticksuffix="%",
        yaxis_title=None,
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Fill rate: %{x:.1f}%<extra></extra>"
    )
    st.plotly_chart(
        style_fig(fig, 350),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ---------- Network mix and supplier view ----------
left, right = st.columns([1, 1], gap="medium")

with left:
    st.markdown(
        '<div class="section-label">Inventory & demand by product family</div>',
        unsafe_allow_html=True,
    )
    cat = filtered.groupby("product_category", as_index=False).agg(
        Ordered=("units_ordered", "sum"),
        Inventory=("inventory_on_hand", "sum"),
    )
    cat_long = cat.melt(
        id_vars="product_category",
        value_vars=["Ordered", "Inventory"],
        var_name="Measure",
        value_name="Units",
    )
    fig = px.bar(
        cat_long,
        x="product_category",
        y="Units",
        color="Measure",
        barmode="group",
        color_discrete_map={"Ordered": BLUE, "Inventory": AMBER},
        template="plotly_white",
    )
    fig.update_layout(xaxis_title=None, legend_title=None)
    st.plotly_chart(
        style_fig(fig, 320),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    st.markdown(
        '<div class="section-label">Supplier lead-time exposure</div>',
        unsafe_allow_html=True,
    )
    sup = filtered.groupby("supplier", as_index=False).agg(
        Orders=("units_ordered", "sum"),
        Lead_Time=("lead_time_days", "mean"),
        Actual=("actual_delivery_days", "mean"),
        Promise=("promised_delivery_days", "mean"),
    )
    sup = sup.sort_values("Orders", ascending=False).head(12)
    fig = px.bar(
        sup.sort_values("Lead_Time", ascending=True),
        x="Lead_Time",
        y="supplier",
        orientation="h",
        color="Lead_Time",
        color_continuous_scale=["#247C73", "#C6973E", "#B64C5A"],
        hover_data={
            "Orders": ":,.0f",
            "Actual": ":.1f",
            "Promise": ":.1f",
        },
        template="plotly_white",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Average lead time (days)",
        yaxis_title=None,
    )
    st.plotly_chart(
        style_fig(fig, 320),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ---------- End-to-end network flow ----------
st.markdown(
    '<div class="section-label">End-to-end network flow · source → plant → distribution → market</div>',
    unsafe_allow_html=True,
)

flow_base = (
    filtered.groupby(
        ["region", "plant", "distribution_center"], as_index=False
    )["units_shipped"]
    .sum()
)
flow_base = flow_base[flow_base["units_shipped"] > 0].nlargest(
    160, "units_shipped"
)

if not flow_base.empty:
    labels = list(
        dict.fromkeys(
            flow_base["region"].tolist()
            + flow_base["plant"].tolist()
            + flow_base["distribution_center"].tolist()
        )
    )
    idx = {label: i for i, label in enumerate(labels)}
    sources = [idx[x] for x in flow_base["region"]] + [
        idx[x] for x in flow_base["plant"]
    ]
    targets = [idx[x] for x in flow_base["plant"]] + [
        idx[x] for x in flow_base["distribution_center"]
    ]
    values = flow_base["units_shipped"].tolist() * 2

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=14,
                thickness=17,
                color=[
                    TEAL if "Hub" in x else BLUE if "Plant" in x else "#71869A"
                    for x in labels
                ],
                line=dict(color="#FFFFFF", width=1),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(36,124,115,0.24)",
            ),
        )
    )
    fig.update_layout(**PLOT_LAYOUT, height=380)
    fig.update_layout(margin=dict(l=8, r=8, t=18, b=8))
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )
else:
    st.info("No shipped volume is available to display in the network flow.")

# ---------- Exceptions and downloadable view ----------
st.markdown(
    '<div class="section-label">Decision desk · exceptions to investigate</div>',
    unsafe_allow_html=True,
)

exceptions = filtered.copy()
exceptions["fill_rate"] = np.where(
    exceptions["units_ordered"] > 0,
    exceptions["units_shipped"] / exceptions["units_ordered"],
    np.nan,
)
exceptions["late_days"] = (
    exceptions["actual_delivery_days"] - exceptions["promised_delivery_days"]
)
exceptions["priority"] = np.select(
    [
        exceptions["fill_rate"].lt(.8) | exceptions["late_days"].gt(7),
        exceptions["fill_rate"].lt(.95) | exceptions["late_days"].gt(0),
    ],
    ["HIGH", "WATCH"],
    default="ON TRACK",
)
exceptions["issue"] = np.select(
    [
        exceptions["fill_rate"].lt(.8),
        exceptions["late_days"].gt(7),
        exceptions["late_days"].gt(0),
        exceptions["inventory_on_hand"].eq(0),
    ],
    [
        "Severe fulfillment gap",
        "Delivery >7 days late",
        "Delivery past promise",
        "Zero inventory reported",
    ],
    default="Review demand / supply signals",
)

priority_order = {"HIGH": 0, "WATCH": 1, "ON TRACK": 2}
exceptions["_priority_sort"] = exceptions["priority"].map(priority_order)
exceptions = (
    exceptions.sort_values(
        ["_priority_sort", "late_days"], ascending=[True, False]
    )
    .drop(columns="_priority_sort")
)

show_cols = [
    "priority",
    "issue",
    "date",
    "region",
    "country",
    "sku",
    "supplier",
    "units_ordered",
    "units_shipped",
    "late_days",
    "inventory_on_hand",
]
visible_cols = [c for c in show_cols if c in exceptions.columns]
show = exceptions[visible_cols].head(100).rename(
    columns={
        "priority": "Priority",
        "issue": "Signal",
        "date": "Date",
        "region": "Region",
        "country": "Country",
        "sku": "SKU",
        "supplier": "Supplier",
        "units_ordered": "Ordered",
        "units_shipped": "Shipped",
        "late_days": "Late days",
        "inventory_on_hand": "Inventory",
    }
)

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    height=360,
    column_config={
        "Priority": st.column_config.TextColumn(),
        "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Ordered": st.column_config.NumberColumn(format="%d"),
        "Shipped": st.column_config.NumberColumn(format="%d"),
        "Late days": st.column_config.NumberColumn(format="%.1f"),
        "Inventory": st.column_config.NumberColumn(format="%d"),
    },
)

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download filtered rows as CSV",
    data=csv_data,
    file_name="nexus_filtered_supply_chain.csv",
    mime="text/csv",
)

st.caption(
    "Metric notes: Fill rate = shipped ÷ ordered units. On-time delivery = orders "
    "delivered on/before promise. Forecast WAPE uses ordered quantity as actual demand. "
    "Workbook field mappings are heuristic; validate column meanings before making "
    "operational decisions."
)
