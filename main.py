# -*- coding: utf-8 -*-
"""
机场货邮吞吐量数据分析与智能预测系统
================================================
技术栈：Python + Streamlit + pandas + numpy + pyecharts
功能模块：数据管理、可视化分析、多模型智能预测
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyecharts import options as opts
from pyecharts.charts import Line, Pie, HeatMap, Scatter

# =============================================================================
# 0. 全局配置与页面美化
# =============================================================================
st.set_page_config(
    page_title="机场货邮数据分析与智能预测系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded" # 强制默认展开侧边栏
)

st.markdown("""
<style>
    /* 隐藏多余菜单，但保留 header 以显示侧边栏折叠按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp { background-color: #F4F7F9 !important; }
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color: #2C3E50 !important; }
    
    /* 顶部主标题渐变，去掉了 Emoji */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0B3D6B, #1E6FA8) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0px;
        padding-bottom: 5px;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        color: #6C7A89 !important;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    
    /* 卡片式指标设计 */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1E6FA8;
    }
    div[data-testid="stMetric"] * { color: #0B3D6B !important; }
    
    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] * { color: #2C3E50 !important; }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #1E6FA8 !important;
        color: #FFFFFF !important;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover { background-color: #0B3D6B !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. 数据初始化
# =============================================================================
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "airport_cargo.csv")
AIRPORTS = ["牡丹江海浪机场", "佳木斯东郊机场", "黑河瑷珲机场"]

BASE_LEVEL = {"牡丹江海浪机场": 1250.0, "佳木斯东郊机场": 2150.0, "黑河瑷珲机场": 130.0}
SEASON_FACTOR = {
    "牡丹江海浪机场": [0.82, 0.78, 0.85, 0.92, 0.96, 1.00, 1.00, 1.02, 1.18, 1.30, 1.25, 1.08],
    "佳木斯东郊机场": [0.80, 0.76, 0.84, 0.90, 0.95, 0.98, 0.98, 1.00, 1.15, 1.28, 1.22, 1.05],
    "黑河瑷珲机场": [0.72, 0.70, 0.78, 0.88, 1.05, 1.15, 1.18, 1.12, 1.05, 0.95, 0.85, 0.76],
}
GDP_GROWTH_BASE = {
    "牡丹江海浪机场": [3.0, 6.8, 3.0, 4.2, 3.9, 4.0],
    "佳木斯东郊机场": [2.5, 6.5, 3.5, 4.5, 4.9, 4.7],
    "黑河瑷珲机场": [3.5, 6.2, 3.0, 4.0, 3.3, 4.4],
}
FOREIGN_RATIO_BASE = {"牡丹江海浪机场": 32.0, "佳木斯东郊机场": 24.0, "黑河瑷珲机场": 41.0}

COLUMNS = ["机场名称", "年份", "月份", "货邮吞吐量(吨)", "货机架次", "外贸货占比(%)", "区域GDP增速(%)", "月平均风速(m/s)"]

def st_pyecharts(chart, key: str = None):
    try:
        h = int(str(chart.height).replace("px", ""))
    except Exception:
        h = 400
    components.html(chart.render_embed(), height=h, scrolling=False)

def rerun():
    try: st.rerun()
    except AttributeError: st.experimental_rerun()

def _generate_demo_data() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(20240501)
    rows = []
    for ap in AIRPORTS:
        base, gdp_seq, fb = BASE_LEVEL[ap], GDP_GROWTH_BASE[ap], FOREIGN_RATIO_BASE[ap]
        for yi, year in enumerate(range(2020, 2026)):
            annual_trend = 1.0 + (gdp_seq[yi] / 100.0) * 0.55
            for month in range(1, 13):
                season = SEASON_FACTOR[ap][month - 1]
                cargo = max(round(base * (annual_trend ** yi) * season * rng.normal(1.0, 0.055), 2), 1.0)
                sorties = max(int(cargo / rng.uniform(2.8, 4.2)), 2)
                foreign = round(float(np.clip(fb + (season - 1.0) * 18 + rng.normal(0, 2.5), 3, 78)), 1)
                wind = round(float(max(3.6 + 1.4 * np.sin((month - 3) / 12 * 2 * np.pi) ** 2 + rng.normal(0, 0.35), 1.2)), 1)
                rows.append([ap, year, month, cargo, sorties, foreign, round(gdp_seq[yi] + rng.normal(0, 0.25), 2), wind])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    return df

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return _generate_demo_data()

def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

# =============================================================================
# 2. 算法模块
# =============================================================================
def model_elasticity(base, gdp_rate, elasticity, years):
    m = gdp_rate / 100.0
    factor = 1 + elasticity * m
    return [base * (factor ** n) for n in range(1, years + 1)]

def model_linear_regression(historical_y, years):
    x = np.arange(len(historical_y))
    y = np.array(historical_y)
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return [max(a * (len(historical_y) - 1 + i) + b, 0) for i in range(1, years + 1)]

def model_gm11(historical_y, years):
    x0 = np.array(historical_y, dtype=float)
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    B = np.vstack([-z1, np.ones(n-1)]).T
    Y = x0[1:].reshape(-1, 1)
    try:
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        a, b = params[0][0], params[1][0]
        def predict_k(k): return (x0[0] - b/a) * np.exp(-a * k) + b/a
        preds = []
        for i in range(1, years + 1):
            k = n - 1 + i
            preds.append(max(predict_k(k) - predict_k(k-1), 0))
        return preds
    except np.linalg.LinAlgError:
        return model_linear_regression(historical_y, years)

def model_holt(historical_y, years, alpha=0.3, beta=0.1):
    x = np.array(historical_y, dtype=float)
    if len(x) < 2: return [x[-1] if len(x)>0 else 0] * years
    level, trend = x[0], x[1] - x[0]
    for i in range(1, len(x)):
        last_level = level
        level = alpha * x[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
    return [max(level + i * trend, 0) for i in range(1, years + 1)]

# =============================================================================
# 3. 页面顶部与侧边栏
# =============================================================================
st.markdown('<div class="main-title">机场货邮吞吐量数据分析与智能预测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">多维数据管理 · 交互式可视化 · 经典与智能算法预测</div>', unsafe_allow_html=True)

df = load_data()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airport.png", width=70)
    st.markdown("### 功能导航")
    page = st.radio(
        "选择模块",
        ["📊 数据概览", "🛠️ 数据管理", "📈 可视化分析", "🎯 智能预测中心"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("课程设计 · 民航交通工程")

# =============================================================================
# 4. 页面：数据概览
# =============================================================================
if page == "📊 数据概览":
    st.markdown("### 核心数据看板")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("记录总数", f"{len(df):,} 条")
    c2.metric("覆盖机场", f"{df['机场名称'].nunique()} 座")
    c3.metric("时间跨度", f"{df['年份'].min()}—{df['年份'].max()}")
    c4.metric("总货邮量（吨）", f"{df['货邮吞吐量(吨)'].sum():,.0f}")

    st.markdown("### 各机场年度货邮吞吐量汇总（吨）")
    pivot = df.groupby(["机场名称", "年份"])["货邮吞吐量(吨)"].sum().round(1).unstack("年份")
    st.dataframe(pivot, use_container_width=True)

    st.markdown("### 原始数据预览（前 20 行）")
    st.dataframe(df.head(20), use_container_width=True)

# =============================================================================
# 5. 页面：数据管理
# =============================================================================
elif page == "🛠️ 数据管理":
    tab_q, tab_add, tab_edit, tab_del, tab_io = st.tabs(["🔍 查询", "➕ 新增", "✏️ 修改", "🗑️ 删除", "📁 导入/导出"])

    with tab_q:
        c1, c2, c3 = st.columns(3)
        with c1: q_ap = st.selectbox("机场", ["全部"] + sorted(df["机场名称"].unique()), key="q_ap")
        with c2: q_yr = st.selectbox("年份", ["全部"] + sorted(df["年份"].unique()), key="q_yr")
        with c3: q_mo = st.selectbox("月份", ["全部"] + list(range(1, 13)), key="q_mo")
        qdf = df.copy()
        if q_ap != "全部": qdf = qdf[qdf["机场名称"] == q_ap]
        if q_yr != "全部": qdf = qdf[qdf["年份"] == q_yr]
        if q_mo != "全部": qdf = qdf[qdf["月份"] == q_mo]
        st.caption(f"命中 {len(qdf)} 条记录")
        st.dataframe(qdf, use_container_width=True)

    with tab_add:
        with st.form("add_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                a_ap = st.selectbox("机场名称", AIRPORTS, key="a_ap")
                a_yr = st.number_input("年份", 2020, 2035, 2025, step=1, key="a_yr")
            with c2:
                a_mo = st.number_input("月份", 1, 12, 1, step=1, key="a_mo")
                a_cg = st.number_input("货邮吞吐量(吨)", 0.0, 100000.0, 100.0, step=1.0, key="a_cg")
            with c3:
                a_st = st.number_input("货机架次", 0, 100000, 30, step=1, key="a_st")
                a_fr = st.number_input("外贸货占比(%)", 0.0, 100.0, 25.0, step=0.5, key="a_fr")
            c4, c5 = st.columns(2)
            with c4: a_gdp = st.number_input("区域GDP增速(%)", -20.0, 30.0, 4.0, step=0.1, key="a_gdp")
            with c5: a_wd = st.number_input("月平均风速(m/s)", 0.0, 30.0, 3.5, step=0.1, key="a_wd")
            if st.form_submit_button("确认新增", type="primary"):
                dup = (df["机场名称"] == a_ap) & (df["年份"] == int(a_yr)) & (df["月份"] == int(a_mo))
                if dup.any(): st.error("该机场该年月已存在记录，请使用“修改”功能。")
                else:
                    df = pd.concat([df, pd.DataFrame([{
                        "机场名称": a_ap, "年份": int(a_yr), "月份": int(a_mo),
                        "货邮吞吐量(吨)": float(a_cg), "货机架次": int(a_st),
                        "外贸货占比(%)": float(a_fr), "区域GDP增速(%)": float(a_gdp),
                        "月平均风速(m/s)": float(a_wd),
                    }])], ignore_index=True)
                    save_data(df)
                    st.success("新增成功。")
                    rerun()

    with tab_edit:
        options = list(df.index[:200])
        edit_idx = st.selectbox(
            "选择要修改的记录（显示前 200 条）", options=options,
            format_func=lambda i: f"{df.loc[i, '机场名称']} | {int(df.loc[i, '年份'])}-{int(df.loc[i, '月份']):02d} | {df.loc[i, '货邮吞吐量(吨)']:.1f} 吨",
            key="edit_sel",
        )
        row = df.loc[edit_idx]
        with st.form("edit_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                e_cg = st.number_input("货邮吞吐量(吨)", 0.0, 100000.0, float(row["货邮吞吐量(吨)"]), step=1.0)
                e_st = st.number_input("货机架次", 0, 100000, int(row["货机架次"]), step=1)
            with c2:
                e_fr = st.number_input("外贸货占比(%)", 0.0, 100.0, float(row["外贸货占比(%)"]), step=0.5)
                e_gdp = st.number_input("区域GDP增速(%)", -20.0, 30.0, float(row["区域GDP增速(%)"]), step=0.1)
            with c3:
                e_wd = st.number_input("月平均风速(m/s)", 0.0, 30.0, float(row["月平均风速(m/s)"]), step=0.1)
            if st.form_submit_button("保存修改", type="primary"):
                df.loc[edit_idx, "货邮吞吐量(吨)"] = float(e_cg)
                df.loc[edit_idx, "货机架次"] = int(e_st)
                df.loc[edit_idx, "外贸货占比(%)"] = float(e_fr)
                df.loc[edit_idx, "区域GDP增速(%)"] = float(e_gdp)
                df.loc[edit_idx, "月平均风速(m/s)"] = float(e_wd)
                save_data(df)
                st.success("修改已保存。")
                rerun()

    with tab_del:
        del_idx = st.selectbox(
            "选择要删除的记录", options=list(df.index),
            format_func=lambda i: f"{df.loc[i, '机场名称']} | {int(df.loc[i, '年份'])}-{int(df.loc[i, '月份']):02d} | {df.loc[i, '货邮吞吐量(吨)']:.1f} 吨",
            key="del_sel",
        )
        st.warning("删除操作不可撤销。")
        if st.button("确认删除", type="secondary"):
            df = df.drop(index=del_idx).reset_index(drop=True)
            save_data(df)
            st.success("已删除。")
            rerun()

    with tab_io:
        st.markdown("**CSV 批量导入**")
        up = st.file_uploader("上传 CSV", type=["csv"])
        if up is not None:
            try:
                imported = pd.read_csv(up, encoding="utf-8-sig")
                missing = set(COLUMNS) - set(imported.columns)
                if missing: st.error(f"缺少字段：{missing}")
                else:
                    save_data(imported[COLUMNS])
                    st.success(f"导入成功，共 {len(imported)} 条记录。")
                    rerun()
            except Exception as exc:
                st.error(f"导入失败：{exc}")
        st.divider()
        st.markdown("**一键导出**")
        st.download_button(
            label="⬇️ 下载当前数据 CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="airport_cargo_export.csv",
            mime="text/csv",
        )

# =============================================================================
# 6. 页面：可视化分析
# =============================================================================
elif page == "📈 可视化分析":
    scope = st.radio("查看范围", ["全部机场对比"] + sorted(df["机场名称"].unique()), horizontal=True, key="viz_scope")
    plot_df = df if scope == "全部机场对比" else df[df["机场名称"] == scope].copy()

    st.markdown("#### ① 货邮吞吐量月度时序趋势")
    monthly = plot_df.groupby(["年份", "月份"])["货邮吞吐量(吨)"].sum().reset_index().sort_values(["年份", "月份"])
    monthly["时间"] = monthly["年份"].astype(str) + "-" + monthly["月份"].astype(str).str.zfill(2)
    
    line = Line(init_opts=opts.InitOpts(width="100%", height="360px", bg_color="#FFFFFF"))
    line.add_xaxis(monthly["时间"].tolist())
    line.add_yaxis("货邮吞吐量(吨)", monthly["货邮吞吐量(吨)"].round(1).tolist(), is_smooth=True, label_opts=opts.LabelOpts(is_show=False), linestyle_opts=opts.LineStyleOpts(width=2.4, color="#1E6FA8"), itemstyle_opts=opts.ItemStyleOpts(color="#1E6FA8"))
    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=55, interval=5)),
        yaxis_opts=opts.AxisOpts(name="吨"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
    )
    st_pyecharts(line)

    st.markdown("#### ② 货机架次占比")
    pie = Pie(init_opts=opts.InitOpts(width="100%", height="340px", bg_color="#FFFFFF"))
    if scope == "全部机场对比":
        pie_data = df.groupby("机场名称")["货机架次"].sum().sort_values(ascending=False)
        pie.add("", [(str(k), int(v)) for k, v in pie_data.items()], radius=["32%", "62%"], label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
    else:
        pie_data = plot_df.groupby("年份")["货机架次"].sum().sort_index()
        pie.add("", [(f"{k}年", int(v)) for k, v in pie_data.items()], radius=["32%", "62%"], label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
    pie.set_global_opts(legend_opts=opts.LegendOpts(orient="vertical", pos_left="left"), tooltip_opts=opts.TooltipOpts(trigger="item"))
    st_pyecharts(pie)

    st.markdown("#### ③ 月度货运季节热力图")
    heat_df = df if scope == "全部机场对比" else plot_df
    heat_pivot = heat_df.groupby(["年份", "月份"])["货邮吞吐量(吨)"].sum().unstack("月份")
    years_h = heat_pivot.index.tolist()
    heat_values = []
    for yi, yr in enumerate(years_h):
        for m in range(1, 13):
            v = heat_pivot.loc[yr, m] if m in heat_pivot.columns else 0
            if pd.isna(v): v = 0
            heat_values.append([m - 1, yi, round(float(v), 1)])
            
    heat = HeatMap(init_opts=opts.InitOpts(width="100%", height="360px", bg_color="#FFFFFF"))
    heat.add_xaxis([f"{m}月" for m in range(1, 13)])
    heat.add_yaxis("年份", [str(y) for y in years_h], heat_values, label_opts=opts.LabelOpts(is_show=True, font_size=9))
    heat.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(min_=min(v[2] for v in heat_values), max_=max(v[2] for v in heat_values), is_calculable=True, orient="horizontal", pos_left="center"),
        xaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=False)),
    )
    st_pyecharts(heat)

    st.markdown("#### ④ GDP增速—货邮吞吐量相关性散点图")
    sc_df = df.copy() if scope == "全部机场对比" else df[df["机场名称"] == scope].copy()
    sc_df = sc_df.sample(min(len(sc_df), 500), random_state=42)

    scatter = Scatter(init_opts=opts.InitOpts(width="100%", height="400px", bg_color="#FFFFFF"))
    scatter.add_xaxis(sc_df["区域GDP增速(%)"].round(2).tolist())
    for ap in sorted(sc_df["机场名称"].unique()):
        sub = sc_df[sc_df["机场名称"] == ap]
        scatter.add_yaxis(ap, sub["货邮吞吐量(吨)"].round(1).tolist(), symbol_size=7, label_opts=opts.LabelOpts(is_show=False))
    scatter.set_global_opts(
        xaxis_opts=opts.AxisOpts(name="区域GDP增速(%)", type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)),
        yaxis_opts=opts.AxisOpts(name="货邮吞吐量(吨)", type_="value"),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        legend_opts=opts.LegendOpts(pos_top="2%"),
    )
    st_pyecharts(scatter)

# =============================================================================
# 7. 页面：智能预测中心
# =============================================================================
elif page == "🎯 智能预测中心":
    st.markdown("### 货运量多模型智能预测")
    
    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        st.markdown("**① 参数设置**")
        pred_ap = st.selectbox("选择机场", sorted(df["机场名称"].unique()), key="pred_ap")
        ap_df = df[df["机场名称"] == pred_ap]
        
        hist_y = ap_df.groupby("年份")["货邮吞吐量(吨)"].sum().sort_index()
        hist_years = hist_y.index.tolist()
        hist_values = hist_y.values.tolist()
        
        recent_year = int(hist_years[-1])
        default_base = float(hist_values[-1])

        base_cargo = st.number_input("基准年货邮吞吐量（吨）", min_value=1.0, max_value=1_000_000.0, value=round(default_base, 1), step=10.0)
        
        st.markdown("---")
        st.markdown("**弹性系数法专用参数**")
        gdp_growth_pct = st.slider("预测期区域GDP年均增速 M（%）", 0.0, 12.0, 4.0, 0.1)
        elasticity = st.slider("货运弹性系数 E", 0.2, 2.5, 1.10, 0.05)
        
        st.markdown("---")
        horizon = st.slider("预测年限 n（年）", 1, 15, 5, 1)
        years_future = [recent_year + i for i in range(1, horizon + 1)]

    with col_r:
        st.markdown("**② 多模型预测结果对比**")
        st.caption("图表底部可左右滑动缩放，图例可点击隐藏/显示对应曲线。")
        
        preds_elasticity = model_elasticity(base_cargo, gdp_growth_pct, elasticity, horizon)
        preds_linear = model_linear_regression(hist_values, horizon)
        preds_gm11 = model_gm11(hist_values, horizon)
        preds_holt = model_holt(hist_values, horizon)
        
        comp_line = Line(init_opts=opts.InitOpts(width="100%", height="420px", bg_color="#FFFFFF"))
        comp_line.add_xaxis([str(y) for y in hist_years] + [str(y) for y in years_future])
        
        comp_line.add_yaxis("历史真实数据", [round(v, 1) for v in hist_values] + [None] * horizon, is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=3, color="#333333"), itemstyle_opts=opts.ItemStyleOpts(color="#333333"))
        
        models = [
            ("弹性系数法 (专业规划)", preds_elasticity, "#1E6FA8"),
            ("线性回归", preds_linear, "#E74C3C"),
            ("GM(1,1) 灰色预测", preds_gm11, "#2ECC71"),
            ("Holt 指数平滑", preds_holt, "#F39C12")
        ]
        
        for name, preds, color in models:
            comp_line.add_yaxis(name, [None] * len(hist_years) + [round(v, 1) for v in preds], is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed", color=color), itemstyle_opts=opts.ItemStyleOpts(color=color))
            
        comp_line.set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(name="吨"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(type_="scroll", pos_top="0%", pos_left="center"),
            datazoom_opts=[opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)],
        )
        st_pyecharts(comp_line)

        res_df = pd.DataFrame({
            "年份": years_future,
            "弹性系数法": [round(v, 1) for v in preds_elasticity],
            "线性回归": [round(v, 1) for v in preds_linear],
            "GM(1,1)": [round(v, 1) for v in preds_gm11],
            "Holt平滑": [round(v, 1) for v in preds_holt],
        }).set_index("年份")
        st.dataframe(res_df, use_container_width=True)

    st.divider()
    st.markdown("**③ 弹性系数率定辅助**")
    cal_rows = []
    for ap in sorted(df["机场名称"].unique()):
        ap_year = df[df["机场名称"] == ap].groupby("年份").agg(货邮=("货邮吞吐量(吨)", "sum"), GDP=("区域GDP增速(%)", "mean")).reset_index()
        ap_year["货邮增长率(%)"] = ap_year["货邮"].pct_change() * 100
        ap_year["隐含弹性"] = ap_year["货邮增长率(%)"] / ap_year["GDP"].replace(0, np.nan)
        ap_year["机场名称"] = ap
        cal_rows.append(ap_year.dropna(subset=["隐含弹性"]))
        
    if cal_rows:
        cal_df = pd.concat(cal_rows)[["机场名称", "年份", "货邮增长率(%)", "GDP", "隐含弹性"]].round(2).rename(columns={"GDP": "GDP增速(%)"})
        st.dataframe(cal_df, use_container_width=True, hide_index=True)
