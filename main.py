# -*- coding: utf-8 -*-
"""
机场货邮吞吐量数据分析与智能预测系统
================================================
技术栈：Python + Streamlit + pandas + numpy + pyecharts
功能：数据管理、可视化分析、多模型智能预测（弹性系数、线性回归、灰色预测、指数平滑）
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyecharts import options as opts
from pyecharts.charts import Line, Pie, HeatMap, Scatter
from pyecharts.commons.utils import JsCode

# =============================================================================
# 0. 全局配置与数据初始化
# =============================================================================
st.set_page_config(
    page_title="机场货邮数据分析与智能预测系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 美化界面
st.markdown("""
<style>
    /* 隐藏默认菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局背景和字体 */
    .stApp {
        background-color: #F4F7F9;
    }
    
    /* 顶部大标题渐变 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0B3D6B, #1E6FA8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 10px;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        color: #6C7A89;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    
    /* 卡片式设计 */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1E6FA8;
    }
    
    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #1E6FA8;
        color: white;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0B3D6B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "airport_cargo.csv")
AIRPORTS = ["牡丹江海浪机场", "佳木斯东郊机场", "黑河瑷珲机场"]

# 默认演示参数（用于生成数据）
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
            trend = 1.0 + (gdp_seq[yi] / 100.0) * 0.55
            for month in range(1, 13):
                season = SEASON_FACTOR[ap][month - 1]
                cargo = max(round(base * (trend ** yi) * season * rng.normal(1.0, 0.055), 2), 1.0)
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
# 1. 算法模块 (预测模型)
# =============================================================================
def model_elasticity(base, gdp_rate, elasticity, years):
    """弹性系数法"""
    m = gdp_rate / 100.0
    factor = 1 + elasticity * m
    return [base * (factor ** n) for n in range(1, years + 1)]

def model_linear_regression(historical_y, years):
    """线性回归法 (基于时间序列)"""
    x = np.arange(len(historical_y))
    y = np.array(historical_y)
    # 最小二乘法拟合 y = ax + b
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return [max(a * (len(historical_y) - 1 + i) + b, 0) for i in range(1, years + 1)]

def model_gm11(historical_y, years):
    """GM(1,1) 灰色预测模型 (适用于小样本)"""
    x0 = np.array(historical_y, dtype=float)
    n = len(x0)
    # 级比检验 (简化，若数据为非负递增，一般可通过)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    B = np.vstack([-z1, np.ones(n-1)]).T
    Y = x0[1:].reshape(-1, 1)
    
    try:
        # 计算发展系数 a 和灰作用量 b
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        a, b = params[0][0], params[1][0]
        
        # 预测公式
        def predict_k(k):
            return (x0[0] - b/a) * np.exp(-a * k) + b/a
        
        preds = []
        for i in range(1, years + 1):
            k = n - 1 + i
            pred_x1 = predict_k(k)
            pred_x1_prev = predict_k(k-1)
            # 累减还原
            preds.append(max(pred_x1 - pred_x1_prev, 0))
        return preds
    except np.linalg.LinAlgError:
        # 矩阵奇异时退回线性回归
        return model_linear_regression(historical_y, years)

def model_holt(historical_y, years, alpha=0.3, beta=0.1):
    """Holt 指数平滑法 (带趋势)"""
    x = np.array(historical_y, dtype=float)
    if len(x) < 2:
        return [x[-1] if len(x)>0 else 0] * years
    
    level = x[0]
    trend = x[1] - x[0]
    for i in range(1, len(x)):
        last_level = level
        level = alpha * x[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
    
    return [max(level + i * trend, 0) for i in range(1, years + 1)]


# =============================================================================
# 2. 页面头部与侧边栏
# =============================================================================
st.markdown('<div class="main-title">✈️ 机场货邮吞吐量数据分析与智能预测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">多维数据管理 · 交互式可视化 · 经典与智能预测算法集成</div>', unsafe_allow_html=True)

df = load_data()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airport.png", width=80)
    st.markdown("### 系统导航")
    page = st.radio(
        "选择功能模块",
        ["📊 数据概览", "🛠️ 数据管理", "📈 可视化分析", "🎯 智能预测中心"],
        label_visibility="collapsed",
    )
    st.divider()
    st.info("💡 提示：系统已接入 4 种主流预测模型，可在预测中心对比结果。")

# =============================================================================
# 3. 页面：数据概览
# =============================================================================
if page == "📊 数据概览":
    st.markdown("### 核心数据看板")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("记录总数", f"{len(df):,} 条")
    c2.metric("覆盖机场", f"{df['机场名称'].nunique()} 座")
    c3.metric("时间跨度", f"{df['年份'].min()}—{df['年份'].max()}")
    c4.metric("总货邮量", f"{df['货邮吞吐量(吨)'].sum():,.0f} 吨")

    st.markdown("### 年度货邮吞吐量汇总")
    pivot = df.groupby(["机场名称", "年份"])["货邮吞吐量(吨)"].sum().round(1).unstack("年份")
    st.dataframe(pivot.style.highlight_max(axis=1, color='#D4E6F1'), use_container_width=True)
    
    st.markdown("### 原始数据明细 (前 20 行)")
    st.dataframe(df.head(20), use_container_width=True)

# =============================================================================
# 4. 页面：数据管理
# =============================================================================
elif page == "🛠️ 数据管理":
    tab_q, tab_add, tab_edit, tab_del, tab_io = st.tabs(["🔍 查询", "➕ 新增", "✏️ 修改", "🗑️ 删除", "📁 导入/导出"])

    with tab_q:
        c1, c2, c3 = st.columns(3)
        q_ap = c1.selectbox("机场筛选", ["全部"] + sorted(df["机场名称"].unique()))
        q_yr = c2.selectbox("年份筛选", ["全部"] + sorted(df["年份"].unique()))
        q_mo = c3.selectbox("月份筛选", ["全部"] + list(range(1, 13)))
        
        qdf = df.copy()
        if q_ap != "全部": qdf = qdf[qdf["机场名称"] == q_ap]
        if q_yr != "全部": qdf = qdf[qdf["年份"] == q_yr]
        if q_mo != "全部": qdf = qdf[qdf["月份"] == q_mo]
        
        st.info(f"共命中 {len(qdf)} 条记录")
        st.dataframe(qdf, use_container_width=True)

    with tab_add:
        with st.form("add_form"):
            c1, c2, c3 = st.columns(3)
            a_ap = c1.selectbox("机场名称", AIRPORTS)
            a_yr = c2.number_input("年份", 2020, 2035, 2025)
            a_mo = c3.number_input("月份", 1, 12, 1)
            
            a_cg = c1.number_input("货邮吞吐量(吨)", 0.0, 100000.0, 100.0)
            a_st = c2.number_input("货机架次", 0, 100000, 30)
            a_fr = c3.number_input("外贸货占比(%)", 0.0, 100.0, 25.0)
            
            a_gdp = c1.number_input("区域GDP增速(%)", -20.0, 30.0, 4.0)
            a_wd = c2.number_input("月平均风速(m/s)", 0.0, 30.0, 3.5)
            
            if st.form_submit_button("确认新增", type="primary"):
                if not ((df["机场名称"] == a_ap) & (df["年份"] == a_yr) & (df["月份"] == a_mo)).any():
                    new_row = pd.DataFrame([{"机场名称": a_ap, "年份": int(a_yr), "月份": int(a_mo), "货邮吞吐量(吨)": float(a_cg), "货机架次": int(a_st), "外贸货占比(%)": float(a_fr), "区域GDP增速(%)": float(a_gdp), "月平均风速(m/s)": float(a_wd)}])
                    df = pd.concat([df, new_row], ignore_index=True)
                    save_data(df)
                    st.success("新增成功！")
                    rerun()
                else:
                    st.error("记录已存在！")

    with tab_edit:
        edit_idx = st.selectbox("选择记录", options=list(df.index[:200]), format_func=lambda i: f"{df.loc[i, '机场名称']} - {df.loc[i, '年份']}年{df.loc[i, '月份']}月")
        row = df.loc[edit_idx]
        with st.form("edit_form"):
            c1, c2, c3 = st.columns(3)
            e_cg = c1.number_input("货邮吞吐量(吨)", value=float(row["货邮吞吐量(吨)"]))
            e_st = c2.number_input("货机架次", value=int(row["货机架次"]))
            e_fr = c3.number_input("外贸货占比(%)", value=float(row["外贸货占比(%)"]))
            e_gdp = c1.number_input("GDP增速(%)", value=float(row["区域GDP增速(%)"]))
            e_wd = c2.number_input("风速(m/s)", value=float(row["月平均风速(m/s)"]))
            if st.form_submit_button("保存修改", type="primary"):
                df.loc[edit_idx, ["货邮吞吐量(吨)", "货机架次", "外贸货占比(%)", "区域GDP增速(%)", "月平均风速(m/s)"]] = [e_cg, e_st, e_fr, e_gdp, e_wd]
                save_data(df)
                st.success("修改已保存！")
                rerun()

    with tab_del:
        del_idx = st.selectbox("选择删除记录", options=list(df.index), format_func=lambda i: f"{df.loc[i, '机场名称']} - {df.loc[i, '年份']}年{df.loc[i, '月份']}月")
        if st.button("确认删除", type="secondary"):
            df = df.drop(index=del_idx).reset_index(drop=True)
            save_data(df)
            st.success("已删除！")
            rerun()

    with tab_io:
        st.markdown("**批量导入 CSV**")
        up = st.file_uploader("上传文件", type=["csv"])
        if up:
            try:
                imported = pd.read_csv(up, encoding="utf-8-sig")
                if set(COLUMNS).issubset(imported.columns):
                    save_data(imported[COLUMNS])
                    st.success("导入成功！")
                    rerun()
                else:
                    st.error("列名不匹配")
            except Exception as e:
                st.error(f"导入失败：{e}")
                
        st.divider()
        st.download_button("⬇️ 导出当前数据", data=df.to_csv(index=False).encode('utf-8-sig'), file_name="cargo_data.csv", mime="text/csv")

# =============================================================================
# 5. 页面：可视化分析
# =============================================================================
elif page == "📈 可视化分析":
    scope = st.radio("数据范围", ["全部机场对比"] + sorted(df["机场名称"].unique()), horizontal=True)
    plot_df = df if scope == "全部机场对比" else df[df["机场名称"] == scope].copy()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📈 货邮吞吐量时序趋势")
        monthly = plot_df.groupby(["年份", "月份"])["货邮吞吐量(吨)"].sum().reset_index().sort_values(["年份", "月份"])
        monthly["时间"] = monthly["年份"].astype(str) + "-" + monthly["月份"].astype(str).str.zfill(2)
        
        line = Line(init_opts=opts.InitOpts(height="350px", bg_color="transparent"))
        line.add_xaxis(monthly["时间"].tolist())
        line.add_yaxis("货邮吞吐量", monthly["货邮吞吐量(吨)"].round(1).tolist(), is_smooth=True, 
                       label_opts=opts.LabelOpts(is_show=False), 
                       linestyle_opts=opts.LineStyleOpts(width=3, color="#1E6FA8"),
                       itemstyle_opts=opts.ItemStyleOpts(color="#1E6FA8"))
        line.set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45, interval=5)),
            yaxis_opts=opts.AxisOpts(name="吨"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)]
        )
        st_pyecharts(line)

    with col2:
        st.markdown("#### 🥧 货机架次占比")
        pie = Pie(init_opts=opts.InitOpts(height="350px", bg_color="transparent"))
        if scope == "全部机场对比":
            pie_data = df.groupby("机场名称")["货机架次"].sum().sort_values(ascending=False)
            pie.add("", [(str(k), int(v)) for k, v in pie_data.items()], radius=["35%", "65%"], label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
        else:
            pie_data = plot_df.groupby("年份")["货机架次"].sum().sort_index()
            pie.add("", [(f"{k}年", int(v)) for k, v in pie_data.items()], radius=["35%", "65%"], label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
        pie.set_global_opts(legend_opts=opts.LegendOpts(orient="horizontal", pos_bottom="0%"))
        st_pyecharts(pie)

    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🔥 货运季节热力图")
        heat_df = df if scope == "全部机场对比" else plot_df
        heat_pivot = heat_df.groupby(["年份", "月份"])["货邮吞吐量(吨)"].sum().unstack("月份")
        years_h = heat_pivot.index.tolist()
        heat_values = [[m-1, yi, round(float(heat_pivot.loc[yr, m]) if m in heat_pivot.columns and not pd.isna(heat_pivot.loc[yr, m]) else 0, 1)] 
                       for yi, yr in enumerate(years_h) for m in range(1, 13)]
        
        heat = HeatMap(init_opts=opts.InitOpts(height="320px", bg_color="transparent"))
        heat.add_xaxis([f"{m}月" for m in range(1, 13)])
        heat.add_yaxis("年份", [str(y) for y in years_h], heat_values, label_opts=opts.LabelOpts(is_show=True, font_size=10))
        heat.set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=min(v[2] for v in heat_values), max_=max(v[2] for v in heat_values), orient="horizontal", pos_bottom="0%", is_calculable=True),
            xaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(splitarea_opts=opts.SplitAreaOpts(is_show=False))
        )
        st_pyecharts(heat)

    with col4:
        st.markdown("#### 🎯 GDP增速—货邮相关性散点")
        sc_df = df if scope == "全部机场对比" else plot_df
        sc_df = sc_df.sample(min(len(sc_df), 500))
        
        scatter = Scatter(init_opts=opts.InitOpts(height="320px", bg_color="transparent"))
        scatter.add_xaxis(sc_df["区域GDP增速(%)"].round(2).tolist())
        for ap in sorted(sc_df["机场名称"].unique()):
            sub = sc_df[sc_df["机场名称"] == ap]
            scatter.add_yaxis(ap, sub["货邮吞吐量(吨)"].round(1).tolist(), symbol_size=8, label_opts=opts.LabelOpts(is_show=False))
            
        scatter.set_global_opts(
            xaxis_opts=opts.AxisOpts(name="GDP增速(%)", type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)),
            yaxis_opts=opts.AxisOpts(name="货邮吞吐量(吨)", type_="value"),
            legend_opts=opts.LegendOpts(pos_top="0%")
        )
        st_pyecharts(scatter)

# =============================================================================
# 6. 页面：智能预测中心
# =============================================================================
elif page == "🎯 智能预测中心":
    st.markdown("### 多模型预测系统")
    
    # 获取历史基准数据
    pred_ap = st.selectbox("选择预测对象机场", sorted(df["机场名称"].unique()))
    ap_df = df[df["机场名称"] == pred_ap]
    
    # 按年汇总历史数据
    hist_y = ap_df.groupby("年份")["货邮吞吐量(吨)"].sum().sort_index()
    hist_years = hist_y.index.tolist()
    hist_values = hist_y.values.tolist()
    
    recent_year = hist_years[-1]
    base_cargo = hist_values[-1]
    
    # 预测参数设置
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.slider("预测年限 (年)", 1, 15, 5)
    gdp_growth = c2.slider("GDP年均增速 (%)", 0.0, 12.0, 4.0, 0.1)
    elasticity = c3.slider("弹性系数 E", 0.2, 2.5, 1.1, 0.05)
    forecast_year_start = recent_year + 1
    years_future = list(range(forecast_year_start, forecast_year_start + horizon))
    
    # 执行 4 种预测模型
    preds_elasticity = model_elasticity(base_cargo, gdp_growth, elasticity, horizon)
    preds_linear = model_linear_regression(hist_values, horizon)
    preds_gm11 = model_gm11(hist_values, horizon)
    preds_holt = model_holt(hist_values, horizon)
    
    # 图表对比展示
    st.markdown("#### 📊 模型预测结果对比")
    
    # 构建对比图
    comp_line = Line(init_opts=opts.InitOpts(height="450px", bg_color="transparent"))
    comp_line.add_xaxis([str(y) for y in hist_years] + [str(y) for y in years_future])
    
    # 历史数据线
    comp_line.add_yaxis(
        "历史真实数据", 
        [round(v, 1) for v in hist_values] + [None] * horizon, 
        is_smooth=True, 
        linestyle_opts=opts.LineStyleOpts(width=3, color="#333333"),
        itemstyle_opts=opts.ItemStyleOpts(color="#333333")
    )
    
    # 4条预测线
    models = [
        ("弹性系数法 (专业规划)", preds_elasticity, "#1E6FA8"),
        ("线性回归", preds_linear, "#E74C3C"),
        ("GM(1,1) 灰色预测", preds_gm11, "#2ECC71"),
        ("Holt 指数平滑", preds_holt, "#F39C12")
    ]
    
    for name, preds, color in models:
        comp_line.add_yaxis(
            name, 
            [None] * len(hist_years) + [round(v, 1) for v in preds], 
            is_smooth=True, 
            linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed", color=color),
            itemstyle_opts=opts.ItemStyleOpts(color=color)
        )
        
    comp_line.set_global_opts(
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(name="吞吐量(吨)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="0%")
    )
    st_pyecharts(comp_line)
    
    # 数据表格展示
    st.markdown("#### 📋 预测数据明细 (单位：吨)")
    res_df = pd.DataFrame({
        "年份": years_future,
        "弹性系数法": [round(v, 1) for v in preds_elasticity],
        "线性回归": [round(v, 1) for v in preds_linear],
        "GM(1,1)": [round(v, 1) for v in preds_gm11],
        "Holt平滑": [round(v, 1) for v in preds_holt],
    }).set_index("年份")
    
    st.dataframe(res_df.style.highlight_max(axis=1, color='#D4E6F1'), use_container_width=True)
    
    st.info("💡 **方法说明**：\n"
            "- **弹性系数法**：基于民航规划规范，\(Y_t = Y_0(1+E \\times M)^n\)。\n"
            "- **线性回归**：基于最小二乘法拟合历史时间序列趋势。\n"
            "- **GM(1,1)**：适合小样本（如新建/支线机场）的灰色系统预测。\n"
            "- **Holt平滑**：考虑数据趋势的指数平滑算法，适合短期预测。")
