import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

# ==================== 1. 页面基础配置 ====================
st.set_page_config(page_title="净泥智控 - 水质智能分析平台", page_icon="💧", layout="wide")

# ==================== 2. 全局状态初始化 ====================
if "page" not in st.session_state:
    st.session_state.page = "main"

if "history_data" not in st.session_state:
    st.session_state.history_data = pd.DataFrame(columns=[
        "序号", "监测时间", "进水流量Q(m³)", "进水BOD5", "进水SS", "进水COD",
        "进水NH3-N", "进水TN", "进水TP", "进水pH值",
        "出水BOD5", "出水SS", "出水COD", "出水NH3-N", "出水TN", "出水TP", "出水pH值",
        "AI水质解析与建议"
    ])

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = time.time()
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "row_counter" not in st.session_state:
    st.session_state.row_counter = 0
if "theme" not in st.session_state:
    st.session_state.theme = "plotly_dark"
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False


# ==================== 3. 缓存模型训练与SHAP计算（含防缓存残缺修复） ====================
@st.cache_resource(show_spinner="正在初始化模型与数据...")
def load_and_train_models():
    """模拟历史数据，训练4个模型并计算SHAP值，只在首次加载时消耗内存"""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        "进水流量": np.random.normal(50000, 5000, n_samples),
        "进水COD": np.random.normal(250, 40, n_samples),
        "进水BOD5": np.random.normal(105, 10, n_samples),
        "进水SS": np.random.normal(255, 15, n_samples),
        "进水NH3-N": np.random.normal(26, 3, n_samples),
        "进水TP": np.random.normal(4.2, 0.5, n_samples),
        "进水TN": np.random.normal(32, 2, n_samples),
        "水温": np.random.normal(18.5, 0.8, n_samples),
    })

    y_fm = 0.18 + (X["进水BOD5"] / 1000 - X["进水SS"] / 20000) + np.random.normal(0, 0.01, n_samples)

    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(alpha=0.1),
        "RF": RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    }

    trained_models = {}
    metrics = {}
    shap_values_dict = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y_fm, test_size=0.2, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        metrics[name] = {
            "R²": round(r2, 3),
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "MAPE": round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2)
        }

        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test[:50])
        shap_values_dict[name] = shap_values

    # ============ 新增：防止云端缓存残缺的兜底逻辑 ============
    if "Linear" not in trained_models:
        st.warning("检测到云端缓存异常，正在重新初始化模型...")
        st.cache_resource.clear()
        return load_and_train_models()
    # =======================================================

    return trained_models, metrics, shap_values_dict, X, y_fm, X_test, y_test


# ==================== 4. 实时数据生成 ====================
def generate_realtime_row():
    st.session_state.row_counter += 1
    inflow_q = round(random.uniform(45000, 55000), 2)
    in_bod = round(random.uniform(90, 120), 2)
    in_ss = round(random.uniform(230, 280), 2)
    in_cod = round(random.uniform(150, 320), 2)
    in_nh3 = round(random.uniform(20, 35), 2)
    in_tn = round(random.uniform(28, 36), 2)
    in_tp = round(random.uniform(3.2, 5.2), 2)
    in_ph = round(random.uniform(6.4, 6.8), 2)
    out_bod = round(random.uniform(3.5, 5.5), 2)
    out_ss = round(random.uniform(0.5, 2.0), 2)
    out_cod = round(random.uniform(9.0, 15.0), 2)
    out_nh3 = round(random.uniform(0.4, 1.2), 2)
    out_tn = round(random.uniform(9.0, 15.0), 2)
    out_tp = round(random.uniform(0.05, 0.25), 2)
    out_ph = round(random.uniform(6.2, 6.6), 2)

    if out_cod > 12:
        ai_advice = f"进水COD偏高({in_cod}mg/L)，出水COD({out_cod})接近限值。建议加大生化池曝气量，适当增加碳源投加，关注污泥活性。"
    elif out_tn > 13:
        ai_advice = f"出水TN({out_tn}mg/L)偏高。建议检查内回流比，适当补充碳源，强化反硝化脱氮效果。"
    elif in_ph < 6.5:
        ai_advice = f"进水pH({in_ph})偏低，可能影响生化处理。建议投加碱性药剂调节，维持在6.5-7.5之间。"
    else:
        ai_advice = "各项指标运行平稳，出水水质达标。建议维持当前工艺参数，持续监测COD和氨氮变化。"

    return {
        "序号": st.session_state.row_counter,
        "监测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "进水流量Q(m³)": inflow_q, "进水BOD5": in_bod, "进水SS": in_ss, "进水COD": in_cod,
        "进水NH3-N": in_nh3, "进水TN": in_tn, "进水TP": in_tp, "进水pH值": in_ph,
        "出水BOD5": out_bod, "出水SS": out_ss, "出水COD": out_cod,
        "出水NH3-N": out_nh3, "出水TN": out_tn, "出水TP": out_tp, "出水pH值": out_ph,
        "AI水质解析与建议": ai_advice
    }


# ==================== 5. 主界面 ====================
def show_main():
    models, metrics, shap_values_dict, X, y_fm, X_test, y_test = load_and_train_models()

    # ----- 侧边栏 -----
    with st.sidebar:
        st.header("⚙️ 系统设置与数据接入")

        theme_choice = st.radio("🎨 界面主题", ["🌙 暗黑模式", "☀️ 明亮模式"], index=0)
        st.session_state.theme = "plotly_dark" if "暗黑" in theme_choice else "plotly_white"

        st.divider()
        st.subheader("🔌 实时数据接入")
        data_source = st.radio("数据源选择", ["模拟实时数据", "手动输入"], index=0)

        if data_source == "手动输入":
            st.number_input("进水流量", value=50000, key="manual_q")
            st.number_input("进水COD", value=280, key="manual_cod")
            st.number_input("进水氨氮", value=25.0, key="manual_nh3")
            if st.button("▶️ 手动记录本次数据", type="primary"):
                st.session_state.predicted = True
                new_row = generate_realtime_row()
                st.session_state.history_data = pd.concat([st.session_state.history_data, pd.DataFrame([new_row])], ignore_index=True)
                st.toast("✅ 手动数据已记录！")
        else:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("▶️ 启动采集", type="primary", use_container_width=True):
                    st.session_state.predicted = True
                    st.session_state.is_paused = False
                    st.toast("⏰ 采集已开始...")
            with col_btn2:
                if st.button("⏸️ 暂停采集", use_container_width=True):
                    st.session_state.is_paused = True
                    st.toast("⏸️ 采集已暂停，数据保留。")

            if st.button("🗑️ 清空所有数据并重置", use_container_width=True):
                st.session_state.history_data = pd.DataFrame(columns=st.session_state.history_data.columns)
                st.session_state.row_counter = 0
                st.session_state.predicted = False
                st.session_state.is_paused = False
                st.session_state.last_run_time = time.time()
                st.toast("✅ 数据已清空，系统已重置！")
                st.rerun()

        st.divider()
        st.subheader("⏱️ 实时自动输出设置")
        col_val, col_unit = st.columns([2, 1])
        with col_val:
            interval_val = st.number_input("时间间隔", min_value=1, value=5, step=1)
        with col_unit:
            interval_unit = st.selectbox("单位", ["秒", "分钟", "小时"], index=0)
        update_interval = interval_val * {"秒": 1, "分钟": 60, "小时": 3600}[interval_unit]
        st.caption(f"当前设定：每 {interval_val} {interval_unit} 自动追加一次数据")

    # ----- 主区域标题 -----
    st.markdown("<h2 style='text-align: center; color: #3B82F6;'>💧 污水处理智能分析平台 v6.0</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B;'>实时数据监控与AI智能解析系统</p>", unsafe_allow_html=True)
    st.divider()

    # ----- 自动定时追加（含暂停判断） -----
    if st.session_state.predicted and data_source == "模拟实时数据":
        if not st.session_state.get("is_paused", False):
            st_autorefresh(interval=1000, key="data_refresh")
            if time.time() - st.session_state.last_run_time >= update_interval:
                st.session_state.last_run_time = time.time()
                new_row = generate_realtime_row()
                st.session_state.history_data = pd.concat([st.session_state.history_data, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state.history_data = st.session_state.history_data.tail(10)
                st.toast(f"⏰ {new_row['监测时间']} 已自动追加新数据！")

    # ==================== 5大标签页 ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 实时数据与AI解析", "⏳ 水质时间序列",
        "📈 特征重要性分析", "🤖 模型评价与对比", "🔍 SHAP解释"
    ])

    # ---- Tab 1：实时数据 + AI 解析 ----
    with tab1:
        st.subheader("📊 进出水水质实时监控与AI智能建议")
        if st.session_state.history_data.empty:
            st.info("💡 请在左侧选择数据源并开始采集。")
        else:
            latest = st.session_state.history_data.iloc[-1]
            with st.container(border=True):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("进水COD", f"{latest['进水COD']} mg/L", "-2%")
                col_b.metric("出水COD", f"{latest['出水COD']} mg/L", "-5%")
                col_c.metric("出水氨氮", f"{latest['出水NH3-N']} mg/L", "-1%")
                col_d.metric("出水总氮", f"{latest['出水TN']} mg/L", "+2%")
            with st.container(border=True):
                st.markdown("### 🤖 AI 大模型水质解析建议")
                st.success(f"**当前工况建议：** {latest['AI水质解析与建议']}")

            csv = st.session_state.history_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载实时数据 (CSV)", csv, "实时水质数据.csv", "text/csv", type="primary")
            st.dataframe(st.session_state.history_data, use_container_width=True, height=400)

    # ---- Tab 2：时间序列 ----
    with tab2:
        st.subheader("⏳ 进出水COD时间序列趋势")
        dates = pd.date_range(start="2026-08-15", periods=30, freq="D")
        ts_data = pd.DataFrame({
            "日期": dates,
            "进水COD": np.random.normal(250, 40, 30),
            "出水COD": np.random.normal(12, 3, 30)
        })
        fig_ts = px.line(ts_data, x="日期", y=["进水COD", "出水COD"], title="近30天进出水COD变化趋势")
        fig_ts.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ---- Tab 3：特征重要性分析 ----
    with tab3:
        st.subheader("📈 特征重要性与斯皮尔曼相关性分析")

        st.markdown("### 🔥 斯皮尔曼相关性热力图（各指标间相关性）")
        corr_matrix = X.corr(method='spearman')
        fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', title="Spearman Correlation Heatmap")
        fig_heat.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("### 🎯 特征重要性（判断影响污泥龄的最大变量）")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            selected_model = st.selectbox("选择模型", ["Linear", "Lasso", "RF", "XGBoost"])
        with col_m2:
            st.write("")

        if selected_model == "Linear":
            importance = np.abs(models["Linear"].coef_)
        elif selected_model == "Lasso":
            importance = np.abs(models["Lasso"].coef_)
        elif selected_model == "RF":
            importance = models["RF"].feature_importances_
        else:
            importance = models["XGBoost"].feature_importances_

        features = X.columns.tolist()
        fig_fi = px.bar(x=importance, y=features, orientation='h', title=f"{selected_model} - F/M Ratio Feature Importance")
        fig_fi.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ---- Tab 4：模型评价与对比 ----
    with tab4:
        st.subheader("🤖 模型性能评价与对比分析")

        st.markdown("### 📉 预测值 vs 实测值（散点图）")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        y_pred_linear = models["Linear"].predict(X_test)
        y_pred_lasso = models["Lasso"].predict(X_test)
        y_pred_rf = models["RF"].predict(X_test)
        y_pred_xgb = models["XGBoost"].predict(X_test)

        with col_s1:
            fig_s1 = px.scatter(x=y_test, y=y_pred_linear, trendline="ols", title=f"Linear (R²={metrics['Linear']['R²']})")
            fig_s1.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            fig_s1.update_layout(template=st.session_state.theme)
            st.plotly_chart(fig_s1, use_container_width=True)
        with col_s2:
            fig_s2 = px.scatter(x=y_test, y=y_pred_lasso, trendline="ols", title=f"Lasso (R²={metrics['Lasso']['R²']})")
            fig_s2.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            fig_s2.update_layout(template=st.session_state.theme)
            st.plotly_chart(fig_s2, use_container_width=True)
        with col_s3:
            fig_s3 = px.scatter(x=y_test, y=y_pred_rf, trendline="ols", title=f"RF (R²={metrics['RF']['R²']})")
            fig_s3.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            fig_s3.update_layout(template=st.session_state.theme)
            st.plotly_chart(fig_s3, use_container_width=True)
        with col_s4:
            fig_s4 = px.scatter(x=y_test, y=y_pred_xgb, trendline="ols", title=f"XGBoost (R²={metrics['XGBoost']['R²']})")
            fig_s4.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            fig_s4.update_layout(template=st.session_state.theme)
            st.plotly_chart(fig_s4, use_container_width=True)

        st.markdown("### 📊 模型评价指标对比")
        metric_choice = st.selectbox("选择评价指标", ["R²", "RMSE", "MAE", "MAPE"])
        model_names = list(metrics.keys())
        values = [metrics[m][metric_choice] for m in model_names]

        col_bar, col_table = st.columns([2, 1])
        with col_bar:
            fig_bar = px.bar(x=model_names, y=values, title=f"{metric_choice} 对比", color=model_names)
            fig_bar.update_layout(template=st.session_state.theme)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_table:
            df_metrics = pd.DataFrame(metrics).T
            st.dataframe(df_metrics)
            st.download_button("📥 下载评价指标表", df_metrics.to_csv().encode('utf-8-sig'), "模型评价指标.csv", "text/csv")

        st.markdown("### 📦 误差分布（箱线图）")
        df_error = pd.DataFrame({
            "模型": ["Linear"]*len(y_test) + ["Lasso"]*len(y_test) + ["RF"]*len(y_test) + ["XGBoost"]*len(y_test),
            "绝对误差": np.concatenate([
                np.abs(y_test - y_pred_linear),
                np.abs(y_test - y_pred_lasso),
                np.abs(y_test - y_pred_rf),
                np.abs(y_test - y_pred_xgb)
            ])
        })
        fig_box = px.box(df_error, x="模型", y="绝对误差", color="模型", title="Absolute Error Distribution")
        fig_box.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_box, use_container_width=True)

    # ---- Tab 5：SHAP分析 ----
    with tab5:
        st.subheader("🔍 SHAP 模型可解释性分析")
        shap_model = st.selectbox("选择SHAP分析的模型", ["Linear", "Lasso", "RF", "XGBoost"])
        shap_vals = shap_values_dict[shap_model]

        st.markdown("### 🐝 SHAP 蜂群图 (特征分布影响)")
        fig_bee = go.Figure()
        for i, feature in enumerate(X.columns):
            fig_bee.add_trace(go.Scatter(
                x=shap_vals.values[:, i],
                y=[feature]*len(shap_vals.values),
                mode='markers',
                marker=dict(size=8, color=shap_vals.values[:, i], colorscale='RdBu_r'),
                showlegend=False
            ))
        fig_bee.update_layout(title="SHAP Beeswarm Plot", xaxis_title="SHAP Value", yaxis_title="Feature")
        fig_bee.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_bee, use_container_width=True)

        st.markdown("### 📊 SHAP 条形图 (特征平均贡献度)")
        mean_shap = np.abs(shap_vals.values).mean(axis=0)
        fig_bar_shap = px.bar(x=mean_shap, y=X.columns, orientation='h', title="SHAP Feature Importance (Mean |SHAP|)")
        fig_bar_shap.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_bar_shap, use_container_width=True)

        st.markdown("### 📋 SHAP值数据表格")
        df_shap = pd.DataFrame(shap_vals.values, columns=X.columns)
        st.dataframe(df_shap.head(10))
        st.download_button("📥 下载SHAP值分析表", df_shap.to_csv(index=False).encode('utf-8-sig'), "SHAP分析表.csv", "text/csv")

    st.markdown("<br><p style='text-align: center; color: #64748B;'>© 2026 驰星队 · 马鞍山学院</p>", unsafe_allow_html=True)


# ==================== 6. 路由控制 ====================
if st.session_state.page == "welcome":
    show_welcome()
else:
    show_main()
