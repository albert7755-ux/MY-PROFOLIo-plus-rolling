import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
import gspread

# ==========================================
# 設定
# ==========================================
# 存續期間計算函式
def calc_modified_duration(coupon_rate, ytm, years_to_maturity, freq=2):
    """計算修正存續期間（Modified Duration）"""
    try:
        if years_to_maturity <= 0 or coupon_rate <= 0:
            return None
        c = coupon_rate / freq  # 每期票息
        n = int(years_to_maturity * freq)  # 總期數
        r = ytm / freq  # 每期殖利率
        if r <= 0:
            return None
        # 計算 Macaulay Duration
        pv_sum = 0
        price = 0
        for t in range(1, n + 1):
            cf = c if t < n else c + 1  # 最後一期加回本金
            pv = cf / (1 + r) ** t
            pv_sum += t * pv
            price += pv
        if price <= 0:
            return None
        macaulay = pv_sum / price / freq
        modified = macaulay / (1 + ytm / freq)
        return modified
    except:
        return None

def calc_convexity(coupon_rate, ytm, years_to_maturity, freq=2):
    """計算凸性（Convexity）"""
    try:
        if years_to_maturity <= 0 or coupon_rate <= 0:
            return None
        c = coupon_rate / freq
        n = int(years_to_maturity * freq)
        r = ytm / freq
        if r <= 0:
            return None
        conv_sum = 0
        price = 0
        for t in range(1, n + 1):
            cf = c if t < n else c + 1
            pv = cf / (1 + r) ** t
            conv_sum += t * (t + 1) * pv
            price += pv
        if price <= 0:
            return None
        convexity = conv_sum / (price * (1 + r) ** 2 * freq ** 2)
        return convexity
    except:
        return None
st.set_page_config(page_title="智能投資組合優化器", layout="wide")

BOND_DRIVE_FOLDER_ID = "1k0RxJn5KKCTWdTEDZqq0Q5hnfwkuPgGK"
FUND_DRIVE_FOLDER_ID = "1i1-zUzLNnuwo2NVWijubvBICLbladZQO"
BOND_MASTER_NAME = " bond_master"

# ==========================================
# 🔐 登入
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 系統登入")
    password = st.text_input("🔑 請輸入系統密碼", type="password")
    if password:
        if password == "5428":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("⛔ 密碼錯誤")
    st.stop()

# ==========================================
# Google Drive 連線
# ==========================================
@st.cache_resource
def get_gcp_credentials():
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        # 支援 JSON 字串格式
        secret = st.secrets["gcp_service_account"]
        if isinstance(secret, str):
            creds_info = json.loads(secret)
        else:
            creds_info = dict(secret)
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        return creds
    except Exception as e:
        st.error(f"❌ Google 憑證錯誤：{e}")
        return None

@st.cache_resource
def get_gspread_client():
    creds = get_gcp_credentials()
    if creds:
        return gspread.authorize(creds)
    return None

@st.cache_data(ttl=3600)
def get_drive_files(folder_id):
    """列出 Drive 資料夾內所有檔案"""
    try:
        creds = get_gcp_credentials()
        if not creds:
            return {}
        creds.refresh(Request())
        headers = {"Authorization": f"Bearer {creds.token}"}
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "files(id, name)",
            "pageSize": 500,
        }
        resp = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=headers, params=params
        )
        return {f["name"]: f["id"] for f in resp.json().get("files", [])}
    except Exception as e:
        return {}

@st.cache_data(ttl=3600)
def load_bond_master():
    """讀取 bond_master"""
    try:
        client = get_gspread_client()
        if not client:
            return pd.DataFrame()
        drive_files = get_drive_files(BOND_DRIVE_FOLDER_ID)
        master_id = drive_files.get(BOND_MASTER_NAME) or drive_files.get("bond_master")
        if not master_id:
            return pd.DataFrame()
        sh = client.open_by_key(master_id)
        ws = sh.get_worksheet(0)
        data = ws.get_all_values()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_bond_prices(sheet_name, file_id):
    """讀取單一債券的價格歷史"""
    try:
        client = get_gspread_client()
        if not client:
            return pd.Series(dtype=float)
        sh = client.open_by_key(file_id)
        ws = sh.get_worksheet(0)
        data = ws.get_all_values()
        if len(data) < 2:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data[1:], columns=['time', 'close'])
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna()
        df = df.set_index('time')['close']
        return df
    except Exception as e:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def load_fund_prices(file_id):
    """讀取基金 NAV"""
    try:
        creds = get_gcp_credentials()
        if not creds:
            return pd.Series(dtype=float)
        creds.refresh(Request())
        headers = {"Authorization": f"Bearer {creds.token}"}
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers=headers
        )
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        # 找日期和NAV欄位
        date_col = None
        nav_col = None
        for col in df.columns:
            if any(k in col.lower() for k in ['date', '日期', 'time']):
                date_col = col
            if any(k in col.lower() for k in ['nav', 'close', '淨值', 'price']):
                nav_col = col
        if not date_col or not nav_col:
            # 假設第一欄是日期，第二欄是NAV
            date_col = df.columns[0]
            nav_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[nav_col] = pd.to_numeric(df[nav_col], errors='coerce')
        df = df.dropna(subset=[date_col, nav_col])
        series = df.set_index(date_col)[nav_col]
        return series
    except Exception as e:
        return pd.Series(dtype=float)

# ==========================================
# 主程式
# ==========================================
st.title("📐 最適投資組合優化器")
st.markdown("結合債券、基金、自選股票/ETF，計算最適配置比例")

# 載入債券清單
with st.spinner("載入債券清單..."):
    df_master = load_bond_master()
    drive_files_bond = get_drive_files(BOND_DRIVE_FOLDER_ID)
    drive_files_fund = get_drive_files(FUND_DRIVE_FOLDER_ID)

# ==========================================
# 側邊欄：標的選擇
# ==========================================
st.sidebar.header("📋 標的選擇")

# 債券選擇
bond_options = []
if not df_master.empty:
    for _, row in df_master.iterrows():
        filename = str(row.get('檔名', '')).strip()
        name = str(row.get('債券名稱', '')).strip()
        coupon = str(row.get('票息率', '')).strip()
        maturity = str(row.get('到期日', '')).strip()

        # 組合顯示名稱
        label = name
        if coupon and coupon not in ['', 'nan']:
            try:
                if '%' in coupon:
                    label += f" {coupon}"  # 已是百分比字串，直接用
                else:
                    label += f" {float(coupon)*100:.2f}%"
            except:
                pass
        if maturity and maturity not in ['', 'nan']:
            label += f" {maturity[:4]}"  # 只顯示年份

        bond_options.append({
            "label": label,
            "filename": filename,
            "name": name,
        })

bond_labels = [b["label"] for b in bond_options]
selected_bond_labels = st.sidebar.multiselect(
    f"債券（{len(bond_labels)} 檔）",
    bond_labels,
    max_selections=15,
)

# 基金選擇
fund_options = []
for fname, fid in drive_files_fund.items():
    if fname.endswith('.csv'):
        fund_options.append({"label": fname.replace('.csv', ''), "file_id": fid})

fund_labels = [f["label"] for f in fund_options]
selected_fund_labels = st.sidebar.multiselect(
    f"基金（{len(fund_labels)} 檔）",
    fund_labels,
    max_selections=10,
)

# 股票/ETF
st.sidebar.text_area(
    "自選股票/ETF（每行一個代號）",
    key="stock_input",
    placeholder="AAPL\nNVDA\nSPY",
    height=100,
)
stock_tickers = [t.strip().upper() for t in st.session_state.get("stock_input", "").split() if t.strip()]

total_selected = len(selected_bond_labels) + len(selected_fund_labels) + len(stock_tickers)
st.sidebar.info(f"已選擇 **{total_selected}** 個標的（債券 {len(selected_bond_labels)} + 基金 {len(selected_fund_labels)} + 股票 {len(stock_tickers)}）")

# ==========================================
# 側邊欄：回測設定
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 回測設定")

years = st.sidebar.selectbox("回測期間", [1, 2, 3, 5, 10], index=2)

opt_method = st.sidebar.radio(
    "優化目標",
    ["🚀 最大夏普比率", "🛡️ 最小風險", "🎯 鎖定目標報酬"],
)

target_return = 0.0
if "目標報酬" in opt_method:
    target_return = st.sidebar.slider("目標年化報酬率", 1.0, 30.0, 8.0, 0.5) / 100

initial_investment = st.sidebar.number_input("初始本金 (USD)", value=100000, step=10000)
risk_free_rate = st.sidebar.number_input("無風險利率 (%)", value=4.5, step=0.1) / 100

# 融資設定
st.sidebar.markdown("---")
use_margin = st.sidebar.checkbox("開啟融資模式")
loan_ratio, margin_rate, leverage = 0.0, 0.0, 1.0
if use_margin:
    loan_ratio = st.sidebar.slider("融資成數", 0.1, 0.9, 0.6, 0.1)
    margin_rate = st.sidebar.number_input("融資年利率 (%)", 2.0, 15.0, 6.0, 0.1) / 100
    leverage = 1 / (1 - loan_ratio)
    st.sidebar.info(f"槓桿：**{leverage:.1f} 倍**")

# ==========================================
# 開始計算
# ==========================================
if st.sidebar.button("🚀 開始計算", type="primary"):
    if total_selected < 2:
        st.error("請至少選擇 2 個標的！")
        st.stop()

    with st.spinner("載入數據中..."):
        all_series = {}
        errors = []

        # 載入債券
        for label in selected_bond_labels:
            bond_info = next((b for b in bond_options if b["label"] == label), None)
            if not bond_info:
                errors.append(f"找不到 {label}")
                continue
            filename = bond_info["filename"]
            name = bond_info["name"]
            # 找對應的 Drive 檔案
            file_id = None
            for possible in [f"{filename}, 1D", filename]:
                if possible in drive_files_bond:
                    file_id = drive_files_bond[possible]
                    break
            if not file_id:
                errors.append(f"找不到 {name} 的數據")
                continue
            series = load_bond_prices(filename, file_id)
            if series.empty:
                errors.append(f"{name} 數據為空")
            else:
                all_series[name] = series

        # 載入基金
        for label in selected_fund_labels:
            fund_info = next((f for f in fund_options if f["label"] == label), None)
            if not fund_info:
                continue
            series = load_fund_prices(fund_info["file_id"])
            if series.empty:
                errors.append(f"{label} 基金數據為空")
            else:
                all_series[label] = series

        # 載入股票
        if stock_tickers:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365 * years + 365)
            try:
                stock_data = yf.download(stock_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
                if 'Close' in stock_data.columns:
                    df_stocks = stock_data['Close']
                else:
                    df_stocks = stock_data
                if isinstance(df_stocks, pd.Series):
                    df_stocks = df_stocks.to_frame(name=stock_tickers[0])
                for ticker in df_stocks.columns:
                    s = df_stocks[ticker].dropna()
                    if not s.empty:
                        if s.index.tz is not None:
                            s.index = s.index.tz_localize(None)
                        all_series[ticker] = s
            except Exception as e:
                errors.append(f"股票數據載入失敗：{e}")

        if errors:
            for err in errors:
                st.warning(f"⚠️ {err}")

        if len(all_series) < 2:
            st.error("有效標的不足 2 個，無法計算！")
            st.stop()

    with st.spinner("計算最適配置..."):
        # 合併所有數據，取交集日期，過去 N 年
        cutoff = datetime.today() - timedelta(days=365 * years)
        df_all = pd.DataFrame()
        for name, series in all_series.items():
            series.index = pd.to_datetime(series.index)
            series = series[series.index >= cutoff]
            series.name = name
            if df_all.empty:
                df_all = series.to_frame()
            else:
                df_all = df_all.join(series, how='inner')

        df_all = df_all.dropna()
        if df_all.empty or len(df_all) < 30:
            st.error("交集數據不足，請調整回測期間或標的！")
            st.stop()

        tickers = df_all.columns.tolist()
        returns = df_all.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(tickers)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = [1/num_assets] * num_assets

        # 優化
        if "最小風險" in opt_method:
            strategy_name = "🛡️ 最小風險"
            strategy_color = "green"
            def obj(w): return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            res = minimize(obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        elif "最大夏普" in opt_method:
            strategy_name = "🚀 最大夏普"
            strategy_color = "red"
            def obj(w):
                r = np.sum(mean_returns * w)
                v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                return -(r - risk_free_rate) / v
            res = minimize(obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            strategy_name = f"🎯 目標報酬 {target_return:.1%}"
            strategy_color = "blue"
            def obj(w): return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            cons = constraints + [{'type': 'ineq', 'fun': lambda w: np.sum(mean_returns * w) - target_return}]
            res = minimize(obj, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

        optimal_weights = res.x
        optimal_weights = np.maximum(optimal_weights, 0)
        optimal_weights /= optimal_weights.sum()

        # 計算回測績效
        normalized = df_all / df_all.iloc[0]
        raw_port = (normalized * optimal_weights).sum(axis=1)

        # 融資調整
        if use_margin:
            days_arr = np.arange(len(raw_port))
            debt = leverage - 1
            daily_rate = margin_rate / 365
            position = raw_port * leverage
            interest = pd.Series(days_arr * debt * daily_rate, index=raw_port.index)
            port_val = position - debt - interest
        else:
            port_val = raw_port.copy()
        port_val.name = strategy_name

        # 統計
        total_ret = port_val.iloc[-1] - 1
        daily_ret = port_val.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (daily_ret.mean() * 252 - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        roll_max = port_val.cummax()
        mdd = ((port_val - roll_max) / roll_max).min()

        # 年化報酬（排除當年）
        ann_prices = port_val.resample('YE').last()
        ann_rets = ann_prices.pct_change().dropna()
        curr_yr = datetime.now().year
        if curr_yr in ann_rets.index.year:
            ann_rets_clean = ann_rets[ann_rets.index.year != curr_yr]
        else:
            ann_rets_clean = ann_rets
        avg_annual = ann_rets_clean.mean() if not ann_rets_clean.empty else total_ret

    # ==========================================
    # 顯示結果
    # ==========================================
    st.success(f"✅ 計算完成！策略：{strategy_name}，數據期間：{df_all.index[0].date()} ~ {df_all.index[-1].date()}，共 {len(df_all)} 個交易日")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 建議配置")
        df_w = pd.DataFrame({'標的': tickers, '權重': optimal_weights})
        df_w = df_w[df_w['權重'] > 0.001].sort_values('權重', ascending=False)
        df_w['顯示'] = df_w['權重'].apply(lambda x: f"{x:.1%}")
        st.table(df_w[['標的', '顯示']].rename(columns={'顯示': '配置'}))

        fig_pie = px.pie(df_w, values='權重', names='標的', hole=0.4)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📈 資產成長回測")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=port_val.index, y=port_val * initial_investment,
            name=strategy_name, line=dict(color=strategy_color, width=3)
        ))
        # 個別標的
        for ticker in tickers:
            s = normalized[ticker] * initial_investment
            fig.add_trace(go.Scatter(
                x=s.index, y=s, name=ticker,
                line=dict(width=1), opacity=0.5
            ))
        fig.update_layout(
            yaxis_title="資產價值 (USD)",
            hovermode="x unified", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPI
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總報酬", f"{total_ret:.2%}")
        c2.metric("平均年報酬", f"{avg_annual:.2%}")
        c3.metric("年化波動", f"{ann_vol:.2%}")
        c4.metric("最大回撤", f"{mdd:.2%}")

        c5, c6 = st.columns(2)
        c5.metric("夏普比率", f"{sharpe:.2f}")
        c6.metric("期末資產", f"${port_val.iloc[-1] * initial_investment:,.0f}")

    # 債券存續期間分析
    bond_duration_data = []
    for label in selected_bond_labels:
        bond_info = next((b for b in bond_options if b["label"] == label), None)
        if not bond_info:
            continue
        # 從 master 找票息率和到期日
        row = df_master[df_master['債券名稱'] == bond_info["name"]]
        if row.empty:
            continue
        row = row.iloc[0]
        try:
            coupon_str = str(row.get('票息率', '')).strip()
            if '%' in coupon_str:
                coupon = float(coupon_str.replace('%', '')) / 100
            else:
                coupon = float(coupon_str)
            maturity_str = str(row.get('到期日', '')).strip()
            if not maturity_str or maturity_str == 'nan':
                continue
            maturity_date = pd.to_datetime(maturity_str)
            years_to_mat = (maturity_date - datetime.today()).days / 365
            if years_to_mat <= 0:
                continue
            # 用當前價格反推 YTM（簡化：用近似法）
            name = bond_info["name"]
            if name in df_all.columns:
                current_price = df_all[name].iloc[-1] / 100  # 換算為面額比例
            else:
                current_price = 1.0
            # 簡化 YTM 近似
            ytm_approx = (coupon + (1 - current_price) / years_to_mat) / ((1 + current_price) / 2)
            ytm_approx = max(0.001, ytm_approx)

            dur = calc_modified_duration(coupon, ytm_approx, years_to_mat)
            conv = calc_convexity(coupon, ytm_approx, years_to_mat)

            bond_duration_data.append({
                "債券名稱": name,
                "票息率": f"{coupon*100:.2f}%",
                "到期日": maturity_str[:10],
                "剩餘年期": f"{years_to_mat:.1f}年",
                "估計YTM": f"{ytm_approx*100:.2f}%",
                "修正存續期間": f"{dur:.2f}" if dur else "N/A",
                "凸性": f"{conv:.2f}" if conv else "N/A",
                "殖利率↑1% 價格變化": f"{-dur*0.01*100:.2f}%" if dur else "N/A",
            })
        except:
            continue

    if bond_duration_data:
        st.markdown("---")
        st.subheader("📐 債券存續期間分析")
        df_dur = pd.DataFrame(bond_duration_data)

        # 加權平均存續期間
        try:
            dur_values = [float(r["修正存續期間"]) for r in bond_duration_data if r["修正存續期間"] != "N/A"]
            bond_weights_dur = [optimal_weights[tickers.index(r["債券名稱"])] for r in bond_duration_data
                               if r["修正存續期間"] != "N/A" and r["債券名稱"] in tickers]
            if dur_values and bond_weights_dur:
                total_w = sum(bond_weights_dur)
                if total_w > 0:
                    weighted_dur = sum(d*w for d,w in zip(dur_values, bond_weights_dur)) / total_w
                    st.metric("📊 投資組合加權平均存續期間", f"{weighted_dur:.2f} 年",
                             help="殖利率每上升1%，投資組合價格約下降此數值%")
        except:
            pass

        st.dataframe(df_dur, use_container_width=True)
        st.caption("註：YTM 為根據當前市價估算，修正存續期間代表殖利率變動1%時的價格敏感度。")

    # 年度報酬表
    st.markdown("---")
    st.subheader("📅 年度報酬對照")
    df_yearly = pd.DataFrame()
    df_yearly[strategy_name] = port_val.resample('YE').last().pct_change()
    for ticker in tickers:
        df_yearly[ticker] = normalized[ticker].resample('YE').last().pct_change()
    df_yearly = df_yearly.dropna(how='all')
    df_yearly.index = df_yearly.index.year
    df_yearly = df_yearly.sort_index(ascending=False)

    # 加平均行
    avg_row = df_yearly[df_yearly.index != curr_yr].mean()
    df_avg = avg_row.to_frame(name="🔥 平均報酬").T
    df_final = pd.concat([df_avg, df_yearly])

    st.dataframe(
        df_final.style.format("{:.2%}")
        .background_gradient(cmap='RdYlGn', vmin=-0.3, vmax=0.3),
        use_container_width=True
    )

    # 蒙地卡羅
    st.markdown("---")
    with st.expander("🔮 蒙地卡羅模擬（未來情境）", expanded=False):
        sim_years = years
        n_sim = 1000
        dt = 1/252
        days_sim = int(sim_years * 252)
        mu = avg_annual
        sigma = ann_vol

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1, (days_sim, n_sim))
        paths = initial_investment * np.exp(np.cumsum(drift + diffusion, axis=0))
        paths = np.vstack([np.full((1, n_sim), initial_investment), paths])

        future_dates = [datetime.today() + timedelta(days=i * 365/252) for i in range(days_sim + 1)]
        p05 = np.percentile(paths, 5, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p95 = np.percentile(paths, 95, axis=1)

        fig_mc = go.Figure()
        for i in range(min(50, n_sim)):
            fig_mc.add_trace(go.Scatter(x=future_dates, y=paths[:, i], mode='lines',
                line=dict(color='lightgrey', width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))
        fig_mc.add_trace(go.Scatter(x=future_dates, y=p05, name='悲觀 (5%)', line=dict(color='red', width=1)))
        fig_mc.add_trace(go.Scatter(x=future_dates, y=p50, name='中性 (50%)', line=dict(color='blue', width=2),
            fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
        fig_mc.add_trace(go.Scatter(x=future_dates, y=p95, name='樂觀 (95%)', line=dict(color='green', width=1),
            fill='tonexty', fillcolor='rgba(0,128,0,0.1)'))
        fig_mc.update_layout(title=f"未來 {sim_years} 年情境模擬", yaxis_title="資產價值 ($)", height=400)
        st.plotly_chart(fig_mc, use_container_width=True)

        cagr = lambda v: (v / initial_investment) ** (1/sim_years) - 1
        st.markdown(f"""
**{sim_years} 年後預測（{n_sim} 次模擬）：**
- 🟢 樂觀 (95%)：**${p95[-1]:,.0f}**（年化 {cagr(p95[-1]):.2%}）
- 🔵 中性 (50%)：**${p50[-1]:,.0f}**（年化 {cagr(p50[-1]):.2%}）
- 🔴 悲觀 (5%)：**${p05[-1]:,.0f}**（年化 {cagr(p05[-1]):.2%}）
        """)
