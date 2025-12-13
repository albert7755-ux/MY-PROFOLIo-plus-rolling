import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 設定網頁標題 ---
st.set_page_config(page_title="智能投資組合優化器", layout="wide")
st.title('📈 智能投資組合優化器 (年度報酬分析版)')
st.markdown("""
此工具提供華爾街等級的投資組合分析，包含 **風險控管**、**融資模擬**、**基準對照** 與 **年度績效回測**。
""")

# --- 2. 參數設定 ---
st.sidebar.header('參數設定')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()

# 基準指數設定
st.sidebar.markdown("---")
st.sidebar.header("⚖️ 基準指數設定 (Benchmark)")
bench_input = st.sidebar.text_input(
    '基準代號與權重 (格式: 代號:%)', 
    'SPY:60 AGG:40', 
    help="請用冒號指定權重，並用空白隔開。\n例如：\n1. 股債平衡: SPY:60 AGG:40\n2. 純美股: SPY"
)

years = st.sidebar.slider('回測年數', 1, 20, 10)
risk_free_rate = 0.02 

# --- 融資設定 ---
st.sidebar.markdown("---")
st.sidebar.header("💰 融資設定 (Margin)")
use_margin = st.sidebar.checkbox("開啟融資回測模式")

if use_margin:
    loan_ratio = st.sidebar.slider("融資成數 (銀行借款比例)", 0.0, 0.9, 0.6, 0.1)
    margin_rate = st.sidebar.number_input("融資年利率 (%)", 2.0, 15.0, 6.0, 0.1) / 100
    self_fund_ratio = 1 - loan_ratio
    leverage = 1 / self_fund_ratio if self_fund_ratio > 0 else 1
    st.sidebar.info(f"槓桿倍數：**{leverage:.1f} 倍**")
else:
    loan_ratio = 0.0
    margin_rate = 0.0
    leverage = 1.0

# --- 3. 核心邏輯 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的。")
    else:
        with st.spinner('正在進行年度績效結算...'):
            try:
                # ==========================
                # A. 數據準備
                # ==========================
                end_date = datetime.today()
                start_date = end_date - timedelta(days=365*years + 365) 
                
                # 1. 下載使用者投資組合
                data = yf.download(user_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                if 'Close' in data.columns:
                    df_close = data['Close']
                else:
                    df_close = data
                
                df_close.dropna(inplace=True)
                
                if df_close.empty:
                    st.error("無法抓取投資組合數據。")
                    st.stop()
                
                tickers = df_close.columns.tolist()

                # 2. 下載與合成 Benchmark
                bench_config = []
                try:
                    items = bench_input.strip().split()
                    for item in items:
                        if ':' in item:
                            parts = item.split(':')
                            ticker = parts[0].upper()
                            weight = float(parts[1])
                        else:
                            ticker = item.upper()
                            weight = 100.0 
                        bench_config.append({'ticker': ticker, 'weight': weight})
                    
                    total_bench_w = sum([x['weight'] for x in bench_config])
                    if total_bench_w == 0: total_bench_w = 1
                    for x in bench_config:
                        x['weight'] /= total_bench_w
                    
                    bench_tickers = [x['ticker'] for x in bench_config]
                    bench_weights = [x['weight'] for x in bench_config]

                except Exception as e:
                    st.error(f"基準指數格式錯誤: {e}")
                    st.stop()

                bench_data_raw = yf.download(bench_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                if 'Close' in bench_data_raw.columns:
                    df_bench_raw = bench_data_raw['Close']
                else:
                    df_bench_raw = bench_data_raw
                
                if isinstance(df_bench_raw, pd.Series):
                    df_bench_raw = df_bench_raw.to_frame(name=bench_tickers[0])

                # 日期對齊
                common_index = df_close.index.intersection(df_bench_raw.index)
                df_close = df_close.loc[common_index]
                df_bench_raw = df_bench_raw.loc[common_index]
                
                if df_bench_raw.empty:
                    normalized_bench = None
                    df_bench_combined = None
                else:
                    bench_daily_ret = df_bench_raw.pct_change().fillna(0)
                    try:
                        aligned_bench_ret = bench_daily_ret[bench_tickers]
                        composite_bench_ret = aligned_bench_ret.dot(bench_weights)
                    except:
                        composite_bench_ret = bench_daily_ret.mean(axis=1)

                    normalized_bench = (1 + composite_bench_ret).cumprod()
                    normalized_bench.name = "基準指數 (Benchmark)"
                    
                    # 建立 Benchmark 的股價 DataFrame (為了算年報酬)
                    df_bench_combined = pd.DataFrame(normalized_bench)
                    df_bench_combined.columns = [f"基準({bench_input})"]

                # 統計數據
                returns = df_close.pct_change().dropna()
                cov_matrix = returns.cov() * 252
                mean_returns = returns.mean() * 252
                corr_matrix = returns.corr()
                
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets
                
                normalized_prices = df_close / df_close.iloc[0]

                # 函數庫
                def calculate_mdd(series):
                    roll_max = series.cummax()
                    drawdown = (series - roll_max) / roll_max
                    return drawdown.min()

                def calculate_margin_equity(raw_portfolio_value, leverage, loan_ratio, annual_rate):
                    if leverage == 1:
                        return raw_portfolio_value
                    
                    debt = leverage - 1
                    daily_rate = annual_rate / 365 
                    position_value = raw_portfolio_value * leverage
                    interest_cost = pd.Series(np.arange(len(raw_portfolio_value)) * debt * daily_rate, index=raw_portfolio_value.index)
                    margin_equity = position_value - debt - interest_cost
                    return margin_equity

                st.success("運算完成！")

                # ==========================
                # B. 分頁顯示
                # ==========================
                tab1, tab2 = st.tabs(["🛡️ 最小風險組合", "🚀 最大夏普組合"])

                # --- 繪圖函數 ---
                def plot_performance(port_val, strategy_name, color):
                    bench_label = f"基準 ({bench_input})"
                    fig = px.line(port_val, title=f'資產成長回測')
                    fig.update_traces(line=dict(color=color, width=3), name=strategy_name)
                    
                    if normalized_bench is not None:
                        aligned_bench = normalized_bench.reindex(port_val.index).fillna(method='ffill')
