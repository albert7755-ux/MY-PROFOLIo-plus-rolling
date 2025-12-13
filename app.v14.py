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
st.title('📈 智能投資組合優化器 (年度回測分流版)')
st.markdown("""
此工具會自動計算最佳權重，並回測該權重在過去每一年的真實報酬率。
""")

# --- 2. 參數設定 ---
st.sidebar.header('1. 標的選擇')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()

st.sidebar.header('2. 基準指數 (Benchmark)')
bench_input = st.sidebar.text_input(
    '基準代號與權重 (格式: 代號:%)', 
    'SPY:60 AGG:40', 
    help="用於比較的市場基準。"
)

years = st.sidebar.slider('回測年數', 1, 20, 10)
risk_free_rate = 0.02 

# --- 融資設定 ---
st.sidebar.markdown("---")
st.sidebar.header("3. 融資設定 (Margin)")
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
        with st.spinner('正在進行 AI 運算與年度回測...'):
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
                
                # 強制移除時區
                if df_close.index.tz is not None:
                    df_close.index = df_close.index.tz_localize(None)

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
                
                # Benchmark 強制移除時區
                if df_bench_raw.index.tz is not None:
                    df_bench_raw.index = df_bench_raw.index.tz_localize(None)

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
                    df_bench_combined = pd.DataFrame(normalized_bench)
                    df_bench_combined.columns = [f"基準({bench_input})"]

                # 3. 計算統計數據
                returns = df_close.pct_change().dropna()
                cov_matrix = returns.cov() * 252
                mean_returns = returns.mean() * 252
                corr_matrix = returns.corr()
                normalized_prices = df_close / df_close.iloc[0]
                
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets

                # 4. 定義共用函數
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

                # ==========================
                # B. ★ 先進行所有策略計算
                # ==========================
                
                # --- 1. 最小風險 (Min Risk) ---
                def min_variance(weights, cov_matrix):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                res_min = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                   method='SLSQP', bounds=bounds, constraints=constraints)
                w_min = res_min.x
                
                raw_port_val_min = (normalized_prices * w_min).sum(axis=1)
                margin_port_val_min = calculate_margin_equity(raw_port_val_min, leverage, loan_ratio, margin_rate)
                margin_port_val_min.name = "🛡️ 最小風險組合"

                # --- 2. 最大夏普 (Max Sharpe) ---
                def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
                    p_ret = np.sum(mean_returns * weights)
                    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return - (p_ret - rf) / p_vol
                
                args = (mean_returns, cov_matrix, risk_free_rate)
                res_sharpe = minimize(neg_sharpe_ratio, init_guess, args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                w_sharpe = res_sharpe.x
                
                raw_port_val_sharpe = (normalized_prices * w_sharpe).sum(axis=1)
                margin_port_val_sharpe = calculate_margin_equity(raw_port_val_sharpe, leverage, loan_ratio, margin_rate)
                margin_port_val_sharpe.name = "🚀 最大夏普組合"

                st.success("AI 運算完成！")

                # ==========================
                # C. 定義年度報酬顯示函數 (分流顯示的核心)
                # ==========================
                def display_annual_returns(portfolio_series, portfolio_name):
                    st.markdown(f"#### 📅 {portfolio_name} - 年度報酬回測")
                    
                    # 1. 準備該策略的 DataFrame
                    df_port = portfolio_series.to_frame(name=portfolio_name)
                    
                    # 2. 合併數據：個股 + Benchmark + 該策略
                    data_list = [df_close, df_port]
                    if df_bench_combined is not None:
                        data_list.append(df_bench_combined)
                    
                    df_all = pd.concat(data_list, axis=1)
                    if df_all.index.tz is not None:
                         df_all.index = df_all.index.tz_localize(None)
                    
                    # 3. 計算年報酬
                    ann_prices = df_all.resample('Y').last()
                    ann_ret = ann_prices.pct_change().dropna()
                    
                    ann_ret.index = ann_ret.index.year
                    ann_ret = ann_ret.sort_index(ascending=False)
                    
                    # 4. 顯示表格 (只用 Heatmap，不加底色以免字看不到)
                    st.dataframe(
                        ann_ret.style.format("{:.2%}")
                        .background_gradient(cmap='RdYlGn', vmin=-0.3, vmax=0.3)
                    )
                    st.caption("註：深綠色代表大賺 (>30%)，深紅色代表大賠 (<-30%)。")

                # ==========================
                # D. 分頁顯示
                # ==========================
                tab1, tab2 = st.tabs(["🛡️ 最小風險組合 (保守)", "🚀 最大夏普組合 (積極)"])

                with tab1:
                    st.subheader("🛡️ 最小風險組合")
                    col1_1, col1_2 = st.columns([1, 2])
                    with col1_1:
                        clean_w = [round(w, 4) if w > 0.0001 else 0.0 for w in w_min]
                        df_min = pd.DataFrame({'標的': tickers, '配置': clean_w})
                        df_min['顯示權重'] = df_min['配置'].apply(lambda x: f"{x:.1%}")
                        df_min = df_min.sort_values('配置', ascending=False)
                        st.table(df_min[['標的', '顯示權重']])
                        
                        fig_pie = px.pie(df_min[df_min['配置']>0], values='配置', names='標的', hole=0.4)
                        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col1_2:
                        fig = px.line(margin_port_val_min, title='資產成長回測')
                        fig.update_traces(line=dict(color='green', width=3))
                        if normalized_bench is not None:
                             aligned_bench = normalized_bench.reindex(margin_port_val_min.index).ffill()
                             if aligned_bench.iloc[0] > 0: aligned_bench = aligned_bench / aligned_bench.iloc[0]
                             fig.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench, mode='lines', name=f'基準 ({bench_input})', line=dict(color='gray', width=2, dash='dash')))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        total_ret = margin_port_val_min.iloc[-1] - 1
                        cagr = (margin_port_val_min.iloc[-1])**(1/years) - 1 if margin_port_val_min.iloc[-1] > 0 else -1
                        mdd = calculate_mdd(margin_port_val_min)
                        st.markdown("### 💰 回測結果")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("總報酬率", f"{total_ret:.2%}")
                        c2.metric("年化報酬", f"{cagr:.2%}")
                        c3.metric("最大回撤", f"{mdd:.2%}", delta_color="inverse")
                    
                    st.divider()
                    # ★ 在此分頁只顯示「最小風險」的年度表
                    display_annual_returns(margin_port_val_min, "🛡️ 最小風險組合")

                with tab2:
                    st.subheader("🚀 最大夏普組合")
                    col2_1, col2_2 = st.columns([1, 2])
                    with col2_1:
                        clean_w_s = [round(w, 4) if w > 0.0001 else 0.0 for w in w_sharpe]
                        df_sharpe = pd.DataFrame({'標的': tickers, '配置': clean_w_s})
                        df_sharpe['顯示權重'] = df_sharpe['配置'].apply(lambda x: f"{x:.1%}")
                        df_sharpe = df_sharpe.sort_values('配置', ascending=False)
                        st.table(df_sharpe[['標的', '顯示權重']])
                        
                        fig_pie_s = px.pie(df_sharpe[df_sharpe['配置']>0], values='配置', names='標的', hole=0.4)
                        fig_pie_s.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie_s, use_container_width=True)

                    with col2_2:
                        fig_s = px.line(margin_port_val_sharpe, title='資產成長回測')
                        fig_s.update_traces(line=dict(color='red', width=3))
                        if normalized_bench is not None:
                             aligned_bench = normalized_bench.reindex(margin_port_val_sharpe.index).ffill()
                             if aligned_bench.iloc[0] > 0: aligned_bench = aligned_bench / aligned_bench.iloc[0]
                             fig_s.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench, mode='lines', name=f'基準 ({bench_input})', line=dict(color='gray', width=2, dash='dash')))
                        st.plotly_chart(fig_s, use_container_width=True)
                        
                        total_ret_s = margin_port_val_sharpe.iloc[-1] - 1
                        cagr_s = (margin_port_val_sharpe.iloc[-1])**(1/years) - 1 if margin_port_val_sharpe.iloc[-1] > 0 else -1
                        mdd_s = calculate_mdd(margin_port_val_sharpe)
                        st.markdown("### 💰 回測結果")
                        cs1, cs2, cs3 = st.columns(3)
                        cs1.metric("總報酬率", f"{total_ret_s:.2%}")
                        cs2.metric("年化報酬", f"{cagr_s:.2%}")
                        cs3.metric("最大回撤", f"{mdd_s:.2%}", delta_color="inverse")
                    
                    st.divider()
                    # ★ 在此分頁只顯示「最大夏普」的年度表
                    display_annual_returns(margin_port_val_sharpe, "🚀 最大夏普組合")

                # ==========================
                # E. 滾動報酬 (這個放最下面大家都看得到)
                # ==========================
                st.markdown("---")
                with st.expander("📊 個股滾動報酬與勝率分析", expanded=False):
                    rolling_periods = {'3個月': 63, '6個月': 126, '1年': 252, '3年': 756, '5年': 1260, '10年': 2520}
                    rolling_data = []
                    for ticker in tickers:
                        row = {'標的': ticker}
                        for name, window in rolling_periods.items():
                            if len(df_close) > window:
                                roll_ret = df_close[ticker].pct_change(window).dropna()
                                win_rate = (roll_ret > 0).mean()
                                row[name] = win_rate
                            else: row[name] = np.nan 
                        time_to_100 = "> 10 年"
                        for y in range(1, 11):
                            window = y * 252
                            if len(df_close) > window:
                                min_ret = df_close[ticker].pct_change(window).min()
                                if min_ret > 0:
                                    time_to_100 = f"{y} 年"
                                    break
                        row['必勝持有期'] = time_to_100
                        rolling_data.append(row)
                    df_roll = pd.DataFrame(rolling_data)
                    st.dataframe(df_roll.style.format({
                        '3個月': '{:.0%}', '6個月': '{:.0%}', '1年': '{:.0%}', 
                        '3年': '{:.0%}', '5年': '{:.0%}', '10年': '{:.0%}'
                    }).background_gradient(subset=list(rolling_periods.keys()), cmap='RdYlGn', vmin=0, vmax=1))

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
else:
    st.info("請在左側輸入股票代號並按下「開始計算」")

# --- 免責聲明 ---
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **免責聲明**")
st.sidebar.caption("""
本工具僅供市場分析與模擬參考，不構成任何投資建議或邀約。
""")
