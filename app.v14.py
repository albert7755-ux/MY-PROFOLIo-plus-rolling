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
st.title('📈 智能投資組合優化器 (解鎖高報酬版)')
st.markdown("""
此工具會自動計算最佳權重，並根據實際數據長度回測真實報酬率、波動度與勝率。
""")

# --- 2. 參數設定 ---
st.sidebar.header('1. 標的選擇')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()

st.sidebar.header('2. 基準指數 (Benchmark)')
bench_input = st.sidebar.text_input(
    '基準代號與權重 (格式: 代號:%)', 
    'SPY:60 AGG:40', 
    help="用於比較的市場基準 (僅用於年度報酬比較與走勢圖)。"
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
    # 避免分母為 0
    if self_fund_ratio <= 0.01: self_fund_ratio = 0.01
    leverage = 1 / self_fund_ratio
    st.sidebar.info(f"槓桿倍數：**{leverage:.1f} 倍**")
else:
    loan_ratio = 0.0
    margin_rate = 0.0
    leverage = 1.0

# --- 優化目標 ---
st.sidebar.markdown("---")
st.sidebar.header("4. 優化目標 (Optimization)")
opt_method = st.sidebar.radio(
    "請選擇配置策略：",
    ("🛡️ 最小風險 (保守)", "🚀 最大夏普 (CP值高)", "🎯 鎖定目標報酬 (積極)")
)

target_return = 0.0
if opt_method == "🎯 鎖定目標報酬 (積極)":
    # ★ 修改處：上限從 30.0 改為 100.0，預設改為 15.0
    target_return = st.sidebar.slider("您想要的年化報酬率 (CAGR)", 1.0, 100.0, 15.0, 0.5) / 100
    st.sidebar.caption("系統將自動補償波動耗損，力求回測結果貼近此目標。")

# --- 3. 核心邏輯 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的。")
    else:
        with st.spinner('正在進行 AI 運算與多維度回測...'):
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
                
                if df_bench_raw.index.tz is not None:
                    df_bench_raw.index = df_bench_raw.index.tz_localize(None)

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
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets

                # 4. 共用函數
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

                def calculate_cagr(series):
                    days = (series.index[-1] - series.index[0]).days
                    actual_years = days / 365.25
                    if actual_years < 0.1: return 0 
                    total_ret = series.iloc[-1]
                    if total_ret <= 0: return -1
                    return (total_ret)**(1/actual_years) - 1

                def calculate_vol(series):
                    daily_ret = series.pct_change().dropna()
                    return daily_ret.std() * np.sqrt(252)

                # ==========================
                # B. 策略運算核心
                # ==========================
                optimal_weights = []
                strategy_name = ""
                strategy_color = ""

                if "最小風險" in opt_method:
                    strategy_name = "🛡️ 最小風險組合"
                    strategy_color = "green"
                    def min_variance(weights, cov_matrix):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    res = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                   method='SLSQP', bounds=bounds, constraints=constraints)
                    optimal_weights = res.x

                elif "最大夏普" in opt_method:
                    strategy_name = "🚀 最大夏普組合"
                    strategy_color = "red"
                    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
                        p_ret = np.sum(mean_returns * weights)
                        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return - (p_ret - rf) / p_vol
                    res = minimize(neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
                    optimal_weights = res.x

                elif "目標報酬" in opt_method:
                    strategy_name = f"🎯 目標報酬組合 ({target_return:.1%})"
                    strategy_color = "blue"
                    max_possible_ret = mean_returns.max()
                    
                    if target_return > max_possible_ret:
                        st.warning(f"⚠️ 提示：目標 ({target_return:.1%}) 超過歷史極限，改為 {max_possible_ret:.1%}。")
                        target_return = max_possible_ret - 0.001

                    def min_variance(weights, cov_matrix):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    # CAGR 近似公式校正
                    def target_constraint(weights):
                        p_ret = np.sum(mean_returns * weights) 
                        p_var = np.dot(weights.T, np.dot(cov_matrix, weights)) 
                        geo_ret_approx = p_ret - 0.5 * p_var
                        return geo_ret_approx - target_return

                    constraints.append({'type': 'eq', 'fun': target_constraint})
                    
                    res = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                   method='SLSQP', bounds=bounds, constraints=constraints)
                    
                    if not res.success:
                         # 解不出來時退回算術平均
                         constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return}]
                         res = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                        method='SLSQP', bounds=bounds, constraints=constraints)
                    
                    optimal_weights = res.x

                # 計算結果
                raw_port_val = (normalized_prices * optimal_weights).sum(axis=1) 
                margin_port_val = calculate_margin_equity(raw_port_val, leverage, loan_ratio, margin_rate) 
                margin_port_val.name = strategy_name

                st.success(f"運算完成！策略：{strategy_name}")

                # ==========================
                # C. 顯示區塊
                # ==========================
                
                # 1. 配置與走勢
                col_top1, col_top2 = st.columns([1, 2])
                with col_top1:
                    st.subheader("📊 建議配置權重")
                    clean_w = [round(w, 4) if w > 0.0001 else 0.0 for w in optimal_weights]
                    df_weights = pd.DataFrame({'標的': tickers, '配置': clean_w})
                    df_weights['顯示權重'] = df_weights['配置'].apply(lambda x: f"{x:.1%}")
                    df_weights = df_weights.sort_values('配置', ascending=False)
                    st.table(df_weights[['標的', '顯示權重']])
                    
                    fig_pie = px.pie(df_weights[df_weights['配置']>0], values='配置', names='標的', hole=0.4)
                    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_top2:
                    st.subheader("📈 資產成長回測")
                    fig = px.line(margin_port_val, title=f'{strategy_name} vs Benchmark')
                    fig.update_traces(line=dict(color=strategy_color, width=3))
                    if normalized_bench is not None:
                            aligned_bench = normalized_bench.reindex(margin_port_val.index).ffill()
                            if aligned_bench.iloc[0] > 0: aligned_bench = aligned_bench / aligned_bench.iloc[0]
                            fig.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench, mode='lines', name=f'基準 ({bench_input})', line=dict(color='gray', width=2, dash='dash')))
                    st.plotly_chart(fig, use_container_width=True)

                    # 寬敞排版
                    total_ret = margin_port_val.iloc[-1] - 1
                    real_cagr = calculate_cagr(margin_port_val)
                    real_vol = calculate_vol(margin_port_val)
                    mdd = calculate_mdd(margin_port_val)

                    r1c1, r1c2 = st.columns(2)
                    r1c1.metric("總報酬率", f"{total_ret:,.2%}")
                    r1c2.metric("年化報酬 (CAGR)", f"{real_cagr:.2%}")
                    
                    r2c1, r2c2 = st.columns(2)
                    r2c1.metric("年化波動", f"{real_vol:.2%}")
                    r2c2.metric("最大回撤 (MDD)", f"{mdd:.2%}", delta_color="inverse")

                # 2. 融資效益視覺化
                st.markdown("---")
                st.subheader("💰 融資效益視覺化 (以投資 100 元為例)")
                st.caption("直觀比較：融資前後的「本金放大效果」與「最終獲利金額」。")
                
                col_v1, col_v2 = st.columns(2)
                
                initial_own = 100
                total_pos_initial = initial_own * leverage 
                loan_amt = total_pos_initial - initial_own 
                
                end_val_no_margin = initial_own * raw_port_val.iloc[-1]
                end_val_margin = initial_own * margin_port_val.iloc[-1]

                with col_v1:
                    fig_cap = go.Figure()
                    fig_cap.add_trace(go.Bar(name='自有本金', x=['無融資'], y=[initial_own], text=[f"${initial_own}"], textposition='auto', marker_color='#2ca02c'))
                    fig_cap.add_trace(go.Bar(name='自有本金', x=['有融資'], y=[initial_own], text=[f"${initial_own}"], textposition='auto', marker_color='#2ca02c', showlegend=False))
                    fig_cap.add_trace(go.Bar(name='銀行借款', x=['有融資'], y=[loan_amt], text=[f"${loan_amt:.0f}"], textposition='auto', marker_color='#d62728'))
                    fig_cap.update_layout(barmode='stack', title=f'初始本金比較 (放大 {leverage:.1f} 倍)', height=350, yaxis_title="金額 ($)", showlegend=True)
                    st.plotly_chart(fig_cap, use_container_width=True)

                with col_v2:
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Bar(x=['無融資', '有融資'], y=[end_val_no_margin, end_val_margin], text=[f"${end_val_no_margin:,.0f}", f"${end_val_margin:,.0f}"], textposition='auto', marker_color=['#1f77b4', '#ff7f0e']))
                    profit_diff = end_val_margin - end_val_no_margin
                    title_text = f'期末淨值比較 (融資多賺 ${profit_diff:,.0f})' if profit_diff > 0 else f'期末淨值比較 (融資少賺 ${abs(profit_diff):,.0f})'
                    fig_res.update_layout(title=title_text, height=350, yaxis_title="期末價值 ($)")
                    st.plotly_chart(fig_res, use_container_width=True)

                # 3. 年度報酬表
                st.markdown("---")
                st.subheader(f"📅 年度報酬回測 ({strategy_name})")
                
                df_port_col = margin_port_val.to_frame(name=strategy_name)
                data_list = [df_close, df_port_col]
                if df_bench_combined is not None:
                    data_list.append(df_bench_combined)
                
                df_all = pd.concat(data_list, axis=1)
                if df_all.index.tz is not None: df_all.index = df_all.index.tz_localize(None)
                
                ann_prices = df_all.resample('Y').last()
                ann_ret = ann_prices.pct_change().dropna()
                
                avg_ret = ann_ret.mean()
                ann_ret.index = ann_ret.index.astype(str)
                
                df_avg = avg_ret.to_frame(name="🔥 平均報酬 (Avg)").T
                final_annual_df = pd.concat([df_avg, ann_ret.sort_index(ascending=False)])

                table_height = (len(final_annual_df) + 1) * 35 + 3

                st.dataframe(
                    final_annual_df.style.format("{:.2%}")
                    .background_gradient(cmap='RdYlGn', vmin=-0.3, vmax=0.3),
                    height=table_height,
                    use_container_width=True
                )
                st.caption("註：最上方列為歷年平均報酬率。")

                # 4. 滾動勝率
                st.markdown("---")
                st.subheader(f"📊 滾動持有勝率分析 ({strategy_name})")
                
                rolling_periods = {'3個月': 63, '6個月': 126, '1年': 252, '3年': 756, '5年': 1260, '10年': 2520}
                rolling_rows = []

                def get_rolling_stats(series, name):
                    row = {'標的': name}
                    for period_name, window in rolling_periods.items():
                        if len(series) > window:
                            roll_ret = series.pct_change(window).dropna()
                            win_rate = (roll_ret > 0).mean()
                            row[period_name] = win_rate
                        else:
                            row[period_name] = np.nan
                    time_to_100 = "> 10 年"
                    for y in range(1, 11):
                        window = y * 252
                        if len(series) > window:
                            min_ret = series.pct_change(window).min()
                            if min_ret > 0:
                                time_to_100 = f"{y} 年"
                                break
                    row['必勝持有期'] = time_to_100
                    return row

                rolling_rows.append(get_rolling_stats(margin_port_val, f"🏆 {strategy_name}"))
                for ticker in tickers:
                    rolling_rows.append(get_rolling_stats(df_close[ticker], ticker))

                df_roll = pd.DataFrame(rolling_rows)
                st.dataframe(
                    df_roll.style.format({
                        '3個月': '{:.0%}', '6個月': '{:.0%}', '1年': '{:.0%}', 
                        '3年': '{:.0%}', '5年': '{:.0%}', '10年': '{:.0%}'
                    })
                    .background_gradient(subset=list(rolling_periods.keys()), cmap='RdYlGn', vmin=0, vmax=1)
                )

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
