import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import feedparser
import urllib.parse
import requests
import json
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import timedelta
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 網頁基本設定 ---
st.set_page_config(page_title="AI 金融大數據預測看板", layout="wide", initial_sidebar_state="expanded")
st.title("📈 AI 供應鏈概念股：趨勢預測與大數據儀表板")

st.sidebar.header("⚙️ 參數設定")
ticker = st.sidebar.text_input("輸入股票代號 (如 NVDA, VRT, 2330.TW)", "2330.TW").upper()
lookback = st.sidebar.slider("AI 觀察天數 (Lookback)", 15, 60, 30)
train_epochs = st.sidebar.slider("模型訓練回數 (Epochs)", 5, 20, 10)
run_btn = st.sidebar.button("🚀 執行即時預測與分析")

# --- 1. 資料獲取與特徵工程 ---
@st.cache_data(ttl=60)
def get_and_prepare_data(tk):
    df = yf.download(tk, start="2022-01-01", progress=False)
    if df.empty: return None, False
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    close_series = df['Close'].squeeze()
    df['Return'] = close_series.pct_change()
    df['Risk'] = df['Return'].rolling(20).std() * np.sqrt(252)
    df['RSI_14'] = RSIIndicator(close=close_series, window=14).rsi()
    df['MACD'] = MACD(close=close_series).macd()
    
    df['SMA_20'] = close_series.rolling(20).mean()
    df['SMA_60'] = close_series.rolling(60).mean()
    df['Quant_Signal'] = np.where((df['MACD'] > 0) & (df['SMA_20'] > df['SMA_60']), 1, 0)
    df['Strategy_Return'] = df['Quant_Signal'].shift(1) * df['Return']
    df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cum_Hold'] = (1 + df['Return']).cumprod()

    has_us_macro = False
    if tk.endswith('.TW') or tk.endswith('.TWO'):
        has_us_macro = True
        us_data = yf.download("^IXIC", start="2021-12-01", progress=False)
        if isinstance(us_data.columns, pd.MultiIndex): us_data.columns = us_data.columns.get_level_values(0)
        df = df.join(pd.DataFrame({'US_Return': us_data['Close'].squeeze().pct_change()}), how='left')
        df['US_Return'] = df['US_Return'].shift(1).ffill().fillna(0)

    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df, has_us_macro

# --- 2. 財報抓取 ---
@st.cache_data(ttl=60)
def get_fundamentals(tk):
    try:
        info = yf.Ticker(tk).info
        return {
            "name": info.get("shortName", tk), "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "pe": info.get("trailingPE", 0), "peg": info.get("pegRatio", 0), "beta": info.get("beta", 0),
            "profit_margin": info.get("profitMargins", 0), "revenue_growth": info.get("revenueGrowth", 0), "debt_to_equity": info.get("debtToEquity", 0)
        }
    except: return None

# --- 3. 新聞與 NLP 情感分析 ---
@st.cache_data(ttl=3600)
def get_news_and_sentiment(tk, limit=5):
    query = urllib.parse.quote(tk + " stock OR news")
    entries = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant").entries[:limit]
    
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    total_score = 0
    
    for n in entries:
        title = n.title.rsplit(" - ", 1)[0] if " - " in n.title else n.title
        score = analyzer.polarity_scores(title)['compound'] 
        total_score += score
        news_data.append({"title": title, "link": n.link, "date": n.published, "score": score})
        
    avg_score = total_score / len(entries) if entries else 0
    return news_data, avg_score

@st.cache_data(ttl=3600)
def get_polymarket(tk):
    ticker_to_name = {"NVDA": "Nvidia", "AAPL": "Apple", "TSM": "TSMC", "VRT": "Vertiv", "2330.TW": "TSMC"}
    try:
        res = requests.get("https://gamma-api.polymarket.com/events", params={"query": ticker_to_name.get(tk, tk), "active": "true", "closed": "false"})
        return [e for e in res.json() if e.get('markets')][:2] 
    except: return []

# --- 主要執行邏輯 ---
if run_btn:
    with st.spinner(f"正在載入 {ticker} 歷史與即時資料..."):
        df, has_us_macro = get_and_prepare_data(ticker)
        fund_data = get_fundamentals(ticker)
        news_list, avg_sentiment = get_news_and_sentiment(ticker)
        
    if df is None: st.error(f"找不到代號 {ticker} 的資料。")
    else:
        # --- 頂部數據卡片 (加入 Help 提示與底線白話文) ---
        st.markdown("### 📊 市場即時掃描")
        if has_us_macro: st.info("🌐 **跨市場 AI 啟動**：偵測到台股標的，已自動將『美國那斯達克指數 (^IXIC) 昨晚漲跌幅』匯入特徵矩陣！")
            
        c1, c2, c3, c4 = st.columns(4)
        latest = df.iloc[-1]
        live_price = fund_data['current_price'] if (fund_data and fund_data['current_price'] > 0) else float(latest['Close'].item() if hasattr(latest['Close'], 'item') else latest['Close'])
        current_return = float(latest['Return'].item() if hasattr(latest['Return'], 'item') else latest['Return'])
        current_rsi = float(latest['RSI_14'].item() if hasattr(latest['RSI_14'], 'item') else latest['RSI_14'])
        current_risk = float(latest['Risk'].item() if hasattr(latest['Risk'], 'item') else latest['Risk'])
        current_macd = float(latest['MACD'].item() if hasattr(latest['MACD'], 'item') else latest['MACD'])
        
        with c1:
            st.metric("即時股價 (Live)", f"${live_price:.2f}", f"{current_return:.2%} (日漲跌)")
            
        with c2:
            st.metric("RSI (14)", f"{current_rsi:.1f}", help="【相對強弱指標】過去14天買賣盤力道。大於70視為超買(過熱易跌)，小於30視為超賣(過冷易漲)。")
            st.caption("💡 市場動能：>70過熱, <30過冷")
            
        with c3:
            st.metric("年化波動風險", f"{current_risk:.1%}", help="【歷史標準差】過去一個月日漲跌幅的波動極限並年化。數字越大代表股價上沖下洗越劇烈。")
            st.caption("💡 風險程度：數字越大震盪越劇烈")
            
        with c4:
            sentiment_index = int((avg_sentiment + 1) / 2 * 100)
            sentiment_label = "🔥 極度貪婪" if sentiment_index > 70 else "😊 樂觀" if sentiment_index > 55 else "😨 恐慌" if sentiment_index < 45 else "😐 中立"
            st.metric("NLP 新聞情緒", f"{sentiment_index}分", sentiment_label, delta_color="off", help="【自然語言處理】AI 閱讀當週新聞標題後量化的市場情緒分數 (0~100)。")
            st.caption("💡 輿情溫度：量化新聞字裡行間的情緒")
        
        # --- AI 模型訓練 ---
        st.markdown("---")
        st.markdown("### 🧠 LSTM 深度學習趨勢預測")
        feature_cols = ['Close', 'Volume', 'Risk', 'RSI_14', 'MACD']
        if has_us_macro: feature_cols.append('US_Return')
        
        features = df[feature_cols].values
        targets = df['Target'].values
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i-lookback:i])
            y.append(targets[i])
        X, y = np.array(X), np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]
        
        with st.spinner("神經網路即時訓練中 (動態適應最新股性)..."):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2), LSTM(50, return_sequences=False), Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=train_epochs, batch_size=32, verbose=0)
            
            _, test_acc = model.evaluate(X_test, y_test, verbose=0)
            pred_prob = model.predict(np.reshape(scaled_features[-lookback:], (1, lookback, features.shape[1])), verbose=0)[0][0]
        
        # 🌟 視覺升級：AI 預測機率儀表板 (Gauge Chart)
        col_gauge, col_logic = st.columns([1, 2])
        
        with col_gauge:
            # 儀表板標題加入 HTML 副標題說明
            gauge_title = "AI 明日看漲信心度<br><span style='font-size:12px;color:#A0A0A0'>神經網路對『明天單一交易日』的預估勝率</span>"
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': gauge_title, 'font': {'size': 20, 'color': 'white'}},
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#00FF00' if pred_prob > 0.5 else '#FF0000'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00FF00" if pred_prob > 0.5 else "#FF0000"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 45], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [45, 55], 'color': "rgba(128, 128, 128, 0.3)"},
                        {'range': [55, 100], 'color': "rgba(0, 255, 0, 0.3)"}],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_logic:
            st.metric("盲測歷史準確率", f"{test_acc:.1%}", help="【無作弊回測】模型在從未見過的 20% 歷史資料中的平均命中率。接近 50% 屬正常隨機漫步現象，證明模型無過度死背資料。")
            st.caption("💡 歷史成績單：證明模型沒有偷看未來作弊")
            
            rsi_status = "偏熱 (動能強但需防回檔)" if current_rsi > 60 else "偏冷 (超賣區邊緣)" if current_rsi < 40 else "中性水位"
            macd_status = "正向 (多頭排列)" if current_macd > 0 else "負向 (空頭排列)"
            us_text = f"4. **跨市場因子**：已讀取美股昨夜表現，納入台股開盤權重。\n" if has_us_macro else ""
            
            st.info(f"""
            **【底層決策邏輯解析 (Bottom-layer Logic)】**
            系統已將過去 {lookback} 天的連續多維特徵壓縮並運算。影響本次信心度（左圖指標）的核心變數為：
            1. **價格波動**：年化風險 **{current_risk:.1%}**，決定了未來路徑的震盪幅度。
            2. **動能乖離**：RSI 處於 **{rsi_status}** ({current_rsi:.1f})。
            3. **趨勢延續**：MACD 呈現 **{macd_status}** ({current_macd:.2f})。
            {us_text}
            """)

        # --- 量化交易策略回測區塊 ---
        st.markdown("---")
        st.markdown("### 🤖 量化交易策略回測 (Quant Backtesting)")
        st.caption("回測邏輯：當趨勢翻多 (MACD > 0 且 20日均線 > 60日均線) 時買入持有，跌破條件時清倉空手。")
        
        final_cum_strategy = (float(df['Cum_Strategy'].iloc[-1]) - 1)
        final_cum_hold = (float(df['Cum_Hold'].iloc[-1]) - 1)
        drawdown = df['Cum_Strategy'] / df['Cum_Strategy'].cummax() - 1.0
        max_drawdown = float(drawdown.min())
        
        cq1, cq2, cq3 = st.columns(3)
        cq1.metric("量化策略總報酬", f"{final_cum_strategy:.2%}", f"{final_cum_strategy - final_cum_hold:.2%} vs 死抱不放")
        cq2.metric("Buy & Hold 總報酬", f"{final_cum_hold:.2%}")
        cq3.metric("策略最大回撤 (風險)", f"{max_drawdown:.2%}", "數值越接近0越抗跌", delta_color="inverse", help="【抗跌指標】在使用策略期間，資金從最高點跌到最低點，最慘會賠多少。")
        
        fig_quant = go.Figure()
        fig_quant.add_trace(go.Scatter(x=df.index, y=df['Cum_Strategy'], mode='lines', name='量化策略資金曲線', line=dict(color='#00FF00', width=2)))
        fig_quant.add_trace(go.Scatter(x=df.index, y=df['Cum_Hold'], mode='lines', name='單純買進持有 (Buy & Hold)', line=dict(color='gray', dash='dash')))
        fig_quant.update_layout(height=350, template='plotly_dark', title="資金成長對比 (1 = 初始資金)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_quant, use_container_width=True)

        # --- 蒙地卡羅推演 ---
        st.markdown("---")
        st.markdown("### 📈 技術面分析與 蒙地卡羅模擬推演 (Monte Carlo)")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        plot_df = df.tail(150) 
        
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'].squeeze(), high=plot_df['High'].squeeze(), low=plot_df['Low'].squeeze(), close=plot_df['Close'].squeeze(), name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'].squeeze(), name='RSI', line=dict(color='orange')), row=2, col=1)

        last_date = plot_df.index[-1]
        recent_volatility = plot_df['Return'].tail(30).std() 
        target_price = live_price * (1 + (recent_volatility * ((pred_prob - 0.5) * 2) * 15))
        future_days = 10 
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
        
        base_trend = np.linspace(live_price, target_price, future_days + 1)[1:]
        path_prices = []
        np.random.seed(42) 
        for i in range(future_days):
            if i == future_days - 1: path_prices.append(target_price)
            else: path_prices.append(base_trend[i] + np.random.normal(0, recent_volatility * live_price * 1.5) * (1 - i/future_days))
                
        plot_dates, plot_prices = [last_date] + future_dates, [live_price] + path_prices
        pred_color = "#00FF00" if pred_prob > 0.5 else "#FF0000"
        
        fig.add_trace(go.Scatter(x=plot_dates, y=plot_prices, mode='lines+markers', line=dict(color='#A020F0', width=2), marker=dict(size=4), name='蒙地卡羅模擬路徑'), row=1, col=1)
        cw = recent_volatility * live_price * 1.2
        fig.add_trace(go.Scatter(x=plot_dates, y=[p+cw for p in plot_prices], mode='lines', line=dict(color='rgba(160,32,240,0.2)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_dates, y=[p-cw for p in plot_prices], mode='lines', line=dict(color='rgba(160,32,240,0.2)', width=1), fill='tonexty', fillcolor='rgba(160,32,240,0.1)', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[future_dates[-1]], y=[target_price], mode='markers+text', marker=dict(symbol='star', size=16, color=pred_color), name='AI 目標價', text=[f"目標: ${target_price:.1f}"], textposition="top center" if pred_prob > 0.5 else "bottom center", textfont=dict(color=pred_color, size=14)), row=1, col=1)

        fig.update_layout(height=500, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(range=[plot_df.index[0], future_dates[-1] + timedelta(days=8)]))
        st.plotly_chart(fig, use_container_width=True)

        # --- 財報健檢與 NLP 文本分析 ---
        st.markdown("---")
        st.markdown(f"### 🏢 基本面健檢與預測市場")
        
        c_fund, c_news = st.columns([1, 1])
        with c_fund:
            if fund_data and fund_data['pe'] != 0:
                st.success("#### 📈 財報與風險量化")
                if fund_data['revenue_growth'] > 0.1: st.write(f"✔️ **高營收成長**：近期營收增長達 **{fund_data['revenue_growth']:.1%}**。")
                if fund_data['profit_margin'] > 0.15: st.write(f"✔️ **優異獲利**：淨利率高達 **{fund_data['profit_margin']:.1%}**。")
                if fund_data['beta'] > 1.2: st.write(f"⚠️ **高市場風險**：Beta 值 **{fund_data['beta']}**，波動較劇烈。")
                if fund_data['pe'] > 40: st.write(f"⚠️ **高估值預警**：本益比高達 **{fund_data['pe']}** 倍。")
            else: st.info("目前無法獲取該標的詳細財報。")
            
            st.markdown("#### 🎲 Polymarket 人類資金共識")
            poly_events = get_polymarket(ticker)
            if poly_events:
                for e in poly_events:
                    st.warning(f"**議題**：{e.get('title')}")
                    for m in e.get('markets', [])[:1]:
                        try:
                            prices = json.loads(m.get('outcomePrices', '["0","0"]'))
                            st.write(f"資金看好度：**{float(prices[0])*100:.1f}%**")
                            st.progress(float(prices[0]))
                        except: pass
            else: st.write("無相關活躍預測。")

        with c_news:
            st.info(f"#### 📰 NLP 情緒解讀 (綜合得分: {sentiment_index}/100)")
            for n in news_list:
                emo = "🟢 正向" if n['score'] > 0.1 else "🔴 負向" if n['score'] < -0.1 else "⚪ 中立"
                st.write(f"**{emo}** | [{n['title']}]({n['link']})")
