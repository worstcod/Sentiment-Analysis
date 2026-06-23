from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import feedparser
import pandas as pd
import datetime
import time
import urllib.parse
from functools import lru_cache
import os
import re

# ---------------------------------------------------------
# Global AI Sentiment Model Initialization (RAM Optimized)
# ---------------------------------------------------------
# Automatically disable heavy AI model if running on Render free tier (512MB RAM limit)
# to prevent kernel OOM (Out Of Memory) SIGKILLs, unless FORCE_AI_MODE is set.
disable_ai = os.environ.get("DISABLE_AI_MODE", "false").lower() == "true"
on_render = os.environ.get("RENDER") == "true"
force_ai = os.environ.get("FORCE_AI_MODE", "false").lower() == "true"

if disable_ai or (on_render and not force_ai):
    sentiment_analyzer = None
    nlp_engine_name = "Lexicon Fallback Mode (Low Memory Mode)"
    print("Low Memory Mode Active: PyTorch and FinBERT imports bypassed to stay under 512MB RAM.")
else:
    try:
        # Set PyTorch thread count to 1 to minimize memory footprint on low-resource hosting
        import torch
        torch.set_num_threads(1)
        
        from transformers import pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        nlp_engine_name = "FinBERT (AI Mode)"
        print("Success: FinBERT pipeline loaded globally.")
    except Exception as e:
        sentiment_analyzer = None
        nlp_engine_name = f"Lexicon Fallback Mode (Model Load Error: {str(e)[:100]})"
        print(f"Warning: Failed to load FinBERT pipeline ({e}). Reverting to Lexicon Fallback.")

# Simple, high-performance financial lexicon-based sentiment analyzer
def analyze_lexicon_sentiment(text):
    """
    Counts positive and negative financial terms.
    Returns compatible dict: {'label': 'positive/negative/neutral'}
    """
    pos_words = {
        'bullish', 'up', 'growth', 'rise', 'rising', 'gain', 'gains', 'profit', 'profits',
        'benefit', 'benefits', 'positive', 'surge', 'surging', 'soar', 'soaring', 'jump', 'jumped',
        'climb', 'climbed', 'rebound', 'rebounding', 'recovery', 'recovering', 'higher', 'increase',
        'increased', 'advances', 'advance', 'beat', 'beats', 'beating', 'strong', 'stronger',
        'optimistic', 'optimism', 'expand', 'expansion', 'expanding', 'deal', 'agreement', 'reopen',
        'reopens', 'reopening', 'boost', 'boosted', 'booming', 'rally', 'rallied'
    }
    neg_words = {
        'bearish', 'down', 'fall', 'falling', 'fell', 'drop', 'dropping', 'dropped', 'loss', 'losses',
        'lose', 'losing', 'warn', 'warns', 'warning', 'negative', 'decline', 'declining', 'declined',
        'slump', 'slumping', 'slumped', 'plunge', 'plunging', 'plunged', 'crash', 'crashing', 'crashed',
        'lower', 'decrease', 'decreased', 'slip', 'slipping', 'slipped', 'shrink', 'shrinking',
        'shrank', 'deficit', 'debt', 'inflation', 'cut', 'cuts', 'cutting', 'blockade', 'blockades',
        'blockading', 'halt', 'halts', 'halting', 'risk', 'risks', 'risky', 'weak', 'weaker',
        'pessimistic', 'pessimism', 'recession', 'recessionary', 'crisis', 'hit', 'hits', 'hitting',
        'conflict', 'escalation', 'disruption'
    }
    
    words = text.lower().split()
    clean_words = [w.strip(".,!?;:()\"'") for w in words]
    
    pos_count = sum(1 for w in clean_words if w in pos_words)
    neg_count = sum(1 for w in clean_words if w in neg_words)
    
    if pos_count > neg_count:
        return {'label': 'positive'}
    elif neg_count > pos_count:
        return {'label': 'negative'}
    else:
        return {'label': 'neutral'}

# ---------------------------------------------------------
# System Design: Rate-Limit Caching (lru_cache)
# ---------------------------------------------------------
# Caching yfinance downloads and feedparser fetches dynamically using string inputs.
# This prevents IP blocks/throttling when users toggle parameters repeatedly.

def get_annualization_factor(ticker):
    ticker = ticker.upper()
    if '-USD' in ticker:
        return 365.0
    elif '=X' in ticker:
        return 260.0
    else:
        return 252.0

def calculate_p_value(car_values):
    n = len(car_values)
    if n < 2:
        return 1.0
    mean_val = sum(car_values) / n
    var_val = sum((x - mean_val) ** 2 for x in car_values) / (n - 1)
    std_val = var_val ** 0.5
    if std_val == 0:
        return 0.0 if mean_val != 0 else 1.0
    se = std_val / (n ** 0.5)
    t_stat = mean_val / se
    
    # Normal approximation for two-tailed p-value
    import math
    abs_t = abs(t_stat)
    y = abs_t * (1.5976 + 0.07056 * abs_t * abs_t)
    try:
        phi = 1.0 / (1.0 + math.exp(-y))
    except OverflowError:
        phi = 1.0
    p_val = 2.0 * (1.0 - phi)
    return p_val

def compute_pro_metrics(price_series_full, news_payload, ticker, start_date_str, end_date_str):
    """
    Computes Sharpe Ratio, Market Beta via full buffered regression, Cumulative Abnormal
    Returns (CAR) Event Study paths with statistical significance tests, and backtests
    the Sentiment Strategy with transaction commissions and slippage frictions.
    """
    # Slice inside to get filtered window for returns
    price_series_filtered = price_series_full[(price_series_full.index >= pd.to_datetime(start_date_str)) & (price_series_full.index <= pd.to_datetime(end_date_str))]
    
    # 1. Full-history Returns
    returns_full = price_series_full.pct_change().dropna()
    daily_returns_filtered = price_series_filtered.pct_change().dropna()
    
    # 2. Sharpe Ratio (assuming risk free rate Rf = 4.0% per annum)
    ann_factor = get_annualization_factor(ticker)
    if not daily_returns_filtered.empty:
        mean_ret = daily_returns_filtered.mean()
        std_ret = daily_returns_filtered.std()
        if std_ret > 0:
            sharpe = float((mean_ret - (0.04 / ann_factor)) / std_ret * (ann_factor ** 0.5))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
        
    # 3. Market Beta (regressed over the full buffered window)
    if '-USD' in ticker.upper():
        bench_symbol = 'BTC-USD'
    else:
        bench_symbol = '^GSPC'
        
    beta = 1.0
    alpha = 0.0
    bench_series_full = None
    
    if ticker.upper() != bench_symbol:
        bench_res = fetch_market_data_cached(bench_symbol, start_date_str, end_date_str)
        if bench_res:
            bench_raw = bench_res['raw_prices']
            bench_indices = [pd.to_datetime(t, unit='s') for t, _ in bench_raw]
            bench_prices = [p for _, p in bench_raw]
            bench_series_full = pd.Series(bench_prices, index=bench_indices)
            bench_returns_full = bench_series_full.pct_change().dropna()
            
            # Align full buffered returns
            combined_full = pd.concat([returns_full, bench_returns_full], axis=1).dropna()
            if len(combined_full) > 5:
                cov = combined_full.iloc[:, 0].cov(combined_full.iloc[:, 1])
                var = combined_full.iloc[:, 1].var()
                if var > 0:
                    beta = float(cov / var)
                    alpha = float(combined_full.iloc[:, 0].mean() - beta * combined_full.iloc[:, 1].mean())
                    
    # 4. Calculate daily Abnormal Returns (AR) over the filtered window
    aligned_df = pd.DataFrame(index=price_series_filtered.index)
    aligned_df['returns'] = price_series_filtered.pct_change().fillna(0.0)
    
    if bench_series_full is not None:
        bench_series_filtered = bench_series_full[(bench_series_full.index >= pd.to_datetime(start_date_str)) & (bench_series_full.index <= pd.to_datetime(end_date_str))]
        bench_returns_filtered = bench_series_filtered.pct_change().fillna(0.0)
        aligned_df['bench_returns'] = aligned_df.index.map(bench_returns_filtered).fillna(0.0)
    else:
        aligned_df['bench_returns'] = 0.0
        
    aligned_df['abnormal_returns'] = aligned_df['returns'] - (alpha + beta * aligned_df['bench_returns'])

    # 5. Lead-Lag Cross-Correlation
    lead_lag_corrs = []
    sent_data = {}
    for item in news_payload:
        d = pd.to_datetime(item['MappedDate']).date()
        score = item['Sentiment Score']
        if d not in sent_data:
            sent_data[d] = []
        sent_data[d].append(score)
        
    daily_sent = {d: sum(scores)/len(scores) for d, scores in sent_data.items()}
    daily_sent_series = pd.Series(daily_sent)
    
    # Use returns_filtered for Lead-Lag
    if not daily_returns_filtered.empty:
        ret_series = daily_returns_filtered.copy()
        ret_series.index = ret_series.index.date
        df = pd.DataFrame({'returns': ret_series})
        df['sentiment'] = daily_sent_series
        df['sentiment'] = df['sentiment'].fillna(0.0)
        
        for k in [-3, -2, -1, 0, 1, 2, 3]:
            shifted = df['sentiment'].shift(k)
            corr_val = df['returns'].corr(shifted)
            lead_lag_corrs.append({
                'lag': k,
                'correlation': float(corr_val) if not pd.isna(corr_val) else 0.0
            })
    else:
        for k in [-3, -2, -1, 0, 1, 2, 3]:
            lead_lag_corrs.append({'lag': k, 'correlation': 0.0})
            
    # 6. Event Study (Cumulative Abnormal Return paths rebased to 0% at t=0)
    event_paths = {'Bullish': [], 'Bearish': []}
    for item in news_payload:
        label = item['Sentiment Label']
        if label not in ['Bullish', 'Bearish']:
            continue
        mapped_date = pd.to_datetime(item['MappedDate'])
        if mapped_date in aligned_df.index:
            event_idx = aligned_df.index.get_loc(mapped_date)
            if isinstance(event_idx, slice):
                event_idx = event_idx.start
            elif not isinstance(event_idx, int):
                try:
                    event_idx = int(event_idx)
                except Exception:
                    try:
                        event_idx = [i for i, val in enumerate(event_idx) if val][0]
                    except Exception:
                        continue
                        
            # Accumulate Abnormal Returns to calculate CAR relative to day 0
            path = {}
            for offset in range(-10, 11):
                if offset == 0:
                    path[offset] = 0.0
                elif offset > 0:
                    target_idx = event_idx + offset
                    if target_idx < len(aligned_df):
                        path[offset] = float(aligned_df['abnormal_returns'].iloc[event_idx + 1 : target_idx + 1].sum() * 100)
                    else:
                        path[offset] = None
                else: # offset < 0
                    target_idx = event_idx + offset
                    if target_idx >= 0:
                        path[offset] = float(-aligned_df['abnormal_returns'].iloc[target_idx : event_idx].sum() * 100)
                    else:
                        path[offset] = None
            event_paths[label].append(path)
            
    avg_bullish_path = []
    avg_bearish_path = []
    for offset in range(-10, 11):
        bull_vals = [p[offset] for p in event_paths['Bullish'] if p[offset] is not None]
        avg_bull = sum(bull_vals)/len(bull_vals) if bull_vals else 0.0
        avg_bullish_path.append({'offset': offset, 'return': avg_bull})
        
        bear_vals = [p[offset] for p in event_paths['Bearish'] if p[offset] is not None]
        avg_bear = sum(bear_vals)/len(bear_vals) if bear_vals else 0.0
        avg_bearish_path.append({'offset': offset, 'return': avg_bear})
        
    # Calculate Statistical Significance (p-values at t = +5 days)
    bull_car_5 = [p[5] for p in event_paths['Bullish'] if p[5] is not None]
    bear_car_5 = [p[5] for p in event_paths['Bearish'] if p[5] is not None]
    
    p_val_bull = calculate_p_value(bull_car_5)
    p_val_bear = calculate_p_value(bear_car_5)
        
    # 7. Backtester Simulation (with transaction costs and slippage frictions)
    strategy_df = pd.DataFrame(index=price_series_filtered.index)
    strategy_df['returns'] = price_series_filtered.pct_change().fillna(0.0)
    
    sentiment_map = daily_sent_series.copy()
    sentiment_map.index = pd.to_datetime(sentiment_map.index)
    strategy_df['sentiment'] = strategy_df.index.map(sentiment_map).fillna(0.0)
    
    signals = []
    current_sig = 0
    days_since_signal = 0
    for idx, row in strategy_df.iterrows():
        sent = row['sentiment']
        if sent > 0.15:
            current_sig = 1
            days_since_signal = 0
        elif sent < -0.15:
            current_sig = -1
            days_since_signal = 0
        else:
            if current_sig != 0:
                days_since_signal += 1
                if days_since_signal >= 5:
                    current_sig = 0
        signals.append(current_sig)
        
    strategy_df['signal'] = signals
    strategy_df['signal_lagged'] = strategy_df['signal'].shift(1).fillna(0)
    strategy_df['signal_lagged_prev'] = strategy_df['signal_lagged'].shift(1).fillna(0)
    
    # Transaction cost = 0.15% (commission + slippage friction) on position turnover
    strategy_df['friction'] = (strategy_df['signal_lagged'] - strategy_df['signal_lagged_prev']).abs() * 0.0015
    strategy_df['strat_returns'] = strategy_df['signal_lagged'] * strategy_df['returns'] - strategy_df['friction']
    
    strategy_df['cum_bh'] = (1 + strategy_df['returns']).cumprod() * 100
    strategy_df['cum_strat'] = (1 + strategy_df['strat_returns']).cumprod() * 100
    
    backtest_payload = []
    for idx, row in strategy_df.iterrows():
        backtest_payload.append({
            'date': idx.strftime('%Y-%m-%d'),
            'cum_bh': float(row['cum_bh']),
            'cum_strat': float(row['cum_strat']),
            'signal': int(row['signal_lagged'])
        })
        
    final_bh = float(strategy_df['cum_bh'].iloc[-1]) - 100.0 if not strategy_df.empty else 0.0
    final_strat = float(strategy_df['cum_strat'].iloc[-1]) - 100.0 if not strategy_df.empty else 0.0
    
    # Calculate Sortino and Calmar Ratios
    sortino = 0.0
    calmar = 0.0
    if not strategy_df.empty and len(strategy_df) > 1:
        mean_strat = strategy_df['strat_returns'].mean()
        ann_strat_return = mean_strat * ann_factor
        
        # 1. Sortino Ratio (Downside deviation relative to risk free rate)
        downside_diff = strategy_df['strat_returns'] - (0.04 / ann_factor)
        downside_diff = downside_diff.clip(upper=0.0)
        downside_std = downside_diff.std(ddof=1)
        downside_vol_ann = downside_std * (ann_factor ** 0.5)
        if downside_vol_ann > 0:
            sortino = float((ann_strat_return - 0.04) / downside_vol_ann)
            
        # 2. Calmar Ratio (Annualized return relative to maximum drawdown)
        rolling_peak = strategy_df['cum_strat'].cummax()
        strat_dd = (strategy_df['cum_strat'] - rolling_peak) / rolling_peak
        max_strat_dd = float(abs(strat_dd.min()))
        if max_strat_dd > 0:
            calmar = float(ann_strat_return / max_strat_dd)
            
    return {
        'sharpe_ratio': sharpe,
        'beta': beta,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'lead_lag_correlations': lead_lag_corrs,
        'event_study_bullish': avg_bullish_path,
        'event_study_bearish': avg_bearish_path,
        'p_value_bullish': p_val_bull,
        'p_value_bearish': p_val_bear,
        'backtest_data': backtest_payload,
        'final_bh_return': final_bh,
        'final_strat_return': final_strat
    }

def fetch_market_data_cached(ticker, start_date_str, end_date_str):
    """
    Downloads, calculations (SMA, Volatility, Max Drawdown), and returns serialized dict.
    Buffered by 35 days for a complete 20-day SMA trace from the start date.
    """
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return None
        
    start_buffered = start_date - datetime.timedelta(days=35)
    end_buffered = end_date + datetime.timedelta(days=2)
    
    try:
        price_df = yf.download(ticker, start=start_buffered, end=end_buffered)
    except Exception:
        return None
        
    if price_df.empty:
        return None
        
    # Standardize DataFrame columns (flatten MultiIndex)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
        
    # Ensure tz-naive index and fill missing days
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    price_df = price_df.ffill().bfill()
    
    # Calculate 20-day SMA on the full buffered dataset
    price_df['SMA_20'] = price_df['Close'].rolling(window=20).mean()
    
    # Filter back to user's desired date range
    mask = (price_df.index >= pd.to_datetime(start_date)) & (price_df.index <= pd.to_datetime(end_date))
    price_df_filtered = price_df[mask]
    
    if price_df_filtered.empty:
        return None
        
    # Calculate performance change
    first_price = float(price_df_filtered['Close'].iloc[0])
    last_price = float(price_df_filtered['Close'].iloc[-1])
    perf_metric = ((last_price - first_price) / first_price) * 100
    
    # Advanced Quant Metrics: Volatility and Max Drawdown
    volatility = 0.0
    max_dd = 0.0
    if len(price_df_filtered) > 1:
        # Volatility = Annualized Standard Deviation of Daily Percentage Returns
        daily_returns = price_df_filtered['Close'].pct_change().dropna()
        if not daily_returns.empty:
            daily_vol = daily_returns.std()
            ann_factor = get_annualization_factor(ticker)
            volatility = float(daily_vol * (ann_factor ** 0.5) * 100)
        else:
            volatility = 0.0
        # Max Drawdown = Biggest peak-to-trough drop within the event window
        rolling_max = price_df_filtered['Close'].cummax()
        drawdown = (price_df_filtered['Close'] - rolling_max) / rolling_max
        max_dd = float(drawdown.min() * 100)
        
    price_list = []
    for idx, row in price_df_filtered.iterrows():
        price_list.append({
            'date': idx.strftime('%Y-%m-%d'),
            'price': float(row['Close']),
            'sma': float(row['SMA_20']) if not pd.isna(row['SMA_20']) else None,
            'volume': float(row['Volume']) if 'Volume' in price_df_filtered.columns else 0.0
        })
        
    # Reconstruct raw prices for nearest-trading-day lookup (hashing helper)
    raw_prices = [(int(idx.timestamp()), float(row['Close'])) for idx, row in price_df.iterrows()]
    
    return {
        'price_data': price_list,
        'performance_metric': perf_metric,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'raw_prices': raw_prices
    }

@lru_cache(maxsize=64)
def fetch_news_cached(keyword, start_date_str, end_date_str):
    """
    Scrapes Google News RSS for keyword, parses XML via feedparser, cleans HTML,
    and returns a serialized list of dicts within requested date boundaries.
    """
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return []
        
    # Support multiple keywords separated by '+'
    keywords = [k.strip() for k in keyword.split('+') if k.strip()]
    if len(keywords) > 1:
        keyword_query = " OR ".join(f'"{k}"' for k in keywords)
        query = f"({keyword_query}) after:{start_date_str} before:{end_date_str}"
    else:
        query = f'"{keyword}" after:{start_date_str} before:{end_date_str}'

    encoded_query = urllib.parse.quote(query)
    feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    articles = []
    try:
        req = urllib.request.Request(
            feed_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        feed = feedparser.parse(xml_data)
        
        for entry in feed.entries:
            title = entry.get('title', '')
            link = entry.get('link', '')
            summary_html = entry.get('description', '')
            pub_struct = entry.get('published_parsed')
            
            if not title or not pub_struct:
                continue
                
            pub_dt = datetime.datetime(*pub_struct[:6])
            pub_date = pub_dt.date()
            
            if start_date <= pub_date <= end_date:
                source_name = "Google News"
                clean_title = title
                if " - " in title:
                    parts = title.split(" - ")
                    source_name = parts[-1]
                    clean_title = " - ".join(parts[:-1])
                
                clean_summary = summary_html
                if "<" in summary_html:
                    try:
                        from bs4 import BeautifulSoup
                        clean_summary = BeautifulSoup(summary_html, 'html.parser').get_text()
                    except Exception:
                        clean_summary = re.sub(r'<[^>]+>', '', summary_html)
                        clean_summary = clean_summary.replace('&nbsp;', ' ').replace('&amp;', '&')
                
                if len(clean_summary) > 200:
                    clean_summary = clean_summary[:200] + "..."
                    
                articles.append({
                    'title': clean_title,
                    'link': link,
                    'date_epoch': int(pub_dt.timestamp()),
                    'date_str': pub_dt.strftime('%Y-%m-%d %H:%M'),
                    'source': source_name,
                    'summary': clean_summary
                })
    except Exception as e:
        print(f"Error parsing news feed: {e}")
        
    articles.sort(key=lambda x: x['date_epoch'], reverse=True)
    return articles

# ---------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------
app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def home():
    """Renders the HTML Frontend."""
    return render_template('index.html')

@app.route('/guide')
def guide():
    """Renders the User & Operations Guide."""
    return render_template('guide.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    POST Endpoint. Expects JSON containing ticker, keyword, start_date, and end_date.
    Returns market history, technicals, indicators, news, and quant metrics.
    Supports comma-separated tickers and return correlations.
    """
    data = request.get_json() or {}
    ticker = data.get('ticker', '').strip().upper()
    keyword = data.get('keyword', '').strip()
    start_date_str = data.get('start_date', '').strip()
    end_date_str = data.get('end_date', '').strip()
    
    if not ticker or not keyword or not start_date_str or not end_date_str:
        return jsonify({'error': 'Missing parameters: ticker, keyword, start_date, and end_date are required.'}), 400
        
    try:
        datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format. Dates must be YYYY-MM-DD.'}), 400
        
    # Parse tickers
    tickers = [t.strip() for t in ticker.split(',') if t.strip()]
    if not tickers:
        return jsonify({'error': 'No tickers provided.'}), 400

    # 1. Fetch Bounded News Articles (cached)
    articles = fetch_news_cached(keyword, start_date_str, end_date_str)

    if len(tickers) == 1:
        # Single ticker mode (original behavior)
        single_ticker = tickers[0]
        market_res = fetch_market_data_cached(single_ticker, start_date_str, end_date_str)
        if not market_res:
            return jsonify({'error': f"Ticker '{single_ticker}' is invalid or returned no market data."}), 400
            
        raw_prices = market_res['raw_prices']
        indices = [pd.to_datetime(t, unit='s') for t, _ in raw_prices]
        prices = [p for _, p in raw_prices]
        price_series = pd.Series(prices, index=indices)
        
        news_payload = []
        if articles:
            headlines = [a['title'] for a in articles]
            predictions = []
            if sentiment_analyzer is not None:
                try:
                    predictions = sentiment_analyzer(headlines, truncation=True)
                except Exception as e:
                    print(f"FinBERT evaluation failed: {e}. Reverting to Lexicon fallback.")
                    predictions = [analyze_lexicon_sentiment(h) for h in headlines]
            else:
                predictions = [analyze_lexicon_sentiment(h) for h in headlines]
                
            for art, pred in zip(articles, predictions):
                label = pred['label'].lower()
                if 'positive' in label:
                    sent_label = "Bullish"
                    sent_score = 1
                elif 'negative' in label:
                    sent_label = "Bearish"
                    sent_score = -1
                else:
                    sent_label = "Neutral"
                    sent_score = 0
                    
                art_ts = pd.to_datetime(art['date_epoch'], unit='s')
                idx = price_series.index.get_indexer([art_ts], method='bfill')[0]
                if idx == -1:
                    idx = price_series.index.get_indexer([art_ts], method='ffill')[0]
                closest_date = price_series.index[idx]
                price_val = float(price_series.iloc[idx])
                
                news_payload.append({
                    'Date': art['date_str'],
                    'Headline': art['title'],
                    'Link': art['link'],
                    'Sentiment Label': sent_label,
                    'Sentiment Score': sent_score,
                    'Price': price_val,
                    'MappedDate': closest_date.strftime('%Y-%m-%d'),
                    'Source': art['source'],
                    'Summary': art['summary']
                })
                
        pro_metrics = compute_pro_metrics(price_series, news_payload, single_ticker, start_date_str, end_date_str)
        
        return jsonify({
            'is_comparison': False,
            'price_data': market_res['price_data'],
            'news_data': news_payload,
            'performance_metric': market_res['performance_metric'],
            'volatility': market_res['volatility'],
            'max_drawdown': market_res['max_drawdown'],
            'engine_active': nlp_engine_name,
            'pro_metrics': pro_metrics
        })
        
    else:
        # Multiple tickers mode
        comparison_res = {}
        valid_tickers = []
        primary_price_series = None
        primary_ticker = None
        
        for t in tickers:
            res = fetch_market_data_cached(t, start_date_str, end_date_str)
            if res:
                comparison_res[t] = res
                valid_tickers.append(t)
                if primary_price_series is None:
                    primary_ticker = t
                    raw_prices = res['raw_prices']
                    indices = [pd.to_datetime(ts, unit='s') for ts, _ in raw_prices]
                    prices = [p for _, p in raw_prices]
                    primary_price_series = pd.Series(prices, index=indices)
                    
        if not comparison_res:
            return jsonify({'error': f"None of the tickers in {tickers} returned valid market data."}), 400
            
        # Fallback to single mode if only one ticker ended up valid
        if len(valid_tickers) == 1:
            single_ticker = valid_tickers[0]
            res = comparison_res[single_ticker]
            raw_prices = res['raw_prices']
            indices = [pd.to_datetime(t, unit='s') for t, _ in raw_prices]
            prices = [p for _, p in raw_prices]
            price_series = pd.Series(prices, index=indices)
            
            news_payload = []
            if articles:
                headlines = [a['title'] for a in articles]
                predictions = []
                if sentiment_analyzer is not None:
                    try:
                        predictions = sentiment_analyzer(headlines, truncation=True)
                    except Exception as e:
                        predictions = [analyze_lexicon_sentiment(h) for h in headlines]
                else:
                    predictions = [analyze_lexicon_sentiment(h) for h in headlines]
                    
                for art, pred in zip(articles, predictions):
                    label = pred['label'].lower()
                    if 'positive' in label:
                        sent_label = "Bullish"
                        sent_score = 1
                    elif 'negative' in label:
                        sent_label = "Bearish"
                        sent_score = -1
                    else:
                        sent_label = "Neutral"
                        sent_score = 0
                        
                    art_ts = pd.to_datetime(art['date_epoch'], unit='s')
                    idx = price_series.index.get_indexer([art_ts], method='bfill')[0]
                    if idx == -1:
                        idx = price_series.index.get_indexer([art_ts], method='ffill')[0]
                    closest_date = price_series.index[idx]
                    price_val = float(price_series.iloc[idx])
                    
                    news_payload.append({
                        'Date': art['date_str'],
                        'Headline': art['title'],
                        'Link': art['link'],
                        'Sentiment Label': sent_label,
                        'Sentiment Score': sent_score,
                        'Price': price_val,
                        'MappedDate': closest_date.strftime('%Y-%m-%d'),
                        'Source': art['source'],
                        'Summary': art['summary']
                    })
            pro_metrics = compute_pro_metrics(price_series, news_payload, single_ticker, start_date_str, end_date_str)
            
            return jsonify({
                'is_comparison': False,
                'price_data': res['price_data'],
                'news_data': news_payload,
                'performance_metric': res['performance_metric'],
                'volatility': res['volatility'],
                'max_drawdown': res['max_drawdown'],
                'engine_active': nlp_engine_name,
                'pro_metrics': pro_metrics
            })
            
        # Complete Multi-Asset Logic
        news_payload = []
        if articles and primary_price_series is not None:
            headlines = [a['title'] for a in articles]
            predictions = []
            if sentiment_analyzer is not None:
                try:
                    predictions = sentiment_analyzer(headlines, truncation=True)
                except Exception as e:
                    predictions = [analyze_lexicon_sentiment(h) for h in headlines]
            else:
                predictions = [analyze_lexicon_sentiment(h) for h in headlines]
                
            for art, pred in zip(articles, predictions):
                label = pred['label'].lower()
                if 'positive' in label:
                    sent_label = "Bullish"
                    sent_score = 1
                elif 'negative' in label:
                    sent_label = "Bearish"
                    sent_score = -1
                else:
                    sent_label = "Neutral"
                    sent_score = 0
                    
                art_ts = pd.to_datetime(art['date_epoch'], unit='s')
                idx = primary_price_series.index.get_indexer([art_ts], method='bfill')[0]
                if idx == -1:
                    idx = primary_price_series.index.get_indexer([art_ts], method='ffill')[0]
                closest_date = primary_price_series.index[idx]
                price_val = float(primary_price_series.iloc[idx])
                
                news_payload.append({
                    'Date': art['date_str'],
                    'Headline': art['title'],
                    'Link': art['link'],
                    'Sentiment Label': sent_label,
                    'Sentiment Score': sent_score,
                    'Price': price_val,
                    'MappedDate': closest_date.strftime('%Y-%m-%d'),
                    'Source': art['source'],
                    'Summary': art['summary']
                })
                
        # Calculate Pearson Return Correlation Matrix
        returns_dict = {}
        for t in valid_tickers:
            res = comparison_res[t]
            raw = res['raw_prices']
            ts_idx = [pd.to_datetime(ts, unit='s') for ts, _ in raw]
            pr = [p for _, p in raw]
            s = pd.Series(pr, index=ts_idx)
            s_filtered = s[(s.index >= pd.to_datetime(start_date_str)) & (s.index <= pd.to_datetime(end_date_str))]
            if len(s_filtered) > 1:
                returns_dict[t] = s_filtered.pct_change().dropna()
                
        correlation_matrix = {}
        if len(returns_dict) > 1:
            returns_df = pd.DataFrame(returns_dict)
            corr_df = returns_df.corr()
            corr_df = corr_df.where(pd.notnull(corr_df), None)
            correlation_matrix = corr_df.to_dict()
            
        price_comparison = {}
        perf_metrics = {}
        volatilities = {}
        drawdowns = {}
        
        for t in valid_tickers:
            res = comparison_res[t]
            price_comparison[t] = res['price_data']
            perf_metrics[t] = res['performance_metric']
            volatilities[t] = res['volatility']
            drawdowns[t] = res['max_drawdown']
            
        # Compute other pro metrics for the primary asset:
        primary_price_series_filtered = primary_price_series[(primary_price_series.index >= pd.to_datetime(start_date_str)) & (primary_price_series.index <= pd.to_datetime(end_date_str))]
        
        # Calculate individual betas relative to primary reference asset (which behaves as the market index)
        betas = {}
        primary_returns = returns_dict.get(primary_ticker) if returns_dict else None
        if primary_returns is None and len(primary_price_series_filtered) > 1:
            primary_returns = primary_price_series_filtered.pct_change().dropna()
            
        for t in valid_tickers:
            if t == primary_ticker:
                betas[t] = 1.0
            else:
                if t in returns_dict and primary_returns is not None:
                    # Align returns
                    combined = pd.concat([returns_dict[t], primary_returns], axis=1).dropna()
                    if len(combined) > 1:
                        cov = combined.iloc[:, 0].cov(combined.iloc[:, 1])
                        var = combined.iloc[:, 1].var()
                        betas[t] = float(cov / var) if var > 0 else 1.0
                    else:
                        betas[t] = 1.0
                else:
                    betas[t] = 1.0
                    
        pro_metrics = compute_pro_metrics(primary_price_series, news_payload, primary_ticker, start_date_str, end_date_str)
        pro_metrics['betas'] = betas
        
        return jsonify({
            'is_comparison': True,
            'price_comparison': price_comparison,
            'news_data': news_payload,
            'performance_metrics': perf_metrics,
            'volatilities': volatilities,
            'max_drawdowns': drawdowns,
            'valid_tickers': valid_tickers,
            'primary_ticker': primary_ticker,
            'correlation_matrix': correlation_matrix,
            'engine_active': nlp_engine_name,
            'pro_metrics': pro_metrics
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
