# Macro-Economic Sentiment Terminal: Technical Documentation & Architecture Guide

This document provides a comprehensive technical overview of the **Macro-Economic Sentiment Terminal**—what the project is, how it functions under the hood, the mathematical and algorithmic calculations it performs, and how the various data pipelines and NLP models are integrated.

---

## 1. Project Overview

The **Macro-Economic Sentiment Terminal** is a lightweight, self-contained quantitative analysis dashboard. Its primary objective is to visually correlate **asset price action** with **geopolitical and macro-economic news events** and their corresponding **AI-calculated sentiment**.

### Primary Use Case
Analyzing how specific macro events (e.g., "Fed Rate Cuts", "Strait of Hormuz blockade", "Inflation reports") impact financial assets (Stocks, Cryptocurrencies, Commodities, Forex) over custom historical timelines.

---

## 2. Core Architecture

The terminal is designed as a decoupled, full-stack application:

```mermaid
graph TD
    subgraph Frontend (templates/index.html)
        UI[TailwindCSS Controls UI] -->|fetch request POST| API_Client[Vanilla JS Fetch Client]
        API_Client -->|JSON payload: ticker, keyword, dates| API_Route
        PlotlyChart[Plotly.js Chart Canvas] <-->|Interactive Price/SMA/Volume Subplot| API_Client
        NewsFeed[Scrollable News Cards] <-->|Render badge & link| API_Client
        CSV_Export[CSV Download Engine] <-->|Serialize & Download Blob| API_Client
    end

    subgraph Backend (app.py - Flask Server)
        API_Route[/api/analyze] -->|yf.download| MarketData[yfinance Pricing Engine]
        API_Route -->|feedparser query| RSS_Engine[Google News RSS Feed]
        
        RSS_Engine -->|Raw XML| BS4_Clean[BeautifulSoup Parser]
        BS4_Clean -->|Filtered Titles| NLP_Engine{NLP Sentiment Classifier}
        
        NLP_Engine -->|FinBERT Model| FinBERT[Hugging Face pipeline]
        NLP_Engine -->|OOM / Load Fallback| Lexicon[Financial Lexicon Dictionary]
        
        FinBERT -->|Bullish / Bearish / Neutral| Merger[Nearest-Neighbor Date Merger]
        Lexicon -->|Bullish / Bearish / Neutral| Merger
        MarketData -->|Close Prices| Merger
        
        Merger -->|JSON response payload| API_Client
    end
```

### Stack Components
1. **Frontend:** Vanilla HTML5, TailwindCSS (rendered via CDN for speed and styling control), and Plotly.js (CDN) for high-performance chart overlays and volume subplots.
2. **Backend:** Flask (REST API), Flask-CORS (Cross-Origin Resource sharing protection), and Pandas (for series indexing and date mappings).
3. **Data Ingestion:** `yfinance` (market quotes) and `feedparser` (Google News RSS parsing).
4. **Sentiment Core:** Hugging Face `transformers` (FinBERT model pipeline) and a rule-based backup lexicon.

---

## 3. Mathematical Calculations & Performance Metrics

The application performs several statistical, technical, and quantitative calculations on the ingested data:

### A. Asset Performance Metric (% Delta)
The percentage price change of the asset is calculated strictly within the user-specified date range $[T_{\text{start}}, T_{\text{end}}]$:

$$\% \Delta P = \left( \frac{P_{\text{last}} - P_{\text{first}}}{P_{\text{first}}} \right) \times 100$$

Where:
- $P_{\text{first}}$ is the daily closing price of the asset on the first available trading day in the filtered dataset ($\ge T_{\text{start}}$).
- $P_{\text{last}}$ is the closing price on the last available trading day in the dataset ($\le T_{\text{end}}$).
- The values are retrieved from a forward-filled and backward-filled pandas series to prevent empty data points on weekends or holidays.

### B. Daily Return Volatility (Standard Deviation)
To assess how volatile an asset was during the news event, the system calculates the standard deviation of the asset's daily returns during the event window:

$$R_t = \frac{P_t - \max_{\tau \le t} P_\tau}{\max_{\tau \le t} P_\tau}$$

Wait, standard deviation of daily returns is:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

$$\sigma_{\text{daily}} = \sqrt{\frac{1}{M-1} \sum_{t=1}^{M} (R_t - \bar{R})^2} \times 100$$

Where:
- $R_t$ is the return of the asset on day $t$.
- $M$ is the number of trading days in the event window.
- $\bar{R}$ is the average daily return over the event window.
- The result $\sigma_{\text{daily}}$ is expressed as a percentage and represents the asset's dispersion of daily price movements.

### C. Maximum Drawdown (Max DD)
Maximum Drawdown measures the largest historical peak-to-trough drop in the asset's price during the selected event window:

$$\text{Drawdown}(t) = \frac{P_t - \max_{\tau \le t} P_\tau}{\max_{\tau \le t} P_\tau}$$

$$\text{Max Drawdown} = \min_{t} \left( \text{Drawdown}(t) \right) \times 100$$

Where:
- $\max_{\tau \le t} P_\tau$ is the historical maximum closing price up to day $t$ within the event window.
- The drawdown is calculated daily, and the minimum value (the deepest trough) is selected as the Maximum Drawdown.

### D. 20-Day Simple Moving Average (SMA-20)
To overlay technical support levels on the price chart, a rolling 20-day Simple Moving Average is calculated:

$$\text{SMA}_{20}(t) = \frac{1}{20} \sum_{i=0}^{19} P_{t-i}$$

To prevent the first 19 days of the user's chart from displaying NaN (empty) values for the SMA line, the backend requests a **35 calendar day buffer** prior to $T_{\text{start}}$ from `yfinance`. This primes the rolling calculations before the dataset is sliced and returned.

### E. NLP Sentiment Score Mapping
Headlines are processed through the active sentiment classifier. To map textual sentiment categories to numerical values for graph plotting and averaging, we use a discrete mapping function:

$$S(a) = \begin{cases} 
      1.0 & \text{if } L(a) = \text{"positive" (Bullish)} \\
     -1.0 & \text{if } L(a) = \text{"negative" (Bearish)} \\
      0.0 & \text{if } L(a) = \text{"neutral" (Neutral)} 
   \end{cases}$$

Where $L(a)$ is the predicted label for article $a$.

### F. Aggregate Sentiment Index
The overall macroeconomic sentiment trend for the selected keyword and timeline is calculated as the arithmetic mean of the sentiment scores of all articles in the dataset:

$$I_{\text{sentiment}} = \frac{1}{N} \sum_{i=1}^{N} S(a_i)$$

Where:
- $N$ is the total count of news articles matching the keyword within the date range.
- $S(a_i)$ is the score mapping for the $i$-th article.
- The index bounds are $[-1.0, 1.0]$. The terminal classifies the aggregate market sentiment as:
  - **Bullish:** $I_{\text{sentiment}} > 0.15$
  - **Bearish:** $I_{\text{sentiment}} < -0.15$
  - **Neutral:** $-0.15 \le I_{\text{sentiment}} \le 0.15$

---

## 4. News Ingestion & Scraper Pipeline

To keep the application 100% free and rate-limit-free, it constructs dynamically bounded queries against Google News RSS.

1. **Query Construction:**
   Instead of scraping a general feed and filtering manually, the app injects search operators:
   $$\text{Query} = \text{Keyword} + \text{ " after:"} + T_{\text{start}} + \text{ " before:"} + T_{\text{end}}$$
2. **HTTP Fetching:**
   The URL `https://news.google.com/rss/search?q={Query}&hl=en-US&gl=US&ceid=US:en` is fetched using `urllib.request` with a custom `User-Agent` string to avoid standard `403 Forbidden` scraping protections.
3. **Feedparser Parsing:**
   The raw XML byte string is loaded by `feedparser.parse()`.
4. **HTML Cleaning:**
   The description fields of RSS items often contain nested HTML table elements, links, and unescaped entities (e.g. `&nbsp;`, `&amp;`). Using strict XML parsers raises `ParseError`. The terminal uses `BeautifulSoup(description, "html.parser").get_text()` with a regex fallback to extract clean text.

---

## 5. Nearest-Neighbor Date Alignment

A major problem in quantitative mapping is aligning sub-daily news timestamps (published at any hour in UTC timezone) with standard daily stock market closing prices (defined on specific trading dates, excluding weekends and holidays).

The terminal solves this using a **nearest-neighbor lookup algorithm** in Pandas:

1. **Standardize Timestamps:**
   Convert the article's UTC timestamp to a timezone-naive datetime object:
   $$t_{\text{article}} \to \text{Timestamp without timezone}$$
2. **Nearest Trading Date Index Lookup:**
   Query the index of the asset price DataFrame (which is a sorted DatetimeIndex):
   $$\text{idx} = \text{price\_df.index.get\_indexer}([t_{\text{article}}], \text{method='nearest'})[0]$$
3. **Price Retrieval:**
   Lookup the date and close price at that index:
   $$T_{\text{trading}} = \text{price\_df.index[idx]}$$
   $$P_{\text{trading}} = \text{price\_df.loc}[T_{\text{trading}}, \text{'Close'}]$$
4. **Result:**
   The news event is plotted at coordinates $(T_{\text{trading}}, P_{\text{trading}})$. This overlays the marker **directly on top** of the asset price line on the Plotly canvas.

---

## 6. System Design: Rate-Limit Caching (`lru_cache`)

Repeatedly querying external feeds like Yahoo Finance or Google News RSS for the same ticker/dates can lead to IP address temporary blocks or rate limits.

The terminal implements a strict caching design:
1. **LRU Cache Decorators:**
   `fetch_market_data_cached` and `fetch_news_cached` are decorated with `@lru_cache(maxsize=64)`.
2. **Lightweight Data Serialization:**
   Pandas DataFrames are not hashable and can lead to memory bloat if cached directly. To prevent this, the backend serializes market data into primitive python types (dictionaries and tuple lists) before saving to the LRU cache.
3. **Nearest-Neighbor Reconstruction:**
   When the Flask endpoint parses requests, it reconstructs the cached raw price tuple list `(timestamp, price)` back into a lightweight Pandas series dynamically. This supports nearest-date alignments instantly without making network requests.

---

## 7. Interactive Subplots & Export Mechanics (Frontend)

### A. Price and Volume Subplots
The chart integrates trading volume below the main price line.
- The Price, SMA-20, and News traces map to `yaxis: 'y'` (domain `[0.33, 1.0]`).
- The Volume trace maps to `yaxis2: 'y2'` (domain `[0.0, 0.23]`).
- They share the default `xaxis: 'x'`. Plotly.js links their zoom and hover actions automatically.
- **Volume Coloring:** Volume bar colors indicate the direction of price returns:
  - Green (`rgba(0, 230, 118, 0.3)`): Day Close $\ge$ Prior Close (buying pressure).
  - Red (`rgba(255, 23, 68, 0.3)`): Day Close $<$ Prior Close (selling pressure).

### B. CSV Export
To serve enterprise users, the dashboard features client-side export handlers:
- **Price Data Export:** Serializes date, close price, SMA-20, and volume.
- **News Data Export:** Serializes publication dates, headlines, URLs, sentiment classifications, score weights, mapped asset prices, sources, and summaries. All text fields containing quotes or commas are escaped via RFC 4180 algorithms.

---

## 8. NLP Sentiment Core & Memory Optimization

Deploying deep learning models on a 1GB RAM budget (Render free tier) requires strict memory management:
1. **Global Lifecycle Caching:** The transformer model (`ProsusAI/finbert`) is loaded once at module import. Route handlers share this single instance without re-instantiating the weights.
2. **Single-Thread Execution:** We restrict PyTorch thread allocation to a single thread:
   `torch.set_num_threads(1)`
   This prevents PyTorch from spinning up multiple CPU worker threads that multiply memory usage.
3. **No Heavy Frontend Frameworks:** By serving standard HTML/JS directly via Flask instead of using Dash or Streamlit, the server-side memory overhead of layout generation is completely eliminated.
4. **Lexicon Fallback Engine:** Features an automatic try-except wrapper. If the Hugging Face hub is rate-limited or fails to load, it falls back to a high-speed local rules-based sentiment pipeline.

---

## 9. Professional Quantitative Terminal Features

To support enterprise-grade quant workflows, the terminal incorporates a dedicated **Pro Quant Terminal** engine. This section explains the math, calculations, and operational principles behind each metric, alongside tab execution instructions.

---

### A. Annualized Sharpe Ratio
The Sharpe Ratio measures "risk-adjusted return". It helps investors understand if the returns of a selected asset are high enough to justify the price swings (volatility) endured.

#### 1. Mathematical Formulation
$$\text{Sharpe Ratio} = \frac{R_a - R_f}{\sigma_a}$$

Where:
- $R_a$ is the annualized return of the asset, computed by scaling the average daily return ($\bar{R}_{\text{daily}}$) by the annual trading session count ($N$):
  $$R_a = \bar{R}_{\text{daily}} \times N$$
- $R_f$ is the annualized risk-free rate. The terminal defaults to **$4.0\%$** ($0.04$), representing the risk-free yield available on US government short-term treasury bills.
- $\sigma_a$ is the annualized volatility of daily returns, computed by scaling the daily returns standard deviation ($\sigma_{\text{daily}}$) by $\sqrt{N}$:
  $$\sigma_a = \sigma_{\text{daily}} \times \sqrt{N}$$
- $N$ is the standard session count: $N = 365$ for Cryptocurrencies, $N = 260$ for Forex, and $N = 252$ for Stocks/Commodities.

#### 2. Interpretation Guide
- **$\text{Sharpe} < 0.0$:** Suboptimal. The asset underperformed safe government bonds.
- **$0.0 \le \text{Sharpe} < 1.0$:** Acceptable. Typical return profile for moderate risks.
- **$\text{Sharpe} \ge 1.0$:** Good. Outstanding return generation relative to standard price swings.
- **$\text{Sharpe} \ge 2.0$:** Excellent. Highly efficient capital deployment.

---

### A1. Annualized Sortino Ratio
The Sortino Ratio measures the strategy's risk-adjusted return relative to **downside deviation** (penalizing only negative price movements).

#### 1. Mathematical Formulation
$$\text{Sortino Ratio} = \frac{R^{strat}_a - R_f}{\sigma_d}$$

Where:
- $R^{strat}_a$ is the annualized strategy return:
  $$R^{strat}_a = \bar{R}^{strat}_{\text{daily}} \times N$$
- $R_f$ is the annualized risk-free rate ($0.04$).
- $\sigma_d$ is the annualized downside volatility, calculated as the sample standard deviation of daily strategy returns ($R^{strat}_t$) clipped at a maximum of the daily risk-free rate ($R_f / N$):
  $$\sigma_d = \sqrt{\text{Variance}\left( \min\left(0, R^{strat}_t - \frac{R_f}{N}\right) \right) \times N}$$
- $N$ is the standard session count: $365$ for Cryptocurrencies, $260$ for Forex, and $252$ for Stocks/Commodities.

#### 2. Interpretation Guide
- **$\text{Sortino} < 0.0$:** Suboptimal. Underperformed safe treasury yields.
- **$0.0 \le \text{Sortino} < 1.0$:** Standard risk profile.
- **$\text{Sortino} \ge 1.0$:** Good downside risk-adjusted return.
- **$\text{Sortino} \ge 2.0$:** Excellent downside protection and return efficiency.

---

### A2. Strategy Calmar Ratio
The Calmar Ratio evaluates the strategy's return relative to its worst historical peak-to-trough drawdown risk.

#### 1. Mathematical Formulation
$$\text{Calmar Ratio} = \frac{R^{strat}_a}{MDD^{strat}}$$

Where:
- $R^{strat}_a$ is the annualized strategy return.
- $MDD^{strat}$ is the absolute maximum strategy peak-to-trough drawdown calculated from the compounding strategy equity curve:
  $$MDD^{strat} = \max_t \left( \frac{\max_{\tau \le t} \text{Equity}(\tau) - \text{Equity}(t)}{\max_{\tau \le t} \text{Equity}(\tau)} \right)$$

#### 2. Interpretation Guide
- **$\text{Calmar} < 0.0$:** Negative strategy return.
- **$0.0 \le \text{Calmar} < 1.0$:** Low return relative to worst-case historical drops.
- **$\text{Calmar} \ge 1.0$:** Good return-to-drawdown efficiency.
- **$\text{Calmar} \ge 2.0$:** Outstanding risk-adjusted reward profile.

---

### B. Systematic Risk (Market Beta) & Market Alpha
Beta ($\beta$) and Alpha ($\alpha$) measure an asset's risk and return profile relative to a market benchmark index. Beta represents systematic sensitivity to market-wide price swings, while Alpha represents the asset's idiosyncratic average excess return.

#### 1. Mathematical Formulation
Using the full buffered daily return series, we run an Ordinary Least Squares (OLS) linear regression of the asset daily returns ($R_{\text{asset}, t}$) against the benchmark daily returns ($R_{\text{benchmark}, t}$):

$$R_{\text{asset}, t} = \alpha + \beta R_{\text{benchmark}, t} + \epsilon_t$$

Where:
- $\beta$ is computed as the ratio of covariance to benchmark variance:
  $$\beta = \frac{\text{Covariance}(R_{\text{asset}}, R_{\text{benchmark}})}{\text{Variance}(R_{\text{benchmark}})}$$
- $\alpha$ is the intercept of the regression line, representing average excess return when the benchmark return is zero:
  $$\alpha = \bar{R}_{\text{asset}} - \beta \bar{R}_{\text{benchmark}}$$
- $\epsilon_t$ is the residual error term.
- **Benchmark Selection:**
  - **Single Asset Mode:** The benchmark is automatically set to the S&P 500 (`^GSPC`) for equities, commodities, forex, and indices, or to Bitcoin (`BTC-USD`) for cryptocurrencies.
  - **Comparison Mode:** The benchmark defaults to the first ticker entered in the custom list. The systematic risk of all other tickers is evaluated relative to this primary ticker.

#### 2. Interpretation Guide
- **$\beta = 1.0$:** The asset moves in perfect lockstep with the benchmark.
- **$\beta > 1.0$:** High systematic sensitivity. The asset amplifies broad market swings (e.g., technology stocks or highly volatile altcoins).
- **$0.0 < \beta < 1.0$:** Defensive asset. Insulated from severe benchmark shocks (e.g., gold or utility stocks).
- **$\beta < 0.0$:** Inverse relationship. Moves in the opposite direction of the benchmark (e.g., short ETFs or negative correlation hedges).

---

### C. News Sentiment Lead-Lag Analysis
Lead-Lag analysis checks if news sentiment predicts asset returns or if news articles merely react to price trends that already occurred.

#### 1. Mathematical Formulation
For each lag day $k \in \{-3, -2, -1, 0, 1, 2, 3\}$, we shift the daily news sentiment series ($S_t$) by $k$ and calculate the Pearson correlation coefficient with the daily returns series ($R_t$):

$$\rho_k = \frac{\sum_{t} (R_t - \bar{R})(S_{t-k} - \bar{S})}{\sqrt{\sum_{t} (R_t - \bar{R})^2 \sum_{t} (S_{t-k} - \bar{S})^2}}$$

Where:
- $S_t$ is the average sentiment score on day $t$ (Bullish = +1.0, Bearish = -1.0, Neutral = 0.0; filled with 0.0 on days with no news).
- $R_t$ is the daily percentage price return on day $t$.

#### 2. Interpretation Guide (Chart Lags)
- **Positive Lags ($+1d, +2d, +3d$):** Correlates past news sentiment with today's returns. Positive correlations indicate that **News Sentiment LEADS (predicts) Price**.
- **Lag $0d$ (Contemporaneous):** Measures same-day market reactions to breaking stories.
- **Negative Lags ($-1d, -2d, -3d$):** Correlates future news sentiment with today's returns. Positive correlations indicate that **Price LEADS News** (news reports are delayed reactions to price swings).

---

### D. Event Study & Cumulative Abnormal Returns (CAR)
An event study isolates the impact of specific news events and tracks the average trajectory of abnormal returns surrounding the event announcement day.

#### 1. Daily Abnormal Returns ($AR_{it}$)
For a news event $i$, the abnormal return on trading day $t$ within the event window is calculated by subtracting the expected return (modeled via OLS parameters) from the actual return:

$$AR_{it} = R_{it} - (\alpha_i + \beta_i R_{mt})$$

Where:
- $R_{it}$ is the actual return of the asset on day $t$ for event $i$.
- $R_{mt}$ is the return of the market index on day $t$.
- $\alpha_i$ and $\beta_i$ are the OLS regression parameters estimated over the full history.

#### 2. Cumulative Abnormal Returns ($CAR_i(\tau)$)
The cumulative abnormal return is computed by accumulating the abnormal returns over the event window relative to the announcement day ($t=0$, where $CAR_i(0) = 0$):

$$CAR_i(\tau) = \begin{cases} 
      \sum_{j=1}^{\tau} AR_{i, t+j} \times 100 & \text{if } \tau > 0 \\
      0.0 & \text{if } \tau = 0 \\
      -\sum_{j=\tau}^{-1} AR_{i, t+j} \times 100 & \text{if } \tau < 0 
   \end{cases}$$

We calculate the average trajectory by averaging $CAR_i(\tau)$ across all events in each category (Bullish or Bearish):

$$\overline{CAR}(\tau) = \frac{1}{n} \sum_{i=1}^{n} CAR_i(\tau)$$

#### 3. Statistical Significance p-value at $t = +5$ Days
To determine if cumulative post-event price trends are statistically significant or merely random noise, the terminal calculates the two-tailed p-value at the $t = +5$ day horizon.

1. **Calculate Sample Standard Deviation ($s$) and Standard Error ($SE$):**
   $$s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (CAR_i(5) - \overline{CAR}(5))^2}$$
   $$SE = \frac{s}{\sqrt{n}}$$
   *(where $n$ is the count of events in the Bullish or Bearish category).*

2. **Compute the $t$-statistic:**
   $$t = \frac{\overline{CAR}(5)}{SE}$$

3. **Compute Two-Tailed p-value using standard normal CDF logistic approximation:**
   $$p\text{-value} = 2 \times (1 - \Phi(|t|))$$
   $$\Phi(|t|) \approx \frac{1}{1 + e^{-y}} \quad \text{where } y = |t| \times (1.5976 + 0.07056 \times |t|^2)$$

A $p$-value $< 0.05$ indicates that the cumulative abnormal movement is statistically significant (we reject the null hypothesis of zero abnormal return at a $95\%$ confidence level), marked with an asterisk (`*`) on the chart legend.

---

### E. Friction-Adjusted Sentiment Strategy Backtest
Backtests the performance of a systematic sentiment-driven quantitative strategy, incorporating realistic transaction costs to prevent overestimating returns.

#### 1. Trading Logic
- **Daily Signal Generation:**
  $$Signal_t = \begin{cases} 
        1.0 & \text{if } S_t > 0.15 \text{ (Go Long)} \\
       -1.0 & \text{if } S_t < -0.15 \text{ (Go Short)} \\
        0.0 & \text{otherwise}
     \end{cases}$$
- **Causality Lag:** The trade signal is lagged by 1 trading day ($Signal_{t-1}$) relative to asset returns ($R_t$) to prevent look-ahead bias, meaning trades execute at the market close *after* news sentiment is parsed.
- **Time-Decay Exit:** If no news occurs for 5 consecutive trading days, the strategy automatically liquidates the position to cash ($Signal_t = 0$) to manage holding risk.
- **Friction Cost Model:** A 0.15% transaction cost (incorporating a 0.10% broker commission fee and a 0.05% slippage spread) is applied on the change in position size (turnover):
  $$\text{Friction}_t = |Signal_{t-1} - Signal_{t-2}| \times 0.0015$$
  - Switching from Long ($+1$) to Short ($-1$) triggers a turnover of $|+1 - (-1)| = 2$ units, incurring a friction penalty of $2 \times 0.15\% = 0.30\%$.
  - Switching from Long ($+1$) to Cash ($0$) triggers a turnover of $|+1 - 0| = 1$ unit, incurring a friction penalty of $1 \times 0.15\% = 0.15\%$.

#### 2. Strategy Returns & Compounding Curves
The daily returns of the sentiment strategy ($R^{strat}_t$) are computed as:

$$R^{strat}_t = Signal_{t-1} \times R_t - \text{Friction}_t$$

The compounding equity curves (rebased to start at 100) are computed as:

$$\text{BH\_Equity}(t) = \prod_{i=1}^{t} (1 + R_i) \times 100$$
$$\text{Strat\_Equity}(t) = \prod_{i=1}^{t} (1 + R^{strat}_i) \times 100$$

---

### F. How to Navigate the Workspace Tabs

The user interface segregates visualizations and analytics into two distinct workspaces:

1. **📊 Market Overview Tab:**
   - **Main Price Canvas:** Plots the asset price line, overlaid with the 20-day Simple Moving Average (gold dashed line) and individual news event markers. News markers are color-coded: **Green triangles** (Bullish sentiment), **Red triangles** (Bearish sentiment), and **Gray circles** (Neutral sentiment).
   - **Volume Subplot:** Renders bar charts of daily trading volume. Bars are color-coded by daily returns (green for positive returns, red for negative returns) to visually isolate buying pressure and panic selling volumes.
   - **Contextual News Feed:** Provides a list of matching news items with links, sources, summaries, and sentiment scores.
   - **Correlation Matrix:** (Visible in comparison mode) Displays returns correlation coefficients.

2. **⚡ Pro Quant Terminal Tab:**
   - **Metrics Cards:** Displays the **Annualized Sharpe Ratio** (risk-adjusted return), **Annualized Sortino Ratio** (downside risk-adjusted return), **Strategy Calmar Ratio** (return relative to drawdown), the **Systematic Beta** (risk sensitivity relative to benchmarks or primary ticker), and the **Backtest Net Return** compared to standard Buy & Hold.
   - **Lead-Lag Analysis Chart:** Plots the cross-correlation bars for shifts $t-3$ to $t+3$.
   - **Event Study Chart:** Plots average abnormal return paths, displaying the calculated p-values and significance flags in the legend.
   - **Backtest Simulator Chart:** Plots the strategy equity growth curve vs. the benchmark buy-and-hold index.
