# Macro-Economic Sentiment Terminal: Technical Documentation & Quantitative Masterclass

Welcome! This document functions as an educational guide and technical blueprint for the **Macro-Economic Sentiment Terminal**. It is structured to act as an expert quantitative instructor, explaining not just *how* each subsystem is built, but *why* it was designed that way using the **"5 Whys and 1 How"** framework, backed by detailed mathematical formulations and step-by-step numerical examples.

---

## Table of Contents
1. **[Data Ingestion: Google News RSS Scraper]**
2. **[Data Cleaning: BeautifulSoup HTML Parser]**
3. **[Semantic Analysis: FinBERT AI & Lexicon Fallback]**
4. **[Date Mappings: Nearest-Neighbor Trading Date Alignment]**
5. **[System Design: LRU Cache Performance Optimization]**
6. **[Technicals: Rolling Simple Moving Average (SMA-20)]**
7. **[Risk Analytics: Annualized Daily Return Volatility]**
8. **[Risk Analytics: Maximum Drawdown]**
9. **[Performance: Annualized Sharpe Ratio]**
10. **[Performance: Annualized Sortino Ratio]**
11. **[Performance: Strategy Calmar Ratio]**
12. **[Systematic Risk: OLS Market Beta & Alpha Regression]**
13. **[Lead-Lag Analysis: Pearson Cross-Correlations]**
14. **[Event Studies: Cumulative Abnormal Returns (CAR) & p-Value Significance]**
15. **[Strategy Simulation: Friction-Adjusted Backtester]**

---

## 1. Data Ingestion: Google News RSS Scraper

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need news ingestion?**
    Financial markets do not move in a vacuum. Price changes are driven by incoming information. To analyze the relationship between macroeconomic events (e.g. interest rates, tariffs) and asset prices, we must collect historical news headlines.
*   **Why 2: Why do we use Google News RSS instead of a standard Google search?**
    Scraping raw Google search results yields messy HTML, ads, and search page changes that break scraper code. Google News RSS feeds provide structured XML data, making it easier to parse titles, links, and publication dates reliably.
*   **Why 3: Why do we structure search queries with `after:` and `before:` tags?**
    Downloading all news for a keyword and then filtering it locally is highly inefficient and wastes memory. By passing query parameters like `Inflation after:2026-06-01 before:2026-06-15` directly to Google, we offload the date filtering to Google's servers and only ingest the relevant subset.
*   **Why 4: Why do we override the request headers with a custom `User-Agent`?**
    Standard Python requests library sends `Python-urllib/X.Y` in the user-agent header. Google's firewalls block default script headers with `403 Forbidden` errors to prevent scraping. Changing the header to look like a desktop web browser (Mozilla/5.0) bypasses these blocks.
*   **Why 5: Why do we wrap the scraping process in `feedparser` instead of a custom XML parser?**
    RSS and Atom XML feeds have many formatting variations (e.g., date formats, tag structures). `feedparser` is a robust library that automatically normalizes different RSS variants into a standard, clean Python dictionary structure.
*   **How: How is it computed/executed?**
    We construct the query URL, request the feed, and parse the XML:
    1.  **Formulate Query:** `Query = (Inflation OR CPI) after:2026-06-01 before:2026-06-15`
    2.  **Encode URL:** Convert spaces and symbols to hex codes: `https://news.google.com/rss/search?q=%28Inflation%20OR%20CPI%29...`
    3.  **Fetch & Parse:**
        ```python
        req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0...'})
        with urllib.request.urlopen(req) as resp:
            feed = feedparser.parse(resp.read())
        ```

---

## 2. Data Cleaning: BeautifulSoup HTML Parser

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need data cleaning?**
    Google News RSS description fields contain raw HTML markup, tables, hyperlinks, and images. Feeding raw HTML into NLP engines or displaying it on a UI causes formatting breaks, garbled text, and security vulnerabilities (XSS).
*   **Why 2: Why do we use BeautifulSoup instead of regular expressions?**
    HTML is not a regular language. Parsing HTML with regex (e.g. `/<.*?>/`) is fragile and fails on nested tags, script blocks, or unclosed elements. BeautifulSoup builds a proper parse tree, allowing us to extract clean text safely.
*   **Why 3: Why do we specify the `"html.parser"` engine?**
    Python has a built-in HTML parser that requires no external C-dependencies. This ensures that the terminal remains lightweight, compiles easily across Windows/Linux, and loads quickly on Render's 512MB RAM server.
*   **Why 4: Why do we decode HTML entities like `&nbsp;` and `&amp;`?**
    XML encoders convert special symbols to entities to prevent parsing errors. If left uncleaned, these symbols pollute NLP text inputs (e.g., analyzing `&quot;bad&quot;` instead of `"bad"` degrades sentiment accuracy).
*   **Why 5: Why do we truncate summaries to 200 characters?**
    Short summaries improve UI scannability. Storing large text blobs in memory wastes RAM. Keeping summaries short keeps the API JSON payload small and load times fast.
*   **How: How is it computed/executed?**
    *   **Input text:** `<div>Apple CEO announced <a href="...">iPhone 17</a> details &amp; specs.</div>`
    *   **Parsing:**
        1. BeautifulSoup parses the HTML structure.
        2. `.get_text()` strips out the `<div>` and `<a>` tags.
        3. Entity decoding converts `&amp;` back to `&`.
    *   **Output text:** `Apple CEO announced iPhone 17 details & specs.`

---

## 3. Semantic Analysis: FinBERT AI & Lexicon Fallback

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need semantic analysis?**
    A computer cannot natively understand if a sentence is good or bad news. Semantic analysis translates human headlines into a numerical score (positive/negative/neutral) that quantitative math algorithms can process.
*   **Why 2: Why do we use FinBERT as our primary model?**
    General NLP models trained on Wikipedia or movie reviews fail on financial terminology. For example, the word *"crude"* is negative in common english (e.g. *"crude joke"*), but neutral/bullish in finance (*"crude oil prices rose"*). FinBERT was trained specifically on financial text (Financial PhraseBank).
*   **Why 3: Why do we set `torch.set_num_threads(1)`?**
    PyTorch automatically spins up multiple CPU threads to process tensor math. On shared low-memory servers (like Render's Free Tier), this thread overhead spikes CPU usage and triggers Out-Of-Memory (OOM) kernel kills. Forcing it to 1 thread keeps the memory footprint tiny.
*   **Why 4: Why do we build a Lexicon Fallback engine?**
    If the Hugging Face hub is down, or if the server starts on a machine with less than 512MB RAM, loading the heavy FinBERT weights will cause the server to crash. The lexicon engine provides a zero-memory, rule-based fallback to guarantee the server starts up.
*   **Why 5: Why do we map labels to numerical scores (`1.0`, `-1.0`, `0.0`)?**
    Calculations like averages, correlations, and regressions require numeric inputs. Mapping labels (Bullish $\to 1.0$, Bearish $\to -1.0$, Neutral $\to 0.0$) allows us to compute quantitative averages.
*   **How: How is it computed/executed?**
    *   **FinBERT Pipeline:** Headline is passed to the Hugging Face model:
        *   `Headline:` *"Federal Reserve cuts interest rates by 50 basis points."*
        *   `FinBERT Output:` `[{'label': 'positive', 'score': 0.942}]`
        *   `Numerical Mapping:` Sentiment Score = `+1.0` (Bullish).
    *   **Lexicon Fallback Pipeline:**
        *   `Headline:` *"Markets slump amid inflation fears."*
        *   Word check against sets: `slump` matches `neg_words`, `inflation` matches `neg_words`.
        *   `Counts:` `pos_count = 0`, `neg_count = 2`.
        *   `Result:` Sentiment Score = `-1.0` (Bearish).

---

## 4. Date Mappings: Nearest-Neighbor Trading Date Alignment

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need date alignment?**
    Geopolitical and macroeconomic news can break at any time (e.g. Sunday afternoon or midnight UTC). However, stock markets are only open on standard trading days (Monday–Friday, 9:30 AM–4:00 PM EST). We must map the news date to a day when prices actually exist.
*   **Why 2: Why do we map weekend news forward to Monday instead of backward to Friday?**
    Mapping weekend news to Friday's close violates **causality**. Friday's market closing price cannot react to news that has not happened yet. The news must be mapped to Monday's close to preserve the timeline of cause and effect.
*   **Why 3: Why do we use a binary timezone-naive index?**
    Timezones (UTC, EST, BST) cause alignment issues due to Daylight Saving Time changes. Converting all timestamps to timezone-naive datetime structures prevents alignment errors.
*   **Why 4: Why do we use pandas `get_indexer(..., method='nearest')` or `method='bfill'`?**
    If an article is published at midnight on a Tuesday, there is no exact price data for that second. `nearest` or `bfill` finds the closest trading date on the index, ensuring that the news is plotted on a valid market day.
*   **Why 5: Why do we use `bfill` (backward fill) for weekend news?**
    `bfill` stands for backward fill, which looks *forward* in time to find the next available trading day. It ensures that news published on a Saturday or Sunday is aligned with Monday's market session, preventing look-ahead bias.
*   **How: How is it computed/executed?**
    *   **Asset DatetimeIndex (sorted):** `[2026-06-12 (Fri), 2026-06-15 (Mon), 2026-06-16 (Tue)]`
    *   **News Article Timestamp:** `2026-06-13 14:30:00 (Sat)`
    *   **Execution:**
        1. Convert Sat to timezone-naive timestamp.
        2. Query index using `bfill` (find next available):
           `idx = price_df.index.get_indexer([Sat], method='bfill')[0]`
        3. Index maps to `2026-06-15 (Mon)`.
    *   **Output:** The news is mapped to Monday's price point.

---

## 5. System Design: LRU Cache Performance Optimization

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need caching?**
    Ingesting prices from Yahoo Finance and headlines from Google News requires external network requests. If multiple users query the same assets repeatedly, the API will hit rate limits, leading to IP blocks and slow page load times.
*   **Why 2: Why do we use an Least Recently Used (LRU) cache?**
    An LRU cache keeps active queries in memory. When the cache limit is reached, it automatically discards the oldest, least-visited queries, preventing memory leak issues.
*   **Why 3: Why do we cache primitive Python types (lists/dicts) instead of Pandas DataFrames?**
    Pandas DataFrames are complex, mutable objects that consume substantial memory. Caching them directly can cause memory bloat. Caching primitive JSON-serializable types keeps the cache footprint small.
*   **Why 4: Why do we separate market data and news data caching?**
    Market price history is identical for all users, whereas news searches can vary based on keyword inputs. Separating them ensures that a price query for `AAPL` can remain cached even if the keyword changes from `iPhone` to `Earnings`.
*   **Why 5: Why do we restrict cache `maxsize` to 64?**
    A higher cache size increases RAM usage. A limit of 64 provides a balance, keeping API response times fast (under 5ms for cached queries) while staying within Render's 512MB RAM limit.
*   **How: How is it computed/executed?**
    *   **Python decorator:**
        ```python
        @lru_cache(maxsize=64)
        def fetch_news_cached(keyword, start_date, end_date):
            # Downloads and returns list of articles
        ```
    *   **First call:** Executed in ~1.5 seconds (network request).
    *   **Second call:** Returns instantly in `< 1ms` from memory.

---

## 6. Technicals: Rolling Simple Moving Average (SMA-20)

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need the SMA-20?**
    Asset prices are noisy. Daily price moves contain random fluctuations that make trends hard to see. A moving average smooths the price line, helping users identify the underlying trend.
*   **Why 2: Why do we calculate it over a rolling 20-day window?**
    A 20-day window represents roughly 1 calendar month of trading sessions. It is a standard technical support level used by traders to identify short-term trends.
*   **Why 3: Why does the backend fetch a 35-day buffer of historical data?**
    Calculating a 20-day moving average requires 20 preceding days of price data. If we only fetched data starting on the user's selected date, the first 19 days of the SMA chart would be empty (NaN). Fetching a 35-day calendar buffer primes the rolling window.
*   **Why 4: Why do we use a simple arithmetic average instead of an exponential one?**
    Simple moving averages weight all 20 days equally. It is easy to calculate, computationally lightweight, and serves as a reliable baseline without complex calculations.
*   **Why 5: Why do we calculate the SMA on the backend rather than the frontend?**
    Calculating technical indicators on the backend reduces the volume of raw data sent to the frontend. The browser only needs to render the pre-calculated points, improving dashboard responsiveness.
*   **How: How is it computed/executed?**
    *   **Window Prices:** Last 20 days: `[100, 101, 102, ..., 119]` (Sum = 2,190)
    *   **Calculation:**
        $$\text{SMA}_{20} = \frac{1}{20} \sum_{i=0}^{19} P_{t-i} = \frac{2190}{20} = 109.5$$
    *   **Output:** The SMA value plotted for day $t$ is \$109.5.

---

## 7. Risk Analytics: Annualized Daily Return Volatility

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need volatility?**
    Raw returns do not tell the whole story. A 10% return on a stable asset (low risk) is different from a 10% return on a highly volatile asset (high risk). Volatility measures this risk profile.
*   **Why 2: Why do we calculate standard deviation over daily returns instead of daily prices?**
    Asset prices trend upward or downward, which distorts standard deviation calculations. Daily returns represent stationary percentage changes, which are statistically stable and comparable across assets.
*   **Why 3: Why do we *annualize* volatility?**
    A daily volatility of 1.5% is hard to compare across different assets. Annualizing translates daily volatility into a standard 1-year horizon, making it easy to compare different asset classes.
*   **Why 4: Why do we multiply by $\sqrt{N}$ instead of $N$ to annualize?**
    Daily returns are assumed to be independent random variables. Variance scales linearly with time ($N$), and standard deviation is the square root of variance, so it scales with $\sqrt{N}$.
*   **Why 5: Why do we use different session counts ($N$) for different asset classes?**
    Cryptocurrencies trade 24/7 ($N=365$), Forex pairs trade 5 days a week plus overnight ($N=260$), and traditional Stock exchanges trade 252 sessions a year. Using the correct $N$ prevents underestimating or overestimating volatility.
*   **How: How is it computed/executed?**
    *   **Inputs:** Average daily return $\bar{R} = 0.10\%$, standard deviation of daily returns $\sigma_{\text{daily}} = 1.20\%$, ticker = `AAPL` ($N=252$).
    *   **Calculation:**
        $$\sigma_{\text{annualized}} = \sigma_{\text{daily}} \times \sqrt{N} = 1.20\% \times \sqrt{252} = 1.20\% \times 15.8745 = 19.05\%$$
    *   **Interpretation:** The asset has an annualized volatility of 19.05%.

---

## 8. Risk Analytics: Maximum Drawdown

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need Maximum Drawdown?**
    Volatility does not measure peak-to-trough risk. An asset can have low daily volatility but still undergo a slow, steady 50% decline. Maximum Drawdown measures the worst-case loss scenario.
*   **Why 2: Why do we use a rolling cumulative maximum peak as our denominator?**
    A drawdown is measured relative to the highest price peak achieved *before* that day. Using a rolling peak ($RollingMax_t$) ensures that we measure the decline from the actual historical high point.
*   **Why 3: Why do we calculate drawdown on a daily basis instead of weekly or monthly?**
    Weekly or monthly prices smooth out price spikes. Daily calculations capture the exact intraday or day-to-day panic selling points.
*   **Why 4: Why do we focus on the *minimum* value of the drawdown series?**
    The minimum value represents the largest peak-to-trough drop. It is the absolute maximum loss you would have experienced if you bought at the absolute peak and sold at the absolute trough.
*   **Why 5: Why do we express it as a percentage?**
    Percentage terms allow us to compare drawdowns across assets with different prices (e.g. comparing a \$10 drop on a \$100 stock to a \$1,000 drop on a \$60,000 cryptocurrency).
*   **How: How is it computed/executed?**
    *   **Price Series:** `[100, 120, 110, 90, 130]`
    *   **Rolling Peak Series:** `[100, 120, 120, 120, 130]`
    *   **Drawdowns:**
        *   Day 1: $(100 - 100) / 100 = 0.0\%$
        *   Day 2: $(120 - 120) / 120 = 0.0\%$
        *   Day 3: $(110 - 120) / 120 = -8.33\%$
        *   Day 4: $(90 - 120) / 120 = -25.0\%$
        *   Day 5: $(130 - 130) / 130 = 0.0\%$
    *   **Maximum Drawdown:** $\min(0, 0, -8.33\%, -25.0\%, 0) = -25.0\%$.

---

## 9. Performance: Annualized Sharpe Ratio

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need the Sharpe Ratio?**
    Investors need to know if an asset's returns are high enough to justify the price swings (volatility) endured. The Sharpe Ratio measures this risk-adjusted performance.
*   **Why 2: Why do we subtract the risk-free rate ($R_f$) from the annualized return?**
    You can earn a risk-free return (e.g. 4% in government treasury bills) with zero risk. We only care about the *excess* return the asset generates for taking active risk.
*   **Why 3: Why do we divide the excess return by the annualized volatility?**
    Dividing by volatility penalizes assets that achieve their returns through huge, stressful price fluctuations, standardizing return efficiency per unit of risk.
*   **Why 4: Why do we annualize both return and volatility before dividing?**
    Daily returns and daily volatilities are too small to interpret. Annualization converts both to a standard 1-year horizon, making comparison easy.
*   **Why 5: Why do we default to a $4.0\%$ risk-free rate?**
    A rate of 4.0% represents the yield available on US government short-term treasury bills, serving as a reliable benchmark for risk-free capital.
*   **How: How is it computed/executed?**
    *   **Inputs:** Average daily return $\bar{R} = 0.05\%$, daily volatility $\sigma = 1.0\%$, session count $N = 252$, risk-free rate $R_f = 4.0\%$.
    *   **Calculation:**
        1.  **Annualized Return:** $R_a = 0.05\% \times 252 = 12.60\%$
        2.  **Excess Return:** $R_a - R_f = 12.60\% - 4.0\% = 8.60\%$
        3.  **Annualized Volatility:** $\sigma_a = 1.0\% \times \sqrt{252} = 15.87\%$
        4.  **Sharpe Ratio:** $\text{Sharpe} = 8.60\% / 15.87\% = 0.54$.

---

## 10. Performance: Annualized Sortino Ratio

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need the Sortino Ratio?**
    The Sharpe Ratio penalizes all volatility, including upside volatility (gains). Investors want upside volatility. The Sortino Ratio solves this by only penalizing downside risk (losses).
*   **Why 2: Why do we clip positive return deviations at zero?**
    Clipping positive excess returns at zero removes them from the standard deviation calculation, isolating the variance of losses.
*   **Why 3: Why do we subtract the daily risk-free rate ($R_f / N$) from daily returns before clipping?**
    To isolate downside risk, we must define it relative to our target benchmark. Any return below the daily risk-free rate is considered a downside deviation.
*   **Why 4: Why is it preferred for trading strategies that run long and short?**
    Long/Short strategies often experience large upside price spikes. The Sortino Ratio prevents these profitable swings from skewing the risk metrics.
*   **Why 5: Why does a higher Sortino ratio indicate a safer trading strategy?**
    A high Sortino ratio indicates that the strategy's returns are achieved with minimal large drawdowns or losses.
*   **How: How is it computed/executed?**
    *   **Inputs:** Daily strategy returns series: `[-1.0%, 2.0%, -0.5%, 1.5%]`, $N=252$, $R_f = 4.0\%$.
    *   **Daily Target Return:** $4\% / 252 = 0.01587\%$
    *   **Deviations ($R_t - R_f/N$):** `[-1.016%, 1.984%, -0.516%, 1.484%]`
    *   **Clipped Downside Deviations (clip at 0):** `[-1.016%, 0.0%, -0.516%, 0.0%]`
    *   **Downside Std ($s_d$):** Standard deviation of clipped series = `0.485%`
    *   **Annualized Downside Volatility ($\sigma_d$):** $0.485\% \times \sqrt{252} = 7.70\%$
    *   **Annualized Strategy Return:** Average return ($0.50\%$) $\times 252 = 126.0\%$ (using simplified mean)
    *   **Sortino Ratio:** $(126.0\% - 4.0\%) / 7.70\% = 15.84$.

---

## 11. Performance: Strategy Calmar Ratio

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need the Calmar Ratio?**
    While the Sortino Ratio measures the frequency and size of daily losses, it does not capture the worst cumulative peak-to-trough drop. The Calmar Ratio evaluates return relative to worst-case drawdown risk.
*   **Why 2: Why do we use the absolute maximum drawdown as the denominator?**
    Using the maximum drawdown ($MDD$) represents the "worst-case scenario" risk. Dividing annualized return by $MDD$ measures the reward per unit of maximum tail risk.
*   **Why 3: Why do we use annualized strategy return as the numerator?**
    Using annualized returns standardizes performance over a 1-year horizon, making it easy to compare Calmar ratios across different backtest periods.
*   **Why 4: Why is Calmar highly scrutinized by hedge funds?**
    Hedge funds want to avoid large drawdowns. A strategy with a high Sharpe ratio but a massive drawdown is risky. Calmar flags strategies that carry high tail-risk.
*   **Why 5: Why do we set Calmar to 0.0 if the maximum drawdown is 0.0?**
    If the strategy never experiences a drawdown (e.g. flat cash), the Calmar calculation would divide by zero. We set it to 0.0 to prevent crashes.
*   **How: How is it computed/executed?**
    *   **Inputs:** Annualized Strategy Return = $15.0\%$, Maximum Strategy Drawdown = $10.0\%$.
    *   **Calculation:**
        $$\text{Calmar} = \frac{\text{Annualized Strategy Return}}{\text{Maximum Drawdown}} = \frac{15.0\%}{10.0\%} = 1.5$$
    *   **Interpretation:** The strategy generates 1.5 units of return for every unit of worst-case drawdown risk.

---

## 12. Systematic Risk: OLS Market Beta & Alpha Regression

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need Beta and Alpha?**
    We need to separate how much of an asset's return is driven by the general market (Beta) versus how much is unique to the asset itself (Alpha).
*   **Why 2: Why do we use Ordinary Least Squares (OLS) regression?**
    OLS regression is the standard mathematical method to fit a straight line through a scatter plot of returns, minimizing the sum of squared differences.
*   **Why 3: Why do we estimate coefficients over the full history instead of the event window?**
    Estimating Beta over a short event window is noisy and unstable. Estimating over the full history provides a reliable baseline for systematic risk.
*   **Why 4: Why do we change the benchmark based on the asset class?**
    Equities fluctuate with the stock market (S&P 500), while cryptocurrencies follow Bitcoin. Using the correct benchmark isolates the relevant market risk.
*   **Why 5: Why do we calculate covariance and variance manually?**
    Calculating them manually using Python/Pandas operations avoids importing heavy libraries like SciPy, keeping the application lightweight and memory-efficient.
*   **How: How is it computed/executed?**
    *   **Inputs:** Asset returns $R_a$, benchmark returns $R_m$.
    *   **OLS Formulas:**
        $$\beta = \frac{\text{Covariance}(R_a, R_m)}{\text{Variance}(R_m)}$$
        $$\alpha = \bar{R}_a - \beta \bar{R}_m$$
    *   **Example:** $\text{Cov}(R_a, R_m) = 0.00012$, $\text{Var}(R_m) = 0.00010$, $\bar{R}_a = 0.06\%$, $\bar{R}_m = 0.04\%$.
        $$\beta = \frac{0.00012}{0.00010} = 1.2$$
        $$\alpha = 0.06\% - (1.2 \times 0.04\%) = 0.06\% - 0.048\% = 0.012\%$$
    *   **Interpretation:** The asset is 20% more volatile than the market ($\beta = 1.2$), and generates $0.012\%$ of daily outperformance ($\alpha$).

---

## 13. Lead-Lag Analysis: Pearson Cross-Correlations

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need Lead-Lag analysis?**
    We need to determine if news sentiment predicts price movements (leads) or if news stories merely react to price trends that already occurred (lags).
*   **Why 2: Why do we shift the sentiment series instead of the returns series?**
    Shifting the sentiment series by $k$ days allows us to correlate past news ($k > 0$) or future news ($k < 0$) with today's returns.
*   **Why 3: Why do we restrict the shifts to a window of -3 to +3 days?**
    News sentiment has a short half-life. Price reactions typically occur within 1 to 3 days of an announcement.
*   **Why 4: Why do we use the Pearson correlation coefficient?**
    Pearson correlation measures the strength and direction of a linear relationship between two variables, ranging from -1.0 to +1.0.
*   **Why 5: Why do we fill days with no news with a sentiment score of 0.0?**
    On days with no news, sentiment is neutral. Storing it as 0.0 prevents empty data points (NaNs) from breaking the correlation calculations.
*   **How: How is it computed/executed?**
    *   **Returns ($R_t$):** `[1.0%, -0.5%, 2.0%, 0.0%]`
    *   **Sentiment ($S_t$):** `[0.5, 0.0, -0.2, 0.8]`
    *   **Calculation for Lag $k=+1$:**
        *   Shift Sentiment by 1 day: `[NaN, 0.5, 0.0, -0.2]`
        *   Align with returns: `[-0.5%, 2.0%, 0.0%]` vs `[0.5, 0.0, -0.2]`
        *   Pearson correlation is computed over these aligned pairs.

---

## 14. Event Studies: Cumulative Abnormal Returns (CAR) & p-Value Significance

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need Abnormal Returns ($AR$)?**
    General market movements distort price reactions. If an asset rises 3% on a day when the market rises 3%, the abnormal price reaction is zero. We must isolate the unique impact of the news.
*   **Why 2: Why do we sum abnormal returns over the event window to calculate CAR?**
    News events can trigger gradual price adjustments (drift). Summing returns over the window captures the cumulative price reaction before and after the event.
*   **Why 3: Why do we rebase the event day ($t=0$) return to exactly 0.0%?**
    Rebasing forces the chart to display the cumulative change relative to the event day, making it easy to compare bullish and bearish paths.
*   **Why 4: Why do we calculate a p-value at the $t = +5$ day horizon?**
    We need to verify if the post-event price trend is statistically significant or merely random noise. A 5-day window captures the short-term impact.
*   **Why 5: Why do we use a logistic normal CDF approximation?**
    Calculating p-values requires standard normal cumulative distribution functions. Using a high-precision logistic approximation:
    $$\Phi(|t|) \approx \frac{1}{1 + e^{-y}} \quad \text{where } y = |t| \times (1.5976 + 0.07056 \times |t|^2)$$
    gives an absolute error $< 0.0002$, avoiding importing heavy libraries like SciPy.
*   **How: How is it computed/executed?**
    *   **Inputs:** Average CAR at day +5 = $1.50\%$, sample standard deviation $s = 3.0\%$, count $n = 36$ events.
    *   **Calculation:**
        1.  **Standard Error:** $SE = s / \sqrt{n} = 3.0\% / 6 = 0.50\%$
        2.  **t-statistic:** $t = 1.50\% / 0.50\% = 3.0$
        3.  **Logistic y:** $y = 3.0 \times (1.5976 + 0.07056 \times 9) = 3.0 \times (1.5976 + 0.63504) = 6.69792$
        4.  **CDF $\Phi(3.0)$:** $1 / (1 + e^{-6.69792}) = 1 / (1 + 0.00123) = 0.99877$
        5.  **p-value:** $2 \times (1 - 0.99877) = 0.00246$ ($0.25\%$)
    *   **Interpretation:** The p-value ($0.25\%$) is $< 5.0\%$. The post-event trend is highly significant.

---

## 15. Strategy Simulation: Friction-Adjusted Backtester

### The "5 Whys" & "1 How" Framework

*   **Why 1: Why do we need backtest simulation?**
    A trading signal is useless unless we can simulate how it performs over time. Backtesting evaluates the strategy's historical performance.
*   **Why 2: Why do we lag signals by 1 trading day ($Signal_{t-1}$)?**
    We cannot execute a trade before the news is published. Lagging by 1 day ensures the trade executes at the market close *after* news sentiment is parsed, preventing look-ahead bias.
*   **Why 3: Why do we implement a 5-day time-decay exit?**
    News sentiment has a decaying impact on prices. If no news occurs for 5 consecutive days, the strategy liquidates the position to cash to manage holding risk.
*   **Why 4: Why do we deduct a 0.15% transaction friction penalty?**
    Trading incurs broker commissions and bid-ask spread slippage. Ignoring these costs makes backtests look overly profitable. Deducting 0.15% on position changes ensures realistic results.
*   **Why 5: Why do we calculate compounding equity curves starting at 100?**
    Compounding returns reflect the reinvestment of profits. Starting at 100 standardizes the equity curve, making it easy to compare strategy growth relative to a buy-and-hold index.
*   **How: How is it computed/executed?**
    *   **Signal Changes:**
        *   Day 1: Signal changes from `0` to `1` (Long position entered). Turnover = $|1 - 0| = 1$.
        *   Friction Cost: $1 \times 0.15\% = 0.15\%$.
        *   Day 2: Signal remains `1`. Turnover = $|1 - 1| = 0$. Friction Cost = $0\%$.
        *   Day 3: Signal changes from `1` to `-1` (Short position entered). Turnover = $|-1 - 1| = 2$.
        *   Friction Cost: $2 \times 0.15\% = 0.30\%$.
    *   **Strategy Return:**
        *   Day 1: Return = $1.2\%$. Net Strategy Return = $(1 \times 1.2\%) - 0.15\% = 1.05\%$.
    *   **Compounding:**
        *   Equity Day 1: $100 \times (1 + 0.0105) = 101.05$.
