# 📈 Macro-Economic Sentiment Terminal

An interactive, light-weight quantitative dashboard that maps global news headlines and their AI-calculated sentiment directly onto asset price action. 

Built using a **Flask Backend** and a custom **Vanilla HTML/JS Frontend** styled with **TailwindCSS** and plotted using **Plotly.js**. Optimized for low-memory hosting platforms like Render's free tier (1GB RAM limits).

---

## 🌟 Key Features

- **Custom UI Layout:** Fully structured HTML/JS interface without the overhead of Streamlit or Dash.
- **Universal Ticker Lookup:** Supports any asset class on Yahoo Finance (Stocks, Cryptocurrencies, Commodities, Forex) via ticker input (e.g., `AAPL`, `BTC-USD`, `BZ=F`, `EURUSD=X`).
- **Keyword-Based News Ingestion:** Searches breaking headlines matching custom event keywords. Supports multiple keywords separated by a plus sign (`+`) using search `OR` query logic (e.g., `Inflation+Interest Rate`).
- **Multi-Asset Comparison & Rebasing:** Compare multiple asset prices on a single rebased chart (starting at 100) and view a dynamically calculated Pearson Daily Return Correlation Matrix.
- **100% Free - No API Keys:** 
  - Market price data is fetched via `yfinance`.
  - News headlines are parsed via `feedparser` querying Google News RSS.
- **⚡ Pro Quant Terminal Workspace:**
  * **Annualized Sharpe Ratio:** Measure risk-adjusted returns net of a $4.0\%$ risk-free rate.
  * **Systematic Beta & Alpha Regression:** OLS linear regression of asset returns against benchmark indexes to isolate systematic risk exposures.
  * **Lead-Lag Cross-Correlation:** Pearson correlations shifted from $-3$ to $+3$ days to determine if news sentiment leads price action or merely reacts to it.
  * **Event Study abnormal trajectories:** Tracks Cumulative Abnormal Returns (CAR) paths net of market index returns around announcements.
  * **Statistical Significance p-values:** Calculates two-tailed p-values for CAR at $t = +5$ days using sample standard error and a high-precision normal CDF logistic approximation (significance at $p < 0.05$ marked with an asterisk `*`).
  * **Friction-Adjusted Backtest Simulator:** Evaluates a systematic sentiment trading model featuring a 1-day causality lag execution, 5-day decay exit-to-cash, and 0.15% turnover transaction costs (0.10% commission + 0.05% slippage).
- **RAM Optimization:** FinBERT (`ProsusAI/finbert`) is loaded into memory exactly once at server startup (globally) to protect against memory limit violations on 1GB RAM servers.
- **Lexicon Fallback Engine:** Features an automatic try-except wrapper. If the Hugging Face hub is rate-limited or fails to load, it falls back to a high-speed local rules-based sentiment pipeline.
- **Plotly.js Chart Overlay:** Interactive line charts displaying close prices with sentiment-categorized news scatter marks (Bullish/Bearish/Neutral) and rolling SMA-20 technical support line.
- **Scrollable News Feed:** Scrollable cards listing dates, sources, badges, and summaries.
- **User & Operations Guide:** A detailed quantitative guide hosted locally at `/guide` explaining all calculations and math.

---

## 🛠️ Tech Stack

- **Backend:** Flask, Flask-CORS, yfinance, feedparser, pandas, transformers, torch, gunicorn
- **Frontend:** HTML5, TailwindCSS (via CDN), Plotly.js (via CDN)

---

## 🚀 Local Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/macro-sentiment-terminal.git
cd macro-sentiment-terminal
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
Launch the Flask development server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 🌐 Deployment to Render.com

This project is fully compatible with **Render.com**'s free web service tier.

1. **Create a Web Service** on Render connected to your GitHub repository.
2. **Environment settings:**
   - **Runtime:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
3. **Optimizing Memory:** The start command automatically uses Gunicorn to run the Flask application. PyTorch has been set to single-threaded mode (`torch.set_num_threads(1)`) inside `app.py` to prevent memory thrashing and stay strictly within the 1GB RAM limit.
