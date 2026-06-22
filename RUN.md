# Running the Macro-Economic Sentiment Terminal

This guide outlines how to run the project locally or deploy it to a production hosting service.

---

## 💻 Option 1: Running Locally (Windows, macOS, Linux)

### 1. Clone the Repository
Open a terminal and run:
```bash
git clone https://github.com/worstcod/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create and Activate a Virtual Environment (Recommended)
This ensures dependencies do not conflict with other Python projects on your system.
* **On Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application
Run the Flask server:
```bash
python app.py
```

### 5. Access the Dashboard
Once the console prints that the server is active, open a web browser and navigate to:
👉 **`http://localhost:5000`**

---

## 🌐 Option 2: Deploying to Render.com (100% Free)

The application is fully optimized to run within Render's free tier web service (which enforces a strict **1GB RAM limit**).

### Configuration Steps:
1. Create a free account on [Render.com](https://render.com) and link your GitHub account.
2. Click **New +** > **Web Service** in the dashboard.
3. Select the **`worstcod/Sentiment-Analysis`** repository.
4. Input the following configuration details:
   * **Runtime:** `Python`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app`
5. Click **Deploy Web Service**.

### ⚡ RAM Optimizations Included:
* **Global Model Initialization:** Hugging Face FinBERT is loaded into memory exactly once at startup, preventing OOM crashes from multiple worker threads.
* **Thread Throttling:** PyTorch is limited to a single CPU thread (`torch.set_num_threads(1)`) to minimize memory usage.
* **Automated Lexicon Fallback:** If the system detects OOM conditions or Hugging Face hub timeout errors, it automatically falls back to a high-speed local lexicon rule engine.

---

## 📖 Operational Guide

1. **Asset Selection:** Select a preset stock, crypto, index, or commodity (or enter a custom ticker like `AAPL` or `BTC-USD`). To compare multiple assets, type them separated by commas (e.g. `AAPL,MSFT,NVDA`).
2. **Event Keyword:** Search for news stories using a keyword (e.g. `inflation`). To search multiple keywords, use a plus sign (`+`) for OR logic (e.g. `inflation+interest`).
3. **Date Range:** Select start and end dates.
4. **Analysis:** Click **Generate Analysis** to load the charts. Toggle between the **📊 Market Overview** and **⚡ Pro Quant Terminal** tabs to analyze price movements, event studies, lead-lag correlations, and strategy backtest equity curves.
