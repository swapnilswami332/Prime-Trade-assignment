# Hyperliquid Trader Sentiment Analysis

## Objective

Investigate whether **Crypto Fear & Greed** sentiment is associated with **Hyperliquid trader behavior** (activity, direction, position size) and **closed trade profitability** in an observational sample of on-chain accounts.

**Research question:** Does market sentiment relate to how these traders trade, and to their per-trade closed PnL?

## Dataset

| File | Description |
|------|-------------|
| `data/historical_data.csv` | ~211k Hyperliquid fills for **32 unique accounts**, May 2023 – May 2025, **246 coins** |
| `data/fear_greed_index.csv` | Daily Fear & Greed index (`classification`, numeric `value`) |

**Schema notes (actual columns, not the generic assignment template):**

- Trades: `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`, `Side`, `Timestamp IST`, `Direction`, `Closed PnL`, `Fee`, …
- **No `leverage` column** in this extract — risk is proxied with **`Size USD`** buckets and PnL volatility.
- Long/short behavior uses **`Direction`** (`Open Long`, `Close Short`, etc.), not `Side` alone (`BUY`/`SELL`).

Original CSVs are kept unmodified under `data/`.

## Methodology

1. Parse trade dates from `Timestamp IST` (day-level merge key).
2. Map sentiment `classification` → ordered `sentiment_score` (Extreme Fear = 0 … Extreme Greed = 4).
3. Left-join sentiment onto each trade by calendar date.
4. Trader metrics: total/avg closed PnL, win rate, PnL std, min 30 trades for “reliable” rankings.
5. Regime analysis: PnL, win rate, volume, avg size, long vs short mix by sentiment.
6. **Mann–Whitney U** test: closed PnL in **Fear** vs **Greed** days (trade-level; not independent — see limitations).
7. Visualizations saved under `outputs/figures/`.

Reproducible batch run (no Jupyter required):

```powershell
cd Hyperliquid-Trader-Sentiment-Analysis
..\.venv\Scripts\python.exe scripts\run_analysis.py
```

Interactive analysis: `notebooks/analysis.ipynb`.

## Key findings (from this data)

1. **Coverage:** 211,218 / 211,224 trades match a sentiment day; 6 trades fall on dates without index rows.

2. **Profitability by sentiment (per trade, association only):**
   - Highest **average closed PnL** and **win rate** occur in **Extreme Greed** (~$68/trade, ~46.5% wins).
   - **Fear** shows the next-strongest average (~$54/trade) and highest **aggregate** volume.
   - **Extreme Fear** has the weakest win rate (~37%) and lower average PnL (~$35/trade).
   - Do **not** interpret this as “greed causes profits” — see limitations.

3. **Position sizing:** Average **Size USD** is **largest in Fear** (~$7.8k) and **smallest in Extreme Greed** (~$3.1k). Sentiment score vs size USD correlation ≈ **−0.032** (weak).

4. **Direction mix:** In **Fear / Extreme Fear / Neutral**, **long-related** directions dominate (~61–66% of trades). In **Greed**, long share drops to **~42%** with more short-related activity — a clear behavioral shift, not a proof of causality.

5. **Size vs outcomes:** Larger **Size USD** buckets show **higher average closed PnL** but **mixed win rates** (e.g. 10k–50k bucket ~37% win rate vs smaller buckets ~41–42%).

6. **Traders:** Only **32 accounts** — results describe **this cohort**, not all Hyperliquid users. Top account by total closed PnL: `0xb1231a4a…` (~$2.14M); worst among ≥30 trades: `0x8170715b…` (~−$168k).

7. **Statistics:** Fear vs Greed closed PnL distributions differ (**p ≈ 1.3×10⁻⁶⁸**, Mann–Whitney). Trade-level sentiment vs PnL correlation ≈ **0.006** — regime and direction stories matter more than a single linear correlation.

## Trading insights (actionable hypotheses)

- **Regime-aware sizing:** When average size is elevated (e.g. Fear) but win rate is lower in some buckets, consider **tighter size caps** rather than leaning on sentiment alone.
- **Direction monitoring:** If internal flow shows **crowded long exposure in fear** and **more shorts in greed**, contrarian or hedging rules may be worth **backtesting** — not validated here.
- **Risk flags:** Combine **extreme sentiment + large Size USD + high PnL volatility** for surveillance (similar to a leverage alert where leverage is unavailable).
- **Cohort analytics:** With few accounts, **per-trader** dashboards beat market-wide claims.

## Limitations

- Observational data; **no causal** claims.
- **Closed PnL** per fill ≠ full portfolio performance (fees partial; funding/slippage unclear).
- **Daily sentiment** vs **intraday** trades.
- **Repeated trades / same account** → non-independent observations for inferential tests.
- **Sample selection:** 32 traders — generalization limited.
- **No leverage** in file; size is an imperfect risk proxy.

## How to run

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

Or: `scripts/run_analysis.py` → `outputs/findings.json`, `outputs/trader_summary.csv`, `outputs/figures/*.png`.
