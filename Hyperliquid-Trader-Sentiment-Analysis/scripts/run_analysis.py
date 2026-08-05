"""Run: .venv\\Scripts\\python.exe scripts/run_analysis.py"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "figures"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_MAP = {name: i for i, name in enumerate(SENTIMENT_ORDER)}
LONG_DIRS = {"Open Long", "Close Long", "Long > Short", "Buy"}
SHORT_DIRS = {"Open Short", "Close Short", "Short > Long", "Sell"}
SIZE_BINS = [0, 500, 2000, 10000, 50000, np.inf]
SIZE_LABELS = ["<500", "500-2k", "2k-10k", "10k-50k", "50k+"]


def _trade_date(ts: str) -> str | None:
    for fmt in ("%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S"):
        try:
            return datetime.strptime(ts.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _position_side(direction: str) -> str:
    if direction in LONG_DIRS:
        return "Long"
    if direction in SHORT_DIRS:
        return "Short"
    return "Other"


def load_sentiment(path: Path) -> dict:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["classification"].strip().title()
            out[row["date"].strip()] = {
                "Classification": cls,
                "sentiment_score": SENTIMENT_MAP.get(cls),
            }
    return out


def load_trades(path: Path, sentiment: dict) -> tuple[list[dict], dict]:
    rows, accounts, coins = [], set(), set()
    unmatched, dmin, dmax, n_raw = 0, None, None, 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_raw += 1
            d = _trade_date(row["Timestamp IST"])
            if not d:
                continue
            dmin, dmax = (d, d) if dmin is None else (min(dmin, d), max(dmax, d))
            accounts.add(row["Account"])
            coins.add(row["Coin"])
            try:
                pnl, size = float(row["Closed PnL"]), float(row["Size USD"])
            except ValueError:
                continue
            sent = sentiment.get(d)
            if not sent:
                unmatched += 1
            cls = sent["Classification"] if sent else None
            score = sent["sentiment_score"] if sent else None
            rows.append(
                {
                    "account": row["Account"],
                    "date": d,
                    "Classification": cls,
                    "sentiment_score": score,
                    "closedPnL": pnl,
                    "size_usd": size,
                    "position_side": _position_side(row["Direction"]),
                    "win": pnl > 0,
                }
            )
    meta = {
        "trade_rows": n_raw,
        "unique_traders": len(accounts),
        "unique_coins": len(coins),
        "trade_date_range": [dmin, dmax],
        "trades_missing_sentiment": unmatched,
    }
    return rows, meta


def _by_sentiment(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for r in rows:
        if r["Classification"]:
            buckets[r["Classification"]].append(r)
    stats = []
    for cls in SENTIMENT_ORDER:
        b = buckets.get(cls)
        if not b:
            continue
        pnls = [x["closedPnL"] for x in b]
        sizes = [x["size_usd"] for x in b]
        stats.append(
            {
                "Classification": cls,
                "trades": len(b),
                "total_pnl": sum(pnls),
                "avg_pnl": float(np.mean(pnls)),
                "win_rate": float(np.mean([x["win"] for x in b])),
                "avg_size_usd": float(np.mean(sizes)),
                "total_volume_usd": sum(sizes),
            }
        )
    return stats


def _trader_summary(rows: list[dict], min_trades: int = 30) -> tuple[list[dict], list[dict]]:
    by_acc = defaultdict(list)
    for r in rows:
        by_acc[r["account"]].append(r)
    summaries = []
    for acc, trs in by_acc.items():
        pnls = np.array([t["closedPnL"] for t in trs])
        std = float(pnls.std(ddof=1)) if len(trs) > 1 else 0.0
        mean = float(pnls.mean())
        summaries.append(
            {
                "account": acc,
                "total_pnl": float(pnls.sum()),
                "avg_pnl": mean,
                "trades": len(trs),
                "wins": int((pnls > 0).sum()),
                "win_rate": float((pnls > 0).mean()),
                "pnl_std": std,
                "avg_size_usd": float(np.mean([t["size_usd"] for t in trs])),
            }
        )
    summaries.sort(key=lambda x: x["total_pnl"], reverse=True)
    reliable = [s for s in summaries if s["trades"] >= min_trades]
    return summaries, reliable


def _save_bar(labels, values, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="steelblue")
    ax.set_title(title)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sentiment = load_sentiment(DATA / "fear_greed_index.csv")
    rows, meta = load_trades(DATA / "historical_data.csv", sentiment)
    meta["trades_with_sentiment_match"] = meta["trade_rows"] - meta["trades_missing_sentiment"]
    meta["note_no_leverage_column"] = True

    sent_perf = _by_sentiment(rows)
    summaries, reliable = _trader_summary(rows)

    # Size buckets
    bucket_idx = np.digitize([r["size_usd"] for r in rows], SIZE_BINS[1:-1])
    size_analysis = []
    for i, label in enumerate(SIZE_LABELS):
        subset = [rows[j] for j in range(len(rows)) if bucket_idx[j] == i]
        if not subset:
            continue
        size_analysis.append(
            {
                "bucket": label,
                "trades": len(subset),
                "avg_pnl": float(np.mean([x["closedPnL"] for x in subset])),
                "win_rate": float(np.mean([x["win"] for x in subset])),
            }
        )

    matched = [r for r in rows if r["sentiment_score"] is not None]
    scores = np.array([r["sentiment_score"] for r in matched])
    pnls = np.array([r["closedPnL"] for r in matched])
    sizes = np.array([r["size_usd"] for r in matched])

    fear = [r["closedPnL"] for r in rows if r["Classification"] == "Fear"]
    greed = [r["closedPnL"] for r in rows if r["Classification"] == "Greed"]
    mw_stat, mw_p = mannwhitneyu(fear, greed, alternative="two-sided")

    fields = [
        "account", "total_pnl", "avg_pnl", "trades", "wins",
        "win_rate", "pnl_std", "avg_size_usd",
    ]
    with (OUT / "trader_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: s[k] for k in fields} for s in summaries)

    labels = [x["Classification"] for x in sent_perf]
    for metric, title, fname in [
        ("avg_pnl", "Average closed PnL by sentiment", "avg_pnl_by_sentiment.png"),
        ("win_rate", "Win rate by sentiment", "win_rate_by_sentiment.png"),
        ("avg_size_usd", "Average size USD by sentiment", "avg_size_by_sentiment.png"),
    ]:
        _save_bar(labels, [x[metric] for x in sent_perf], title, fname)

    # Long/short stacked
    counts = {c: defaultdict(int) for c in SENTIMENT_ORDER}
    for r in rows:
        if r["Classification"]:
            counts[r["Classification"]][r["position_side"]] += 1
    long_v = [counts[c]["Long"] for c in labels]
    short_v = [counts[c]["Short"] for c in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, long_v, label="Long")
    ax.bar(x, short_v, bottom=long_v, label="Short")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.set_title("Long vs short-related trades by sentiment")
    fig.tight_layout()
    fig.savefig(FIG / "long_short_by_sentiment.png", dpi=150)
    plt.close(fig)

    findings = {
        "dataset": meta,
        "sentiment_performance": sent_perf,
        "size_usd_bucket_analysis": size_analysis,
        "mann_whitney_fear_vs_greed_pnl": {"statistic": float(mw_stat), "p_value": float(mw_p)},
        "correlation_sentiment_score_vs_pnl": float(np.corrcoef(scores, pnls)[0, 1]),
        "correlation_sentiment_score_vs_size_usd": float(np.corrcoef(scores, sizes)[0, 1]),
        "top5_traders": summaries[:5],
        "bottom5_traders_min30trades": sorted(reliable, key=lambda x: x["total_pnl"])[:5],
    }
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
