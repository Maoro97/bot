"""
Web dashboard for the Polymarket arbitrage bot.
Shows live stats, recent opportunities, trade history, and P&L.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from flask import Flask, render_template_string

from utils.logger import logger

# Shared state — updated by the bot, read by the dashboard
_state = {
    "scans": 0,
    "opportunities_found": 0,
    "trades_attempted": 0,
    "trades_succeeded": 0,
    "estimated_profit": 0.0,
    "markets_loaded": 0,
    "dry_run": True,
    "started_at": "",
    "recent_opportunities": [],  # last 50
    "recent_trades": [],         # last 50
    "errors": [],                # last 20
}
_MAX_RECENT = 50
_MAX_ERRORS = 20

app = Flask(__name__)


def update_stats(stats) -> None:
    _state["scans"] = stats.scans
    _state["opportunities_found"] = stats.opportunities_found
    _state["trades_attempted"] = stats.trades_attempted
    _state["trades_succeeded"] = stats.trades_succeeded
    _state["estimated_profit"] = stats.estimated_profit_usdc


def add_opportunity(opp) -> None:
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "market": opp.condition_id[:20] + "...",
        "direction": opp.direction.value,
        "yes": f"${opp.yes_price:.4f}",
        "no": f"${opp.no_price:.4f}",
        "net_profit": f"${opp.net_profit:.4f}",
        "pct": f"{opp.net_profit*100:.2f}%",
        "size": f"${opp.max_size_usdc:.2f}",
    }
    _state["recent_opportunities"].insert(0, entry)
    _state["recent_opportunities"] = _state["recent_opportunities"][:_MAX_RECENT]


def add_trade(result) -> None:
    opp = result.opportunity
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "market": opp.condition_id[:20] + "...",
        "direction": opp.direction.value,
        "success": result.success,
        "yes_order": result.yes_order_id or "DRY RUN",
        "no_order": result.no_order_id or "DRY RUN",
        "profit": f"${opp.net_profit * opp.max_size_usdc:.4f}",
        "error": result.error or "",
    }
    _state["recent_trades"].insert(0, entry)
    _state["recent_trades"] = _state["recent_trades"][:_MAX_RECENT]


def add_error(msg: str) -> None:
    _state["errors"].insert(0, {"time": time.strftime("%H:%M:%S"), "msg": msg})
    _state["errors"] = _state["errors"][:_MAX_ERRORS]


def set_meta(dry_run: bool, markets: int) -> None:
    _state["dry_run"] = dry_run
    _state["markets_loaded"] = markets
    _state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="10">
<title>Polymarket Arb Bot</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background:#0d1117; color:#e6edf3; padding:20px; }
  .header { display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; }
  .header h1 { font-size:24px; }
  .badge { padding:4px 12px; border-radius:12px; font-size:13px; font-weight:600; }
  .badge-dry { background:#f0883e; color:#000; }
  .badge-live { background:#3fb950; color:#000; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:16px; margin-bottom:24px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }
  .card .label { font-size:12px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; }
  .card .value { font-size:28px; font-weight:700; margin-top:4px; }
  .card .value.green { color:#3fb950; }
  .card .value.blue { color:#58a6ff; }
  .card .value.orange { color:#f0883e; }
  table { width:100%; border-collapse:collapse; margin-bottom:24px; }
  th { text-align:left; padding:8px 12px; font-size:12px; color:#8b949e; text-transform:uppercase;
       border-bottom:1px solid #30363d; letter-spacing:1px; }
  td { padding:8px 12px; border-bottom:1px solid #21262d; font-size:14px; }
  .section { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:24px; }
  .section h2 { font-size:16px; margin-bottom:12px; color:#58a6ff; }
  .success { color:#3fb950; }
  .fail { color:#f85149; }
  .empty { color:#8b949e; text-align:center; padding:20px; }
  .footer { text-align:center; font-size:12px; color:#484f58; margin-top:20px; }
</style>
</head>
<body>
<div class="header">
  <h1>Polymarket Arbitrage Bot</h1>
  <div>
    <span class="badge {{ 'badge-dry' if state.dry_run else 'badge-live' }}">
      {{ 'DRY RUN' if state.dry_run else 'LIVE' }}
    </span>
  </div>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Scans</div>
    <div class="value blue">{{ state.scans }}</div>
  </div>
  <div class="card">
    <div class="label">Markets</div>
    <div class="value blue">{{ state.markets_loaded }}</div>
  </div>
  <div class="card">
    <div class="label">Opportunities</div>
    <div class="value orange">{{ state.opportunities_found }}</div>
  </div>
  <div class="card">
    <div class="label">Trades</div>
    <div class="value">{{ state.trades_succeeded }}/{{ state.trades_attempted }}</div>
  </div>
  <div class="card">
    <div class="label">Est. Profit</div>
    <div class="value green">${{ "%.4f"|format(state.estimated_profit) }}</div>
  </div>
</div>

<div class="section">
  <h2>Recent Opportunities</h2>
  {% if state.recent_opportunities %}
  <table>
    <tr><th>Time</th><th>Market</th><th>Direction</th><th>YES</th><th>NO</th><th>Profit</th><th>%</th><th>Size</th></tr>
    {% for o in state.recent_opportunities %}
    <tr>
      <td>{{ o.time }}</td><td>{{ o.market }}</td><td>{{ o.direction }}</td>
      <td>{{ o.yes }}</td><td>{{ o.no }}</td>
      <td class="success">{{ o.net_profit }}</td><td>{{ o.pct }}</td><td>{{ o.size }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <div class="empty">No opportunities detected yet</div>
  {% endif %}
</div>

<div class="section">
  <h2>Recent Trades</h2>
  {% if state.recent_trades %}
  <table>
    <tr><th>Time</th><th>Market</th><th>Direction</th><th>Status</th><th>YES Order</th><th>NO Order</th><th>Profit</th></tr>
    {% for t in state.recent_trades %}
    <tr>
      <td>{{ t.time }}</td><td>{{ t.market }}</td><td>{{ t.direction }}</td>
      <td class="{{ 'success' if t.success else 'fail' }}">{{ 'OK' if t.success else 'FAIL' }}</td>
      <td><code>{{ t.yes_order }}</code></td><td><code>{{ t.no_order }}</code></td>
      <td class="success">{{ t.profit }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <div class="empty">No trades executed yet</div>
  {% endif %}
</div>

{% if state.errors %}
<div class="section">
  <h2>Recent Errors</h2>
  <table>
    <tr><th>Time</th><th>Error</th></tr>
    {% for e in state.errors %}
    <tr><td>{{ e.time }}</td><td class="fail">{{ e.msg }}</td></tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<div class="footer">
  Started: {{ state.started_at }} | Auto-refreshes every 10s
</div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE, state=_state)


@app.route("/api/stats")
def api_stats():
    return _state


def start_dashboard(port: int = 5000) -> None:
    """Run Flask in a background daemon thread."""
    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info("Dashboard running on http://0.0.0.0:{}", port)
