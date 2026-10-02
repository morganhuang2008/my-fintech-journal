"""交易紀錄追蹤與復盤檢討平台 — Streamlit single-file app."""
from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, date, time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from ta.momentum import StochasticOscillator
from ta.trend import MACD


st.set_page_config(page_title="Trade Journal | 台股復盤", page_icon="📈", layout="wide")

FEE_RATE = 0.001425
TAX_RATE = 0.003
DEFAULT_DISCOUNT = 0.60
TRADE_COLUMNS = [
    "id", "ticker", "trade_datetime", "side", "price", "shares", "logic", "psychology",
    "note", "fee", "tax", "created_at",
]


def money(value: float) -> str:
    return f"NT${value:,.0f}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def fee_for(price: float, shares: int, discount: float) -> float:
    """Taiwan broker fee: each order has a NT$20 floor."""
    return float(max(20, round(price * shares * FEE_RATE * discount)))


def blank_trade(ticker: str, day: str, side: str, price: float, shares: int,
                logic: str, psychology: str, note: str, discount: float) -> dict:
    fee = fee_for(price, shares, discount)
    return {
        "id": str(uuid.uuid4()), "ticker": ticker, "trade_datetime": day,
        "side": side, "price": price, "shares": shares, "logic": logic,
        "psychology": psychology, "note": note, "fee": fee,
        "tax": round(price * shares * TAX_RATE) if side == "賣出" else 0.0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def init_state() -> None:
    if "trades" in st.session_state:
        return
    d = DEFAULT_DISCOUNT
    st.session_state.discount = d
    st.session_state.watchlist = {"2330.TW": "台積電", "2454.TW": "聯發科"}
    st.session_state.trades = [
        blank_trade("2330.TW", "2025-01-06 09:15", "買入", 1060, 1000, "技術面突破", "冷靜執行", "突破月線後分批進場，成交量同步放大。", d),
        blank_trade("2330.TW", "2025-02-18 10:05", "買入", 1095, 500, "籌碼面", "冷靜執行", "外資連三日買超，按原訂計畫加碼。", d),
        blank_trade("2330.TW", "2025-03-10 13:20", "賣出", 1160, 800, "技術面突破", "紀律停利", "達到第一段目標價，先落袋並保留部位。", d),
        blank_trade("2454.TW", "2025-01-14 09:40", "買入", 1380, 300, "消息面", "FOMO追高", "財報題材強勢，但進場位置偏高，需檢討追價。", d),
        blank_trade("2454.TW", "2025-02-03 11:10", "買入", 1320, 300, "技術面突破", "冷靜執行", "回測支撐止穩後補倉，設定明確停損。", d),
        blank_trade("2454.TW", "2025-03-04 10:30", "賣出", 1415, 300, "籌碼面", "紀律停利", "反彈至壓力區減碼，執行良好。", d),
    ]


def trades_df(ticker: str | None = None) -> pd.DataFrame:
    data = pd.DataFrame(st.session_state.trades)
    if data.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    data["trade_datetime"] = pd.to_datetime(data["trade_datetime"])
    for c in ["price", "shares", "fee", "tax"]:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    if ticker:
        data = data[data.ticker == ticker]
    return data.sort_values(["trade_datetime", "created_at"], kind="stable").reset_index(drop=True)


def fifo_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Allocate each sell to the oldest open buys. Returns sells, open lots, matched lots."""
    lots, matches, sell_rows = [], [], []
    for _, row in df.iterrows():
        r = row.to_dict()
        if r["side"] == "買入":
            r["remaining"] = float(r["shares"])
            lots.append(r)
            continue
        left, pnl, held_weight, matched = float(r["shares"]), 0.0, 0.0, 0.0
        sell_cost_per_share = (float(r["fee"]) + float(r["tax"])) / max(float(r["shares"]), 1)
        for lot in lots:
            if left <= 0:
                break
            qty = min(left, lot["remaining"])
            buy_cost_per_share = (lot["price"] * lot["shares"] + lot["fee"]) / max(lot["shares"], 1)
            piece = qty * (r["price"] - sell_cost_per_share - buy_cost_per_share)
            days = max(0, (r["trade_datetime"] - lot["trade_datetime"]).total_seconds() / 86400)
            pnl += piece
            held_weight += days * qty
            matched += qty
            matches.append({"ticker": r["ticker"], "qty": qty, "pnl": piece, "holding_days": days,
                            "logic": lot["logic"], "psychology": lot["psychology"], "sell_id": r["id"]})
            lot["remaining"] -= qty
            left -= qty
        # Oversold entries remain visible rather than inventing an unknown historical cost basis.
        sell_rows.append({**r, "realized_pnl": pnl, "matched_qty": matched,
                          "holding_days": held_weight / matched if matched else 0.0,
                          "unmatched_qty": left})
    return pd.DataFrame(sell_rows), [x for x in lots if x["remaining"] > 0], matches


@st.cache_data(ttl=900, show_spinner=False)
def market_data(ticker: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        return raw
    except Exception:
        return pd.DataFrame()


def style() -> None:
    st.markdown("""<style>
    .stApp {background: radial-gradient(circle at 15% 0%, #15274b 0%, #070b16 42%, #04060c 100%); color:#ecf5ff;}
    [data-testid="stSidebar"] {background:#0a1020; border-right:1px solid #243a60;}
    .hero {padding:2.2rem; border:1px solid #315d9c; border-radius:20px; background:linear-gradient(115deg,#0e1c37,#101224 65%,#132d3a); box-shadow:0 0 30px #1261a633;}
    .hero h1 {margin:0;color:#f3f8ff; font-size:2.45rem;} .hero p {color:#9bb7d9; font-size:1.08rem;}
    [data-testid="stMetric"] {background:#0d1628; border:1px solid #253f67; border-radius:14px; padding:14px;}
    .ticker-card {padding:1rem; border:1px solid #275180; border-radius:13px; background:#0c1628; min-height:106px;}
    .hint {color:#ffd77a; padding:.7rem; border-left:3px solid #f4a900; background:#251d0b; border-radius:6px;}
    </style>""", unsafe_allow_html=True)


def csv_bytes() -> bytes:
    frame = trades_df()
    frame["metadata_json"] = json.dumps({"watchlist": st.session_state.watchlist, "discount": st.session_state.discount}, ensure_ascii=False)
    return frame.to_csv(index=False).encode("utf-8-sig")


def backup_box(prefix: str) -> None:
    st.markdown('<div class="hint">提醒：系統異動後請記得點擊儲存，下載 CSV 備份您的交易與檢討紀錄！</div>', unsafe_allow_html=True)
    st.download_button("💾 儲存與匯出資料 (CSV)", csv_bytes(), "trade_journal_backup.csv", "text/csv", key=f"download_{prefix}", use_container_width=True)
    uploaded = st.file_uploader("📥 匯入歷史備份 (CSV)", type="csv", key=f"upload_{prefix}")
    if uploaded is not None:
        try:
            restored = pd.read_csv(uploaded)
            missing = set(TRADE_COLUMNS) - set(restored.columns)
            if missing:
                st.error("備份檔缺少欄位：" + ", ".join(sorted(missing)))
            else:
                st.session_state.trades = restored[TRADE_COLUMNS].fillna("").to_dict("records")
                if "metadata_json" in restored and restored.metadata_json.notna().any():
                    meta = json.loads(restored.metadata_json.dropna().iloc[0])
                    st.session_state.watchlist = meta.get("watchlist", st.session_state.watchlist)
                    st.session_state.discount = float(meta.get("discount", DEFAULT_DISCOUNT))
                for ticker in restored.ticker.unique():
                    st.session_state.watchlist.setdefault(ticker, ticker)
                st.success("備份已還原，畫面將更新。")
                st.rerun()
        except Exception as exc:
            st.error(f"無法匯入備份：{exc}")


def latest_price(ticker: str, data: pd.DataFrame | None = None) -> float | None:
    data = market_data(ticker) if data is None else data
    return float(data["Close"].iloc[-1]) if not data.empty else None


def page_explore() -> None:
    st.markdown('<div class="hero"><h1>PORTFOLIO // JOURNAL</h1><p>把每一次進出場，轉化為下一次更穩健的決策。</p></div>', unsafe_allow_html=True)
    st.write("")
    query = st.text_input("搜尋台股標的（yfinance 格式）", placeholder="例如：2330.TW")
    if query:
        symbol = query.upper().strip()
        if st.button(f"加入 {symbol} 至自選清單"):
            st.session_state.watchlist.setdefault(symbol, symbol)
            st.success(f"已加入 {symbol}")
    st.subheader("熱門標的排行榜")
    popular = [("2330.TW", "台積電", "半導體製造"), ("2454.TW", "聯發科", "IC 設計"), ("2317.TW", "鴻海", "電子代工"), ("2308.TW", "台達電", "電源與 AI"), ("2881.TW", "富邦金", "金融")]
    cols = st.columns(5)
    for col, (ticker, name, sector) in zip(cols, popular):
        with col:
            st.markdown(f'<div class="ticker-card"><b>{name}</b><br><span style="color:#79b7ff">{ticker}</span><br><small>{sector}</small></div>', unsafe_allow_html=True)
            if st.button("加入自選", key=f"popular_{ticker}"):
                st.session_state.watchlist.setdefault(ticker, name)
                st.rerun()


def chart(ticker: str, df: pd.DataFrame, indicators: str) -> go.Figure:
    price = market_data(ticker)
    rows = 2 if indicators != "不顯示" else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28] if rows == 2 else None)
    if not price.empty:
        fig.add_trace(go.Candlestick(x=price.index, open=price.Open, high=price.High, low=price.Low, close=price.Close, name="K 線", increasing_line_color="#30d5a5", decreasing_line_color="#ff5c7a"), row=1, col=1)
        close = price.Close.astype(float)
        if indicators == "成交量":
            fig.add_trace(go.Bar(x=price.index, y=price.Volume, marker_color="#4e8cff", name="Volume"), row=2, col=1)
        elif indicators == "KD 指標":
            kd = StochasticOscillator(price.High, price.Low, close)
            fig.add_trace(go.Scatter(x=price.index, y=kd.stoch(), name="K", line=dict(color="#39d5ff")), row=2, col=1)
            fig.add_trace(go.Scatter(x=price.index, y=kd.stoch_signal(), name="D", line=dict(color="#ffd166")), row=2, col=1)
        elif indicators == "乖離率 (BIAS)":
            bias = (close / close.rolling(20).mean() - 1) * 100
            fig.add_trace(go.Scatter(x=price.index, y=bias, name="BIAS 20", line=dict(color="#c084fc")), row=2, col=1)
        elif indicators == "MACD":
            macd = MACD(close)
            fig.add_trace(go.Bar(x=price.index, y=macd.macd_diff(), name="Histogram", marker_color="#7186ff"), row=2, col=1)
            fig.add_trace(go.Scatter(x=price.index, y=macd.macd(), name="MACD", line=dict(color="#39d5ff")), row=2, col=1)
            fig.add_trace(go.Scatter(x=price.index, y=macd.macd_signal(), name="Signal", line=dict(color="#ffb703")), row=2, col=1)
    for side, color, symbol in [("買入", "#24d6a5", "triangle-up"), ("賣出", "#ff5577", "triangle-down")]:
        tx = df[df.side == side]
        if not tx.empty:
            hover = [f"<b>{side}</b><br>成交價：{r.price:,.2f}<br>股數：{int(r.shares):,}<br>手續費：{money(r.fee)}<br>策略：{r.logic}<br>心理：{r.psychology}<br>復盤：{r.note}" for r in tx.itertuples()]
            fig.add_trace(go.Scatter(x=tx.trade_datetime, y=tx.price, mode="markers", name=side, marker=dict(size=14, symbol=symbol, color=color, line=dict(width=1,color="#ffffff")), text=hover, hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.update_layout(template="plotly_dark", height=620, paper_bgcolor="#09111f", plot_bgcolor="#09111f", margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig


def page_tracker() -> None:
    st.title("📈 交易標的追蹤")
    tickers = list(st.session_state.watchlist)
    choice = st.radio("自選標的快速切換", tickers, horizontal=True, format_func=lambda x: f"{st.session_state.watchlist[x]} · {x}")
    add_col, ind_col = st.columns([1, 2])
    with add_col:
        new = st.text_input("新增自選標的", placeholder="例如 2308.TW").upper().strip()
        if st.button("＋ 新增自選") and new:
            st.session_state.watchlist.setdefault(new, new)
            st.rerun()
    with ind_col:
        indicator = st.selectbox("圖表下方技術指標", ["成交量", "KD 指標", "乖離率 (BIAS)", "MACD", "不顯示"])
    df = trades_df(choice)
    sells, lots, _ = fifo_analysis(df)
    price_data = market_data(choice)
    current = latest_price(choice, price_data)
    cost_basis = sum(l["remaining"] * l["price"] + l["fee"] * l["remaining"] / l["shares"] for l in lots)
    holding_qty = sum(l["remaining"] for l in lots)
    unrealized = (holding_qty * current - fee_for(current, int(holding_qty), st.session_state.discount) - holding_qty * current * TAX_RATE - cost_basis) if current and holding_qty else 0.0
    realized = sells.realized_pnl.sum() if not sells.empty else 0.0
    invested = sum(df[df.side == "買入"].price * df[df.side == "買入"].shares + df[df.side == "買入"].fee)
    a,b,c,d = st.columns(4)
    a.metric("總投入資金", money(invested))
    b.metric("已實現損益（FIFO）", money(realized), pct(realized / max(invested,1)*100))
    c.metric("未實現損益", money(unrealized), pct(unrealized / max(cost_basis,1)*100) if holding_qty else "—")
    d.metric("目前持股 / 現價", f"{holding_qty:,.0f} 股", money(current) if current else "行情暫不可用")
    st.plotly_chart(chart(choice, df, indicator), use_container_width=True)
    with st.expander("➕ 新增交易紀錄", expanded=False):
        with st.form("trade_form", clear_on_submit=True):
            c1,c2,c3,c4 = st.columns(4)
            trade_day = c1.date_input("交易日期", value=date.today())
            trade_time = c2.time_input("交易時間", value=time(9, 0))
            side = c3.selectbox("類型", ["買入", "賣出"])
            shares = c4.number_input("股數", min_value=1, value=1000, step=100)
            c5,c6,c7 = st.columns(3)
            trade_price = c5.number_input("成交價格", min_value=0.01, value=float(current or 100), step=0.5)
            logic = c6.selectbox("進場邏輯", ["技術面突破", "消息面", "籌碼面", "價值面", "風險控管出場"])
            psychology = c7.selectbox("交易心理標籤", ["冷靜執行", "FOMO追高", "恐慌砍倉", "紀律停利", "猶豫觀望"])
            note = st.text_area("該筆交易專屬檢討筆記 (Trade Note)", placeholder="當時的判斷、情緒與下次可改善之處…")
            if st.form_submit_button("儲存交易紀錄", use_container_width=True):
                dt = datetime.combine(trade_day, trade_time).strftime("%Y-%m-%d %H:%M")
                st.session_state.trades.append(blank_trade(choice, dt, side, float(trade_price), int(shares), logic, psychology, note, st.session_state.discount))
                st.success("已加入交易紀錄")
                st.rerun()
    st.subheader("交易紀錄明細")
    shown = df.copy()
    if not sells.empty:
        pnl_map = sells.set_index("id").realized_pnl
        shown["net_pnl"] = shown.id.map(pnl_map).fillna(0.0)
    else: shown["net_pnl"] = 0.0
    st.dataframe(shown[["trade_datetime","side","price","shares","fee","tax","logic","psychology","note","net_pnl"]], use_container_width=True, hide_index=True, column_config={"net_pnl": st.column_config.NumberColumn("淨損益（FIFO）", format="NT$ %.0f")})


def page_analytics() -> None:
    st.title("📊 總體績效與檢討")
    df = trades_df(); sells, _, matches = fifo_analysis(df)
    matched = pd.DataFrame(matches)
    total = sells.realized_pnl.sum() if not sells.empty else 0.0
    win = matched[matched.pnl > 0] if not matched.empty else matched
    loss = matched[matched.pnl < 0] if not matched.empty else matched
    win_rate = len(win) / len(matched) * 100 if len(matched) else 0
    payoff = win.pnl.mean() / abs(loss.pnl.mean()) if len(win) and len(loss) else None
    avg_days = matched.holding_days.mean() if len(matched) else 0
    a,b,c,d = st.columns(4)
    a.metric("累積總淨損益", money(total)); b.metric("總勝率", f"{win_rate:.1f}%")
    c.metric("盈虧比", f"{payoff:.2f}" if payoff is not None else "—"); d.metric("平均持股天數", f"{avg_days:.1f} 天")
    if matched.empty:
        st.info("尚無可配對的買賣交易；新增賣出紀錄後即可分析。")
    else:
        # fifo_analysis preserves the opening trade's strategy and mindset, so
        # these charts genuinely answer which *entry* decisions work best.
        matched["win"] = matched.pnl > 0
        logic_stats = matched.groupby("logic").agg(net_pnl=("pnl","sum"), win_rate=("win","mean")).reset_index(); logic_stats.win_rate *= 100
        psych_stats = matched.groupby("psychology").agg(net_pnl=("pnl","sum"), win_rate=("win","mean")).reset_index(); psych_stats.win_rate *= 100
        left,right = st.columns(2)
        with left:
            st.plotly_chart(go.Figure(go.Pie(labels=logic_stats.logic, values=logic_stats.net_pnl.abs(), hole=.48, textinfo="label+percent")).update_layout(template="plotly_dark", title="依進場／出場邏輯：損益權重"), use_container_width=True)
            st.plotly_chart(go.Figure(go.Bar(x=logic_stats.logic, y=logic_stats.win_rate, marker_color="#30d5a5")).update_layout(template="plotly_dark", title="依策略：勝率 %", yaxis_ticksuffix="%"), use_container_width=True)
        with right:
            st.plotly_chart(go.Figure(go.Pie(labels=psych_stats.psychology, values=psych_stats.net_pnl.abs(), hole=.48, textinfo="label+percent")).update_layout(template="plotly_dark", title="依心理標籤：損益權重"), use_container_width=True)
            st.plotly_chart(go.Figure(go.Bar(x=psych_stats.psychology, y=psych_stats.net_pnl, marker_color="#ff7c98")).update_layout(template="plotly_dark", title="依心理標籤：淨損益", yaxis_title="NT$"), use_container_width=True)
    st.subheader("進度安全備份區")
    backup_box("analytics")


init_state(); style()
with st.sidebar:
    st.title("⚡ TRADE OS")
    page = st.radio("導覽", ["🏠 平台探索", "📈 交易標的追蹤", "📊 總體績效與檢討"])
    st.divider()
    st.caption("台股成本設定")
    discount_pct = st.number_input("手續費折扣率（折）", min_value=0.1, max_value=1.0, value=float(st.session_state.discount), step=0.05)
    st.session_state.discount = float(discount_pct)
    st.caption("手續費 0.1425% × 折扣，單筆最低 NT$20；賣出稅 0.3%。")
    st.divider(); backup_box("sidebar")

if page.startswith("🏠"): page_explore()
elif page.startswith("📈"): page_tracker()
else: page_analytics()

st.divider()
backup_box("footer")
