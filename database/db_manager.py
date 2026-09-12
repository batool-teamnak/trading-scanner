import sqlite3
from contextlib import closing
import json
from pathlib import Path
from datetime import date, timedelta
import config


def get_connection():
    if config.DB_PATH != ":memory:":
        Path(config.DB_PATH).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(Path(config.DB_PATH).expanduser()) if config.DB_PATH != ":memory:" else config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(get_connection()) as conn:
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bias TEXT,
                grade TEXT,
                score INTEGER,
                conditions TEXT,
                entry REAL,
                sl REAL,
                tp1 REAL,
                tp2 REAL,
                session TEXT,
                chart_path TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT,
                setup_id INTEGER,
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                session TEXT,
                grade TEXT,
                score INTEGER,
                entry REAL,
                sl REAL,
                tp REAL,
                exit_price REAL,
                outcome TEXT,
                rr_achieved REAL,
                notes TEXT,
                chart_path TEXT,
                FOREIGN KEY (setup_id) REFERENCES setups(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                entry_date DATE DEFAULT CURRENT_DATE,
                followed_plan TEXT,
                outcome TEXT,
                moved_stop TEXT,
                better_entry TEXT,
                emotions TEXT,
                notes TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS risk_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                balance REAL,
                equity REAL,
                daily_dd REAL,
                weekly_dd REAL,
                consecutive_losses INTEGER,
                trading_allowed INTEGER
            )
        """)

        conn.commit()


def save_setup(symbol, timeframe, analysis, scoring, chart_path=None):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        levels = analysis.get("levels") or {}
        c.execute("""
            INSERT INTO setups (symbol, timeframe, bias, grade, score, conditions, entry, sl, tp1, tp2, session, chart_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, timeframe,
            analysis.get("bias"),
            scoring.get("grade"),
            scoring.get("score"),
            json.dumps(analysis.get("conditions", [])),
            levels.get("entry"),
            levels.get("sl"),
            levels.get("tp1"),
            levels.get("tp2"),
            analysis.get("session"),
            chart_path,
        ))
        setup_id = c.lastrowid
        conn.commit()
    return setup_id


def log_trade(symbol, direction, entry, sl, tp, grade=None, score=None, session=None, setup_id=None):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO trades (symbol, direction, setup_id, session, grade, score, entry, sl, tp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, direction, setup_id, session, grade, score, entry, sl, tp))
        trade_id = c.lastrowid
        conn.commit()
    return trade_id


def close_trade(trade_id, exit_price, outcome, rr_achieved=None, notes=None):
    if outcome not in ("win", "loss", "breakeven"):
        raise ValueError("outcome must be win, loss or breakeven")
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE trades
            SET exit_price=?, outcome=?, rr_achieved=?, notes=?, closed_at=CURRENT_TIMESTAMP
            WHERE id=? AND closed_at IS NULL
        """, (exit_price, outcome, rr_achieved, notes, trade_id))
        changed = c.rowcount == 1
        conn.commit()
    return changed


def save_journal_entry(trade_id, followed_plan, outcome, moved_stop, better_entry, emotions, notes):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO journal_entries (trade_id, followed_plan, outcome, moved_stop, better_entry, emotions, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, followed_plan, outcome, moved_stop, better_entry,
              json.dumps(emotions) if isinstance(emotions, list) else emotions, notes))
        conn.commit()


def log_risk(balance, equity, risk_status):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO risk_log (balance, equity, daily_dd, weekly_dd, consecutive_losses, trading_allowed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            balance, equity,
            risk_status.get("daily_drawdown"),
            risk_status.get("weekly_drawdown"),
            risk_status.get("consecutive_losses"),
            1 if risk_status.get("allowed") else 0,
        ))
        conn.commit()


def get_recent_losses(n=5):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("SELECT outcome FROM trades WHERE outcome IS NOT NULL ORDER BY closed_at DESC, id DESC LIMIT ?", (n,))
        rows = c.fetchall()
    losses = []
    for row in rows:
        if row["outcome"] == "loss":
            losses.append("loss")
        else:
            break
    return losses


def get_daily_stats(target_date=None):
    target = (target_date or date.today()).isoformat()
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
                   AVG(rr_achieved) as avg_rr
            FROM trades WHERE DATE(opened_at) = ?
        """, (target,))
        row = dict(c.fetchone())
    row["avg_rr"] = round(row["avg_rr"] or 0, 2)
    row["wins"] = row["wins"] or 0
    row["losses"] = row["losses"] or 0
    return row


def get_weekly_stats():
    week_ago = (date.today() - timedelta(days=6)).isoformat()
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
                   AVG(rr_achieved) as avg_rr
            FROM trades WHERE DATE(opened_at) BETWEEN ? AND ?
        """, (week_ago, date.today().isoformat()))
        row = dict(c.fetchone())
    row["avg_rr"] = round(row["avg_rr"] or 0, 2)
    row["wins"] = row["wins"] or 0
    row["losses"] = row["losses"] or 0
    return row


def get_all_trades():
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY opened_at DESC")
        rows = [dict(r) for r in c.fetchall()]
    return rows


def get_active_setups(limit=20):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM setups ORDER BY detected_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in c.fetchall()]
    return rows


def get_journal_entries():
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM journal_entries ORDER BY entry_date DESC")
        rows = [dict(r) for r in c.fetchall()]
    return rows


def get_risk_log(limit=100):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM risk_log ORDER BY logged_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in c.fetchall()]
    return rows
