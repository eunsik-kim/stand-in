"""What has to survive a restart, in one SQLite file on the volume.

- accounts: who has signed in, and whether they are blocked;
- usage: how much each account used today, so a daily cap holds across restarts;
- spend: what the whole server spent today, so the budget is per day rather than per process;
- worlds: each world with its owner, listed per owner instead of by scanning files.

Scenes in progress stay in memory, and images, voices and the reply cache stay as files.
Days are UTC. One connection guarded by a lock is plenty for one machine.
"""
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from standin.llm import BudgetExceeded, Spend

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    sub TEXT PRIMARY KEY, email TEXT, name TEXT,
    first_seen REAL, last_seen REAL, blocked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS usage (
    who TEXT, day TEXT, kind TEXT, count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (who, day, kind));
CREATE TABLE IF NOT EXISTS spend (
    day TEXT PRIMARY KEY, usd REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS account_spend (
    who TEXT, day TEXT, usd REAL NOT NULL DEFAULT 0, PRIMARY KEY (who, day));
CREATE TABLE IF NOT EXISTS worlds (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE INDEX IF NOT EXISTS worlds_by_owner ON worlds (owner, updated DESC);
"""


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _one(self, sql: str, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    # accounts
    def seen(self, sub: str, email: str, name: str) -> None:
        now = time.time()
        with self.lock:
            self.db.execute(
                "INSERT INTO accounts (sub, email, name, first_seen, last_seen) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(sub) DO UPDATE SET email = excluded.email, name = excluded.name, last_seen = excluded.last_seen",
                (sub, email, name, now, now))

    def blocked(self, sub: str) -> bool:
        row = self._one("SELECT blocked FROM accounts WHERE sub = ?", (sub,))
        return bool(row and row[0])

    def block(self, sub: str, on: bool = True) -> None:
        """Works before the account has ever signed in, too: the row waits for it."""
        with self.lock:
            self.db.execute("INSERT INTO accounts (sub, email, name, blocked) VALUES (?, '', '', ?) "
                            "ON CONFLICT(sub) DO UPDATE SET blocked = excluded.blocked", (sub, int(on)))

    # usage
    def take(self, who: str, kind: str, limit: int) -> bool:
        """Count one more `kind` for today, unless that would pass `limit`. True when allowed."""
        day = today()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute("SELECT count FROM usage WHERE who = ? AND day = ? AND kind = ?",
                                      (who, day, kind)).fetchone()
                if row and row[0] >= limit:
                    self.db.execute("ROLLBACK")
                    return False
                self.db.execute("INSERT INTO usage (who, day, kind, count) VALUES (?, ?, ?, 1) "
                                "ON CONFLICT(who, day, kind) DO UPDATE SET count = count + 1", (who, day, kind))
                self.db.execute("COMMIT")
                return True
            except Exception:
                self.db.execute("ROLLBACK")
                raise

    def refund(self, who: str, kind: str) -> None:
        """Give back one `kind` taken today, for a request that was refused after all."""
        with self.lock:
            self.db.execute("UPDATE usage SET count = count - 1 WHERE who = ? AND day = ? AND kind = ? AND count > 0",
                            (who, today(), kind))

    def used(self, who: str) -> dict[str, int]:
        with self.lock:
            rows = self.db.execute("SELECT kind, count FROM usage WHERE who = ? AND day = ?", (who, today())).fetchall()
        return dict(rows)

    # spend
    def add_spend(self, usd: float) -> float:
        day = today()
        with self.lock:
            self.db.execute("INSERT INTO spend (day, usd) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd",
                            (day, usd))
            return self.db.execute("SELECT usd FROM spend WHERE day = ?", (day,)).fetchone()[0]

    def spent_today(self) -> float:
        row = self._one("SELECT usd FROM spend WHERE day = ?", (today(),))
        return row[0] if row else 0.0

    def add_account_spend(self, who: str, usd: float) -> None:
        with self.lock:
            self.db.execute("INSERT INTO account_spend (who, day, usd) VALUES (?, ?, ?) "
                            "ON CONFLICT(who, day) DO UPDATE SET usd = usd + excluded.usd", (who, today(), usd))

    # the admin view
    def spend_days(self, days: int = 7) -> list[tuple[str, float]]:
        """The server's spend for each of the last `days` days, oldest first, zero where nothing was spent."""
        end = datetime.now(timezone.utc)
        keys = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days - 1, -1, -1)]
        with self.lock:
            rows = dict(self.db.execute(f"SELECT day, usd FROM spend WHERE day IN ({','.join('?' * len(keys))})", keys).fetchall())
        return [(k, rows.get(k, 0.0)) for k in keys]

    def accounts_on(self, day: str | None = None) -> list[dict]:
        """Every account with what it used on `day` (today by default), the busiest first."""
        day = day or today()
        with self.lock:
            accounts = self.db.execute("SELECT sub, email, name, first_seen, last_seen, blocked FROM accounts").fetchall()
            counts = self.db.execute("SELECT who, kind, count FROM usage WHERE day = ?", (day,)).fetchall()
            spent = dict(self.db.execute("SELECT who, usd FROM account_spend WHERE day = ?", (day,)).fetchall())
        used: dict[str, dict[str, int]] = {}
        for who, kind, n in counts:
            used.setdefault(who, {})[kind] = n
        out = [{"sub": sub, "email": email, "name": name, "first_seen": first, "last_seen": last, "blocked": bool(blocked),
                "used": used.get(sub, {}), "usd": spent.get(sub, 0.0)} for sub, email, name, first, last, blocked in accounts]
        return sorted(out, key=lambda a: (-a["usd"], -sum(a["used"].values()), -(a["last_seen"] or 0)))

    # worlds
    def save_world(self, wid: str, owner: str, data: dict) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO worlds (id, owner, data, updated) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated = excluded.updated WHERE worlds.owner = excluded.owner",
                (wid, owner, json.dumps(data, ensure_ascii=False), time.time()))

    def load_world(self, wid: str, owner: str) -> dict | None:
        """Only the owner gets a world back."""
        row = self._one("SELECT data FROM worlds WHERE id = ? AND owner = ?", (wid, owner))
        return json.loads(row[0]) if row else None

    def worlds_of(self, owner: str, limit: int = 12) -> list[dict]:
        """The owner's newest worlds, each with `_updated` (when it was last saved)."""
        with self.lock:
            rows = self.db.execute("SELECT data, updated FROM worlds WHERE owner = ? ORDER BY updated DESC LIMIT ?",
                                   (owner, limit)).fetchall()
        return [{**json.loads(data), "_updated": updated} for data, updated in rows]

    def delete_world(self, wid: str, owner: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM worlds WHERE id = ? AND owner = ?", (wid, owner))

    def import_world_files(self, folder: Path) -> int:
        """One-time move of worlds saved as JSON files; files already imported are skipped."""
        moved = 0
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            wid, owner = data.get("id"), data.get("owner", "")
            if not wid or self._one("SELECT 1 FROM worlds WHERE id = ?", (wid,)):
                continue
            with self.lock:
                self.db.execute("INSERT INTO worlds (id, owner, data, updated) VALUES (?, ?, ?, ?)",
                                (wid, owner, json.dumps(data, ensure_ascii=False), path.stat().st_mtime))
            moved += 1
        return moved


class DailySpend(Spend):
    """The server's spend, with the budget counted per UTC day in the store so a restart doesn't reset it.

    `check()` runs before every paid call; it first writes what was spent since the last check.
    Text and image calls are also put on the account `who()` names (the request being served);
    voices and work done in background threads count only toward the server's total.
    """

    def __init__(self, store: Store, daily_budget_usd: float, who: Callable[[], str] = lambda: ""):
        super().__init__(daily_budget_usd)
        self.store = store
        self.who = who
        self.flushed = 0.0
        self.flush_lock = threading.Lock()  # requests check at the same time; each dollar is written once

    def add(self, usage, role: str, seconds: float, model: str = "") -> None:
        before = self.usd
        super().add(usage, role, seconds, model)
        account = self.who()
        if account and self.usd > before:
            self.store.add_account_spend(account, self.usd - before)
        self.flush()  # written now, so a restart before the next call doesn't lose it

    def flush(self) -> float:
        with self.flush_lock:
            delta, self.flushed = self.usd - self.flushed, self.usd
        return self.store.add_spend(delta) if delta > 0 else self.store.spent_today()

    def check(self) -> None:
        total = self.flush()
        if total > self.budget:
            raise BudgetExceeded(f"spent ${total:.4f} today > daily budget ${self.budget:.2f}")
