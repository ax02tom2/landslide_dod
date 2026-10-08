"""案件/使用者管理：Supabase PostgreSQL（雲端）+ SQLite（本機 fallback）。

雲端 Streamlit：使用 Streamlit Secrets 中的 SUPABASE_DB_* 連線 Supabase。
本機若未設定這些 Secrets，仍可使用原本的 .dod_data/cases.sqlite3。

注意：這一版先把「使用者／案件資料」改成持久化 PostgreSQL；
DEM/DSM 等實體檔案仍沿用本機檔案機制，下一階段再接 Supabase Storage。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import time
from copy import deepcopy
from pathlib import Path

try:
    import streamlit as st
except Exception:  # pragma: no cover - 本機純 Python 測試時可沒有 Streamlit
    st = None

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.errors import UniqueViolation
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None
    UniqueViolation = Exception

ROOT = Path(os.environ.get("DOD_DATA_DIR", Path(__file__).resolve().parent / ".dod_data"))
DB = ROOT / "cases.sqlite3"
FILES = ROOT / "files"
ROOT.mkdir(parents=True, exist_ok=True)
FILES.mkdir(parents=True, exist_ok=True)

_INTERNAL_FILES_KEY = "__case_manager_files__"


def _secret(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return str(value)
    if st is not None:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass
    return default


def _supabase_db_configured() -> bool:
    required = (
        "SUPABASE_DB_HOST",
        "SUPABASE_DB_PORT",
        "SUPABASE_DB_NAME",
        "SUPABASE_DB_USER",
        "SUPABASE_DB_PASSWORD",
    )
    return all(_secret(k) for k in required)


def using_supabase() -> bool:
    """回傳目前案件資料庫是否使用 Supabase PostgreSQL。"""
    return _supabase_db_configured() and psycopg is not None


def _sqlite_conn():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def _pg_conn():
    if psycopg is None:
        raise RuntimeError("尚未安裝 psycopg，請重新部署 requirements.txt。")
    return psycopg.connect(
        host=_secret("SUPABASE_DB_HOST"),
        port=int(_secret("SUPABASE_DB_PORT", "5432")),
        dbname=_secret("SUPABASE_DB_NAME", "postgres"),
        user=_secret("SUPABASE_DB_USER"),
        password=_secret("SUPABASE_DB_PASSWORD"),
        sslmode="require",
        row_factory=dict_row,
    )


def _conn():
    return _pg_conn() if using_supabase() else _sqlite_conn()


def init_db():
    if using_supabase():
        # Supabase 的 users / cases 已由 SQL Editor 建立。
        # 不在 App 啟動時重建或修改使用者資料表。
        return
    with _sqlite_conn() as c:
        c.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cases(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          description TEXT DEFAULT '',
          state_json TEXT DEFAULT '{}',
          files_json TEXT DEFAULT '{}',
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_cases_user_updated ON cases(user_id, updated_at DESC);
        """)


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 180_000)
    return "pbkdf2_sha256$180000$" + salt.hex() + "$" + dk.hex()


def _verify(password: str, encoded: str) -> bool:
    try:
        _, rounds, salt_hex, digest = encoded.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def _ts(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    return float(value)


def _pg_json(value):
    # psycopg 3 接受 Python dict/list 作 JSON 需要 Jsonb wrapper；
    # 為避免額外 API，直接送 json 字串，再由 SQL cast 成 jsonb。
    return json.dumps(value, ensure_ascii=False)


def register(username: str, password: str):
    username = username.strip()
    if len(username) < 3:
        return False, "使用者名稱至少 3 個字元。"
    if len(password) < 6:
        return False, "密碼至少 6 個字元。"
    now = time.time()
    password_hash = _hash_password(password)

    if using_supabase():
        try:
            with _pg_conn() as c:
                c.execute(
                    "INSERT INTO public.users(username,password_hash,created_at) VALUES(%s,%s,to_timestamp(%s))",
                    (username, password_hash, now),
                )
            return True, "註冊成功，請登入。"
        except UniqueViolation:
            return False, "這個使用者名稱已存在。"
        except Exception as e:
            return False, f"註冊失敗：{e}"

    try:
        with _sqlite_conn() as c:
            c.execute("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
                      (username, password_hash, now))
        return True, "註冊成功，請登入。"
    except sqlite3.IntegrityError:
        return False, "這個使用者名稱已存在。"


def authenticate(username: str, password: str):
    username = username.strip()
    if using_supabase():
        try:
            with _pg_conn() as c:
                row = c.execute(
                    "SELECT id, username, password_hash FROM public.users WHERE username=%s",
                    (username,),
                ).fetchone()
        except Exception:
            return None
        if row and _verify(password, row["password_hash"]):
            return {"id": str(row["id"]), "username": row["username"]}
        return None

    with _sqlite_conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and _verify(password, row["password_hash"]):
        return {"id": int(row["id"]), "username": row["username"]}
    return None


def _normalize_case(row: dict) -> dict:
    d = dict(row)
    d["id"] = str(d["id"])
    d["user_id"] = str(d["user_id"])
    d["created_at"] = _ts(d["created_at"])
    d["updated_at"] = _ts(d["updated_at"])
    state = d.get("state")
    if not isinstance(state, dict):
        state = {}
    state = deepcopy(state)
    files = state.pop(_INTERNAL_FILES_KEY, {})
    if not isinstance(files, dict):
        files = {}
    d["state"] = state
    d["files"] = files
    return d


def create_case(user_id, name: str, description: str = ""):
    now = time.time()
    name = name.strip() or "未命名案件"
    description = description.strip()

    if using_supabase():
        with _pg_conn() as c:
            row = c.execute(
                """
                INSERT INTO public.cases(user_id,name,description,state,created_at,updated_at)
                VALUES(%s,%s,%s,%s::jsonb,to_timestamp(%s),to_timestamp(%s))
                RETURNING *
                """,
                (user_id, name, description, _pg_json({_INTERNAL_FILES_KEY: {}}), now, now),
            ).fetchone()
        return _normalize_case(row)

    with _sqlite_conn() as c:
        cur = c.execute("INSERT INTO cases(user_id,name,description,created_at,updated_at) VALUES(?,?,?,?,?)",
                        (user_id, name, description, now, now))
        cid = int(cur.lastrowid)
    (FILES / str(user_id) / str(cid)).mkdir(parents=True, exist_ok=True)
    return get_case(user_id, cid)


def list_cases(user_id):
    if using_supabase():
        with _pg_conn() as c:
            rows = c.execute(
                "SELECT id,name,description,created_at,updated_at FROM public.cases WHERE user_id=%s ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                **dict(r),
                "id": str(r["id"]),
                "user_id": str(user_id),
                "created_at": _ts(r["created_at"]),
                "updated_at": _ts(r["updated_at"]),
            }
            for r in rows
        ]

    with _sqlite_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,name,description,created_at,updated_at FROM cases WHERE user_id=? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()]


def get_case(user_id, case_id):
    if using_supabase():
        with _pg_conn() as c:
            row = c.execute(
                "SELECT * FROM public.cases WHERE id=%s AND user_id=%s",
                (case_id, user_id),
            ).fetchone()
        return _normalize_case(row) if row else None

    with _sqlite_conn() as c:
        r = c.execute("SELECT * FROM cases WHERE id=? AND user_id=?", (case_id, user_id)).fetchone()
    if not r:
        return None
    d = dict(r)
    try:
        d["state"] = json.loads(d.pop("state_json") or "{}")
    except Exception:
        d["state"] = {}
    try:
        d["files"] = json.loads(d.pop("files_json") or "{}")
    except Exception:
        d["files"] = {}
    return d


def update_case(user_id, case_id, *, name=None, description=None, state=None, files=None):
    old = get_case(user_id, case_id)
    if not old:
        return None

    now = time.time()
    name = old["name"] if name is None else name.strip() or old["name"]
    description = old["description"] if description is None else description
    state = deepcopy(old["state"] if state is None else state)
    files = old["files"] if files is None else files

    if using_supabase():
        state[_INTERNAL_FILES_KEY] = deepcopy(files)
        with _pg_conn() as c:
            c.execute(
                """
                UPDATE public.cases
                SET name=%s, description=%s, state=%s::jsonb, updated_at=to_timestamp(%s)
                WHERE id=%s AND user_id=%s
                """,
                (name, description, _pg_json(state), now, case_id, user_id),
            )
        return get_case(user_id, case_id)

    with _sqlite_conn() as c:
        c.execute("UPDATE cases SET name=?,description=?,state_json=?,files_json=?,updated_at=? WHERE id=? AND user_id=?",
                  (name, description, json.dumps(state, ensure_ascii=False), json.dumps(files, ensure_ascii=False), now, case_id, user_id))
    return get_case(user_id, case_id)


def delete_case(user_id, case_id):
    if using_supabase():
        with _pg_conn() as c:
            c.execute("DELETE FROM public.cases WHERE id=%s AND user_id=%s", (case_id, user_id))
    else:
        with _sqlite_conn() as c:
            c.execute("DELETE FROM cases WHERE id=? AND user_id=?", (case_id, user_id))
        p = FILES / str(user_id) / str(case_id)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def save_uploaded_file(user_id, case_id, uploaded_file, role: str):
    """目前仍寫入本機檔案；Storage 會在下一階段接入。"""
    if uploaded_file is None:
        return None
    folder = FILES / str(user_id) / str(case_id)
    folder.mkdir(parents=True, exist_ok=True)
    safe = Path(uploaded_file.name).name
    stamp = hashlib.sha256(f"{safe}:{uploaded_file.size}".encode()).hexdigest()[:12]
    path = folder / f"{role}_{stamp}_{safe}"
    if not path.exists():
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return str(path)


def case_file(user_id, case_id, role: str):
    d = get_case(user_id, case_id)
    if not d:
        return None
    p = d.get("files", {}).get(role)
    return p if p and os.path.isfile(p) else None


init_db()
