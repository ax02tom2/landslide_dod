"""案件/使用者管理：SQLite + 案件資料夾。

預設儲存在 DOD_DATA_DIR 環境變數指定的資料夾；未設定時為專案下的 .dod_data。
"""
from __future__ import annotations
import hashlib, hmac, json, os, secrets, sqlite3, time
from pathlib import Path

ROOT = Path(os.environ.get("DOD_DATA_DIR", Path(__file__).resolve().parent / ".dod_data"))
DB = ROOT / "cases.sqlite3"
FILES = ROOT / "files"
ROOT.mkdir(parents=True, exist_ok=True)
FILES.mkdir(parents=True, exist_ok=True)


def _conn():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
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


def register(username: str, password: str):
    username = username.strip()
    if len(username) < 3:
        return False, "使用者名稱至少 3 個字元。"
    if len(password) < 6:
        return False, "密碼至少 6 個字元。"
    now = time.time()
    try:
        with _conn() as c:
            c.execute("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
                      (username, _hash_password(password), now))
        return True, "註冊成功，請登入。"
    except sqlite3.IntegrityError:
        return False, "這個使用者名稱已存在。"


def authenticate(username: str, password: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    if row and _verify(password, row["password_hash"]):
        return {"id": int(row["id"]), "username": row["username"]}
    return None


def create_case(user_id: int, name: str, description: str = ""):
    now = time.time()
    with _conn() as c:
        cur = c.execute("INSERT INTO cases(user_id,name,description,created_at,updated_at) VALUES(?,?,?,?,?)",
                        (user_id, name.strip() or "未命名案件", description.strip(), now, now))
        cid = int(cur.lastrowid)
    (FILES / str(user_id) / str(cid)).mkdir(parents=True, exist_ok=True)
    return get_case(user_id, cid)


def list_cases(user_id: int):
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,name,description,created_at,updated_at FROM cases WHERE user_id=? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()]


def get_case(user_id: int, case_id: int):
    with _conn() as c:
        r = c.execute("SELECT * FROM cases WHERE id=? AND user_id=?", (case_id, user_id)).fetchone()
    if not r:
        return None
    d = dict(r)
    try: d["state"] = json.loads(d.pop("state_json") or "{}")
    except Exception: d["state"] = {}
    try: d["files"] = json.loads(d.pop("files_json") or "{}")
    except Exception: d["files"] = {}
    return d


def update_case(user_id: int, case_id: int, *, name=None, description=None, state=None, files=None):
    old = get_case(user_id, case_id)
    if not old:
        return None
    now = time.time()
    name = old["name"] if name is None else name.strip() or old["name"]
    description = old["description"] if description is None else description
    state = old["state"] if state is None else state
    files = old["files"] if files is None else files
    with _conn() as c:
        c.execute("UPDATE cases SET name=?,description=?,state_json=?,files_json=?,updated_at=? WHERE id=? AND user_id=?",
                  (name, description, json.dumps(state, ensure_ascii=False), json.dumps(files, ensure_ascii=False), now, case_id, user_id))
    return get_case(user_id, case_id)


def delete_case(user_id: int, case_id: int):
    with _conn() as c:
        c.execute("DELETE FROM cases WHERE id=? AND user_id=?", (case_id, user_id))
    p = FILES / str(user_id) / str(case_id)
    if p.exists():
        import shutil; shutil.rmtree(p, ignore_errors=True)


def save_uploaded_file(user_id: int, case_id: int, uploaded_file, role: str):
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


def case_file(user_id: int, case_id: int, role: str):
    d = get_case(user_id, case_id)
    if not d: return None
    p = d.get("files", {}).get(role)
    return p if p and os.path.isfile(p) else None

init_db()
