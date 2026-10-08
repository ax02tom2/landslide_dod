"""崩塌地形變異分析｜案件管理入口

使用 st.navigation 統一管理多頁導覽，避免 st.switch_page() 因部署環境的頁面路徑解析
而出現 PageNotFoundError。案件資料仍由 case_manager 依使用者分開保存。
"""
from __future__ import annotations

from datetime import datetime
import streamlit as st

import case_manager as cm

st.set_page_config(page_title="崩塌地形變異分析", page_icon="⛰️", layout="wide")


def logged_in() -> bool:
    return st.session_state.get("user") is not None


def render_case_home() -> None:
    """案件管理首頁。"""
    if not logged_in():
        st.title("⛰️ 崩塌地形變異分析")
        st.caption("案件管理版｜前處理 → DoD 分析 → 儲存與重新開啟")
        a, b = st.tabs(["登入", "建立使用者"])
        with a:
            u = st.text_input("使用者名稱", key="login_u")
            p = st.text_input("密碼", type="password", key="login_p")
            if st.button("登入", type="primary", width="stretch"):
                user = cm.authenticate(u, p)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("使用者名稱或密碼錯誤。")
        with b:
            u2 = st.text_input("使用者名稱", key="reg_u")
            p2 = st.text_input("密碼（至少 6 碼）", type="password", key="reg_p")
            p3 = st.text_input("再次輸入密碼", type="password", key="reg_p2")
            if st.button("建立使用者", width="stretch"):
                if p2 != p3:
                    st.error("兩次密碼不一致。")
                else:
                    ok, msg = cm.register(u2, p2)
                    (st.success if ok else st.error)(msg)
        st.info("每個使用者只能看到自己建立的案件。案件設定、圈繪範圍與前處理成果會依案件分開保存。")
        return

    user = st.session_state["user"]
    st.title("⛰️ 崩塌地形變異分析")
    st.subheader("案件管理")
    st.write("先建立或開啟案件，再依序執行：**① 前處理｜裁切與縮小 → ② 地形變異分析｜DoD**。")

    cases = cm.list_cases(user["id"])
    if cases:
        st.markdown("### 我的案件")
        for c in cases:
            col1, col2, col3 = st.columns([5, 2, 1])
            updated = datetime.fromtimestamp(c["updated_at"]).strftime("%Y-%m-%d %H:%M")
            col1.markdown(f"**{c['name']}** · 最後更新：{updated}")
            if c.get("description"):
                col1.caption(c["description"])
            # 不再使用 st.switch_page；設定案件後由左側導航前往分析頁。
            if col2.button("開啟案件", key=f"open_{c['id']}", width="stretch"):
                st.session_state["case_id"] = c["id"]
                st.session_state["case"] = cm.get_case(user["id"], c["id"])
                st.session_state["case_opened_notice"] = c["name"]
                st.rerun()
            if col3.button("刪除", key=f"del_{c['id']}"):
                cm.delete_case(user["id"], c["id"])
                if st.session_state.get("case_id") == c["id"]:
                    st.session_state.pop("case_id", None)
                    st.session_state.pop("case", None)
                st.rerun()
    else:
        st.info("目前還沒有案件。")

    opened = st.session_state.get("case_opened_notice")
    if opened:
        st.success(f"已開啟案件：「{opened}」。請從左側選擇 ① 前處理或 ② 地形變異分析。")

    st.divider()
    st.markdown("### 建立新案件")
    name = st.text_input("案件名稱", placeholder="例如：1150706 ○○崩塌地分析")
    desc = st.text_area("案件說明（選填）", placeholder="位置、事件日期、分析目的等")
    if st.button("＋ 建立案件", type="primary"):
        if not name.strip():
            st.error("請輸入案件名稱。")
        else:
            case = cm.create_case(user["id"], name, desc)
            st.session_state["case_id"] = case["id"]
            st.session_state["case"] = case
            st.session_state["case_opened_notice"] = case["name"]
            st.success(f"已建立案件：「{case['name']}」。請從左側進入 ① 前處理。")
            st.rerun()


# Streamlit 原生多頁導覽：這裡才是左側真正的「可點擊」作業順序。
# 不依賴 st.switch_page("pages/...")，因此不會再因 Cloud 路徑解析而 PageNotFound。
nav_pages = {
    "案件": [
        st.Page(render_case_home, title="案件管理", icon="📁", url_path="cases"),
    ],
    "作業順序": [
        st.Page("pages/1_preprocess.py", title="① 前處理｜裁切與縮小", icon="✂️", url_path="preprocess"),
        st.Page("pages/2_analysis.py", title="② 地形變異分析｜DoD", icon="⛰️", url_path="analysis"),
    ],
}

# 導覽只顯示案件管理與兩個工作頁，不顯示 app.py 這種程式檔名。
pg = st.navigation(nav_pages, position="sidebar", expanded=True)

# 使用者資訊放在原生導覽下方。
if logged_in():
    st.sidebar.divider()
    st.sidebar.caption(f"👤 {st.session_state['user']['username']}")
    case_id = st.session_state.get("case_id")
    if case_id:
        case = cm.get_case(st.session_state["user"]["id"], case_id)
        if case:
            st.sidebar.caption(f"📁 目前案件：{case['name']}")
    if st.sidebar.button("登出", width="stretch"):
        st.session_state.clear()
        st.rerun()

pg.run()
