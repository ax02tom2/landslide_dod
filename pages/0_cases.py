"""案件管理頁面：登入、建立、開啟與刪除案件。"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

import case_manager as cm

st.set_page_config(page_title="案件管理", page_icon="📁", layout="wide")


def logged_in() -> bool:
    return st.session_state.get("user") is not None


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
    st.stop()

user = st.session_state["user"]
st.title("⛰️ 崩塌地形變異分析")
st.subheader("案件管理")
st.write("先建立或開啟案件，再依序執行：**① 前處理｜裁切與縮小 → ② 地形變異分析｜DoD**。")

cases = cm.list_cases(user["id"])
if cases:
    st.markdown("### 我的案件")
    for c in cases:
        col1, col2, col3 = st.columns([5, 2, 1])
        updated = datetime.fromtimestamp(c["updated_at"], tz=timezone.utc).astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
        col1.markdown(f"**{c['name']}** · 最後更新：{updated}")
        if c.get("description"):
            col1.caption(c["description"])
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
