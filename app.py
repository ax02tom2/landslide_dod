"""崩塌地形變異分析｜主入口

使用 st.navigation 統一管理多頁導覽。
案件管理本身也是正式註冊的頁面，因此其他頁面可以安全地用 st.page_link 回到案件管理。
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="崩塌地形變異分析",
    page_icon="⛰️",
    layout="wide",
)

nav_pages = {
    "案件": [
        st.Page(
            "pages/0_cases.py",
            title="案件管理",
            icon="📁",
            url_path="cases",
        ),
    ],
    "作業順序": [
        st.Page(
            "pages/1_preprocess.py",
            title="① 前處理｜裁切與縮小",
            icon="✂️",
            url_path="preprocess",
        ),
        st.Page(
            "pages/2_analysis.py",
            title="② 地形變異分析｜DoD",
            icon="⛰️",
            url_path="analysis",
        ),
    ],
}

pg = st.navigation(
    nav_pages,
    position="sidebar",
    expanded=True,
)

# ============================================================
# 工具說明下載
# ============================================================
# 放在側邊欄，所有使用者都可以下載。
manual_path = (
    Path(__file__).parent
    / "docs"
    / "崩塌地形變異分析_工具說明簡報.pptx"
)

if manual_path.is_file():
    st.sidebar.divider()
    st.sidebar.markdown("### 📘 工具說明")

    st.sidebar.download_button(
        "下載工具說明簡報",
        data=manual_path.read_bytes(),
        file_name="崩塌地形變異分析_工具說明簡報.pptx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        ),
        width="stretch",
        help="下載崩塌地形變異分析的操作與方法說明簡報。",
    )

# ============================================================
# 使用者資訊
# ============================================================
if st.session_state.get("user"):
    import case_manager as cm

    st.sidebar.divider()
    st.sidebar.caption(
        f"👤 {st.session_state['user']['username']}"
    )

    case_id = st.session_state.get("case_id")

    if case_id:
        case = cm.get_case(
            st.session_state["user"]["id"],
            case_id,
        )

        if case:
            st.sidebar.caption(
                f"📁 目前案件：{case['name']}"
            )

    if st.sidebar.button("登出", width="stretch"):
        st.session_state.clear()
        st.rerun()

pg.run()
