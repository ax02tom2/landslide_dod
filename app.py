"""崩塌地形變異分析 — 主入口與工作流程導航

使用者只需要依序操作：
① 前處理｜裁切與縮小
② 地形變異分析｜DoD

本檔案本身不作為分析工作頁；它只負責正式入口、品牌標題與頁面導航。
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="崩塌地形變異分析",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 左側正式工作流程
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="
            font-size: 1.18rem;
            font-weight: 700;
            line-height: 1.35;
            margin: 0.2rem 0 0.35rem 0;
        ">
            ⛰️ 崩塌地形變異分析
        </div>
        <div style="
            color: #6b7280;
            font-size: 0.82rem;
            margin-bottom: 1rem;
        ">
            工程分析工作流程
        </div>
        """,
        unsafe_allow_html=True,
    )

# st.navigation 會取代 Streamlit 原本的自動 pages 導航，
# 因此左側不再顯示「app / 1_preprocess / 2_analysis」等程式檔名。
pages = [
    st.Page(
        "pages/1_preprocess.py",
        title="① 前處理｜裁切與縮小",
        icon="📐",
        default=True,
    ),
    st.Page(
        "pages/2_analysis.py",
        title="② 地形變異分析｜DoD",
        icon="⛰️",
    ),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
