import streamlit as st


def set_page_config():
    if "sidebar_state" not in st.session_state:
        st.session_state["sidebar_state"] = "expanded"

    st.set_page_config(
        page_title="DocMind AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state=st.session_state["sidebar_state"],
        menu_items={
            "Get Help": "https://github.com/Ashita-no-Kaushar/DocMind-AI/discussions",
            "Report a bug": "https://github.com/Ashita-no-Kaushar/DocMind-AI/issues",
        },
    )

    st.markdown(
        r"""
    <style>
    /* Hide Streamlit chrome */
    .stDeployButton, [data-testid="stDeployButton"],
    [data-testid="stToolbar"], [data-testid="stHeaderActionElements"],
    #MainMenu, footer { visibility: hidden !important; display: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* Wide enough for the route figure, which is 1120px at full size */
    .block-container {
            max-width: 1240px;
            padding-top: 0.6rem;
            padding-bottom: 1rem;
        }

    /* Brand row. Colours follow the app theme and fall back to the dark
       palette in .streamlit/config.toml, so nothing is invisible if a user
       switches Streamlit to the light theme. */
    .dm-brand {
            display: flex; align-items: baseline; gap: 10px;
            margin: 0 0 6px 0; padding: 0 0 6px 0;
            border-bottom: 1px solid rgba(139, 92, 246, 0.35);
        }
    .dm-brand-mark {
            font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;
            color: var(--text-color, #E6E8EF);
        }
    .dm-brand-sub {
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--primary-color, #8B5CF6);
        }

    /* Status strip under the brand */
    .dm-strip { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 10px 0; }
    .dm-badge {
            display: inline-block; font-size: 0.74rem; line-height: 1.5;
            font-weight: 600; letter-spacing: 0.01em;
            padding: 2px 9px; border-radius: 999px;
            border: 1px solid transparent; white-space: nowrap;
    }
    .dm-ok {
            background: rgba(52, 211, 153, 0.14); color: #6EE7B7;
            border-color: rgba(52, 211, 153, 0.35);
    }
    .dm-idle {
            background: rgba(148, 163, 184, 0.14); color: #CBD5E1;
            border-color: rgba(148, 163, 184, 0.28);
    }
    .dm-warn {
            background: rgba(251, 191, 36, 0.14); color: #FCD34D;
            border-color: rgba(251, 191, 36, 0.35);
    }

    /* Chat rhythm */
    [data-testid="stChatMessage"] { padding: 0.55rem 0.8rem; }
    [data-testid="stChatMessage"] p { line-height: 1.55; }

    /* Map figure sits flush so the route is not boxed in twice */
    [data-testid="stIFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.28); border-radius: 10px;
            overflow: hidden;
        }

    /* Suggestion pills */
    div[data-testid="stPills"] { gap: 6px !important; }
    div[data-testid="stPills"] button {
            border-radius: 999px !important; font-size: 0.83rem !important;
            padding: 6px 12px !important;
        }

    /* Light theme fallback */
    @media (prefers-color-scheme: light) {
        .dm-brand { border-bottom-color: rgba(139, 92, 246, 0.30); }
        .dm-ok {
                background: #e8f3ec; color: #17603a;
                border-color: #c3e2d0;
            }
        .dm-idle {
                background: #eef1f5; color: #495364;
                border-color: #dde2e9;
            }
        .dm-warn {
                background: #fdf1e0; color: #8a5a12;
                border-color: #f0dcb8;
            }
        [data-testid="stIFrame"] { border-color: #e3e8ef; }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
