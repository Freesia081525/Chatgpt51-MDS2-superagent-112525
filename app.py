import os
import io
import base64
import json
import re
from typing import Dict, Any, List

import streamlit as st
import yaml
import plotly.graph_objects as go
import pandas as pd
import networkx as nx

# LLM SDKs
from google import genai  # pip install google-genai
from google.genai import types as genai_types  # for Part / Image
from openai import OpenAI  # pip install openai
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as grok_user, system as grok_system

# PDF and OCR tools
import PyPDF2  # pip install PyPDF2
import pdfplumber  # pip install pdfplumber
from pdf2image import convert_from_bytes  # pip install pdf2image
import pytesseract  # pip install pytesseract
from PIL import Image  # pillow


# ===============================
# BASIC CONFIG
# ===============================
st.set_page_config(
    page_title="Floral Agentic Workflow",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===============================
# LOCALIZATION
# ===============================
def t(key: str, lang: str) -> str:
    TEXT = {
        "app_title": {
            "en": "Floral Agentic Workflow Dashboard",
            "zh": "花語智能代理工作台",
        },
        "upload_label": {
            "en": "Upload a document (PDF, TXT, MD, JSON) or paste text",
            "zh": "上傳文件（PDF/TXT/MD/JSON）或貼上文字",
        },
        "paste_text": {"en": "Or paste text below:", "zh": "或在下方貼上文字："},
        "global_task": {"en": "Global task / question", "zh": "全域任務 / 問題說明"},
        "run_agents": {"en": "Run Selected Agents", "zh": "執行選取的代理"},
        "settings": {"en": "Settings", "zh": "設定"},
        "api_keys": {"en": "API Keys", "zh": "API 金鑰"},
        "theme_settings": {"en": "Theme & Language", "zh": "主題與語言"},
        "dashboard": {"en": "Agent Dashboard", "zh": "代理儀表板"},
        "agents_panel": {"en": "Agents", "zh": "代理清單"},
        "prompt_label": {"en": "Advanced global system prompt", "zh": "進階全域系統提示詞"},
        "max_tokens": {"en": "Max tokens", "zh": "最大 Token 數"},
        "model": {"en": "Model", "zh": "模型"},
        "status": {"en": "Status", "zh": "狀態"},
        "output": {"en": "Output", "zh": "輸出"},
        "no_doc": {"en": "No document or text provided yet.", "zh": "尚未提供任何文件或文字。"},
        "selected_theme": {"en": "Selected Flower Theme", "zh": "目前花卉主題"},
        "spin_wheel": {"en": "Spin Floral Luck Wheel", "zh": "旋轉花語幸運輪"},
        "light_mode": {"en": "Light mode", "zh": "亮色模式"},
        "dark_mode": {"en": "Dark mode", "zh": "深色模式"},
        "language": {"en": "Language", "zh": "介面語言"},
        "english": {"en": "English", "zh": "英文"},
        "traditional_chinese": {"en": "Traditional Chinese", "zh": "繁體中文"},
        "edit_agent_prompt": {"en": "Edit agent prompt", "zh": "編輯代理提示詞"},
        "wow_status": {"en": "WOW Status Indicators", "zh": "WOW 狀態指示"},
        "document_preview": {"en": "Document Preview", "zh": "文件預覽"},
        "token_usage": {"en": "Token Usage (approx.)", "zh": "Token 使用量（約略）"},
        "response_length": {"en": "Response length (chars)", "zh": "回應長度（字元）"},
        "tab_doc": {"en": "Document & OCR", "zh": "文件與 OCR"},
        "tab_agents": {"en": "Agents & Orchestration", "zh": "代理與流程"},
        "tab_dash": {"en": "Dashboard & Outputs", "zh": "儀表板與輸出"},
        "ocr_settings": {"en": "PDF OCR Settings", "zh": "PDF OCR 設定"},
        "ocr_method": {"en": "OCR / Extraction method", "zh": "OCR／文字擷取方法"},
        "ocr_pages": {"en": "Select pages for OCR", "zh": "選取 OCR 頁面"},
        "ocr_run": {"en": "Run OCR on selected pages", "zh": "對選取頁面執行 OCR"},
        "ocr_preview": {"en": "OCR text preview", "zh": "OCR 文字預覽"},
        "ocr_analysis": {"en": "OCR-based Summary & Entities", "zh": "OCR 摘要與實體分析"},
        "ocr_model": {"en": "Analysis model for OCR text", "zh": "OCR 文字分析模型"},
        "ocr_analyze_btn": {"en": "Generate summary & 20 entities", "zh": "產生摘要與 20 顆實體"},
        "ocr_summary": {"en": "OCR Summary (Markdown, keywords in coral)", "zh": "OCR 摘要（Markdown，關鍵字為珊瑚色）"},
        "ocr_entities_table": {"en": "20 Entities with Context", "zh": "20 個實體與其脈絡"},
        "ocr_word_graph": {"en": "Entity Word Graph", "zh": "實體關聯圖"},
    }
    if key not in TEXT:
        return key
    return TEXT[key]["zh"] if lang == "zh" else TEXT[key]["en"]


# ===============================
# THEMES: 20 FLOWER STYLES
# ===============================
FLOWER_THEMES = [
    {
        "id": "sakura_breeze",
        "label": "Sakura Breeze",
        "emoji": "🌸",
        "light": {"bg": "#fff5f8", "fg": "#3b0b19", "accent": "#ff99c8"},
        "dark": {"bg": "#2b0f1b", "fg": "#ffe6f2", "accent": "#ff7aa2"},
    },
    {
        "id": "rose_gold",
        "label": "Rose Gold",
        "emoji": "🌹",
        "light": {"bg": "#fff6f7", "fg": "#4b1114", "accent": "#f75c77"},
        "dark": {"bg": "#2a0d0f", "fg": "#ffe8ec", "accent": "#ff6b81"},
    },
    {
        "id": "lavender_dream",
        "label": "Lavender Dream",
        "emoji": "💜",
        "light": {"bg": "#f4f1ff", "fg": "#22164d", "accent": "#a78bfa"},
        "dark": {"bg": "#1b1433", "fg": "#ede9fe", "accent": "#c4b5fd"},
    },
    {
        "id": "sunflower_glow",
        "label": "Sunflower Glow",
        "emoji": "🌻",
        "light": {"bg": "#fffbea", "fg": "#3b2f0c", "accent": "#fbbf24"},
        "dark": {"bg": "#1f1303", "fg": "#fef3c7", "accent": "#facc15"},
    },
    {
        "id": "lotus_pond",
        "label": "Lotus Pond",
        "emoji": "🪷",
        "light": {"bg": "#ecfdf5", "fg": "#064e3b", "accent": "#22c55e"},
        "dark": {"bg": "#022c22", "fg": "#dcfce7", "accent": "#4ade80"},
    },
    {
        "id": "orchid_mist",
        "label": "Orchid Mist",
        "emoji": "🌺",
        "light": {"bg": "#fdf2ff", "fg": "#3b0764", "accent": "#e879f9"},
        "dark": {"bg": "#2b0b39", "fg": "#fae8ff", "accent": "#f472b6"},
    },
    {
        "id": "peony_blush",
        "label": "Peony Blush",
        "emoji": "🌷",
        "light": {"bg": "#fff1f2", "fg": "#4a041c", "accent": "#fb7185"},
        "dark": {"bg": "#3f0213", "fg": "#ffe4e6", "accent": "#fb7185"},
    },
    {
        "id": "iris_night",
        "label": "Iris Night",
        "emoji": "🪻",
        "light": {"bg": "#eff6ff", "fg": "#111827", "accent": "#6366f1"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#4f46e5"},
    },
    {
        "id": "cherry_meadow",
        "label": "Cherry Meadow",
        "emoji": "🍒",
        "light": {"bg": "#fef2f2", "fg": "#111827", "accent": "#fb7185"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "accent": "#f97316"},
    },
    {
        "id": "camellia_silk",
        "label": "Camellia Silk",
        "emoji": "🌺",
        "light": {"bg": "#fdf2f8", "fg": "#4a044e", "accent": "#ec4899"},
        "dark": {"bg": "#3b0764", "fg": "#fdf2f8", "accent": "#db2777"},
    },
    {
        "id": "magnolia_cloud",
        "label": "Magnolia Cloud",
        "emoji": "🌼",
        "light": {"bg": "#f9fafb", "fg": "#111827", "accent": "#eab308"},
        "dark": {"bg": "#0b1120", "fg": "#e5e7eb", "accent": "#f59e0b"},
    },
    {
        "id": "plum_blossom",
        "label": "Plum Blossom",
        "emoji": "🌸",
        "light": {"bg": "#fef2ff", "fg": "#4a044e", "accent": "#f97316"},
        "dark": {"bg": "#3f0e40", "fg": "#fce7f3", "accent": "#f97316"},
    },
    {
        "id": "gardenia_moon",
        "label": "Gardenia Moon",
        "emoji": "🌙",
        "light": {"bg": "#f9fafb", "fg": "#020617", "accent": "#22c55e"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#22c55e"},
    },
    {
        "id": "wisteria_rain",
        "label": "Wisteria Rain",
        "emoji": "🌧️",
        "light": {"bg": "#eef2ff", "fg": "#1e293b", "accent": "#a855f7"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#8b5cf6"},
    },
    {
        "id": "dahlia_fire",
        "label": "Dahlia Fire",
        "emoji": "🔥",
        "light": {"bg": "#fff7ed", "fg": "#1f2937", "accent": "#f97316"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "accent": "#fb923c"},
    },
    {
        "id": "bluebell_forest",
        "label": "Bluebell Forest",
        "emoji": "🔵",
        "light": {"bg": "#eff6ff", "fg": "#111827", "accent": "#3b82f6"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#60a5fa"},
    },
    {
        "id": "poppy_fields",
        "label": "Poppy Fields",
        "emoji": "🌺",
        "light": {"bg": "#fef2f2", "fg": "#1f2937", "accent": "#ef4444"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "accent": "#f87171"},
    },
    {
        "id": "lotus_dawn",
        "label": "Lotus Dawn",
        "emoji": "🌅",
        "light": {"bg": "#fefce8", "fg": "#1f2937", "accent": "#22c55e"},
        "dark": {"bg": "#1e293b", "fg": "#e5e7eb", "accent": "#10b981"},
    },
    {
        "id": "hibiscus_sunset",
        "label": "Hibiscus Sunset",
        "emoji": "🌇",
        "light": {"bg": "#fff7ed", "fg": "#1f2937", "accent": "#fb7185"},
        "dark": {"bg": "#1f2937", "fg": "#e5e7eb", "accent": "#f97316"},
    },
    {
        "id": "jasmine_night",
        "label": "Jasmine Night",
        "emoji": "🌙",
        "light": {"bg": "#f9fafb", "fg": "#1f2937", "accent": "#22c55e"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#84cc16"},
    },
]


def apply_theme(theme_id: str, dark: bool):
    theme = next((t for t in FLOWER_THEMES if t["id"] == theme_id), FLOWER_THEMES[0])
    palette = theme["dark"] if dark else theme["light"]

    css = f"""
    <style>
    body {{
        background: {palette['bg']} !important;
        color: {palette['fg']} !important;
    }}
    .stApp {{
        background: linear-gradient(135deg, {palette['bg']} 0%, #ffffff11 50%, {palette['bg']} 100%);
    }}
    .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stDataFrame, .stButton > button {{
        color: {palette['fg']} !important;
    }}
    .floral-accent {{
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        background: {palette['accent']}22;
        border: 1px solid {palette['accent']};
        color: {palette['fg']};
        font-weight: 600;
        font-size: 0.8rem;
        text-align: center;
        display: inline-block;
    }}
    .floral-badge-success {{
        background: #16a34a22;
        border-color: #22c55e;
    }}
    .floral-badge-error {{
        background: #b91c1c22;
        border-color: #ef4444;
    }}
    .floral-badge-running {{
        background: #0369a122;
        border-color: #0ea5e9;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return theme


# ===============================
# LOAD AGENTS CONFIG
# ===============================
@st.cache_resource
def load_agents_config(path: str = "agents.yaml") -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("agents", [])


AGENTS_BASE = load_agents_config()

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]


# ===============================
# SESSION STATE INIT
# ===============================
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "theme_id" not in st.session_state:
    st.session_state.theme_id = FLOWER_THEMES[0]["id"]
if "agents" not in st.session_state:
    st.session_state.agents = [
        {
            **agent,
            "status": "Pending",
            "output": "",
            "token_usage": 0,
        }
        for agent in AGENTS_BASE
    ]
if "global_prompt" not in st.session_state:
    st.session_state.global_prompt = (
        "You are part of a multi-agent analysis system called 'Floral Agentic Workflow'."
    )
if "base_doc_text" not in st.session_state:
    st.session_state.base_doc_text = ""
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "ocr_summary_html" not in st.session_state:
    st.session_state.ocr_summary_html = ""
if "ocr_entities" not in st.session_state:
    st.session_state.ocr_entities = []
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_num_pages" not in st.session_state:
    st.session_state.pdf_num_pages = 0
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "gemini": os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENAAI_API_KEY", ""),
        "grok": os.getenv("XAI_API_KEY", ""),
    }

current_theme = apply_theme(st.session_state.theme_id, st.session_state.dark_mode)


def get_accent_color():
    theme = next((t for t in FLOWER_THEMES if t["id"] == st.session_state.theme_id), FLOWER_THEMES[0])
    palette = theme["dark"] if st.session_state.dark_mode else theme["light"]
    return palette["accent"]


def get_full_doc_text() -> str:
    base = st.session_state.base_doc_text or ""
    ocr = st.session_state.ocr_text or ""
    return (base + ("\n\n[OCR]\n" + ocr if ocr else "")).strip()


# ===============================
# SIDEBAR: SETTINGS, THEME, API KEYS
# ===============================
with st.sidebar:
    st.markdown(f"### {t('settings', st.session_state.lang)}")

    lang_choice = st.radio(
        t("language", st.session_state.lang),
        options=["en", "zh"],
        format_func=lambda x: t(
            "english" if x == "en" else "traditional_chinese", st.session_state.lang
        ),
        index=0 if st.session_state.lang == "en" else 1,
    )
    st.session_state.lang = lang_choice

    mode = st.radio(
        t("light_mode", st.session_state.lang) + " / " + t("dark_mode", st.session_state.lang),
        options=["light", "dark"],
        index=1 if st.session_state.dark_mode else 0,
    )
    st.session_state.dark_mode = mode == "dark"
    current_theme = apply_theme(st.session_state.theme_id, st.session_state.dark_mode)

    st.markdown(f"### {t('theme_settings', st.session_state.lang)}")
    theme_labels = [f"{th['emoji']} {th['label']}" for th in FLOWER_THEMES]
    wheel_fig = go.Figure(
        data=[
            go.Pie(
                labels=theme_labels,
                values=[1] * len(FLOWER_THEMES),
                hole=0.4,
                textinfo="none",
            )
        ]
    )
    wheel_fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
    )
    st.plotly_chart(wheel_fig, use_container_width=True)

    if st.button(t("spin_wheel", st.session_state.lang)):
        import random

        st.session_state.theme_id = random.choice(FLOWER_THEMES)["id"]
        current_theme = apply_theme(st.session_state.theme_id, st.session_state.dark_mode)

    current_theme_obj = next(
        (th for th in FLOWER_THEMES if th["id"] == st.session_state.theme_id), FLOWER_THEMES[0]
    )
    st.markdown(
        f"**{t('selected_theme', st.session_state.lang)}:** "
        f"{current_theme_obj['emoji']} {current_theme_obj['label']}"
    )

    st.markdown(f"### {t('api_keys', st.session_state.lang)}")
    if not st.session_state.api_keys["gemini"]:
        gemini_key = st.text_input("Gemini API Key", type="password")
        if gemini_key:
            st.session_state.api_keys["gemini"] = gemini_key

    if not st.session_state.api_keys["openai"]:
        openai_key = st.text_input("OPENAAI/OpenAI API Key", type="password")
        if openai_key:
            st.session_state.api_keys["openai"] = openai_key

    if not st.session_state.api_keys["grok"]:
        grok_key = st.text_input("Grok XAI_API_KEY", type="password")
        if grok_key:
            st.session_state.api_keys["grok"] = grok_key

    st.markdown(
        "<p style='font-size: 0.75rem; opacity:0.8;'>"
        "API keys are kept in session memory only and sent directly to provider APIs."
        "</p>",
        unsafe_allow_html=True,
    )


# ===============================
# LLM CLIENT HELPERS
# ===============================
def get_gemini_client():
    key = st.session_state.api_keys.get("gemini", "")
    if not key:
        raise RuntimeError("Missing Gemini API key")
    return genai.Client(api_key=key)


def get_openai_client():
    key = st.session_state.api_keys.get("openai", "")
    if not key:
        raise RuntimeError("Missing OPENAAI/OpenAI API key")
    return OpenAI(api_key=key)


def get_grok_client():
    key = st.session_state.api_keys.get("grok", "")
    if not key:
        raise RuntimeError("Missing Grok XAI_API_KEY")
    return XAIClient(api_key=key, timeout=3600)


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Unified LLM call across Gemini, OPENAAI/OpenAI-style, and Grok.

    - Gemini: uses google-genai Client, `contents=[prompt]`, `config={...}`.
    - OPENAAI/OpenAI: uses openai.ChatCompletion-style API.
    - Grok: uses xai_sdk chat.create(...).sample().

    Returns:
        {
          "text": str,         # model response
          "usage_approx": int  # approximate token/char usage
        }
    """
    lower = model.lower()

    # ---------- GEMINI ----------
    if lower.startswith("gemini"):
        client = get_gemini_client()

        # Combine system + user into a single text prompt
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"

        # For text-only calls, contents can be a list of strings
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            config={"max_output_tokens": max_tokens},
        )

        text = getattr(resp, "text", "") or ""
        usage_approx = len(prompt) + len(text)
        return {"text": text, "usage_approx": usage_approx}

    # ---------- OPENAAI / OPENAI (gpt-*) ----------
    if lower.startswith("gpt-"):
        client = get_openai_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )

        text = resp.choices[0].message.content
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None and hasattr(usage_obj, "total_tokens"):
            usage_approx = usage_obj.total_tokens
        else:
            usage_approx = len(system_prompt) + len(user_prompt) + len(text)

        return {"text": text, "usage_approx": usage_approx}

    # ---------- GROK (grok-*) ----------
    if lower.startswith("grok-"):
        client = get_grok_client()

        chat = client.chat.create(model=model)
        chat.append(grok_system(system_prompt))
        chat.append(grok_user(user_prompt))

        resp = chat.sample()
        text = str(resp.content)
        usage_approx = len(system_prompt) + len(user_prompt) + len(text)

        return {"text": text, "usage_approx": usage_approx}

    # ---------- UNSUPPORTED ----------
    raise ValueError(f"Unsupported model: {model}")

# ===============================
# PDF HELPERS
# ===============================
def load_pdf_metadata(pdf_bytes: bytes) -> int:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


def preview_pdf_page(pdf_bytes: bytes, page_number: int, dpi: int = 120) -> Image.Image:
    images = convert_from_bytes(
        pdf_bytes,
        first_page=page_number,
        last_page=page_number,
        dpi=dpi,
    )
    return images[0]


def extract_text_pypdf2(pdf_bytes: bytes, page_numbers: List[int]) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for p in page_numbers:
        idx = p - 1
        if idx < 0 or idx >= len(reader.pages):
            continue
        page = reader.pages[idx]
        page_text = page.extract_text() or ""
        texts.append(f"[Page {p}]\n{page_text}")
    return "\n\n".join(texts)


def extract_text_pdfplumber(pdf_bytes: bytes, page_numbers: List[int]) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in page_numbers:
            idx = p - 1
            if idx < 0 or idx >= len(pdf.pages):
                continue
            page = pdf.pages[idx]
            page_text = page.extract_text() or ""
            texts.append(f"[Page {p}]\n{page_text}")
    return "\n\n".join(texts)


def ocr_pdf_pytesseract(pdf_bytes: bytes, page_numbers: List[int], dpi: int = 200) -> str:
    texts = []
    for p in page_numbers:
        images = convert_from_bytes(
            pdf_bytes,
            first_page=p,
            last_page=p,
            dpi=dpi,
        )
        for img in images:
            txt = pytesseract.image_to_string(img)
            texts.append(f"[Page {p}]\n{txt}")
    return "\n\n".join(texts)


def ocr_pdf_gemini(
    pdf_bytes: bytes,
    page_numbers: List[int],
    model: str = "gpt-4o-mini",
) -> str:
    """
    OCR the specified PDF pages using gpt-4o-mini vision via the OPENAAI/OpenAI-style client.

    - Converts each requested page to a JPEG with pdf2image.
    - Encodes JPEG as base64 and sends as an image_url in chat.completions.
    - Returns concatenated plain text with [Page N] markers.

    NOTE: The `model` argument from callers is ignored in favor of 'gpt-4o-mini'
    to avoid changing call sites; this function always uses gpt-4o-mini.
    """
    client = get_openai_client()
    vision_model = "gpt-4o-mini"  # force use of gpt-4o-mini for OCR

    texts: List[str] = []

    for p in page_numbers:
        # Convert the selected page to image(s)
        images = convert_from_bytes(
            pdf_bytes,
            first_page=p,
            last_page=p,
            dpi=200,
        )

        for img in images:
            # Encode page as JPEG bytes and then base64
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

            import base64  # already imported at top of app, but safe if repeated
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{image_b64}"

            # Chat with image using OpenAI/OPENAAI-style vision format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe all legible text from this page. "
                                "Return plain text only, no markdown or commentary."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ]

            resp = client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_tokens=2048,
            )

            page_text = resp.choices[0].message.content or ""
            texts.append(f"[Page {p}]\n{page_text}")

    return "\n\n".join(texts)
# ===============================
# OCR ANALYSIS (SUMMARY + ENTITIES + WORD GRAPH)
# ===============================
def analyze_ocr_text(ocr_text: str, model: str) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert summarization and entity-extraction assistant.\n"
        "- Work ONLY with the provided OCR text.\n"
        "- Be precise and avoid hallucinations.\n"
    )

    user_prompt = f"""
You will perform two tasks on the following OCR text:

1. Create a concise but information-dense Markdown summary.
   - Use headings and bullet points where appropriate.
   - For 10–20 key domain keywords, surround them with @@, e.g. @@keyword@@.
   - Do NOT use any other special markers.

2. Extract exactly 20 important entities.
   - For each entity, produce:
     - name: short canonical name.
     - type: short type label (Person, Organization, Concept, Metric, Date, Location, etc.).
     - context: 1–2 sentence explanation with citation to the OCR text.
     - related: list of 0–5 names of other entities from this set that are most related.

Return ONLY a single JSON object, wrapped in a JSON code block:

```json
{{
  "summary_markdown": "...",
  "entities": [
    {{
      "name": "string",
      "type": "string",
      "context": "string",
      "related": ["string", "..."]
    }}
  ]
}}
```

Do not include any explanations outside the JSON block.

OCR_TEXT:
\"\"\"{ocr_text[:10000]}\"\"\"
"""

    result = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=3000,
    )
    text = result["text"]

    match = re.search(r"```json(.*?)```", text, re.S | re.I)
    if not match:
        match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Could not find JSON block in model output.")

    json_str = match.group(1).strip()
    data = json.loads(json_str)
    return data


def build_word_graph(entities: List[Dict[str, Any]]) -> go.Figure:
    G = nx.Graph()
    names = [e.get("name", "") for e in entities if e.get("name")]
    name_set = set(names)

    for e in entities:
        name = e.get("name")
        if not name:
            continue
        G.add_node(name, type=e.get("type", ""))
        for rel in e.get("related", []) or []:
            if rel in name_set and rel != name:
                G.add_edge(name, rel)

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No entities / edges for word graph",
            xaxis_visible=False,
            yaxis_visible=False,
        )
        return fig

    pos = nx.spring_layout(G, k=0.7, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#94a3b8"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=get_accent_color(),
            size=14,
            line=dict(width=1, color="#ffffff"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        title=t("ocr_word_graph", st.session_state.lang),
    )
    return fig


# ===============================
# MAIN LAYOUT
# ===============================
st.markdown(f"## {t('app_title', st.session_state.lang)}")

col_gt, col_gp = st.columns([1, 1.2])
with col_gt:
    st.markdown(f"#### {t('global_task', st.session_state.lang)}")
    global_task = st.text_area(
        "",
        height=120,
        key="global_task",
        placeholder="e.g., Summarize this document for executives and list open risks.",
    )
with col_gp:
    st.markdown(f"#### {t('prompt_label', st.session_state.lang)}")
    st.session_state.global_prompt = st.text_area(
        "",
        value=st.session_state.global_prompt,
        height=150,
        key="advanced_prompt",
    )

tab_doc, tab_agents, tab_dash = st.tabs(
    [t("tab_doc", st.session_state.lang), t("tab_agents", st.session_state.lang), t("tab_dash", st.session_state.lang)]
)

# ===============================
# TAB 1: DOCUMENT & OCR
# ===============================
with tab_doc:
    st.markdown(f"### {t('upload_label', st.session_state.lang)}")
    uploaded = st.file_uploader(
        "",
        type=["pdf", "txt", "md", "markdown", "json"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        ext = uploaded.name.lower().split(".")[-1]
        if ext == "pdf":
            st.session_state.pdf_bytes = uploaded.read()
            st.session_state.pdf_num_pages = load_pdf_metadata(st.session_state.pdf_bytes)
            st.session_state.base_doc_text = ""
        else:
            st.session_state.pdf_bytes = None
            st.session_state.pdf_num_pages = 0
            raw = uploaded.read().decode("utf-8", errors="ignore")
            if ext == "json":
                try:
                    data = json.loads(raw)
                    st.session_state.base_doc_text = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception:
                    st.session_state.base_doc_text = raw
            else:
                st.session_state.base_doc_text = raw

    if st.session_state.pdf_bytes:
        st.markdown(f"#### PDF ({st.session_state.pdf_num_pages} pages)")

        col_pdf_left, col_pdf_right = st.columns([1.1, 1.3])

        with col_pdf_left:
            preview_page = st.slider(
                "Preview page",
                min_value=1,
                max_value=st.session_state.pdf_num_pages,
                value=1,
                step=1,
            )
            try:
                img = preview_pdf_page(st.session_state.pdf_bytes, preview_page)
                st.image(img, caption=f"Page {preview_page}", use_column_width=True)
            except Exception as e:
                st.warning(f"Preview error: {e}")

        with col_pdf_right:
            st.markdown(f"#### {t('ocr_settings', st.session_state.lang)}")
            ocr_method = st.radio(
                t("ocr_method", st.session_state.lang),
                options=[
                    "pypdf2",
                    "pdfplumber",
                    "pytesseract_image",
                    "llm_gemini",
                ],
                format_func=lambda x: {
                    "pypdf2": "PyPDF2 (fast text extraction)",
                    "pdfplumber": "pdfplumber (layout-aware text extraction)",
                    "pytesseract_image": "pdf2image + pytesseract (image OCR, scanned PDFs)",
                    "llm_gemini": "LLM OCR (Gemini 2.5 Flash vision)",
                }[x],
            )

            pages = list(range(1, st.session_state.pdf_num_pages + 1))
            selected_pages = st.multiselect(
                t("ocr_pages", st.session_state.lang),
                options=pages,
                default=pages,
            )

            if ocr_method == "llm_gemini":
                st.markdown(
                    "<small>LLM OCR currently uses Gemini 2.5 Flash multimodal.</small>",
                    unsafe_allow_html=True,
                )

            if st.button(t("ocr_run", st.session_state.lang)):
                if not selected_pages:
                    st.warning("Please select at least one page.")
                else:
                    with st.spinner("Running OCR / extraction on selected pages..."):
                        try:
                            if ocr_method == "pypdf2":
                                text = extract_text_pypdf2(st.session_state.pdf_bytes, selected_pages)
                            elif ocr_method == "pdfplumber":
                                text = extract_text_pdfplumber(st.session_state.pdf_bytes, selected_pages)
                            elif ocr_method == "pytesseract_image":
                                text = ocr_pdf_pytesseract(st.session_state.pdf_bytes, selected_pages)
                            elif ocr_method == "llm_gemini":
                                text = ocr_pdf_gemini(
                                    st.session_state.pdf_bytes,
                                    selected_pages,
                                    model="gemini-2.5-flash",
                                )
                            else:
                                text = ""
                            if st.session_state.ocr_text:
                                st.session_state.ocr_text += "\n\n" + text
                            else:
                                st.session_state.ocr_text = text
                            st.success("OCR / extraction completed.")
                        except Exception as e:
                            st.error(f"OCR error: {e}")

    st.markdown(f"#### {t('document_preview', st.session_state.lang)}")
    full_doc = get_full_doc_text()
    if full_doc:
        st.text_area("", value=full_doc[:8000], height=220, key="doc_preview")
    else:
        st.info(t("no_doc", st.session_state.lang))

    st.markdown(f"**{t('paste_text', st.session_state.lang)}**")
    pasted = st.text_area(
        "",
        value="",
        height=160,
        key="doc_paste",
        placeholder="Type or paste additional content here...",
    )
    if pasted:
        if st.session_state.base_doc_text:
            st.session_state.base_doc_text += "\n\n" + pasted
        else:
            st.session_state.base_doc_text = pasted

    if st.session_state.ocr_text:
        st.markdown("---")
        st.markdown(f"### {t('ocr_analysis', st.session_state.lang)}")
        with st.expander(t("ocr_preview", st.session_state.lang)):
            st.text_area(
                "",
                value=st.session_state.ocr_text[:8000],
                height=220,
                key="ocr_text_preview",
            )

        analysis_model = st.selectbox(
            t("ocr_model", st.session_state.lang),
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in AVAILABLE_MODELS else 0,
        )

        if st.button(t("ocr_analyze_btn", st.session_state.lang)):
            with st.spinner("Analyzing OCR text to create summary & entities..."):
                try:
                    data = analyze_ocr_text(st.session_state.ocr_text, analysis_model)
                    summary_md = data.get("summary_markdown", "")
                    entities = data.get("entities", [])

                    def coralify(match):
                        word = match.group(1)
                        return f"<span style='color:coral; font-weight:600'>{word}</span>"

                    summary_html = re.sub(r"@@(.*?)@@", coralify, summary_md)
                    st.session_state.ocr_summary_html = summary_html
                    st.session_state.ocr_entities = entities
                    st.success("OCR summary & entities generated.")
                except Exception as e:
                    st.error(f"OCR analysis error: {e}")

        if st.session_state.ocr_summary_html:
            st.markdown(f"#### {t('ocr_summary', st.session_state.lang)}")
            st.markdown(st.session_state.ocr_summary_html, unsafe_allow_html=True)

        if st.session_state.ocr_entities:
            st.markdown(f"#### {t('ocr_entities_table', st.session_state.lang)}")
            df = pd.DataFrame(st.session_state.ocr_entities)
            st.dataframe(df, use_container_width=True)

            st.markdown(f"#### {t('ocr_word_graph', st.session_state.lang)}")
            fig_graph = build_word_graph(st.session_state.ocr_entities)
            st.plotly_chart(fig_graph, use_container_width=True)


# ===============================
# TAB 2: AGENTS & ORCHESTRATION
# ===============================
with tab_agents:
    st.markdown(f"### {t('agents_panel', st.session_state.lang)}")

    if not st.session_state.agents:
        st.warning("No agents found in agents.yaml")
    else:
        for i, agent in enumerate(st.session_state.agents):
            with st.expander(f"{agent.get('name', agent['id'])} [{agent['id']}]"):
                c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
                with c1:
                    st.checkbox(
                        "Enabled",
                        value=agent.get("enabled", True),
                        key=f"agent_enabled_{agent['id']}",
                    )
                with c2:
                    st.session_state.agents[i]["model"] = st.selectbox(
                        t("model", st.session_state.lang),
                        options=AVAILABLE_MODELS,
                        index=AVAILABLE_MODELS.index(agent["model"])
                        if agent["model"] in AVAILABLE_MODELS
                        else 0,
                        key=f"agent_model_{agent['id']}",
                    )
                with c3:
                    st.session_state.agents[i]["max_tokens"] = st.slider(
                        t("max_tokens", st.session_state.lang),
                        min_value=100,
                        max_value=12000,
                        value=int(agent.get("max_tokens", 2048)),
                        step=100,
                        key=f"agent_maxtok_{agent['id']}",
                    )
                with c4:
                    st.markdown(
                        f"<span class='floral-accent'>{t('status', st.session_state.lang)}: "
                        f"{agent.get('status', 'Pending')}</span>",
                        unsafe_allow_html=True,
                    )

                if st.button(t("edit_agent_prompt", st.session_state.lang), key=f"edit_prompt_{agent['id']}"):
                    st.session_state[f"show_prompt_modal_{agent['id']}"] = True

                if st.session_state.get(f"show_prompt_modal_{agent['id']}", False):
                    st.markdown("---")
                    st.markdown(f"**System Prompt for {agent['name']}**")
                    new_prompt = st.text_area(
                        "",
                        value=agent.get("system_prompt", ""),
                        height=200,
                        key=f"prompt_text_{agent['id']}",
                    )
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("Save", key=f"save_prompt_{agent['id']}"):
                            st.session_state.agents[i]["system_prompt"] = new_prompt
                            st.session_state[f"show_prompt_modal_{agent['id']}"] = False
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_prompt_{agent['id']}"):
                            st.session_state[f"show_prompt_modal_{agent['id']}"] = False

    st.markdown("---")
    if st.button(t("run_agents", st.session_state.lang)):
        full_doc = get_full_doc_text()
        if not full_doc and not global_task:
            st.warning("Please provide document text or a task before running agents.")
        else:
            for i, agent in enumerate(st.session_state.agents):
                enabled = st.session_state.get(f"agent_enabled_{agent['id']}", agent.get("enabled", True))
                if not enabled:
                    st.session_state.agents[i]["status"] = "Skipped"
                    continue

                st.session_state.agents[i]["status"] = "Running"
                with st.spinner(f"Running agent: {agent['name']} ({agent['model']})"):
                    try:
                        composed_user_prompt = ""
                        if global_task:
                            composed_user_prompt += f"Global task:\n{global_task}\n\n"
                        if full_doc:
                            composed_user_prompt += "Document content:\n" + full_doc[:12000]
                        else:
                            composed_user_prompt += "(No document provided.)"

                        full_system_prompt = (
                            st.session_state.global_prompt.strip()
                            + "\n\n--- Agent-specific instructions ---\n"
                            + agent.get("system_prompt", "").strip()
                        )

                        result = call_llm(
                            model=agent["model"],
                            system_prompt=full_system_prompt,
                            user_prompt=composed_user_prompt,
                            max_tokens=int(agent.get("max_tokens", 2048)),
                        )
                        st.session_state.agents[i]["output"] = result["text"]
                        st.session_state.agents[i]["token_usage"] = result["usage_approx"]
                        st.session_state.agents[i]["status"] = "Success"
                    except Exception as e:
                        st.session_state.agents[i]["output"] = f"Error: {e}"
                        st.session_state.agents[i]["token_usage"] = 0
                        st.session_state.agents[i]["status"] = "Error"


# ===============================
# TAB 3: DASHBOARD & OUTPUTS
# ===============================
with tab_dash:
    st.markdown(f"### {t('wow_status', st.session_state.lang)} & {t('dashboard', st.session_state.lang)}")

    if st.session_state.agents:
        cols = st.columns(len(st.session_state.agents))
        for col, agent in zip(cols, st.session_state.agents):
            status = agent.get("status", "Pending")
            if status == "Success":
                badge_class = "floral-badge-success"
            elif status == "Error":
                badge_class = "floral-badge-error"
            elif status == "Running":
                badge_class = "floral-badge-running"
            else:
                badge_class = ""
            with col:
                st.markdown(
                    f"<div class='floral-accent {badge_class}'>"
                    f"{agent.get('name', agent['id'])}<br/>"
                    f"<small>Status: {status}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        names = [a.get("name", a["id"]) for a in st.session_state.agents]
        token_usage = [a.get("token_usage", 0) for a in st.session_state.agents]
        resp_lengths = [len(a.get("output", "") or "") for a in st.session_state.agents]

        dash_col1, dash_col2 = st.columns(2)
        with dash_col1:
            st.markdown(f"**{t('token_usage', st.session_state.lang)}**")
            accent = get_accent_color()
            fig_tokens = go.Figure(
                data=[go.Bar(x=names, y=token_usage, marker_color=accent)]
            )
            fig_tokens.update_layout(
                xaxis_title="Agent",
                yaxis_title="Tokens (approx.)",
                height=320,
                margin=dict(l=40, r=20, t=30, b=80),
            )
            st.plotly_chart(fig_tokens, use_container_width=True)

        with dash_col2:
            st.markdown(f"**{t('response_length', st.session_state.lang)}**")
            accent = get_accent_color()
            fig_len = go.Figure(
                data=[go.Bar(x=names, y=resp_lengths, marker_color=accent)]
            )
            fig_len.update_layout(
                xaxis_title="Agent",
                yaxis_title="Characters",
                height=320,
                margin=dict(l=40, r=20, t=30, b=80),
            )
            st.plotly_chart(fig_len, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### {t('output', st.session_state.lang)}")

    for agent in st.session_state.agents:
        with st.expander(f"{agent.get('name', agent['id'])} ({agent['status']})"):
            st.markdown(
                f"**Model:** `{agent['model']}`  |  **Tokens (approx):** {agent.get('token_usage', 0)}"
            )
            st.markdown(agent.get("output", "") or "_No output yet._")
