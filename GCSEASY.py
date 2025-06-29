import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import ast
import random
import openai
import tempfile
import os
import re
import io
import requests
from gtts import gTTS
from datetime import datetime

# --- Helper Functions ---
def clean_tts_text(text):
    if not text.strip():
        return ""
    text = text.replace("$", "").replace("\\", "")
    text = re.sub(r'frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'\1 divided by \2', text)
    text = re.sub(r'(\w+)\^(\w+)', r'\1 to the power of \2', text)
    text = re.sub(r'(\w+)_([a-zA-Z0-9]+)', r'\1 sub \2', text)
    replacements = {
        "*": " times ", "/": " divided by ", "=": " equals ", "+": " plus ", "-": " minus ",
        "±": " plus or minus ", "≈": " approximately equal to ", "≠": " not equal to ",
        "≥": " greater than or equal to ", "≤": " less than or equal to ", ">": " greater than ", "<": " less than ",
        "√": " square root of ", "π": " pi ", "∑": " sum of ", "∞": " infinity ", "θ": " theta ",
        "Δ": " delta ", "∫": " integral of ", "∂": " partial derivative ", "⋅": " dot ",
        "∈": " belongs to ", "∉": " does not belong to ", "∩": " intersection ", "∪": " union ",
        "∀": " for all ", "∃": " there exists ", "→": " approaches ", "≅": " congruent to "
    }
    for symbol, spoken in replacements.items():
        text = text.replace(symbol, spoken)
    return re.sub(r'\s+', ' ', text).strip()

def gtts_text_to_speech(text, lang='en'):
    if not text.strip():
        raise ValueError("No text to speak.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        gTTS(text=text, lang=lang).save(fp.name)
        return fp.name

def format_duration(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"

# --- CONFIG ---
st.set_page_config(page_title="GCSE Maths App", layout="centered")
openai.api_key = st.secrets["openai"]["api_key"]
if st.secrets.get("show_debug", False):
    st.write("🔑 API key loaded:", bool(openai.api_key))

# --- DATA LOADING ---
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1NGilcKWzaVdi2S6PbcVOczI3qCWswbcB"
    response = requests.get(url)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    df = pd.read_parquet(buffer)
    def parse_options(x):
        if isinstance(x, list): return x
        try: return ast.literal_eval(x)
        except: return []
    df["options"] = df["options"].apply(parse_options)
    return df

data = load_data()

# --- UI ELEMENTS ---
st.title("🧠 GCSE Maths App")
read_aloud = st.sidebar.checkbox("🔊 Read Aloud Help Messages", value=True)

st.sidebar.markdown("### TTS Settings")
language_options = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi"}
language_label = st.sidebar.selectbox("🌍 Language", list(language_options.keys()))
language = language_options[language_label]
translate_help = st.sidebar.checkbox("🌐 Translate Help?", value=(language != "en"))

# --- QUIZ LOGIC ---
if "user_name" not in st.session_state:
    name = st.text_input("Enter your name:")
    levels = ["All"] + sorted(data["level"].dropna().astype(str).unique())
    subjects = ["All"] + sorted(data["subject"].dropna().astype(str).unique())
    level = st.selectbox("Choose Level", levels)
    subject = st.selectbox("Choose Subject", subjects)
    if st.button("Start Quiz") and name:
        st.session_state.update({
            "user_name": name,
            "selected_level": level,
            "selected_subject": subject,
            "current_index": 0,
            "results": [],
            "quiz_start_time": int(time.time()),
            "help_response": None
        })
        st.rerun()

# --- FILTER DATA ---
filtered = data.copy()
if "selected_level" in st.session_state and "selected_subject" in st.session_state:
    level = st.session_state.selected_level
    subject = st.session_state.selected_subject
    if level != "All":
        filtered = filtered[filtered["level"].astype(str).str.lower() == level.lower()]
    if subject != "All":
        filtered = filtered[filtered["subject"].astype(str).str.lower() == subject.lower()]
    filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    if filtered.empty:
        st.warning("No questions available for this level and subject. Please choose different filters.")
        if st.button("Restart"):
            st.session_state.clear()
            st.rerun()
        st.stop()

if st.session_state.get("current_index", 0) >= len(filtered):
    st.success("✅ Quiz Finished!")
    df_results = pd.DataFrame(st.session_state.results)

    if not df_results.empty:
        df_results["time_taken_readable"] = df_results["time_taken"].apply(format_duration)
        avg_times = df_results.groupby("level")["time_taken"].mean()
        st.bar_chart(avg_times)
        counts = df_results.groupby(["level", "correct"]).size().unstack(fill_value=0)
        st.bar_chart(counts)

        results_with_readable = df_results.to_dict(orient="records")

        def convert_to_builtin_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_builtin_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_builtin_types(elem) for elem in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        safe_results = convert_to_builtin_types(results_with_readable)

        st.download_button(
            "Download Results JSON",
            data=json.dumps(safe_results, indent=2),
            file_name="results.json",
            mime="application/json"
        )

    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()

# --- Show current question ---
if st.session_state.get("current_index", 0) < len(filtered):
    q = filtered.iloc[st.session_state.current_index]
    st.markdown(f"#### {q['question']}")
    if isinstance(q.get("decoded_image"), dict) and "bytes" in q["decoded_image"]:
        st.image(q["decoded_image"]["bytes"])
    answer_key = f"answer_{st.session_state.current_index}"
    if f"start_time_{answer_key}" not in st.session_state:
        st.session_state[f"start_time_{answer_key}"] = time.time()
        st.session_state["help_response"] = None
    options = q["options"]
    response = st.radio("Choose an answer:", random.sample(options, len(options)), key=answer_key) if options else st.text_input("Your answer:", key=answer_key)
    col1, col2, col3 = st.columns(3)
    submit_clicked = col1.button("Submit", key=f"submit_{answer_key}")
    finish_clicked = col2.button("Finish Quiz", key=f"finish_{answer_key}")
    help_clicked = col3.button("Ask for Help", key=f"help_{answer_key}")

    help_key = f"help_{st.session_state.current_index}_{language if translate_help else 'en'}"
    if help_clicked:
        if help_key in st.session_state:
            st.session_state.help_response = st.session_state[help_key]
        else:
            with st.spinner("Getting guidance from ChatGPT..."):
                help_prompt = f"You are a friendly GCSE maths tutor. {'Respond in ' + language_label.lower() + '.' if translate_help else ''} Question: {q['question']}\n"
                if options: help_prompt += f"Options: {options}\n"
                if subject != "All": help_prompt += f"Subject: {subject}\n"
                if level != "All": help_prompt += f"Level: {level}\n"
                try:
                    res = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful GCSE maths tutor."},
                            {"role": "user", "content": help_prompt}
                        ],
                        max_tokens=200,
                        temperature=0.4
                    )
                    st.session_state[help_key] = st.session_state.help_response = res.choices[0].message.content.strip()
                except Exception as e:
                    st.session_state.help_response = f"Error: {e}"

    if st.session_state.get("help_response"):
        raw_text = st.session_state.help_response.strip()
        help_text = clean_tts_text(raw_text)
        st.info(raw_text)
        audio_key = f"audio_{st.session_state.current_index}_{language}"
        if read_aloud and help_text:
            if audio_key not in st.session_state:
                try:
                    audio_path = gtts_text_to_speech(help_text, lang=language)
                    with open(audio_path, 'rb') as f:
                        st.session_state[audio_key] = {"data": f.read(), "format": "audio/mp3"}
                    os.remove(audio_path)
                except Exception as e:
                    st.warning(f"TTS failed: {e}")
            audio_blob = st.session_state.get(audio_key)
            if audio_blob:
                st.audio(audio_blob["data"], format=audio_blob["format"])

    if submit_clicked and response:
        time_taken = time.time() - st.session_state[f"start_time_{answer_key}"]
        correct = str(response).strip().lower() == str(q["answer"]).strip().lower()
        st.success("Correct!") if correct else st.error(f"Incorrect. Correct answer: {q['answer']}")
        st.session_state.results.append({
            "id": q["id"],
            "question": q["question"],
            "user_answer": response,
            "correct": correct,
            "level": q.get("level", ""),
            "subject": q.get("subject", ""),
            "time_taken": time_taken
        })
        st.session_state.current_index += 1
        st.session_state.help_response = None
        st.rerun()

    if finish_clicked:
        st.session_state.current_index = len(filtered)
        st.session_state.help_response = None
        st.rerun()