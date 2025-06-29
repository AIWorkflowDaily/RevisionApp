import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import ast
import random
import openai
import pyttsx3
from gtts import gTTS
import tempfile
import os
import re

# --- TTS CLEANING ---
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
        "≥": " greater than or equal to ", "≤": " less than or equal to ", ">": " greater than ",
        "<": " less than ", "√": " square root of ", "π": " pi ", "∑": " sum of ", "∞": " infinity ",
        "θ": " theta ", "Δ": " delta ", "∫": " integral of ", "∂": " partial derivative ",
        "⋅": " dot ", "∈": " belongs to ", "∉": " does not belong to ", "∩": " intersection ",
        "∪": " union ", "∀": " for all ", "∃": " there exists ", "→": " approaches ",
        "≅": " congruent to "
    }
    for symbol, spoken in replacements.items():
        text = text.replace(symbol, spoken)

    return re.sub(r'\s+', ' ', text).strip()

# --- TTS FUNCTIONS ---
def gtts_text_to_speech(text, lang='en'):
    if not text.strip():
        raise ValueError("No text to speak.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang=lang)
        tts.save(fp.name)
        return fp.name

def text_to_speech(text, voice_id=None):
    if not text.strip():
        raise ValueError("No text to speak.")
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    if voice_id:
        engine.setProperty('voice', voice_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        engine.save_to_file(text, fp.name)
        engine.runAndWait()
        return fp.name

# --- CONFIG ---
st.set_page_config(page_title="GCSE Maths App", layout="centered")
openai.api_key = st.secrets.get("openai_api_key", "")

@st.cache_data
def load_data():
    df = pd.read_parquet("C:/Users/Bevan/Documents/GCSE/DATASET/mathquestions.parquet")
    def parse_options(x):
        if isinstance(x, list): return x
        try: return ast.literal_eval(x)
        except: return []
    df["options"] = df["options"].apply(parse_options)
    return df

data = load_data()

# --- STYLES ---
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        .stButton>button, .stSelectbox, .stTextInput>div>input { border-radius: 12px; padding: 0.75em 1em; font-size: 16px; }
        .stTextInput>div>input, .stTextInput input { border: 1px solid #ccc; background-color: white; }
        .stRadio > div { border: 1px solid #ccc; border-radius: 12px; padding: 1em; background-color: #fff; }
    </style>
""", unsafe_allow_html=True)

if "quiz_start_time" not in st.session_state:
    st.session_state.quiz_start_time = time.time()

st.title("🧠 GCSE Maths App")

# --- Sidebar Controls ---
read_aloud = st.sidebar.checkbox("🔊 Read Aloud Help Messages", value=True)
st.sidebar.markdown("### TTS Settings")

language_options = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi"
}
language_label = st.sidebar.selectbox("🌍 Language", list(language_options.keys()))
language = language_options[language_label]
translate_help = st.sidebar.checkbox("🌐 Translate Help?", value=(language != "en"))

engine = pyttsx3.init()
voices = engine.getProperty('voices')
voice_names = [f"{v.name} ({v.id})" for v in voices]
selected_voice = st.sidebar.selectbox("🗣 Voice (Offline)", voice_names)
selected_voice_id = voices[voice_names.index(selected_voice)].id

# --- Main Quiz Flow ---
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
            "quiz_start_time": time.time(),
            "help_response": None
        })
        st.rerun()
else:
    total_elapsed = int(time.time() - st.session_state.quiz_start_time)
    st.markdown(f"### ⏱ Total Time: {total_elapsed // 60:02}:{total_elapsed % 60:02}")

    filtered = data.copy()
    level = st.session_state.selected_level
    subject = st.session_state.selected_subject
    if level != "All":
        filtered = filtered[filtered["level"].astype(str).str.lower() == level.lower()]
    if subject != "All":
        filtered = filtered[filtered["subject"].astype(str).str.lower() == subject.lower()]
    filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    if st.session_state.current_index >= len(filtered):
        st.markdown("## ✅ Quiz Finished!")
        df_results = pd.DataFrame(st.session_state.results)
        if not df_results.empty:
            st.markdown("### 📊 Average Time per Question (by Level)")
            avg_times = df_results.groupby("level")["time_taken"].mean()
            fig1, ax1 = plt.subplots()
            avg_times.plot(kind="bar", ax=ax1)
            ax1.set_ylabel("Time (s)")
            st.pyplot(fig1)

            st.markdown("### 📈 Correct vs Incorrect (by Level)")
            counts = df_results.groupby(["level", "correct"]).size().unstack(fill_value=0)
            fig2, ax2 = plt.subplots()
            counts.plot(kind="bar", stacked=True, ax=ax2)
            ax2.set_ylabel("Number of Questions")
            ax2.legend(["Incorrect", "Correct"])
            st.pyplot(fig2)

        serializable_results = json.dumps(st.session_state.results, indent=4, default=str)
        st.download_button("Download Results JSON", data=serializable_results, file_name="results.json", mime="application/json")
        if st.button("Restart"):
            st.session_state.clear()
            st.rerun()
    else:
        q = filtered.iloc[st.session_state.current_index]
        st.markdown(f"#### {q['question']}")
        if isinstance(q.get("decoded_image"), dict) and "bytes" in q["decoded_image"]:
            st.image(q["decoded_image"]["bytes"])

        answer_key = f"answer_{st.session_state.current_index}"
        if f"start_time_{answer_key}" not in st.session_state:
            st.session_state[f"start_time_{answer_key}"] = time.time()
            st.session_state["help_response"] = None

        response = st.radio("Choose an answer:", q["options"], key=answer_key) if isinstance(q["options"], list) and q["options"] else st.text_input("Your answer:", key=answer_key)

        col1, col2, col3 = st.columns(3)
        submit_clicked = col1.button("Submit", key=f"submit_{answer_key}")
        finish_clicked = col2.button("Finish Quiz", key=f"finish_{answer_key}")
        help_clicked = col3.button("Ask for Help", key=f"help_{answer_key}")

        help_key = f"help_{st.session_state.current_index}_{language if translate_help else 'en'}"

        if help_clicked:
            if help_key in st.session_state:
                st.session_state["help_response"] = st.session_state[help_key]
            else:
                with st.spinner("Getting guidance from ChatGPT..."):
                    help_prompt = (
                        f"You are a friendly GCSE maths tutor. {'Respond in ' + language_label.lower() + '.' if translate_help else ''} "
                        f"Give a hint or some guidance for this question:\n\nQuestion: {q['question']}\n"
                    )
                    if isinstance(q["options"], list) and q["options"]:
                        help_prompt += f"Options: {q['options']}\n"
                    if subject != "All": help_prompt += f"Subject: {subject}\n"
                    if level != "All": help_prompt += f"Level: {level}\n"
                    try:
                        response_chat = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful GCSE maths tutor."},
                                {"role": "user", "content": help_prompt}
                            ],
                            max_tokens=200,
                            temperature=0.4
                        )
                        help_response = response_chat.choices[0].message.content.strip()
                        st.session_state[help_key] = help_response
                        st.session_state["help_response"] = help_response
                    except Exception as e:
                        st.session_state["help_response"] = f"Error: {e}"

        if st.session_state.get("help_response"):
            raw_text = st.session_state["help_response"].strip()
            help_text = clean_tts_text(raw_text)
            st.info(raw_text)

            audio_cache_key = f"audio_{st.session_state.current_index}_{language}"
            if read_aloud and help_text:
                if audio_cache_key not in st.session_state:
                    try:
                        audio_path = gtts_text_to_speech(help_text, lang=language)
                        with open(audio_path, 'rb') as audio_file:
                            st.session_state[audio_cache_key] = {"data": audio_file.read(), "format": "audio/mp3"}
                        os.remove(audio_path)
                    except Exception as gtts_err:
                        st.warning(f"gTTS failed: {gtts_err}. Falling back to offline voice.")
                        try:
                            audio_path = text_to_speech(help_text, voice_id=selected_voice_id)
                            with open(audio_path, 'rb') as audio_file:
                                st.session_state[audio_cache_key] = {"data": audio_file.read(), "format": "audio/wav"}
                            os.remove(audio_path)
                        except Exception as pyttsx_err:
                            st.error(f"Offline TTS also failed: {pyttsx_err}")

                audio_blob = st.session_state.get(audio_cache_key)
                if audio_blob:
                    st.audio(audio_blob["data"], format=audio_blob["format"])

        if submit_clicked and response:
            time_taken = time.time() - st.session_state[f"start_time_{answer_key}"]
            correct = str(response).strip().lower() == str(q["answer"]).strip().lower()
            st.success("Correct!") if correct else st.error(f"Incorrect. Correct answer: {q['answer']}")
            st.session_state.results.append({
                "id": q["id"], "question": q["question"], "user_answer": response,
                "correct": correct, "level": q.get("level", ""), "subject": q.get("subject", ""),
                "time_taken": time_taken
            })
            st.session_state.current_index += 1
            st.session_state["help_response"] = None
            st.rerun()

        if finish_clicked:
            st.session_state.current_index = len(filtered)
            st.session_state["help_response"] = None
            st.rerun()