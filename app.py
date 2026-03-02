import streamlit as st
from faster_whisper import WhisperModel
import ollama
import os

# Настройка страницы
st.set_page_config(page_title="AutoDoc: Из аудио в ТЗ", layout="wide")
st.title("🎙️ Автоматическое составление тех. документации")

# Инициализация модели Whisper (локально)
@st.cache_resource
def load_whisper():
    # 'base' — быстро, 'large-v3' — максимально точно
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_whisper()

uploaded_file = st.file_uploader("Выберите аудиофайл (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Сохраняем временный файл
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Начать обработку"):
        # 1. Транскрибация
        with st.status("Шаг 1: Распознавание речи...", expanded=True):
            segments, info = model.transcribe("temp_audio.mp3", beam_size=5)
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
                st.write(f"[{segment.start:.2f}s] {segment.text}")
        
        st.success("Текст успешно извлечен!")
        
        # 2. Формирование документации через Ollama
        with st.status("Шаг 2: Анализ и формирование документации...", expanded=True):
            prompt = f"""
            Ты — технический писатель. На основе следующей расшифровки аудиозаписи обсуждения, 
            составь структурированную техническую документацию (Meeting Minutes).
            Включи следующие разделы:
            1. Тема обсуждения.
            2. Ключевые принятые решения.
            3. Список поставленных задач (Action Items) с ответственными.
            4. Технические требования или ограничения, если они озвучены.
            
            Текст расшифровки:
            {full_text}
            """
            
            response = ollama.generate(model='llama3', prompt=prompt)
            doc_result = response['response']
            st.markdown(doc_result)

        # 3. Возможность скачать результат
        st.download_button(
            label="Скачать документ (.txt)",
            data=doc_result,
            file_name="technical_doc.txt",
            mime="text/plain"
        )
        
        # Удаляем временный файл
        os.remove("temp_audio.mp3")
