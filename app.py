import streamlit as st
import whisper
import os
import re
from collections import Counter
from datetime import datetime
import tempfile
import subprocess
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Загрузка переменных окружения из .env (если есть)
load_dotenv()

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(
    page_title="Tech Doc Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== API КЛЮЧ GEMINI (ПРЯМО В КОДЕ) ==========
# ВНИМАНИЕ: для реальных проектов используйте .env или st.secrets!
GEMINI_API_KEY = "AIzaSyCV0-NgdsnuTrNPeQ_XTR32C-laOYw_B2o"

def get_gemini_api_key():
    """Получает ключ Gemini: сначала из .env, потом из захардкоженного"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = GEMINI_API_KEY  # fallback
    return api_key

# ========== ПРОВЕРКА FFMPEG ==========
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def install_ffmpeg_windows():
    st.markdown("""
    ### 🔧 Установка FFmpeg для Windows:
    
    1. **Скачайте FFmpeg:**  
       https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    
    2. **Распакуйте** в `C:\\ffmpeg`
    
    3. **Добавьте в PATH:**  
       - Нажмите `Win + R`, введите `sysdm.cpl`  
       - Дополнительно → Переменные среды  
       - В Path добавьте `C:\\ffmpeg\\bin`
    
    4. **Перезапустите VS Code и терминал**
    """)

# ========== ЛОКАЛЬНЫЕ ФУНКЦИИ (FALLBACK) ==========
def clean_text(text):
    corrections = {
        'стут': 'студии', 'погоди': 'погода', 'щас': 'сейчас',
        'типа': 'типа', 'кароче': 'короче', 'вообщем': 'в общем',
        'наверно': 'наверное', 'потомучто': 'потому что',
    }
    for wrong, correct in corrections.items():
        text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def extract_key_points(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    important_markers = [r'важно', r'ключевой', r'главный', r'основной', r'решение', r'вывод']
    key_points = []
    for sent in sentences:
        if any(re.search(marker, sent.lower()) for marker in important_markers) or len(sent) > 40:
            key_points.append(sent)
    return key_points[:12]

def extract_keywords(text):
    stop_words = {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'за', 'из', 'о', 'для', 'это', 'что', 'как', 'так', 'но', 'а', 'или', 'очень', 'можно', 'нужно', 'будет', 'есть'}
    words = text.split()
    word_freq = Counter()
    for w in words:
        w_clean = w.lower().strip('.,!?;:()"\'')
        if len(w_clean) > 3 and w_clean not in stop_words:
            word_freq[w_clean] += 1
    return word_freq.most_common(10)

def detect_topic(text):
    topics = {
        'Погода и климат': ['погода', 'температура', 'дождь', 'снег', 'ветер'],
        'Бизнес и финансы': ['компания', 'проект', 'клиент', 'продажи', 'бизнес'],
        'Технологии и IT': ['код', 'программа', 'система', 'данные', 'сервер', 'разработка'],
        'Медицина': ['врач', 'лечение', 'здоровье', 'пациент', 'больница'],
        'Образование': ['школа', 'урок', 'студент', 'учеба', 'экзамен'],
    }
    text_lower = text.lower()
    best_topic = 'Общее обсуждение'
    best_score = 0
    for topic, keywords in topics.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic

def format_documentation(text, filename, topic, key_points, keywords):
    now = datetime.now()
    words_count = len(text.split())
    doc = f"""# 📄 ТЕХНИЧЕСКАЯ ДОКУМЕНТАЦИЯ

## 📋 ОБЩАЯ ИНФОРМАЦИЯ
| Параметр | Значение |
|----------|----------|
| **Тема** | {topic} |
| **Источник** | {filename} |
| **Дата** | {now.strftime('%d.%m.%Y %H:%M:%S')} |
| **Объём** | {words_count} слов |

---

## 🎯 КЛЮЧЕВЫЕ МОМЕНТЫ

"""
    for i, point in enumerate(key_points, 1):
        doc += f"{i}. {point}\n\n"
    doc += f"""## 🔑 КЛЮЧЕВЫЕ СЛОВА

| Слово | Частота |
|-------|---------|
"""
    for word, count in keywords:
        doc += f"| **{word}** | {count} раз |\n"
    doc += f"""
## 📝 ПОЛНЫЙ ТЕКСТ

{text}

---
*Документ сгенерирован автоматически*
"""
    return doc

# ========== ФУНКЦИЯ УЛУЧШЕНИЯ ЧЕРЕЗ GEMINI ==========
def enhance_with_gemini(raw_text: str, model_name: str) -> dict:
    prompt = f"""
Ты — эксперт по технической документации. Перед тобой сырой текст после автоматического распознавания речи (Whisper). В нём могут быть ошибки, повторы, сленг, слова-паразиты.

Твоя задача — превратить текст в **идеально оформленную техническую документацию** по шаблону ниже.

Шаблон документации:
📄 ЗАГОЛОВОК (придумай сам)
📋 ОБЩАЯ ИНФОРМАЦИЯ
Тема: (кратко)

Тип документа: (протокол совещания / ТЗ / отчёт и т.п.)

🎯 ЦЕЛИ ОБСУЖДЕНИЯ
(список)

📌 КЛЮЧЕВЫЕ МОМЕНТЫ И РЕШЕНИЯ
(список)

🔑 ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ
(если есть)

✅ ПРИНЯТЫЕ РЕШЕНИЯ И СЛЕДУЮЩИЕ ШАГИ
(список)

📝 ПОЛНЫЙ ОЧИЩЕННЫЙ ТЕКСТ
(исправленный, без повторов и слов-паразитов)

text

Вот сырой текст:
\"\"\"
{raw_text}
\"\"\"

Верни результат строго в формате JSON со следующими полями:
{{
  "title": "заголовок",
  "topic": "тема",
  "key_points": ["пункт 1", "пункт 2", ...],
  "summary": "краткое резюме (2-3 предложения)",
  "cleaned_text": "полный исправленный текст",
  "full_documentation": "полностью сформированный документ в Markdown, включая все заголовки из шаблона"
}}

Не добавляй никаких лишних комментариев, только JSON.
"""
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text_response = response.text.strip()
    if text_response.startswith("```json"):
        text_response = text_response[7:]
    if text_response.startswith("```"):
        text_response = text_response[3:]
    if text_response.endswith("```"):
        text_response = text_response[:-3]
    result = json.loads(text_response)
    return result

# ========== ОСНОВНОЕ ПРИЛОЖЕНИЕ ==========
def main():
    st.title("📝 Генератор технической документации")
    st.markdown("Преобразует аудиозаписи обсуждений в структурированную документацию с помощью AI")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Настройки")
        model_choice = st.selectbox(
            "Модель распознавания (Whisper):",
            ["tiny", "base", "small"],
            index=1,
            help="tiny - быстро, base - баланс, small - точнее"
        )
        st.markdown("---")
        st.header("📌 Инструкция")
        st.markdown("""
        1. Загрузите аудиофайл
        2. Нажмите кнопку
        3. Дождитесь обработки
        4. Скачайте документацию
        
        **Поддерживаемые форматы:** MP3, WAV, M4A, MP4, FLAC, OGG, WEBM
        """)
        
        st.markdown("---")
        st.header("🔧 Статус")
        if check_ffmpeg():
            st.success("✅ FFmpeg установлен")
        else:
            st.error("❌ FFmpeg не найден")
            install_ffmpeg_windows()
        
        api_key = get_gemini_api_key()
        if api_key:
            st.success("✅ Gemini API ключ найден")
        else:
            st.error("❌ Gemini API ключ не найден")

    col1, col2 = st.columns([2, 1])
    with col1:
        audio_file = st.file_uploader(
            "📁 Загрузите аудиофайл",
            type=["mp3", "wav", "m4a", "mp4", "flac", "ogg", "webm"]
        )
    with col2:
        st.info("""
        **Советы:**
        - Чёткая речь без шума
        - Для длинных записей (30+ мин) используйте Whisper small
        """)
    
    if audio_file and st.button("🚀 Создать документацию", type="primary", use_container_width=True):
        if not check_ffmpeg():
            st.error("❌ FFmpeg не установлен!")
            st.stop()
        
        api_key = get_gemini_api_key()
        if not api_key:
            st.warning("⚠️ Не найден API ключ Gemini. Будет использована локальная обработка.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📥 Сохранение файла...")
            progress_bar.progress(10)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                tmp_file.write(audio_file.getbuffer())
                temp_path = tmp_file.name
            
            status_text.text(f"🤖 Загрузка модели Whisper ({model_choice})...")
            progress_bar.progress(30)
            model = whisper.load_model(model_choice)
            
            status_text.text("🎤 Распознавание речи...")
            progress_bar.progress(50)
            result = model.transcribe(temp_path, language="ru", task="transcribe", temperature=0.0, fp16=False)
            raw_text = result["text"]
            
            status_text.text("✨ Постобработка текста...")
            progress_bar.progress(70)
            
            use_gemini = False
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    models_to_try = [
                        "gemini-2.0-flash-exp",
                        "gemini-2.0-flash-lite",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro",
                        "gemini-pro"
                    ]
                    enhanced = None
                    for model_name in models_to_try:
                        try:
                            status_text.text(f"✨ Пробуем модель {model_name}...")
                            enhanced = enhance_with_gemini(raw_text, model_name=model_name)
                            if enhanced:
                                status_text.text(f"✅ Используется модель {model_name}")
                                break
                        except Exception as e:
                            st.warning(f"Модель {model_name} не работает: {str(e)[:100]}")
                            continue
                    if enhanced:
                        cleaned_text = enhanced["cleaned_text"]
                        topic = enhanced["topic"]
                        key_points = enhanced["key_points"]
                        title = enhanced["title"]
                        full_doc = enhanced["full_documentation"]
                        summary = enhanced["summary"]
                        use_gemini = True
                    else:
                        raise Exception("Ни одна модель не сработала")
                except Exception as e:
                    st.warning(f"⚠️ Ошибка Gemini: {str(e)[:200]}. Используем локальную обработку.")
                    use_gemini = False
            
            if not use_gemini:
                cleaned_text = clean_text(raw_text)
                topic = detect_topic(cleaned_text)
                key_points = extract_key_points(cleaned_text)
                keywords = extract_keywords(cleaned_text)
                full_doc = format_documentation(cleaned_text, audio_file.name, topic, key_points, keywords)
                summary = f"Документ на тему «{topic}» содержит {len(key_points)} ключевых моментов."
                title = "Техническая документация"
            
            os.unlink(temp_path)
            progress_bar.progress(100)
            status_text.text("✅ Готово!")
            
            st.success("✅ Документация успешно создана!")
            
            tab1, tab2, tab3 = st.tabs(["📄 Документация", "📊 Статистика", "💾 Экспорт"])
            
            with tab1:
                st.markdown(full_doc)
            
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего слов", len(cleaned_text.split()))
                    st.metric("Уникальных слов", len(set(cleaned_text.split())))
                with col2:
                    st.metric("Ключевых моментов", len(key_points))
                    st.metric("Использован AI", "Gemini" if use_gemini else "Локальный")
                with col3:
                    st.metric("Тема", topic)
                    st.metric("Заголовок", title[:40] + "..." if len(title) > 40 else title)
                st.caption(f"📌 Резюме: {summary}")
                if not use_gemini and 'keywords' in locals():
                    st.markdown("### 🔑 Частота ключевых слов")
                    for word, count in keywords[:8]:
                        if keywords[0][1] > 0:
                            st.progress(min(count / keywords[0][1], 1.0), text=f"{word}: {count} раз")
            
            with tab3:
                st.markdown("### 💾 Скачать документацию")
                md_filename = f"documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                st.download_button(
                    label="📝 Скачать в Markdown",
                    data=full_doc,
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True
                )
                txt_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button(
                    label="📄 Скачать очищенный текст",
                    data=cleaned_text,
                    file_name=txt_filename,
                    mime="text/plain",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Критическая ошибка: {str(e)}")
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    st.markdown("---")
    st.caption("💡 **Технологии:** Whisper (распознавание) + Google Gemini (структурирование)")

if __name__ == "__main__":
    main()
