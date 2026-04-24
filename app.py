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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pytube import YouTube
import yt_dlp
import requests
from typing import Dict, List, Tuple, Optional
import hashlib
import sqlite3
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка NLTK данных
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ========== КЛАССЫ И ДАТАКЛАССЫ ==========

class ProcessingMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    EXPERT = "expert"

class DocumentType(Enum):
    MEETING_PROTOCOL = "meeting_protocol"
    TECHNICAL_SPEC = "technical_spec"
    REPORT = "report"
    TUTORIAL = "tutorial"
    BRAINSTORM = "brainstorm"
    DECISION_LOG = "decision_log"

@dataclass
class ProcessingResult:
    raw_text: str
    cleaned_text: str
    enhanced_text: str
    summary: str
    title: str
    topic: str
    key_points: List[str]
    decisions: List[str]
    action_items: List[str]
    entities: Dict[str, List[str]]
    sentiment: Dict[str, float]
    statistics: Dict[str, any]
    document_type: DocumentType
    processing_time: float
    used_gemini: bool
    model_used: str
    quality_score: float

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(
    page_title="Advanced AI Documentation Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== API КЛЮЧИ ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCV0-NgdsnuTrNPeQ_XTR32C-laOYw_B2o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# ========== КЛАССЫ ДЛЯ РАСШИРЕННОЙ ОБРАБОТКИ ==========

class DatabaseManager:
    """Управление базой данных для хранения истории обработки"""
    
    def __init__(self, db_path="documentation_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    document_type TEXT,
                    title TEXT,
                    topic TEXT,
                    word_count INTEGER,
                    quality_score REAL,
                    processing_time REAL,
                    used_gemini BOOLEAN,
                    summary TEXT
                )
            """)
    
    def save_record(self, result: ProcessingResult, filename: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO processing_history 
                (timestamp, filename, document_type, title, topic, word_count, 
                 quality_score, processing_time, used_gemini, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                filename,
                result.document_type.value,
                result.title,
                result.topic,
                len(result.cleaned_text.split()),
                result.quality_score,
                result.processing_time,
                result.used_gemini,
                result.summary
            ))
    
    def get_history(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, filename, document_type, title, topic, 
                       word_count, quality_score, processing_time, summary
                FROM processing_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()

class TextAnalyzer:
    """Расширенный анализ текста"""
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Извлечение именованных сущностей"""
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "numbers": [],
            "technologies": []
        }
        
        # Паттерны для извлечения
        patterns = {
            "persons": r'\b([А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)\b(?=\s*(?:сказал|отметил|предложил|добавил))',
            "organizations": r'\b([А-Я]{2,}[а-я]*(?:\s+[А-Я][а-я]+)*)\b(?=\s*(?:компания|фирма|корпорация))',
            "dates": r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b(?:сегодня|завтра|вчера|на следующей неделе)\b',
            "numbers": r'\b\d+(?:[.,]\d+)?\s*(?:тысяч|миллионов|процентов|раз)\b',
            "technologies": r'\b(?:API|SDK|ML|AI|кластер|сервер|база данных|алгоритм|нейросеть)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))[:10]
        
        return entities
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, float]:
        """Анализ тональности текста"""
        positive_words = ['хорошо', 'отлично', 'успешно', 'эффективно', 'позитивно', 'прогресс']
        negative_words = ['проблема', 'ошибка', 'сложно', 'трудно', 'критично', 'сбой']
        neutral_words = ['возможно', 'вероятно', 'рассмотрим', 'обсудим']
        
        words = text.lower().split()
        total = len(words)
        
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        neutral_count = sum(1 for w in words if w in neutral_words)
        
        return {
            "positive": positive_count / max(total, 1),
            "negative": negative_count / max(total, 1),
            "neutral": neutral_count / max(total, 1),
            "polarity": (positive_count - negative_count) / max(total, 1)
        }
    
    @staticmethod
    def extract_action_items(text: str) -> List[str]:
        """Извлечение action items"""
        patterns = [
            r'(?:нужно|необходимо|следует|требуется)\s+([^.!?]+[.!?])',
            r'(?:сделать|выполнить|реализовать|подготовить)\s+([^.!?]+[.!?])',
            r'(?:ответственный|отвечает|назначается)\s+([^.!?]+[.!?])'
        ]
        
        actions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)
        
        return list(set(actions))[:15]
    
    @staticmethod
    def extract_decisions(text: str) -> List[str]:
        """Извлечение принятых решений"""
        patterns = [
            r'(?:решили|постановили|приняли решение|договорились)\s+([^.!?]+[.!?])',
            r'(?:утвердили|одобрили|согласовали)\s+([^.!?]+[.!?])',
            r'(?:выбрали|определили|установили)\s+([^.!?]+[.!?])'
        ]
        
        decisions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            decisions.extend(matches)
        
        return list(set(decisions))[:10]
    
    @staticmethod
    def calculate_quality_score(text: str, key_points: List[str]) -> float:
        """Расчет оценки качества документации"""
        score = 0.0
        
        # Критерии качества
        criteria = {
            "length": min(len(text.split()) / 1000, 1.0) * 0.2,
            "key_points": min(len(key_points) / 10, 1.0) * 0.3,
            "structure": 1.0 if re.search(r'#{2,3}', text) else 0.5 * 0.15,
            "action_items": 1.0 if re.search(r'(?:нужно|необходимо)', text) else 0.5 * 0.15,
            "decisions": 1.0 if re.search(r'(?:решили|постановили)', text) else 0.5 * 0.1,
            "entities": 1.0 if len(re.findall(r'[А-Я][а-я]+(?:\s+[А-Я][а-я]+)?', text)) > 3 else 0.5 * 0.1
        }
        
        score = sum(criteria.values())
        return min(score, 1.0)

class DocumentClassifier:
    """Классификация типа документа"""
    
    @staticmethod
    def classify_document(text: str) -> DocumentType:
        """Определение типа документа на основе содержания"""
        text_lower = text.lower()
        
        patterns = {
            DocumentType.MEETING_PROTOCOL: ['совещание', 'встреча', 'обсуждение', 'участники'],
            DocumentType.TECHNICAL_SPEC: ['техническое задание', 'требования', 'архитектура', 'API', 'интеграция'],
            DocumentType.REPORT: ['отчет', 'результат', 'анализ', 'показатели', 'метрики'],
            DocumentType.TUTORIAL: ['руководство', 'инструкция', 'как сделать', 'пошагово', 'пример'],
            DocumentType.BRAINSTORM: ['идея', 'предложение', 'вариант', 'альтернатива', 'креатив'],
            DocumentType.DECISION_LOG: ['решение', 'выбор', 'утверждение', 'одобрение', 'принято']
        }
        
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[doc_type] = score
        
        if max(scores.values()) == 0:
            return DocumentType.MEETING_PROTOCOL
        
        return max(scores, key=scores.get)

class YouTubeDownloader:
    """Загрузка аудио с YouTube"""
    
    @staticmethod
    def download_audio(url: str, output_path: str = "temp_audio") -> Optional[str]:
        """Скачивание аудио с YouTube"""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                mp3_filename = filename.rsplit('.', 1)[0] + '.mp3'
                
                return mp3_filename if os.path.exists(mp3_filename) else None
        except Exception as e:
            logger.error(f"YouTube download error: {e}")
            return None

class AudioRecorder:
    """Запись аудио с микрофона"""
    
    @staticmethod
    def get_recording_html():
        """Получение HTML компонента для записи аудио"""
        return """
        <div id="recorder">
            <button id="recordBtn" class="record-button">🎙️ Начать запись</button>
            <button id="stopBtn" class="record-button" disabled>⏹️ Остановить</button>
            <div id="timer" style="margin-top: 10px; font-size: 18px;">00:00</div>
            <audio id="audioPlayback" controls style="display: none; margin-top: 10px;"></audio>
        </div>
        
        <script>
        let mediaRecorder;
        let audioChunks = [];
        let startTime;
        let timerInterval;
        
        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const timer = document.getElementById('timer');
        const audioPlayback = document.getElementById('audioPlayback');
        
        recordBtn.onclick = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
                audioPlayback.style.display = 'block';
                
                // Отправка на сервер
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    const event = new CustomEvent('audioRecorded', { detail: base64Audio });
                    window.dispatchEvent(event);
                };
                reader.readAsDataURL(audioBlob);
            };
            
            mediaRecorder.start();
            startTime = Date.now();
            timerInterval = setInterval(updateTimer, 1000);
            
            recordBtn.disabled = true;
            stopBtn.disabled = false;
        };
        
        stopBtn.onclick = () => {
            mediaRecorder.stop();
            clearInterval(timerInterval);
            recordBtn.disabled = false;
            stopBtn.disabled = true;
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        };
        
        function updateTimer() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        </script>
        
        <style>
        .record-button {
            padding: 10px 20px;
            font-size: 16px;
            margin: 5px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
        }
        .record-button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        </style>
        """

class VisualizationManager:
    """Управление визуализациями"""
    
    @staticmethod
    def create_wordcloud(text: str) -> plt.Figure:
        """Создание облака слов"""
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis', max_words=100, 
                             stopwords=set(stopwords.words('russian')))
        wordcloud.generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    
    @staticmethod
    def create_topic_timeline(processing_history: List) -> go.Figure:
        """Создание временной шкалы тем"""
        if not processing_history:
            return go.Figure()
        
        df = pd.DataFrame(processing_history, 
                         columns=['timestamp', 'filename', 'type', 'title', 'topic', 
                                 'words', 'quality', 'time', 'summary'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['quality'],
            mode='lines+markers',
            text=df['topic'],
            name='Качество документации',
            line=dict(color='blue', width=2),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Динамика качества документации',
            xaxis_title='Дата',
            yaxis_title='Оценка качества',
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_processing_stats(stats: Dict) -> go.Figure:
        """Создание статистики обработки"""
        fig = go.Figure(data=[
            go.Bar(
                x=list(stats.keys()),
                y=list(stats.values()),
                marker_color='lightblue',
                text=list(stats.values()),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Статистика обработки',
            xaxis_title='Параметр',
            yaxis_title='Значение',
            showlegend=False
        )
        
        return fig

# ========== РАСШИРЕННЫЕ ФУНКЦИИ GEMINI ==========

def enhance_with_gemini_advanced(raw_text: str, model_name: str, doc_type: DocumentType) -> dict:
    """Расширенная обработка через Gemini с учетом типа документа"""
    
    type_instructions = {
        DocumentType.MEETING_PROTOCOL: """
        Сделай акцент на:
        - Список участников и их роли
        - Хронология обсуждения
        - Принятые решения
        - Дедлайны и ответственные
        """,
        DocumentType.TECHNICAL_SPEC: """
        Сделай акцент на:
        - Технические требования
        - Архитектурные решения
        - Интеграции и API
        - Ограничения и риски
        """,
        DocumentType.REPORT: """
        Сделай акцент на:
        - Ключевые метрики и KPI
        - Достижения и проблемы
        - Рекомендации
        - Сравнение с предыдущими периодами
        """,
        DocumentType.TUTORIAL: """
        Сделай акцент на:
        - Пошаговые инструкции
        - Примеры кода/действий
        - Предупреждения и советы
        - Чек-листы
        """
    }
    
    prompt = f"""
    Ты — эксперт по созданию {doc_type.value} документации.
    
    Инструкции для этого типа документа:
    {type_instructions.get(doc_type, "Создай стандартную техническую документацию")}
    
    Сырой текст:
    \"\"\"
    {raw_text}
    \"\"\"
    
    Создай расширенную документацию в формате JSON:
    {{
        "title": "информативный заголовок",
        "topic": "основная тема (кратко)",
        "key_points": ["список ключевых моментов (8-12 пунктов)"],
        "decisions": ["принятые решения (если есть)"],
        "action_items": ["конкретные задачи с ответственными"],
        "entities": {{
            "persons": ["спикеры и участники"],
            "organizations": ["упомянутые компании"],
            "technologies": ["технологии и инструменты"]
        }},
        "summary": "расширенное резюме (3-5 предложений)",
        "cleaned_text": "полностью очищенный и структурированный текст",
        "full_documentation": "полная документация в Markdown с заголовками, таблицами, списками"
    }}
    
    Используй профессиональный стиль. Добавь эмодзи для улучшения читаемости.
    """
    
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text_response = response.text.strip()
    
    # Очистка ответа от markdown
    if text_response.startswith("```json"):
        text_response = text_response[7:]
    if text_response.startswith("```"):
        text_response = text_response[3:]
    if text_response.endswith("```"):
        text_response = text_response[:-3]
    
    return json.loads(text_response)

# ========== ОСНОВНОЕ ПРИЛОЖЕНИЕ ==========

def main():
    # Инициализация компонентов
    db_manager = DatabaseManager()
    text_analyzer = TextAnalyzer()
    classifier = DocumentClassifier()
    viz_manager = VisualizationManager()
    
    st.title("🤖 Advanced AI Documentation Generator")
    st.markdown("### Превратите аудио в профессиональную документацию с помощью AI")
    st.markdown("---")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Режим обработки
        processing_mode = st.selectbox(
            "Режим обработки",
            [mode.value for mode in ProcessingMode],
            format_func=lambda x: {
                "fast": "🚀 Быстрый (скорость)",
                "balanced": "⚖️ Сбалансированный",
                "accurate": "🎯 Точный (качество)",
                "expert": "🧠 Экспертный (максимум AI)"
            }[x]
        )
        
        # Модель Whisper
        whisper_model = st.selectbox(
            "Модель распознавания",
            ["tiny", "base", "small", "medium", "large"],
            index=2
        )
        
        # Тип документа
        doc_type = st.selectbox(
            "Тип документа (подсказка для AI)",
            [dt.value for dt in DocumentType],
            format_func=lambda x: {
                "meeting_protocol": "📋 Протокол совещания",
                "technical_spec": "🔧 Техническое задание",
                "report": "📊 Отчет",
                "tutorial": "📚 Руководство/Инструкция",
                "brainstorm": "💡 Мозговой штурм",
                "decision_log": "✅ Журнал решений"
            }[x]
        )
        
        st.markdown("---")
        
        # Источник аудио
        st.header("🎵 Источник аудио")
        audio_source = st.radio(
            "Выберите источник",
            ["📁 Загрузить файл", "🔗 YouTube URL", "🎙️ Записать с микрофона"]
        )
        
        st.markdown("---")
        
        # Дополнительные опции
        st.header("🔧 Дополнительно")
        enable_enhanced_analysis = st.checkbox("Включить расширенный анализ", value=True)
        save_to_database = st.checkbox("Сохранять в историю", value=True)
        generate_wordcloud = st.checkbox("Создать облако слов", value=True)
        
        st.markdown("---")
        
        # Статистика
        st.header("📊 Статистика")
        history = db_manager.get_history(10)
        if history:
            avg_quality = sum(h[6] for h in history) / len(history)
            st.metric("Среднее качество", f"{avg_quality:.2%}")
            st.metric("Всего документов", len(history))
    
    # Основная область
    audio_data = None
    
    if audio_source == "📁 Загрузить файл":
        audio_file = st.file_uploader(
            "Выберите аудиофайл",
            type=["mp3", "wav", "m4a", "mp4", "flac", "ogg", "webm"],
            help="Поддерживаются все популярные аудиоформаты"
        )
        if audio_file:
            audio_data = audio_file
        
    elif audio_source == "🔗 YouTube URL":
        youtube_url = st.text_input("Введите URL видео с YouTube")
        if youtube_url and st.button("📥 Загрузить с YouTube"):
            with st.spinner("Загрузка аудио с YouTube..."):
                downloader = YouTubeDownloader()
                audio_path = downloader.download_audio(youtube_url)
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, 'rb') as f:
                        audio_data = BytesIO(f.read())
                    audio_data.name = os.path.basename(audio_path)
                    st.success("Аудио успешно загружено!")
                else:
                    st.error("Не удалось загрузить аудио с YouTube")
    
    elif audio_source == "🎙️ Записать с микрофона":
        st.components.v1.html(AudioRecorder.get_recording_html(), height=200)
        
        # JavaScript для получения аудио
        audio_buffer = st.empty()
        audio_js = """
        <script>
        window.addEventListener('audioRecorded', function(e) {
            const audioData = e.detail;
            const input = document.createElement('input');
            input.type = 'hidden';
            input.id = 'recordedAudio';
            input.value = audioData;
            document.body.appendChild(input);
        });
        </script>
        """
        st.components.v1.html(audio_js)
        
        if st.button("🔄 Использовать запись"):
            st.info("Запись будет обработана после завершения...")
    
    # Кнопка обработки
    if audio_data and st.button("🚀 Создать документацию", type="primary", use_container_width=True):
        start_time = time.time()
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Сохранение временного файла
            status_text.text("📥 Сохранение аудиофайла...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_data.getbuffer())
                temp_path = tmp_file.name
            
            # Распознавание речи
            status_text.text(f"🎤 Распознавание речи (модель: {whisper_model})...")
            progress_bar.progress(30)
            
            model = whisper.load_model(whisper_model)
            result = model.transcribe(temp_path, language="ru", task="transcribe", 
                                     temperature=0.0, fp16=False)
            raw_text = result["text"]
            
            # Очистка текста
            status_text.text("✨ Очистка и предобработка текста...")
            progress_bar.progress(50)
            
            cleaned_text = clean_text(raw_text)
            
            # Определение типа документа
            doc_type_enum = DocumentType(doc_type)
            
            # Обработка через Gemini
            use_gemini = False
            enhanced_result = None
            
            if GEMINI_API_KEY:
                try:
                    status_text.text("🧠 Обработка через Gemini AI...")
                    genai.configure(api_key=GEMINI_API_KEY)
                    
                    # Выбор модели в зависимости от режима
                    model_name = {
                        "fast": "gemini-1.5-flash",
                        "balanced": "gemini-1.5-flash",
                        "accurate": "gemini-1.5-pro",
                        "expert": "gemini-1.5-pro"
                    }.get(processing_mode, "gemini-1.5-flash")
                    
                    enhanced_result = enhance_with_gemini_advanced(cleaned_text, model_name, doc_type_enum)
                    use_gemini = True
                    status_text.text("✅ Обработка Gemini завершена")
                except Exception as e:
                    st.warning(f"Ошибка Gemini: {str(e)[:100]}. Использую локальную обработку.")
            
            # Расширенный анализ
            status_text.text("📊 Проведение расширенного анализа...")
            progress_bar.progress(70)
            
            if enable_enhanced_analysis:
                entities = text_analyzer.extract_entities(cleaned_text)
                sentiment = text_analyzer.analyze_sentiment(cleaned_text)
                action_items = text_analyzer.extract_action_items(cleaned_text)
                decisions = text_analyzer.extract_decisions(cleaned_text)
                key_points = text_analyzer.extract_key_points(cleaned_text) if not use_gemini else enhanced_result.get("key_points", [])
            else:
                entities = {}
                sentiment = {"positive": 0, "negative": 0, "neutral": 0, "polarity": 0}
                action_items = []
                decisions = []
                key_points = extract_key_points(cleaned_text)
            
            # Формирование финальной документации
            status_text.text("📝 Формирование документации...")
            progress_bar.progress(90)
            
            if use_gemini and enhanced_result:
                title = enhanced_result.get("title", "Техническая документация")
                topic = enhanced_result.get("topic", detect_topic(cleaned_text))
                summary = enhanced_result.get("summary", "Документация создана автоматически")
                full_doc = enhanced_result.get("full_documentation", format_documentation(cleaned_text, audio_data.name, topic, key_points, []))
            else:
                title = f"Документация от {datetime.now().strftime('%d.%m.%Y')}"
                topic = detect_topic(cleaned_text)
                summary = f"Документ на тему «{topic}» содержит {len(key_points)} ключевых моментов."
                keywords = extract_keywords(cleaned_text)
                full_doc = format_documentation(cleaned_text, audio_data.name, topic, key_points, keywords)
            
            # Расчет оценки качества
            quality_score = text_analyzer.calculate_quality_score(full_doc, key_points)
            
            # Подготовка результата
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                enhanced_text=full_doc,
                summary=summary,
                title=title,
                topic=topic,
                key_points=key_points,
                decisions=decisions,
                action_items=action_items,
                entities=entities,
                sentiment=sentiment,
                statistics={
                    "word_count": len(cleaned_text.split()),
                    "char_count": len(cleaned_text),
                    "sentence_count": len(sent_tokenize(cleaned_text)),
                    "avg_word_length": sum(len(w) for w in cleaned_text.split()) / len(cleaned_text.split())
                },
                document_type=doc_type_enum,
                processing_time=processing_time,
                used_gemini=use_gemini,
                model_used=whisper_model,
                quality_score=quality_score
            )
            
            # Сохранение в базу данных
            if save_to_database:
                db_manager.save_record(result, audio_data.name)
            
            # Очистка временных файлов
            os.unlink(temp_path)
            progress_bar.progress(100)
            status_text.text("✅ Готово!")
            
            # Отображение результатов
            st.success(f"✅ Документация успешно создана! Качество: {quality_score:.1%}")
            
            # Вкладки результатов
            tabs = st.tabs(["📄 Документация", "📊 Аналитика", "🎨 Визуализации", "🔍 Детальный анализ", "💾 Экспорт"])
            
            with tabs[0]:
                st.markdown(full_doc)
            
            with tabs[1]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего слов", result.statistics["word_count"])
                    st.metric("Качество документа", f"{quality_score:.1%}")
                with col2:
                    st.metric("Ключевых моментов", len(key_points))
                    st.metric("Action items", len(action_items))
                with col3:
                    st.metric("Принятых решений", len(decisions))
                    st.metric("Время обработки", f"{processing_time:.1f} сек")
                
                st.markdown("---")
                st.markdown("### 📊 Детальная статистика")
                stats_df = pd.DataFrame([result.statistics])
                st.dataframe(stats_df)
                
                st.markdown("### 🎯 Принятые решения")
                for decision in decisions[:5]:
                    st.markdown(f"- {decision}")
                
                st.markdown("### 📋 Action Items")
                for action in action_items[:5]:
                    st.markdown(f"- {action}")
            
            with tabs[2]:
                if generate_wordcloud and cleaned_text:
                    st.markdown("### ☁️ Облако слов")
                    fig = viz_manager.create_wordcloud(cleaned_text)
                    st.pyplot(fig)
                    
                    # Тональность
                    st.markdown("### 📈 Анализ тональности")
                    sentiment_df = pd.DataFrame([result.sentiment])
                    st.bar_chart(sentiment_df)
                    
                    # История
                    if save_to_database:
                        st.markdown("### 📈 История обработок")
                        history_data = db_manager.get_history(10)
                        if history_data:
                            fig_timeline = viz_manager.create_topic_timeline(history_data)
                            st.plotly_chart(fig_timeline, use_container_width=True)
            
            with tabs[3]:
                st.markdown("### 🏷️ Извлеченные сущности")
                for entity_type, entities_list in entities.items():
                    if entities_list:
                        st.markdown(f"**{entity_type}:** {', '.join(entities_list[:5])}")
                
                st.markdown("### 🧠 Анализ ключевых тем")
                # TF-IDF анализ
                vectorizer = TfidfVectorizer(max_features=10, stop_words=stopwords.words('russian'))
                tfidf_matrix = vectorizer.fit_transform([cleaned_text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                tfidf_df = pd.DataFrame({
                    'Термин': feature_names,
                    'Важность': tfidf_scores
                }).sort_values('Важность', ascending=False)
                
                st.dataframe(tfidf_df)
                
                st.markdown("### 📝 Примеры ключевых предложений")
                sentences = sent_tokenize(cleaned_text)
                for sent in sentences[:5]:
                    st.markdown(f"- {sent[:200]}...")
            
            with tabs[4]:
                st.markdown("### 💾 Экспорт документации")
                
                # Markdown
                md_filename = f"documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                st.download_button(
                    label="📝 Скачать в Markdown",
                    data=full_doc,
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                    key="md_download"
                )
                
                # JSON
                result_dict = {
                    "title": title,
                    "topic": topic,
                    "summary": summary,
                    "key_points": key_points,
                    "decisions": decisions,
                    "action_items": action_items,
                    "entities": entities,
                    "sentiment": sentiment,
                    "statistics": result.statistics,
                    "quality_score": quality_score,
                    "full_documentation": full_doc
                }
                json_data = json.dumps(result_dict, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📋 Скачать в JSON",
                    data=json_data,
                    file_name=f"documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="json_download"
                )
                
                # TXT
                st.download_button(
                    label="📄 Скачать очищенный текст",
                    data=cleaned_text,
                    file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="txt_download"
                )
                
                # Отчет в HTML
                html_report = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>{title}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #2c3e50; }}
                        .metric {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; }}
                    </style>
                </head>
                <body>
                    <h1>{title}</h1>
                    <div class="metric">Качество: {quality_score:.1%}</div>
                    <div class="metric">Слов: {result.statistics['word_count']}</div>
                    <h2>Резюме</h2>
                    <p>{summary}</p>
                    <h2>Ключевые моменты</h2>
                    <ul>
                        {''.join(f'<li>{point}</li>' for point in key_points[:5])}
                    </ul>
                </body>
                </html>
                """
                st.download_button(
                    label="🌐 Скачать HTML отчет",
                    data=html_report,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="html_download"
                )
        
        except Exception as e:
            st.error(f"❌ Ошибка при обработке: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)
            
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    # Футер
    st.markdown("---")
    st.caption("""
    ### 🚀 Технологии:
    - **Whisper** от OpenAI (распознавание речи)
    - **Google Gemini** (структурирование и анализ)
    - **NLTK** (обработка естественного языка)
    - **Plotly** (визуализация данных)
    - **WordCloud** (облака слов)
    """)

if __name__ == "__main__":
    main()
