# 🚀 Инструкция по запуску локальной системы генерации технической документации

## 📋 Видео-демонстрация проекта

**Ознакомьтесь с работой системы:**  
[![Демо видео проекта](https://img.youtube.com/vi/RUEH16ZKv34/0.jpg)](https://youtu.be/RUEH16ZKv34)


## 🔧 Пошаговая установка

### Шаг 1: Установка Python

#### Windows:
```bash
# Скачайте Python 3.10+ с официального сайта
# https://www.python.org/downloads/

# При установке ОБЯЗАТЕЛЬНО отметьте:
# ☑ "Add Python to PATH"
# ☑ "Install for all users"

# Проверьте установку:
python --version
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv -y
python3 --version
```

#### macOS:
```bash
brew install python@3.10
python3 --version
```

---

### Шаг 2: Установка FFmpeg (критически важно!)

#### Windows:
```bash
# Способ 1 (простой):
# Скачайте сборку: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
# Распакуйте в C:\ffmpeg
# Добавьте C:\ffmpeg\bin в PATH системы

# Способ 2 (через chocolatey):
choco install ffmpeg

# Проверка:
ffmpeg -version
```

#### Linux:
```bash
sudo apt install ffmpeg -y
ffmpeg -version
```

#### macOS:
```bash
brew install ffmpeg
ffmpeg -version
```

---

### Шаг 3: Создание виртуального окружения

```bash
# Создайте папку проекта
mkdir tech-doc-generator
cd tech-doc-generator

# Создайте виртуальное окружение
python -m venv venv

# Активируйте окружение:

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

---

### Шаг 4: Установка зависимостей

Создайте файл `requirements.txt`:

```txt
# Core
streamlit==1.29.0
whisper-openai==20231117
openai-whisper==20231117
python-dotenv==1.0.0

# AI/ML
google-generativeai==0.3.2
torch==2.1.0
torchaudio==2.1.0
transformers==4.36.0

# Data processing
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1

# Visualization
plotly==5.17.0
wordcloud==1.9.2
matplotlib==3.7.2

# Audio processing
yt-dlp==2023.11.14
pytube==15.0.0
pydub==0.25.1
librosa==0.10.0

# Database
sqlite3

# Utils
requests==2.31.0
deep-translator==1.11.4
beautifulsoup4==4.12.2
tqdm==4.66.1

# Optional (for better performance)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # для CUDA
```

Установка:

```bash
pip install -r requirements.txt
# ИЛИ установка по одному:
pip install streamlit whisper openai-whisper python-dotenv google-generativeai pandas plotly wordcloud matplotlib yt-dlp nltk scikit-learn deep-translator

# Установка NLTK данных
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

### Шаг 5: Настройка API ключей

Создайте файл `.env` в корне проекта:

```env
# Google Gemini API Key (получите бесплатно на https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=AIzaSyCV0-NgdsnuTrNPeQ_XTR32C-laOYw_B2o

# OpenAI (опционально, для альтернативной обработки)
OPENAI_API_KEY=sk-your-key-here

# HuggingFace (опционально)
HUGGINGFACE_API_KEY=hf-your-key-here
```

> **Важно:** API ключ уже встроен в код, но для работы нужно зарегистрироваться и получить свой ключ на [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 🚀 Запуск приложения

### Простой запуск (Streamlit):

```bash
# Убедитесь, что виртуальное окружение активировано
# Активация: venv\Scripts\activate (Windows) или source venv/bin/activate (Linux/macOS)

# Запуск приложения
streamlit run app.py

# Приложение откроется в браузере по адресу:
# http://localhost:8501
```

### Запуск с дополнительными опциями:

```bash
# С указанием порта
streamlit run app.py --server.port 8080

# Запуск в production режиме
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false

# С увеличенной памятью (для больших файлов)
streamlit run app.py --server.maxUploadSize 500
```

---

## 🎯 Альтернативные способы запуска

### Запуск через Docker (рекомендуется для production):

Создайте `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Сборка и запуск:

```bash
# Сборка образа
docker build -t tech-doc-generator .

# Запуск контейнера
docker run -p 8501:8501 tech-doc-generator
```

### Запуск с GPU поддержкой (NVIDIA):

```bash
# Установите CUDA и NVIDIA Container Toolkit
# Запуск с GPU
docker run --gpus all -p 8501:8501 tech-doc-generator
```

---

## 📱 Доступ через локальную сеть

### Вариант 1: Автоматический доступ
```bash
# Streamlit автоматически покажет сетевой URL при запуске
# Обычно: http://192.168.1.xxx:8501
```

### Вариант 2: Ручная настройка
```python
# Добавьте в начало app.py:
st.set_page_config(
    page_title="Tech Doc Generator",
    initial_sidebar_state="expanded"
)

# При запуске укажите хост:
# streamlit run app.py --server.address 0.0.0.0
```

---

## 🧪 Тестирование системы

### Создание тестового аудиофайла:

```python
# create_test_audio.py
from gtts import gTTS
import os

text = """
Добрый день, коллеги. Сегодня обсуждаем новый проект по разработке системы генерации документации.
Важно: нужно реализовать транскрибацию аудио с помощью Whisper.
Также необходимо добавить интеграцию с Google Gemini для улучшения качества текста.
Срок сдачи: 15 декабря.
Ответственный: Алексей.
"""

tts = gTTS(text=text, lang='ru')
tts.save("test_meeting.mp3")
print("Тестовый файл создан: test_meeting.mp3")
```

Запуск:
```bash
python create_test_audio.py
```

### Загрузка тестовых файлов:
- Скачайте примеры с [Google Drive](https://drive.google.com/drive/folders/example)
- Или используйте любые записи совещаний в формате MP3/WAV

---

## 🔧 Устранение неполадок

### Проблема 1: FFmpeg не найден

**Ошибка:** `RuntimeError: FFmpeg not found`

**Решение:**
```bash
# Windows: Скачайте и добавьте в PATH как показано выше
# Linux/macOS: sudo apt install ffmpeg или brew install ffmpeg

# Проверка:
where ffmpeg  # Windows
which ffmpeg  # Linux/macOS
```

### Проблема 2: Ошибка памяти

**Ошибка:** `Out of memory` или `CUDA out of memory`

**Решение:**
```python
# В коде app.py измените модель на более легкую:
model = whisper.load_model("tiny")  # вместо "base" или "small"

# Или используйте CPU режим:
model = whisper.load_model("base", device="cpu")
```

### Проблема 3: Медленная работа

**Оптимизации:**
```bash
# 1. Используйте более легкую модель Whisper:
model = whisper.load_model("tiny")  # ~150 MB, быстро

# 2. Включите кэширование:
streamlit run app.py --browser.gatherUsageStats false

# 3. Используйте GPU если доступно:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Проблема 4: Ошибка API Gemini

**Ошибка:** `API key not valid`

**Решение:**
1. Зарегистрируйтесь на [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Получите бесплатный API ключ
3. Обновите `.env` файл или вставьте ключ в код

---

## 📊 Настройка производительности

### Для быстрой обработки (рекомендуемая конфигурация):

```python
# В app.py:
whisper_model = "tiny"  # или "base"
processing_mode = "fast"  # быстрый режим
enable_enhanced_analysis = False  # отключить расширенный анализ
```

### Для максимального качества:

```python
whisper_model = "large"  # самая точная модель
processing_mode = "expert"  # экспертный режим
enable_enhanced_analysis = True
```

---

## 🌐 Доступ через интернет (опционально)

### Использование ngrok для временного доступа:

```bash
# Установка ngrok (бесплатно)
npm install -g ngrok
# ИЛИ скачайте с https://ngrok.com

# Запуск туннеля
ngrok http 8501

# Будет сгенерирован URL типа: https://abc123.ngrok.io
```

### Развертывание на облачном сервере:

```bash
# DigitalOcean / AWS / Google Cloud
# 1. Запустите Ubuntu 20.04+
# 2. Установите Docker и docker-compose
# 3. Клонируйте репозиторий
# 4. Запустите: docker-compose up -d

# Пример docker-compose.yml:
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
```

---

## 📝 Примеры использования

### 1. Обработка записи совещания:
```bash
1. Загрузите файл "meeting_2024_11_20.mp3"
2. Выберите тип документа: "Протокол совещания"
3. Выберите режим: "Сбалансированный"
4. Нажмите "Создать документацию"
5. Через 2-3 минуты получите структурированный протокол
```

### 2. Создание технического задания:
```bash
1. Запишите обсуждение требований к проекту
2. Загрузите аудио
3. Выберите тип: "Техническое задание"
4. Выберите режим: "Экспертный"
5. Получите структурированное ТЗ в формате Markdown
```

---

## 🎯 Проверка работоспособности

### Минимальный тест:

```python
# test_system.py
import streamlit as st
import whisper

def test_whisper():
    try:
        model = whisper.load_model("tiny")
        print("✅ Whisper загружен успешно")
    except Exception as e:
        print(f"❌ Ошибка Whisper: {e}")

def test_gemini():
    import google.generativeai as genai
    try:
        genai.configure(api_key="YOUR_API_KEY")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content("Test")
        print("✅ Gemini доступен")
    except Exception as e:
        print(f"⚠️ Gemini не работает: {e}")

if __name__ == "__main__":
    test_whisper()
    test_gemini()
    print("✅ Система готова к работе!")
```

---

## 📞 Поддержка и помощь

### Полезные ссылки:
- **Видео-демонстрация:** https://youtu.be/RUEH16ZKv34
- **Документация Whisper:** https://github.com/openai/whisper
- **Streamlit документация:** https://docs.streamlit.io
- **Google Gemini API:** https://ai.google.dev

### Сообщения об ошибках:
Создайте issue в репозитории проекта с:
- Лог ошибки
- Конфигурация системы (ОС, RAM, GPU)
- Размер и длительность аудиофайла

---

## ✅ Чек-лист готовности

Перед первым запуском проверьте:

- [ ] Python 3.10+ установлен
- [ ] FFmpeg установлен и доступен в PATH
- [ ] Виртуальное окружение создано и активировано
- [ ] Все зависимости установлены (`pip install -r requirements.txt`)
- [ ] API ключ Gemini получен и добавлен
- [ ] NLTK данные загружены
- [ ] Тестовый аудиофайл создан
- [ ] Streamlit запускается без ошибок

**После выполнения всех шагов система готова к использованию!** 🎉

---

## 🚀 Быстрый старт (для нетерпеливых)

```bash
# Копируйте и вставляйте по очереди:

# 1. Клонирование (или создайте файл app.py с кодом)
git clone https://github.com/your-repo/tech-doc-generator.git
cd tech-doc-generator

# 2. Установка
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate

# 3. Зависимости
pip install streamlit whisper python-dotenv google-generativeai pandas plotly wordcloud matplotlib yt-dlp nltk scikit-learn

# 4. Запуск
streamlit run app.py

# 5. Откройте браузер на http://localhost:8501
```
