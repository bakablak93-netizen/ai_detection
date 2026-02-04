import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# --- КОНФИГУРАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="AI Sentinel Live", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: white; }
    /* Стиль заголовка */
    .main-title {
        text-align: center;
        color: #00ffa3;
        font-size: 40px;
        font-weight: 800;
        text-shadow: 0 0 15px #00ffa3;
        margin-bottom: 10px;
    }
    /* Кнопка START как в приложении */
    div.stButton > button {
        width: 100%;
        height: 60px;
        background-color: #00ffa3 !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 22px !important;
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.5);
    }
    /* Окно статуса */
    .status-panel {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        border: 2px solid #00ffa3;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>AI SENTINEL</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Экранды нақты уақытта талдау</p>", unsafe_allow_html=True)

# --- ЛОГИКА ПОТОКА ---
# Поле для ссылки (скрыто в экспандере, чтобы не портить вид)
with st.expander("Баптаулар (Настройки потока)"):
    stream_url = st.text_input("VDO.ninja сілтемесі:", placeholder="https://vdo.ninja/?v=...")

if stream_url:
    # Загружаем ИИ
    model = YOLO('yolov8n.pt')
    
    # Кнопка СҚАНДАРЛАУ
    if st.button("СҚАНДАРЛАУДЫ БАСТАУ"):
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(stream_url)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Ағын үзілді (Поток прерван)")
                break

            # ИИ Анализ
            results = model(frame, stream=True, conf=0.4, verbose=False)
            
            # Базовый вердикт (для примера)
            is_ai = False
            
            for r in results:
                frame = r.plot() # Рисуем рамки
                
                # Анализ частот (наш детектор)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if score < 40: is_ai = True

            # Визуальный индикатор в интерфейсе
            if is_ai:
                st.markdown("<div style='color: #ff4b4b; text-align: center; font-weight: bold;'>⚠️ AI DETECTED!</div>", unsafe_allow_html=True)
            
            # Вывод видео в "телефонное" окно
            FRAME_WINDOW.image(frame, channels="BGR")
else:
    st.info("Жұмысты бастау үшін жоғарыдағы баптауларға VDO.ninja сілтемесін қойыңыз.")
