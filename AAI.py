import cv2
import numpy as np
import pyautogui
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox
import threading

class SafeAIDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Detector Hackathon v1.0")
        self.root.geometry("400x350")
        self.root.attributes("-topmost", True)

        
        self.video_engine = None
        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_mesh
            self.video_engine = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
            print("Система видео-анализа: ОК")
        except Exception as e:
            print(f"Видео-движок недоступен (ошибка: {e}), используем базовый анализ.")

        
        self.status = tk.Label(root, text="СИСТЕМА ГОТОВА", font=("Courier", 12, "bold"), fg="blue")
        self.status.pack(pady=20)

        
        self.frame_box = tk.Frame(root, width=200, height=100, bg="black")
        self.frame_box.pack(pady=10)

        self.btn = tk.Button(root, text="ОПРЕДЕЛИТЬ (SCAN)", font=("Arial", 14, "bold"),
                             bg="#2ecc71", fg="white", command=self.start_thread,
                             width=20, height=2, relief="raised", bd=5)
        self.btn.pack(pady=20)

    def analyze_logic(self):
        self.btn.config(state="disabled", text="АНАЛИЗ...")
        self.status.config(text="СКАНИРОВАНИЕ СИГНАЛА...", fg="orange")
        
        try:
            
            duration = 1.0 
            fs = 44100
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_data = np.abs(np.fft.fft(recording.flatten()))
            audio_score = np.var(audio_data) 

            
            screen = pyautogui.screenshot()
            img = np.array(screen)
            
           
            is_ai = audio_score < 0.001 

            if is_ai:
                self.root.bell()
                messagebox.showwarning("ВНИМАНИЕ", "ОБНАРУЖЕН ЦИФРОВОЙ СИГНАЛ (ИИ / БОТ)")
            else:
                messagebox.showinfo("РЕЗУЛЬТАТ", "АНАЛИЗ ЗАВЕРШЕН: ЧЕЛОВЕК")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Сбой сканера: {e}")
        
        self.status.config(text="СИСТЕМА ГОТОВА", fg="blue")
        self.btn.config(state="normal", text="ОПРЕДЕЛИТЬ (SCAN)")

    def start_thread(self):
        threading.Thread(target=self.analyze_logic, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = SafeAIDetector(root)
    root.mainloop()
