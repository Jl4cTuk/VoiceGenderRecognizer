import gradio as gr
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Загрузка модели и скейлера
model_path = "src/gender_classification_model.keras"
scaler_path = "src/scaler.pkl"
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Параметры аудио
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 13

def predict_gender(audio_file):
    """
    Предсказание пола по голосу.
    :param audio_file: Путь к аудиофайлу.
    :return: Строка с результатами.
    """
    try:
        # Загружаем аудио и извлекаем MFCC
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
        
        # Преобразуем данные с помощью скейлера и делаем предсказание
        X_scaled = scaler.transform(mfcc_mean)
        y_pred_prob = model.predict(X_scaled)[0][0]
        y_pred = 1 if y_pred_prob > 0.5 else 0
        gender = "Мужской" if y_pred == 1 else "Женский"
        probability = y_pred_prob if y_pred == 1 else 1 - y_pred_prob
        return f"Пол: {gender}, Вероятность: {probability:.2%}"
    except Exception as e:
        return f"Ошибка: {e}"

# Настройка интерфейса Gradio
interface = gr.Interface(
    fn=predict_gender,  # Функция предсказания
    inputs=gr.Audio(type="filepath"),  # Вход: аудиофайл
    outputs="text",  # Выход: текст с результатами
    title="Определение пола по голосу",  # Заголовок
    description="Загрузите файл с голосом (формат .wav или .flac), чтобы определить пол говорящего."  # Описание
)

# Запуск интерфейса
if __name__ == "__main__":
    interface.launch()
