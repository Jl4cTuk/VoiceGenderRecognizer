import os
import warnings
import logging

import librosa
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model

# Настройка логирования
logging.basicConfig(level=logging.ERROR, filename='log.txt', filemode='w')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Уровень логов TensorFlow

# Перенаправление предупреждений в лог
def custom_warn(message, category, filename, lineno, file=None, line=None):
    logging.error(f"{filename}:{lineno}: {category.__name__}: {message}")

warnings.showwarning = custom_warn

# Настройка логирования TensorFlow для подавления системных сообщений
tf.get_logger().setLevel('ERROR')

# Пути
speakers_path = "SPEAKERS.TXT"
train_path = os.path.join("src", "train.csv")
test_path = os.path.join("src", "test.csv")
train_audio_path = "datasets/train-clean-100"
test_audio_path = "datasets/dev-clean"
output_file = "src/train_prepared.npz"
test_output_file = "src/test_prepared.npz"
scaler_path = "src/scaler.pkl"
model_path = "src/gender_classification_model.keras"

# Параметры аудио
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 13

def prepare_csv():
    """
    Подготавливает CSV файлы train.csv и test.csv на основе файла SPEAKERS.TXT.
    Сохраняет файлы в указанные пути.
    """
    def extract(metadata_lines):
        data = {'train': [], 'test': []}
        for line in metadata_lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                continue
            reader_id, gender, subset = parts[0], parts[1], parts[2]
            if subset == "train-clean-100":
                data['train'].append({"speaker": int(reader_id), "gender": 1 if gender == "M" else 0})
            elif subset == "dev-clean":
                data['test'].append({"speaker": int(reader_id), "gender": 1 if gender == "M" else 0})
        return data

    with open(speakers_path, "r") as file:
        metadata = file.readlines()

    data = extract(metadata)

    train_df = pd.DataFrame(data['train']).reset_index(drop=True)
    test_df = pd.DataFrame(data['test']).reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"train.csv сохранен в {train_path}")
    print(f"test.csv сохранен в {test_path}")

def extract_mfcc(audio_path, csv_path, output_file):
    """
    Извлекает MFCC-признаки из аудиоданных и сохраняет их вместе с метками пола.
    :param audio_path: путь к папке с аудио данными
    :param csv_path: путь к CSV файлу с информацией о спикерах
    :param output_file: путь для сохранения выходных данных
    """
    speaker_df = pd.read_csv(csv_path)
    speaker_df.set_index("speaker", inplace=True)

    mfcc_features = []
    labels = []

    # Получаем список папок с говорящими
    speakers = [folder for folder in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, folder))]
    speakers = [speaker for speaker in speakers if int(speaker) in speaker_df.index]

    # Используем tqdm для отображения прогресса
    for speaker_folder in tqdm(speakers, desc=f"Извлечение MFCC из {audio_path}"):
        speaker_path = os.path.join(audio_path, speaker_folder)
        gender = speaker_df.loc[int(speaker_folder), "gender"]

        for root, _, files in os.walk(speaker_path):
            for file in files:
                if file.endswith(".flac"):
                    file_path = os.path.join(root, file)
                    try:
                        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
                        mfcc_mean = np.mean(mfcc.T, axis=0)
                        mfcc_features.append(mfcc_mean)
                        labels.append(gender)
                    except Exception as e:
                        logging.error(f"Ошибка обработки файла {file_path}: {e}")

    # Сохраняем данные
    np.savez(output_file, mfcc=np.array(mfcc_features), gender=np.array(labels))
    print(f"MFCC и метки пола сохранены в файл: {output_file}")

def train_model():
    """
    Обучает модель на подготовленных данных и сохраняет ее вместе со скейлером.
    """
    data = np.load(output_file)
    X, y = data['mfcc'], data['gender']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
    model.save(model_path)
    print(f"Модель сохранена в файл: {model_path}")

def test_model():
    """
    Тестирует модель на тестовых данных и выводит точность.
    Если MFCC признаки тестовых данных уже сохранены, загружает их из файла.
    """
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    if os.path.exists(test_output_file):
        # Загружаем предварительно сохраненные MFCC признаки тестовых данных
        data = np.load(test_output_file)
        X_test, y_test = data['mfcc'], data['gender']
        print(f"Загружены MFCC и метки пола из файла: {test_output_file}")
    else:
        # Извлекаем MFCC признаки и сохраняем их
        extract_mfcc(test_audio_path, test_path, test_output_file)
        data = np.load(test_output_file)
        X_test, y_test = data['mfcc'], data['gender']

    X_test_scaled = scaler.transform(X_test)
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred == y_test)
    print(f"Точность на тестовых данных: {accuracy:.2%}")

def predict_own_voice():
    """
    Позволяет пользователю загрузить свой аудиофайл и получить предсказание пола.
    """
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    audio_file = input("Введите путь к вашему аудио файлу (в формате .wav или .flac): ").strip()
    if not os.path.isfile(audio_file):
        print("Файл не найден. Пожалуйста, проверьте путь к файлу.")
        return
    try:
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X = mfcc_mean.reshape(1, -1)
        X_scaled = scaler.transform(X)
        y_pred_prob = model.predict(X_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        gender = "Мужской" if y_pred[0] == 1 else "Женский"
        probability = y_pred_prob[0][0] if y_pred[0] == 1 else 1 - y_pred_prob[0][0]
        print(f"Предсказанный пол: {gender}, вероятность: {probability:.2%}")
    except Exception as e:
        print(f"Ошибка обработки файла {audio_file}: {e}")

# Основной интерфейс
def main():
    actions = {
        "1": prepare_csv,
        "2": lambda: extract_mfcc(train_audio_path, train_path, output_file),
        "3": train_model,
        "4": test_model,
        "5": predict_own_voice
    }

    while True:
        print("\nВыберите действие:")
        print("1. Подготовить CSV файлы")
        print("2. Извлечь MFCC из тренировочных данных")
        print("3. Обучить модель")
        print("4. Тестировать модель")
        print("5. Проверить свой голос")
        print("6. Выйти")
        choice = input("Введите номер действия: ").strip()

        if choice == "6":
            print("Выход из программы.")
            break
        elif choice in actions:
            actions[choice]()
        else:
            print("Некорректный ввод. Пожалуйста, выберите номер от 1 до 6.")

if __name__ == "__main__":
    main()
