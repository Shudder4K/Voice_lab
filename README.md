##Whisper Speech-to-Text Fine-tuning (Ukrainian TESS)
/n##Опис
Цей проект демонструє повний пайплайн для донавчання моделі Whisper від OpenAI на кастомному датасеті аудіо (наприклад, TESS з українською розміткою). Навчання реалізовано за допомогою PyTorch Lightning для максимальної структури та reproducibility.

В основі лежить:

Власна генерація транскрипцій із імен файлів (Say the word ...).

Автоматичний поділ на train/val/test по списку TEST_IDS.

Додавання фічей та токенізація за допомогою WhisperProcessor.

Гнучкий DataModule для PyTorch Lightning.

Метріки WER (Word Error Rate) та CER (Character Error Rate) для оцінки якості.

Вимоги
Python 3.9–3.12

CUDA-compatible GPU (рекомендовано)

Пакети:
transformers, datasets, torch, torchmetrics, pytorch_lightning

Встановлення
bash
Копіювати
Редагувати
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets torchmetrics pytorch_lightning
Обов'язково перевірити, що версії torch та lightning сумісні з GPU та вашим CUDA.

Датасет
Дата-аудіо зберігаються у папці, наприклад,
C:\Users\User\PycharmProjects\Voice\tess_wav

Ім'я кожного аудіофайлу кодує таргет: наприклад, OAF_back_angry.wav → транскрипт Say the word back.

Поділ на тестовий сет — через перелік TEST_IDS у класі CFG.

Структура коду
WhisperDataModule — створює train/val/test датасети, кастомізує препроцесінг і collate_fn.

WhisperFineTuner — PyTorch Lightning-модуль для навчання, валідації та логування метрик.

remove_columns_if_exists — допоміжна функція, яка безпечно видаляє непотрібні колонки з датасетів.

train_dataloader/val_dataloader/test_dataloader — оптимізовані під lightning функції.

main — точка входу: ініціалізує все та запускає тренування.

Як запустити
Скопіювати свій датасет у потрібну папку.

Вказати шлях у CFG.DATA_DIR.

Запустити файл l4.py:

bash
Копіювати
Редагувати
python l4.py
Або, для Jupyter Notebook, розбити код на клітинки (у файлі вже враховані всі нюанси для роботи в ноутбуці).

Приклад результату
Результати тренування (заповнити після запуску):

python-repl
Копіювати
Редагувати
Epoch 0: train_loss=0.000459, val_WER=0.105, val_CER=0.0251
Epoch 1: train_loss=5e-5, val_WER=0.0234, val_CER=0.00559
...
Пояснення параметрів
CFG.MODEL_NAME — яку модель Whisper використовувати.

CFG.SR — sampling rate (Hz).

CFG.NUM_WORKERS — для DataLoader (на Windows ставте 0 або 1).

MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE — стандартно для глибокого навчання.

gradient_checkpointing_enable — зменшує споживання GPU-пам’яті для великих моделей.

Тестування і кастомні налаштування
Для покращення якості навчання можна експериментувати з:

поділом train/val/test

batch size та lr

більш тривалою треніровкою

Якщо модель починає "завчати" (overfit) — збільшіть розмір датасету або використайте регуляризацію.

Відомі проблеми/поради
На Windows іноді краще ставити num_workers=0 у DataLoader.

Для покращення стабільності використовуйте precision="16-mixed" для сучасних відеокарт.


Результати
Epoch	Train Loss	Val WER	Val CER
0	0.000459	0.105	0.0251
1	0.00005	0.0234	0.00559
2	0.00031	0.0234	0.00559

