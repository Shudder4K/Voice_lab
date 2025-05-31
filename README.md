# Fine-tuning Whisper-small на TESS Toronto Dataset

Цей проєкт виконує донавчання моделі [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) для задачі автоматичного розпізнавання мовлення (ASR) на основі датасету TESS Toronto. Навчання організоване за допомогою PyTorch Lightning.

---

## Опис проєкту

- **Дані:** аудіофайли `.wav` з папки `tess_wav`. Транскрипти генеруються з імен файлів.
- **Модель:** OpenAI Whisper (small)
- **Пайплайн:** PyTorch Lightning + Huggingface Transformers + Torchmetrics
- **Метрики:** WER (Word Error Rate), CER (Character Error Rate)
- **Файл з кодом:** [`l4.py`](l4.py)

---

## Основні параметри

- **Batch size (train/val):** 16 / 8
- **Epochs:** 2 (можна змінити в `CFG`)
- **Sampling rate:** 16kHz
- **GPU:** RTX 3060/4070 (чи інший CUDA-пристрій)
- **Мова:** Англійська (генерується транскрипція “Say the word …”)

---

## Запуск

```bash
# Віртуальне середовище (рекомендується)
python -m venv .venv
source .venv/bin/activate   # або .venv\Scripts\activate на Windows

# Встановити залежності
pip install torch torchvision torchaudio pytorch-lightning transformers datasets torchmetrics

# Запустити навчання
python l4.py

| Epoch | Train Loss | Val WER | Val CER |
| ----- | ---------- | ------- | ------- |
| 0     | 0.000459   | 0.105   | 0.0251  |
| 1     | 5e-5       | 0.0234  | 0.00559 |
| 2     | 0.00031    | 0.0234  | 0.00559 |

.
├── l4.py                # Основний код для тренування
├── tess_wav/            # Аудіофайли
└── README.md            # (цей файл)

