import os
import platform
from typing import List
from datasets import Dataset, Audio
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torchmetrics.text import WordErrorRate, CharErrorRate

# ==== КОНФІГ =====
class CFG:
    TEST_IDS = [
        'toronto_27', 'toronto_46', 'toronto_42', 'toronto_37', 'toronto_89',
        'toronto_43', 'toronto_157', 'toronto_9', 'toronto_156', 'toronto_7',
        'toronto_123', 'toronto_54', 'toronto_67', 'toronto_62', 'toronto_81',
        'toronto_134', 'toronto_148', 'toronto_21', 'toronto_135', 'toronto_166',
        'toronto_58'
    ]
    DATA_DIR = r'C:\Users\User\PycharmProjects\Voice\tess_wav'  # <-- твоя папка з .wav файлами
    PREPROCESSED_DIR = r"C:\Users\User\PycharmProjects\Voice\preprocessed_toronto"
    MODEL_NAME = "openai/whisper-small"
    SR = 16000
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 8
    NUM_WORKERS = 1
    MAX_EPOCHS = 2
    LEARNING_RATE = 1e-5

def get_transcript_from_filename(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    if len(parts) < 2:
        return None
    word = parts[1]
    return f"Say the word {word}"

def build_records(data_dir, test_ids: List[str]):
    records_train = []
    records_test = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.wav'):
            continue
        fid = os.path.splitext(fname)[0]
        transcript = get_transcript_from_filename(fname)
        full_path = os.path.join(data_dir, fname)
        record = {"id": fid, "audio": full_path, "transcript": transcript}
        if fid in test_ids:
            records_test.append(record)
        else:
            records_train.append(record)
    return records_train, records_test

def remove_columns_if_exists(ds, cols):
    for col in cols:
        if col in ds.column_names:
            ds = ds.remove_columns(col)
    return ds

class WhisperDataModule(pl.LightningDataModule):
    def __init__(self, processor, data_dir, test_ids, batch_size=8, val_batch_size=8, num_workers=4, sr=16000):
        super().__init__()
        self.processor = processor
        self.data_dir = data_dir
        self.test_ids = test_ids
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.sr = sr

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_records, test_records = build_records(self.data_dir, self.test_ids)
        val_size = min(64, len(train_records) // 10)
        val_records = train_records[:val_size]
        train_records = train_records[val_size:]
        self.train_dataset = Dataset.from_list(train_records)
        self.val_dataset = Dataset.from_list(val_records)
        self.test_dataset = Dataset.from_list(test_records)

        self.train_dataset = self.train_dataset.cast_column("audio", Audio(sampling_rate=self.sr))
        self.val_dataset = self.val_dataset.cast_column("audio", Audio(sampling_rate=self.sr))
        self.test_dataset = self.test_dataset.cast_column("audio", Audio(sampling_rate=self.sr))

        def preprocess(batch):
            arrays = [x["array"] for x in batch["audio"]]
            inputs = self.processor.feature_extractor(
                arrays, CFG.SR, return_tensors="pt", sampling_rate=CFG.SR
            )
            labels = self.processor.tokenizer(
                batch["transcript"], padding="longest", return_tensors="pt"
            ).input_ids
            batch["input_features"] = inputs.input_features
            batch["labels"] = labels
            return batch

        num_proc = min(self.num_workers, os.cpu_count() or 1)
        self.train_dataset = self.train_dataset.map(
            preprocess, batched=True, num_proc=num_proc
        )
        self.val_dataset = self.val_dataset.map(
            preprocess, batched=True, num_proc=1
        )
        self.test_dataset = self.test_dataset.map(
            preprocess, batched=True, num_proc=1
        )
        self.train_dataset = remove_columns_if_exists(self.train_dataset, ["audio", "transcript", "id"])
        self.val_dataset = remove_columns_if_exists(self.val_dataset, ["audio", "transcript", "id"])
        self.test_dataset = remove_columns_if_exists(self.test_dataset, ["audio", "transcript", "id"])

    def collate_fn(self, batch):
        input_feats = []
        for ex in batch:
            arr = ex["input_features"]
            if isinstance(arr, list):
                arr = torch.tensor(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            input_feats.append(arr)
        labels = [torch.tensor(ex["labels"]) if isinstance(ex["labels"], list) else ex["labels"].squeeze(0) for ex in batch]
        input_feats_padded = torch.nn.utils.rnn.pad_sequence(
            input_feats, batch_first=True, padding_value=0.0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels_padded[labels_padded == self.processor.tokenizer.pad_token_id] = -100
        return {"input_features": input_feats_padded, "labels": labels_padded}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, persistent_workers=False, collate_fn=self.collate_fn
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
            num_workers=0, pin_memory=True, persistent_workers=False, collate_fn=self.collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
            num_workers=0, pin_memory=True, persistent_workers=False, collate_fn=self.collate_fn
        )

class WhisperFineTuner(pl.LightningModule):
    def __init__(self, model_name, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()
        # Only use torch.compile if on Linux
        if hasattr(torch, "compile") and platform.system() != "Windows":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def forward(self, input_features, labels=None):
        return self.model(input_features, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_features"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            batch["input_features"], num_beams=1, max_new_tokens=100
        )
        preds = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = torch.where(batch["labels"] != -100, batch["labels"], self.processor.tokenizer.pad_token_id)
        refs = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        self.wer.update(preds, refs)
        self.cer.update(preds, refs)

    def on_validation_epoch_end(self):
        wer = self.wer.compute()
        cer = self.cer.compute()
        self.log("val_WER", wer, prog_bar=True)
        self.log("val_CER", cer, prog_bar=True)
        self.wer.reset()
        self.cer.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    processor = WhisperProcessor.from_pretrained(CFG.MODEL_NAME)
    data_module = WhisperDataModule(
        processor,
        CFG.DATA_DIR,
        CFG.TEST_IDS,
        batch_size=CFG.TRAIN_BATCH_SIZE,
        val_batch_size=CFG.VAL_BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        sr=CFG.SR
    )
    model = WhisperFineTuner(CFG.MODEL_NAME, lr=CFG.LEARNING_RATE)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=CFG.MAX_EPOCHS,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)
