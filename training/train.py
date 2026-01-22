# ============================================
# DeBERTa-v3-Large for URL(HEAT) Classification
# (Transformers 4.57 호환 + Regularization + EarlyStopping ON)
# - 안정화 패치: gradient checkpointing 비활성화
# ============================================

import os, json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

import torch
from torch import nn

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)

# --- 버전 확인 ---
import transformers, tokenizers, accelerate, huggingface_hub
print(
    "Versions ->",
    "transformers:", transformers.__version__,
    "| tokenizers:", tokenizers.__version__,
    "| accelerate:", accelerate.__version__,
    "| hub:", huggingface_hub.__version__
)

# (Optional) Google Drive (Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as _e:
    print("[Info] Colab이 아니거나 Drive 마운트를 건너뜁니다:", _e)

# ==============================
# 1) CONFIG
# ==============================
class CFG:
    data_path         = "/content/drive/MyDrive/HEAT_release_v1/dataset_cleaned.csv"
    label_map_path    = "/content/drive/MyDrive/HEAT_release_v1/label_map.json"
    class_weight_path = "/content/drive/MyDrive/HEAT_release_v1/class_weight.json"  # 없으면 None

    model_name = "microsoft/deberta-v3-large"
    text_col   = "url"
    label_col  = "label_id"

    # 정규화(Regularization)
    dropout       = 0.2
    attn_dropout  = 0.2
    cls_dropout   = 0.2
    weight_decay  = 0.01

    # 학습 설정
    max_length = 192
    batch_size = 4
    grad_accum = 8

    val_size  = 0.15
    test_size = 0.15
    seed      = 42

    hp_lr        = ["2e-5", "1e-5", "5e-6"]
    hp_epochs    = ["3", "4"]
    hp_scheduler = ["linear", "cosine"]

    output_dir = "/content/drive/MyDrive/capstone/outputs_deberta_v3_reg_457_stable"

# ==============================
# 2) Utilities
# ==============================
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def basic_preprocess(text: str) -> str:
    return str(text).strip().lower()

def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    if all(isinstance(v, (int, float)) for v in lm.values()):
        return {str(k): int(v) for k, v in lm.items()}
    if all(str(k).isdigit() and isinstance(v, str) for k, v in lm.items()):
        return {v: int(k) for k, v in lm.items()}
    raise ValueError(f"Unrecognized label_map format: {lm}")

def load_class_weight(path: Optional[str], label_map: Dict[str,int]) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        cw = json.load(f)
    inv_map = {v:k for k,v in label_map.items()}  # {id: name}
    weights = []
    for lid in range(len(label_map)):
        name_key = inv_map[lid]; id_key = str(lid)
        if name_key in cw: val = float(cw[name_key])
        elif id_key in cw: val = float(cw[id_key])
        else: val = 1.0
        weights.append(val)
    return torch.tensor(weights, dtype=torch.float)

def map_labels(series: pd.Series, label_map: Dict[str,int]) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).map(lambda s: label_map[str(s)])
    ints = series.astype(int)
    max_id = max(label_map.values()); min_id = min(label_map.values())
    if ints.min() < min_id or ints.max() > max_id:
        print(f"[WARN] label id range ({ints.min()}~{ints.max()}) vs map ({min_id}~{max_id})")
    return ints

# ==============================
# 3) TrainingArguments (4.57 호환)
# ==============================
def make_training_args(
    output_dir, batch_size, grad_accum, lr, epochs, seed, fp16, scheduler_kind, weight_decay
):
    common = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(lr),
        num_train_epochs=int(epochs),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        bf16=False,
        seed=seed,
        lr_scheduler_type=scheduler_kind,
        report_to="none",
        save_total_limit=2,

        # 안정성/메모리
        gradient_checkpointing=False,   # ★ OFF (핵심 수정)
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        group_by_length=True,
        torch_compile=False,
        optim="adamw_torch",

        # 정규화
        weight_decay=weight_decay,

        # Trainer가 입력 컬럼을 건드리지 않도록(손실 수동계산 시 안전)
        remove_unused_columns=False,
    )
    # 4.57: evaluation_strategy → eval_strategy
    return TrainingArguments(
        eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500,
        **common
    )

# ==============================
# 4) Dataset
# ==============================
class URLDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length: int = 256):
        self.texts  = [basic_preprocess(t) for t in df[text_col].tolist()]
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length
        )
        enc["labels"] = self.labels[idx]
        return enc

# ==============================
# 5) WeightedTrainer
# ==============================
class WeightedTrainer(Trainer):
    def __init__(self, class_weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Trainer가 모델에 labels를 넘겨서 내부 loss를 만들지 않도록, labels를 분리
        labels = inputs.pop("labels")
        outputs = model(**inputs)  # return_dict=True (default)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weight.to(logits.device) if self.class_weight is not None else None
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_outputs:
            # eval/predict 단계 안전: 그래프 비보존 텐서로 반환
            safe_outputs = {}
            for k, v in outputs.items():
                safe_outputs[k] = v.detach() if torch.is_tensor(v) else v
            return loss, type(outputs)(**safe_outputs)
        return loss

# ==============================
# 6) Metrics
# ==============================
def _auto_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    n_classes = logits.shape[1]
    if n_classes <= 2:
        avg = "binary"
        try:
            probs_pos = torch.tensor(logits).softmax(dim=-1).numpy()[:, 1]
            auc = roc_auc_score(labels, probs_pos)
        except Exception:
            auc = float("nan")
    else:
        avg = "macro"
        try:
            probs = torch.tensor(logits).softmax(dim=-1).numpy()
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=avg, zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc}

def compute_metrics_fn(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return _auto_metrics(np.asarray(logits), np.asarray(labels))

# ==============================
# 7) Error Analysis
# ==============================
def fine_grained_error_analysis(
    df_split: pd.DataFrame, preds: np.ndarray, probs: np.ndarray,
    text_col: str, label_col: str, id2label: Dict[int,str],
    out_dir: str, split_name: str
):
    df = df_split.copy()
    df["pred"] = preds
    df["prob_1"] = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else np.nan
    df["correct"] = (df[label_col].values == df["pred"].values).astype(int)
    df["label_name"] = df[label_col].map(id2label); df["pred_name"]  = df["pred"].map(id2label)

    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, f"{split_name}_predictions.csv"), index=False)
    df[df["correct"] == 0].sort_values("prob_1", ascending=False).to_csv(
        os.path.join(out_dir, f"{split_name}_misclassified.csv"), index=False
    )

    def extract_domain(url: str) -> str:
        u = str(url)
        if "://" in u: u = u.split("://", 1)[1]
        return u.split("/", 1)[0]

    df["domain_extracted"] = df[text_col].apply(extract_domain)
    df.groupby("domain_extracted").agg(
        n=("domain_extracted","count"),
        acc=("correct","mean")
    ).sort_values("n", ascending=False).head(100).to_csv(
        os.path.join(out_dir, f"{split_name}_domain_stats_top100.csv")
    )

    cm = confusion_matrix(df[label_col], df["pred"])
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{id2label[i]}" for i in range(cm.shape[0])],
        columns=[f"pred_{id2label[i]}" for i in range(cm.shape[1])]
    )
    cm_df.to_csv(os.path.join(out_dir, f"{split_name}_confusion_matrix.csv"))

    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(
        df[label_col], df["pred"],
        target_names=target_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).to_csv(os.path.join(out_dir, f"{split_name}_classification_report.csv"))

# ==============================
# 8) Train/Eval once (EarlyStopping ON, gc OFF)
# ==============================
def train_eval_once(
    train_df, val_df, test_df,
    text_col, label_col, model_name, id2label, label2id,
    lr, epochs, scheduler_kind,
    batch_size, grad_accum, max_length, output_dir,
    class_weight_tensor=None, seed=42,
    dropout=0.2, attn_dropout=0.2, cls_dropout=0.2, weight_decay=0.01
):
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
        problem_type="single_label_classification"
    )
    # Dropout 적용
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = float(dropout)
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = float(attn_dropout)
    if hasattr(config, "classifier_dropout") and cls_dropout is not None:
        config.classifier_dropout = float(cls_dropout)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # 안정화 옵션
    model.config.use_cache = False          # generation cache off
    # ★ gradient checkpointing 사용 안 함 (gc OFF)
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()

    torch.backends.cuda.matmul.allow_tf32 = True

    # Datasets
    dtrain = URLDataset(train_df, text_col, label_col, tokenizer, max_length)
    dval   = URLDataset(val_df,   text_col, label_col, tokenizer, max_length)
    dtest  = URLDataset(test_df,  text_col, label_col, tokenizer, max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    fp16 = torch.cuda.is_available()

    args = make_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        epochs=epochs,
        seed=seed,
        fp16=fp16,
        scheduler_kind=scheduler_kind,
        weight_decay=weight_decay,
    )

    print("[DEBUG] model is None? ->", model is None)
    print("[DEBUG] model type     ->", type(model))

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        processing_class=tokenizer,   # 4.57 권장
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        class_weight=class_weight_tensor,
    )

    # EarlyStopping ON
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=1e-4
    ))

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("[Eval] val metrics:", eval_metrics)

    # Test
    test_pred = trainer.predict(dtest)
    logits = test_pred.predictions if not isinstance(test_pred.predictions, tuple) else test_pred.predictions[0]
    probs = torch.tensor(logits).softmax(dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "test_logits.npy"), logits)
    np.save(os.path.join(output_dir, "test_probs.npy"), probs)
    np.save(os.path.join(output_dir, "test_preds.npy"), preds)

    fine_grained_error_analysis(
        test_df, preds, probs, text_col, label_col, id2label,
        output_dir, split_name="test"
    )
    return eval_metrics

# ==============================
# 9) Run
# ==============================
def main():
    cfg = CFG()

    df = pd.read_csv(cfg.data_path, usecols=[cfg.text_col, cfg.label_col])

    label_map = load_label_map(cfg.label_map_path)
    label2id = {str(name): idx for name, idx in label_map.items()}
    id2label = {idx: name for name, idx in label_map.items()}

    if cfg.label_col not in df.columns:
        raise ValueError(f"{cfg.label_col} not found in dataframe columns: {list(df.columns)}")

    df[cfg.label_col] = map_labels(df[cfg.label_col], label_map)

    df_train, df_temp = train_test_split(
        df, test_size=cfg.val_size + cfg.test_size, random_state=cfg.seed, stratify=df[cfg.label_col]
    )
    rel_test = cfg.test_size / (cfg.val_size + cfg.test_size)
    df_val, df_test = train_test_split(
        df_temp, test_size=rel_test, random_state=cfg.seed, stratify=df_temp[cfg.label_col]
    )
    print(f"[Split] train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    cw = load_class_weight(cfg.class_weight_path, label_map)

    lr        = cfg.hp_lr[0]
    epochs    = cfg.hp_epochs[0]
    scheduler = cfg.hp_scheduler[0]

    metrics = train_eval_once(
        df_train, df_val, df_test,
        cfg.text_col, cfg.label_col, cfg.model_name, id2label, label2id,
        lr, epochs, scheduler,
        cfg.batch_size, cfg.grad_accum, cfg.max_length, cfg.output_dir,
        class_weight_tensor=cw, seed=cfg.seed,
        dropout=cfg.dropout, attn_dropout=cfg.attn_dropout, cls_dropout=cfg.cls_dropout,
        weight_decay=cfg.weight_decay
    )
    print("[Done] Metrics:", metrics)

if __name__ == "__main__":
    main()