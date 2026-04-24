import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# =====================
# Config
# =====================
DATA_PATH = "C:/Users/hp/OneDrive/Desktop/FaceSketcher/Module1/module1/data/celeba/list_attr_celeba.csv"
MODEL_PATH = "C:/Users/hp/OneDrive/Desktop/FaceSketcher/Module1/module1/models/bert_celeba"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64        # Increased: attribute text can exceed 32 tokens
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SAMPLE_SIZE = 5000

# FIX 1: num_workers > 0 causes crashes on Windows with DataLoader
# Safe default: 0 on Windows, cpu_count-1 on Linux/Mac
NUM_WORKERS = 0 if os.name == "nt" else max(os.cpu_count() - 1, 1)

os.makedirs(MODEL_PATH, exist_ok=True)
print("Device:", DEVICE)


# =====================
# Load & Preprocess CSV
# =====================
df = pd.read_csv(DATA_PATH)

if "image_id" in df.columns:
    df = df.drop(columns=["image_id"])

# Convert -1 → 0 (CelebA uses -1 for absent attributes)
df = df.replace(-1, 0)

attribute_names = df.columns.tolist()
num_labels = len(attribute_names)
print(f"Attributes found: {num_labels}")

# FIX 2: reset_index after sample to avoid index misalignment
df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)


# =====================
# Convert attributes → text
# FIX 3: Correctly iterate using attribute_names + row values (not row.items())
# =====================
def attributes_to_text(row_values: list, attr_names: list) -> str:
    """Convert a binary attribute vector to a human-readable text string."""
    active = [
        attr.replace("_", " ").lower()
        for attr, val in zip(attr_names, row_values)
        if val == 1
    ]
    # Edge case: if no attributes are active, return a placeholder
    return " ".join(active) if active else "no attributes"


print("Generating text descriptions from attributes...")
texts, labels = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    row_values = row.tolist()
    texts.append(attributes_to_text(row_values, attribute_names))
    labels.append(row_values)   # multi-label targets (list of 0/1 floats)

# =====================
# Train / Val Split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")


# =====================
# Tokenizer & Dataset
# =====================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class CelebADataset(Dataset):
    def __init__(self, texts: list, labels: list):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# FIX 4: pin_memory only helps with CUDA; avoid on CPU
train_loader = DataLoader(
    CelebADataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    CelebADataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)



# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# Training + Validation Loop

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_train_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{EPOCHS} — Training")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()                                   # zero before forward
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"  Train Loss: {avg_train_loss:.4f}")

    # --- Validate ---
    model.eval()
    total_val_loss = 0.0

    print(f"Epoch {epoch + 1}/{EPOCHS} — Validation")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"  Val   Loss: {avg_val_loss:.4f}")


# =====================
# Save
# FIX 6: Save attribute names with a proper column header for easy reloading
# =====================
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

attr_series = pd.Series(attribute_names, name="attribute")
attr_series.to_csv(os.path.join(MODEL_PATH, "attributes.csv"), index=False)

print(f"\nModel saved at: {MODEL_PATH}")