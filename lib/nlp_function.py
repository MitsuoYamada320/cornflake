from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
max_length = 128  # メモリ削減のため短縮

# トークナイザとモデルのロード（軽量辞書 unidic-lite を指定）
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# モデルのロード（半精度演算を有効化）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
model.eval()  # 学習モードをオフ

def text_to_vector(text_list):
    # 文章を BERT ベクトルに変換（バッチ処理でメモリ最適化）
    encoding = tokenizer(text_list, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

    # デバイス（CPU / GPU）へ移動
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        output = model(**encoding)
        last_hidden_state = output.last_hidden_state
        attention_mask = encoding["attention_mask"]

        # attention_mask を考慮した平均プーリング
        masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
        averaged_hidden_state = masked_hidden_state.sum(1) / attention_mask.sum(1, keepdim=True)

    return averaged_hidden_state.cpu().numpy()  # GPU使用時はCPUに戻す

def cos_sim(vector1, vector2):
    # 2D配列に変換
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
    return float(cosine_similarity(vector1, vector2)[0][0])
