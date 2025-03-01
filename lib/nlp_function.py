from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
max_length = 256

#トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

def text_to_vector(text_list):
    #文章を BERT ベクトルに変換
    texts_vector = []
    for text in text_list:
        #truncation = maxの文字数を265とするための引数
        #return_tensors="pt"に設定すると、Pytorch(オープンソース機械学習ライブラリ)に入力可能なtensor型に変換
        encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        attention_mask = encoding["attention_mask"]

        #BERTの出力を処理
        with torch.no_grad():
            # BERTで計算
            output = model(**encoding)
            last_hidden_state = output.last_hidden_state
            averaged_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        #ベクトルを保存
        texts_vector.append(averaged_hidden_state[0].detach().numpy())
    return texts_vector

def cos_sim(vector1, vector2):
    # 2D配列に変換
    vector1 = vector1.reshape(1, -1)  
    vector2 = vector2.reshape(1, -1)
    return float(cosine_similarity(vector1, vector2)[0][0])
