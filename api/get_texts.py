from flask import Blueprint, request, jsonify
import os
import numpy as np
from lib.nlp_function import text_to_vector, cos_sim

api = Blueprint("api", __name__)

# 特徴リストをファイルから読み込む関数
def load_cornflake_features():
    feature_file = os.path.join(os.path.dirname(__file__), "../data/text.txt")
    try:
        with open(feature_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"エラー: {feature_file} が見つかりません")
        return []

# 特徴リストとベクトル化
cornflake_features = load_cornflake_features()
cornflake_vectors = text_to_vector(cornflake_features)

@api.route("/similarity", methods=["POST"])
def similarity():
    data = request.get_json()
    input_text = data.get("text", "")

    if not input_text:
        return jsonify({"error": "テキストが入力されていません"}), 400

    # 類似度の閾値
    threshold_value = 0.8

    # 入力文をベクトル化(nlp_function.py)
    input_vector = text_to_vector([input_text])[0]

    similarity_scores = [
        {"text": cornflake_features[i], "value": float(sim_value)}
        for i, feature_vector in enumerate(cornflake_vectors)
        if (sim_value := cos_sim(feature_vector, input_vector)) >= threshold_value
    ]

    # 類似度が高い順にソート
    similarity_scores.sort(key=lambda x: x["value"], reverse=True)

    response = {
        "message": "それはコーンフレークやね" if similarity_scores else "ほなコーンフレークと違うかー",
        "similar_texts": similarity_scores
    }

    return jsonify(response)
