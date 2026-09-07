import argparse
import csv
import os
from datetime import datetime, timezone

import joblib
import numpy as np

from features import extract_features, extract_features_from_path

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "terrain_classifier_binary.joblib")
REVIEW_QUEUE_PATH = os.path.join(os.path.dirname(__file__), "review_queue.csv")
DEFAULT_MARGIN_THRESHOLD = 0.25
NON_TERRAIN_THRESHOLD = 0.35


class TerrainClassifier:
    def __init__(self, model_path: str = MODEL_PATH, margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
                 non_terrain_threshold: float = NON_TERRAIN_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.margin_threshold = margin_threshold
        self.non_terrain_threshold = non_terrain_threshold

    def _feats_to_vector(self, feats: dict) -> np.ndarray:
        return np.array([[feats[name] for name in self.feature_names]])

    def classify_path(self, path: str) -> dict:
        feats = extract_features_from_path(path)
        result = self._classify_from_feats(feats)
        result["path"] = path
        return result

    def classify_image(self, img: np.ndarray) -> dict:
        feats = extract_features(img)
        return self._classify_from_feats(feats)

    def _classify_from_feats(self, feats: dict) -> dict:
        X = self._feats_to_vector(feats)
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        probs = dict(zip(classes, proba))

        p_mountain = probs.get("mountain", 0.0)
        p_not = probs.get("not_mountain", 0.0)
        margin = abs(p_mountain - p_not)

        label = self.model.predict(X)[0]
        status = "confident" if margin >= self.margin_threshold else "uncertain"
        non_terrain_frac = feats.get("water_area_frac", 0.0) + feats.get("out_of_bounds_area_frac", 0.0)
        non_terrain_override = False
        if non_terrain_frac >= self.non_terrain_threshold and status == "confident":
            status = "uncertain"
            non_terrain_override = True

        return {
            "label": label, "status": status, "probs": probs, "margin": margin,
            "non_terrain_override": non_terrain_override, "non_terrain_frac": non_terrain_frac,
        }


def log_to_review_queue(result: dict, queue_path: str = REVIEW_QUEUE_PATH):
    file_exists = os.path.exists(queue_path)
    with open(queue_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "path", "predicted_label", "margin", "probs"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            result.get("path", ""),
            result["label"],
            f"{result['margin']:.4f}",
            str({k: round(float(v), 4) for k, v in result["probs"].items()}),
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Путь к изображению для классификации")
    parser.add_argument("--log-uncertain", action="store_true",
                         help="Неуверенное предсказание дописать в review_queue.csv")
    args = parser.parse_args()

    clf = TerrainClassifier()
    result = clf.classify_path(args.image_path)

    print(f"Предсказание: {result['label']}  [{result['status']}]")
    if result.get("non_terrain_override"):
        print(f"  (Выставлена метка неуверенности: общая площадь нетеррейна = {result['non_terrain_frac']:.1%} кадра")
    print(f"Разрыв между mountain и not_mountain: {result['margin']:.3f}")
    print("Вероятности по классам:")
    for cls, p in sorted(result["probs"].items(), key=lambda x: -x[1]):
        print(f"  {cls:14s}: {p:.3f}")

    if result["status"] == "uncertain" and args.log_uncertain:
        log_to_review_queue(result)
        print(f"\nНеуверенное предсказание добавлено в {REVIEW_QUEUE_PATH}")