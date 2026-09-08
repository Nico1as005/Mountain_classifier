import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from dataset import build_dataset
from features import FEATURE_NAMES

MODEL_PATH_3CLASS = os.path.join(os.path.dirname(__file__), "models", "terrain_classifier.joblib")
MODEL_PATH_BINARY = os.path.join(os.path.dirname(__file__), "models", "terrain_classifier_binary.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", action="store_true",
        help="Бинарный режим объединяет холмы и равнины в один класс"
    )
    args = parser.parse_args()

    df = build_dataset()

    if args.binary:
        df["label"] = df["label"].apply(lambda x: "mountain" if x == "mountain" else "not_mountain")
        model_path = MODEL_PATH_BINARY
        print("mountain vs not_mountain\n")
    else:
        model_path = MODEL_PATH_3CLASS

    print("Распределение классов:")
    print(df["label"].value_counts())
    print()

    n_per_class = df["label"].value_counts().min()
    if n_per_class < 10:
        print(
            f" Минимальный класс содержит всего {n_per_class} примеров. Результаты не имеют смысла\n"
        )

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    n_splits = min(5, n_per_class) if n_per_class >= 2 else 2
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    try:
        y_pred_cv = cross_val_predict(clf, X, y, cv=skf)
        print(f"Кросс-валидация ({n_splits}-fold)")
        print(classification_report(y, y_pred_cv))
        print("Confusion matrix:")
        labels_sorted = sorted(set(y))
        cm = confusion_matrix(y, y_pred_cv, labels=labels_sorted)
        print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))
        print()

        mistakes = df.loc[y_pred_cv != y, ["path", "label"]].copy()
        mistakes["predicted"] = y_pred_cv[y_pred_cv != y]
        mistakes_path = os.path.join(os.path.dirname(__file__), "misclassified.csv")
        mistakes.to_csv(mistakes_path, index=False)
        print(f"Список ошибочно классифицированных файлов: {mistakes_path}")
        print(f"Всего ошибок: {len(mistakes)} из {len(df)}")
        print("Топ путаниц по парам классов:")
        print(mistakes.groupby(["label", "predicted"]).size().sort_values(ascending=False).to_string())
        print()
    except ValueError as e:
        print(f"Кросс-валидация не удалась: {e}")

    # --- Финальное обучение на всех данных ---
    clf.fit(X, y)

    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("Топ признаков по важности")
    print(importances.to_string())
    print()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, model_path)
    print(f"\nМодель сохранена: {model_path}")


if __name__ == "__main__":
    main()
