"""Command-line interface for training and prediction."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from token_efficient_nlp.model import LocalTextClassifier


def train(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.data, sep=args.separator, encoding=args.encoding)
    frame = data[[args.text_column, args.label_column]].dropna()
    frame = frame[frame[args.label_column] != args.ignore_label]

    train_df, test_df = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=frame[args.label_column],
    )
    model = LocalTextClassifier(
        spacy_model=args.spacy_model,
        random_state=args.random_state,
    )
    model.fit(
        train_df[args.text_column].astype(str).tolist(),
        train_df[args.label_column].astype(str).tolist(),
    )
    predicted = model.predict(test_df[args.text_column].astype(str).tolist())
    print(
        classification_report(
            test_df[args.label_column].astype(str),
            [item.label for item in predicted],
            zero_division=0,
        )
    )
    model.save(args.output)
    print(json.dumps(model.model_card(), ensure_ascii=False, indent=2))


def predict(args: argparse.Namespace) -> None:
    model = LocalTextClassifier.load(args.model)
    result = model.predict_one(args.text)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Token-efficient local-first NLP")
    commands = root.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--text-column", default="text")
    training.add_argument("--label-column", default="label")
    training.add_argument("--ignore-label", default="Não Especificado")
    training.add_argument("--separator", default=",")
    training.add_argument("--encoding", default="utf-8")
    training.add_argument("--spacy-model", default="pt_core_news_sm")
    training.add_argument("--output", default="artifacts/classifier.joblib")
    training.add_argument("--test-size", type=float, default=0.20)
    training.add_argument("--random-state", type=int, default=42)
    training.set_defaults(func=train)

    inference = commands.add_parser("predict")
    inference.add_argument("--model", required=True)
    inference.add_argument("--text", required=True)
    inference.set_defaults(func=predict)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
