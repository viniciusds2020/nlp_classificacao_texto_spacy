from pathlib import Path

from token_efficient_nlp.model import LocalTextClassifier


def test_local_model_trains_predicts_and_reloads(tmp_path: Path):
    texts = [
        "fumaça no painel",
        "cheiro de queimado",
        "piso molhado",
        "risco de tropeço",
        "fogo no equipamento",
        "passarela escorregadia",
    ]
    labels = ["incendio", "incendio", "queda", "queda", "incendio", "queda"]

    model = LocalTextClassifier(spacy_model="modelo_inexistente")
    model.fit(texts, labels)
    prediction = model.predict_one("fumaça no equipamento")

    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = LocalTextClassifier.load(path)

    assert prediction.label in model.labels
    assert 0 <= prediction.confidence <= 1
    assert loaded.labels == model.labels
