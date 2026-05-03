import pandas as pd
from pathlib import Path

try:
    import torch
    from transformers import pipeline
    from tqdm import tqdm
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_HEALTH_WORDS = [
    "terveys", "terveellinen", "epäterveellinen", "vitamiini",
    "dieetti", "ruokavalio", "ravintoarvo", "proteiini", "kalori", "rasva", "sokeri",
]

_SUSTAINABILITY_WORDS = [
    "luomu", "ekologinen", "kestävä", "ilmasto", "päästö",
    "lähiruoka", "ilmastonmuutos", "hiilijalanjälki", "vegaani", "kasvisruoka",
]

_HEALTH_LABELS      = {"Positive": "Pro-healthy",      "Negative": "Anti-healthy"}
_SUSTAIN_LABELS     = {"Positive": "Pro-sustainability", "Negative": "Skeptical"}


def _finalize_category(row: pd.Series) -> str:
    sentiment = row["Sentiment"]
    text = str(row["content"]).lower()

    categories = []
    if any(w in text for w in _HEALTH_WORDS) and sentiment in _HEALTH_LABELS:
        categories.append(_HEALTH_LABELS[sentiment])
    if any(w in text for w in _SUSTAINABILITY_WORDS) and sentiment in _SUSTAIN_LABELS:
        categories.append(_SUSTAIN_LABELS[sentiment])

    return " & ".join(categories) if categories else sentiment


def run_sentiment_analysis() -> None:
    BASE_DIR    = Path(__file__).resolve().parent.parent
    input_file  = BASE_DIR / "data" / "suomi24_STRICT_food_data.csv"
    output_file = BASE_DIR / "data" / "suomi24_sentiment_FINAL_results.csv"

    if not _HAS_TORCH:
        print("VIRHE: torch/transformers/tqdm ei ole asennettu.")
        print("Asenna: pip install torch transformers tqdm")
        return

    if not input_file.exists():
        print(f"VIRHE: Tiedostoa {input_file} ei löydy. Aja suodatinkoodi ensin!")
        return

    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        print("HUOM: NVIDIA GPU:ta ei löydy — ajetaan CPU:lla (hidasta, voi kestää tunteja).")
        device_name = "CPU"
    else:
        device_name = torch.cuda.get_device_name(0)
    print(f"Käytetään laitetta: {device_name}")

    print("Ladataan FinBERT-mallia...")
    sentiment_analyzer = pipeline(
        "text-classification",
        model="fergusq/finbert-finnsentiment",
        device=device,
    )

    df = pd.read_csv(input_file)
    print(f"Ladattu {len(df)} riviä puhdasta ruokadataa.")

    print("Analysoidaan asenteet (batch_size=64)...")
    texts = df["content"].astype(str).tolist()
    results = [
        out["label"].capitalize()
        for out in tqdm(
            sentiment_analyzer(texts, batch_size=64, truncation=True, max_length=512),
            total=len(texts),
        )
    ]
    df["Sentiment"] = results

    print("Luokitellaan laajennetut kategoriat (Health & Sustainability)...")
    df["Opinion_Category"] = df.apply(_finalize_category, axis=1)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Valmis! Tulokset tallennettu: {output_file}")


if __name__ == "__main__":
    run_sentiment_analysis()
