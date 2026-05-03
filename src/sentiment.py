import re
import pandas as pd

# Finnish positive words relevant to healthy/sustainable food discussions
POSITIVE_WORDS = {
    "hyvä", "hyvää", "hyvin", "erinomainen", "upea", "loistava", "mahtava", "terveellinen",
    "ravitseva", "tuore", "luomu", "herkullinen", "maukas", "mainio", "positiivinen",
    "tärkeä", "rakastaa", "pitää", "nauttia", "tyytyväinen", "onnellinen", "toimiva",
    "tehokas", "edullinen", "halpa", "tarjous", "laadukas", "luonnollinen", "puhdas",
    "tasapainoinen", "monipuolinen", "kestävä", "ympäristöystävällinen", "vastuullinen",
    "ekologinen", "paikallinen", "lähiruoka", "kasvispohjainen", "kasvisruoka", "sopiva",
    "suosia", "hyödyllinen", "parempi", "paras", "turvallinen", "terveys", "hyvinvointi",
    "suositella", "kannattaa", "kevyt", "vähärasvainen", "sokeriton", "proteiini",
    "vitamiini", "kuitu", "kuitupitoinen", "ravintoarvo", "vegaani", "vegaaninen",
    "helppo", "vaivaton",
}

# Finnish negative words relevant to healthy/sustainable food discussions
NEGATIVE_WORDS = {
    "huono", "huonosti", "epäterveellinen", "haitallinen", "vaarallinen", "kallis",
    "inhottava", "vastenmielinen", "pahoinvointi", "pahaa", "mauton", "sairaus",
    "lihoaa", "lihominen", "ylipaino", "ylipainoinen", "lihava", "rasvainen", "rasva",
    "sokeripitoinen", "sokeri", "prosessoitu", "keinotekoinen", "lisäaine", "kemikaali",
    "myrkyllinen", "allergia", "allergeeni", "tehotuotanto", "kärsimys", "saastuttava",
    "ongelma", "pettymys", "pettynyt", "ärsyttää", "ärsyttävä", "vastustaa", "turhaa",
    "teurastamo", "eläinkoe", "roskaruoka", "pikaruoka", "eines", "valmisruoka",
    "karkki", "sipsi", "limu", "mikroateria", "epäpuhdas", "saastunut", "pilaantunut",
    "vanha", "vanhentunut", "halveksua", "inhota", "karttaa", "välttää", "varoittaa",
    "liha", "lihansyönti", "nauta", "sika", "broileri",
    "ilmastovaikutus", "hiilijalanjälki",
}


def classify_sentiment(text: str) -> str:
    """Classify text as positive/negative/neutral using a Finnish keyword lexicon."""
    if pd.isna(text) or not str(text).strip():
        return "neutral"

    words = set(re.findall(r"\b\w+\b", str(text).lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def add_sentiment_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'sentiment' column to the dataframe using keyword-based classification."""
    print("Running keyword-based sentiment analysis...")
    df = df.copy()
    df["sentiment"] = df["content"].apply(classify_sentiment)

    counts = df["sentiment"].value_counts()
    total = len(df)
    for label in ("positive", "negative", "neutral"):
        n = counts.get(label, 0)
        print(f"  {label:8s}: {n:>8,}  ({100 * n / total:.1f}%)")

    return df
