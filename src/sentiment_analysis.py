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

    # 2. Määritellään kategoriat - TÄSMÄLLEEN SAMAT LAITETTU TÄNNE (Health vs. Sustainability)
    health_words = [
        "terveys", "terveellinen", "epäterveellinen", "hyvinvointi", "terveellisyys",
        "terveydelle", "terveyshyöty", "sairaus", "oire", "lääkäri", "verenpaine",
        "kolesteroli", "verensokeri", "sydäntauti", "diabetes", "ylipaino", "lihavuus",
        "laihdutus", "painonhallinta", "laihtua", "laihduttaa", "terveemmin", "terveempi",
        "kalori", "kalorit", "kilokalori", "kcal", "energia", "energiantarve",
        "proteiini", "proteiinit", "prode", "hiilihydraatti", "hiilihydraatit", "hiilarit",
        "rasva", "rasvat", "tyydyttynyt", "tyydyttymätön", "transrasva", "omega",
        "sokeri", "sokerit", "piilosokeri", "fruktoosi", "kuitu", "kuidut",
        "vitamiini", "vitamiinit", "ravintoaine", "ravintoaineet", "suojaravintoaine",
        "kalsium", "rauta", "magnesium", "sinkki", "kalium", "natrium", "suola",
        "antioksidantti", "antioksidantit", "hivenaine", "hivenaineet", "b12", "d-vitamiini",
        "c-vitamiini", "foolihappo", "jodi", "dieetti", "ruokavalio", "ravinto", 
        "ruokailutottumus", "keto", "karppaus", "vhh", "vähähiilihydraattinen", 
        "pätkäpaasto", "paasto", "paleo", "gluteeniton", "maidoton", "laktoositon", 
        "keliakia", "allergia", "allergeeni", "vehnätön", "aineenvaihdunta", 
        "ruoansulatus", "suolisto", "mikrobiomi", "vatsa", "turvotus", "närästys", 
        "tulehdus", "immuniteetti", "vastustuskyky", "palautuminen", "lihas", 
        "lihakset", "kunto", "jaksaminen", "vireystila", "ravitseva", "kevytt", 
        "kevyt", "rasvainen", "sokerinen", "suolainen", "myrkky", "lisäaine", 
        "e-koodi", "keinotekoinen", "makeutusaine", "aspartaami", "puhdistava", 
        "detox", "superfood", "tehotuote"
    ]

    sustainability_words = [
        "ekologinen", "ekologisuus", "eko", "kestävä", "kestävyys", "ympäristö",
        "ympäristöystävällinen", "ilmasto", "ilmastonmuutos", "ilmastokriisi",
        "luonto", "luonnonsuojelu", "vihreä", "kierrätys", "hävikki", "ruokahävikki",
        "päästö", "päästöt", "hiilijalanjälki", "hiilinielu", "vesijalanjälki",
        "kasvihuonekaasu", "metaanipäästö", "metaani", "hiilidioksidi", "co2",
        "saaste", "saastuminen", "ilmastoteko", "päästövähennys", "luomu", 
        "luonnonmukainen", "lähiruoka", "kotimainen", "tuotantoeläin",
        "tehotuotanto", "tehomaatalous", "maatalous", "viljely", "torjunta-aine",
        "hyönteismyrkky", "glyfosaatti", "gmo", "geenimuunneltu", "monimuotoisuus",
        "biodiversiteetti", "metsäkato", "sademetsä", "rehu", "soija", "palmuöljy",
        "eettinen", "epäeettinen", "eläinten", "eläinoikeus", "eläinrääkkäys",
        "eläinsuojelu", "häkkikanala", "vapaan", "laiduntava", "luomuliha",
        "teuras", "teurastamo", "kärsimys", "eläinperäinen", "riisto", "reilu", 
        "fairtrade", "vegaani", "vegaaninen", "veganismi", "kasvisruoka", 
        "kasvissyönti", "kasvissyöjä", "kasvis", "kasvipohjainen", "plant-based", 
        "vege", "lihankorvike", "nyhtökaura", "härkis", "soijarouhe", "tofu", 
        "seitan", "kaurajuoma", "kauramaito", "mantelimaito", "kasvimaito", 
        "ilmastodieetti", "pakkaus", "muovi", "muovipakkaus", "kierrätettävä", 
        "biohajoava", "kertakäyttöinen", "mikromuovi", "kestopussi"
    ]

    # TRIGGER-SANAT (Pakottavat "Neutral" -> "Pro" tilaan)
    pro_healthy_triggers = [
        "ravintoaine", "ravintoaineet", "vitamiini", "terveyshyöty", 
        "parantaa", "suojaa", "terveellisempi", "hyväksi", "terveydelle", "ravitseva"
    ]
    
    pro_sustainability_triggers = [
        "ympäristöystävällinen", "kierrätettävä", "luonnonmukainen", 
        "ilmastoteko", "päästövähennys", "hiilinielu"
    ]

    # 3. Tarkistetaan GPU
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

    # 6. Opinion Mining - Luokittelu tehtävänannon mukaisesti (PÄIVITETTY)
    print("Luokitellaan laajennetut kategoriat (Health & Sustainability)...")
    
    def finalize_category(row):
        sentiment = row['Sentiment']
        text = str(row['content']).lower()
        
        is_health = any(word in text for word in health_words)
        is_sustainability = any(word in text for word in sustainability_words)
        
        has_pro_health_trigger = any(trigger in text for trigger in pro_healthy_triggers)
        has_pro_sust_trigger = any(trigger in text for trigger in pro_sustainability_triggers)

        categories = []
        
        # Terveys-luokittelu
        if is_health:
            if sentiment == 'Positive' or (sentiment == 'Neutral' and has_pro_health_trigger): 
                categories.append('Pro-healthy')
            elif sentiment == 'Negative': 
                categories.append('Anti-healthy')
            
        # Kestävyys-luokittelu (Skeptical = Negative asenne kestävyyttä kohtaan)
        if is_sustainability:
            if sentiment == 'Positive' or (sentiment == 'Neutral' and has_pro_sust_trigger): 
                categories.append('Pro-sustainability')
            elif sentiment == 'Negative': 
                categories.append('Skeptical')
            
        # Jos löydettiin luokkia, yhdistetään ne (esim. "Pro-healthy & Pro-sustainability")
        if categories:
            return " & ".join(categories)
            
        # Jos kumpaakaan ei selkeästi löytynyt, jätetään perus sentimentti
        return sentiment

    df['Opinion_Category'] = df.apply(finalize_category, axis=1)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Valmis! Tulokset tallennettu: {output_file}")


if __name__ == "__main__":
    run_sentiment_analysis()
