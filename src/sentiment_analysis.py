import pandas as pd
import torch
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm

def run_sentiment_analysis():
    # 1. Määritetään polut (Käytetään suodatettua ruokadataa!)
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_file = BASE_DIR / "data" / "suomi24_STRICT_food_data.csv"
    output_file = BASE_DIR / "data" / "suomi24_sentiment_FINAL_results.csv"

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
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Käytetään laitetta: {device_name}")

    # 4. Lataa FinBERT
    print("Ladataan FinBERT-mallia...")
    sentiment_analyzer = pipeline(
        "text-classification", 
        model="fergusq/finbert-finnsentiment",
        device=device
    )

    df = pd.read_csv(input_file)
    print(f"Ladattu {len(df)} riviä puhdasta ruokadataa.")

    # 5. Aja asenneanalyysi
    print(f"Analysoidaan asenteet (batch_size=64)...")
    texts = df['content'].astype(str).tolist()
    
    results = []
    for out in tqdm(sentiment_analyzer(texts, batch_size=64, truncation=True, max_length=512), total=len(texts)):
        results.append(out['label'].capitalize())
    
    df['Sentiment'] = results

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

    # 7. Tallennus
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Valmis! Tulokset tallennettu: {output_file}")

if __name__ == "__main__":
    run_sentiment_analysis()