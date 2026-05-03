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

    # 2. Määritellään kategoriat (Health vs. Sustainability)
    health_words = [
        "terveys", "terveellinen", "epäterveellinen", "vitamiini", 
        "dieetti", "ruokavalio", "ravintoarvo", "proteiini", "kalori", "rasva", "sokeri"
    ]
    
    sustainability_words = [
        "luomu", "ekologinen", "kestävä", "ilmasto", "päästö", 
        "lähiruoka", "ilmastonmuutos", "hiilijalanjälki", "vegaani", "kasvisruoka"
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

    # 6. Opinion Mining - Luokittelu tehtävänannon mukaisesti
    print("Luokitellaan laajennetut kategoriat (Health & Sustainability)...")
    
    def finalize_category(row):
        sentiment = row['Sentiment']
        text = str(row['content']).lower()
        
        # Tarkistetaan kumpaan teemaan viesti liittyy
        is_health = any(word in text for word in health_words)
        is_sustainability = any(word in text for word in sustainability_words)

        categories = []
        
        # Terveys-luokittelu
        if is_health:
            if sentiment == 'Positive': categories.append('Pro-healthy')
            elif sentiment == 'Negative': categories.append('Anti-healthy')
            
        # Kestävyys-luokittelu (Skeptical = Negative asenne kestävyyttä kohtaan)
        if is_sustainability:
            if sentiment == 'Positive': categories.append('Pro-sustainability')
            elif sentiment == 'Negative': categories.append('Skeptical')
            
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