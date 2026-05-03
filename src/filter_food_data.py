import pandas as pd
from pathlib import Path

def filter_and_combine_datasets():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"

    input_files = [
        "suomi24_filtered_data_s24_2021.csv",
        "suomi24_filtered_data_s24_2022.csv",
        "suomi24_filtered_data_s24_2023.csv"
    ]
    
    # Yhdistetyn lopputuloksen nimi
    output_file = data_dir / "suomi24_STRICT_food_data.csv"

    # SÄÄNTÖ 1: Ruokaan ja syömiseen liittyvät sanat
    food_words = [
        "ruoka", "syödä", "söin", "syö ", "ateria", "ravinto", "juoda", "juoma", 
        "liha", "kasvis", "vegaani", "proteiini", "hiilihydraatti", "sokeri", 
        "rasva", "maito", "leipä", "vihannes", "hedelmä", "kalori"
    ]

    # SÄÄNTÖ 2: Terveyteen ja kestävyyteen liittyvät sanat
    health_sustainability_words = [
        "terveys", "terveellinen", "epäterveellinen", "luomu", "ekologinen", 
        "kestävä", "ilmasto", "päästö", "lisäaine", "lähiruoka", "ilmastonmuutos",
        "vitamiini", "dieetti", "ruokavalio", "ravintoarvo", "hiilijalanjälki"
    ]

    # SÄÄNTÖ 3: MUSTA LISTA
    blacklist_words = [
        "rokote", "rokotus", "korona", "covid", "thl", "maski", 
        "auto", "bensa", "diesel", "sähköauto", "akku", "lataus", "volvo", 
        "upm", "lakko", "tehdas", "porvari", "duunari", 
        "huora", "seksi", "thaimaa", "lesbo", "homo", 
        "jumala", "paavali", "uskonto"
    ]

    def is_relevant(text):
        if any(bad_word in text for bad_word in blacklist_words):
            return False
        has_food = any(food_word in text for food_word in food_words)
        has_health = any(health_word in text for health_word in health_sustainability_words)
        return has_food and has_health

    all_filtered_dfs = [] # Tähän listaan kerätään kaikkien vuosien suodatetut datat
    total_original = 0

    print("Aloitetaan datan suodatus ja yhdistäminen...")

    for filename in input_files:
        input_path = data_dir / filename
        
        if not input_path.exists():
            print(f"\n[VAROITUS] Tiedostoa ei löydy, ohitetaan: {filename}")
            continue

        print(f"\nKäsitellään: {filename}")
        
        # Ladataan data
        df = pd.read_csv(input_path)
        original_len = len(df)
        total_original += original_len
        
        # Pienet kirjaimet vertailua varten
        df['content_lower'] = df['content'].astype(str).str.lower()

        # Sovelletaan filtteri
        mask = df['content_lower'].apply(is_relevant)
        filtered_df = df[mask].copy()

        # Siivotaan apusarake pois
        filtered_df = filtered_df.drop(columns=['content_lower'])
        
        # Lisätään suodatettu data listaan odottamaan yhdistämistä
        all_filtered_dfs.append(filtered_df)
        
        print(f"  -> Alkuperäinen: {original_len} riviä | Jäljelle jäi: {len(filtered_df)} riviä")

    print(f"\n==============================================")
    print("Yhdistetään tiedostot...")
    
    if all_filtered_dfs:
        # pd.concat yhdistää kaikki listan tiedostot päällekkäin yhdeksi isoksi taulukoksi
        combined_df = pd.concat(all_filtered_dfs, ignore_index=True)
        
        # Tallennetaan lopputulos
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"KAIKKI TIEDOSTOT KÄSITELTY JA YHDISTETTY!")
        print(f"Yhteensä luettu massaa: {total_original} riviä")
        print(f"Lopullisessa yhdistetyssä tiedostossa: {len(combined_df)} puhdasta riviä")
        print(f"Tallennettu nimellä: {output_file.name}")
    else:
        print("Yhtään tiedostoa ei onnistuttu käsittelemään.")

if __name__ == "__main__":
    filter_and_combine_datasets()