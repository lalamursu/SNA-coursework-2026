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

    food_words = [
        "ruoka", "syödä", "söin", "syö ", "ateria", "ravinto", "juoda", "juoma", 
        "liha", "kasvis", "vegaani", "proteiini", "hiilihydraatti", "sokeri", 
        "rasva", "maito", "leipä", "vihannes", "hedelmä", "kalori", "kala", 
        "kana", "juusto", "kananmuna", "vilja", "marja", "pähkinä", "herkku"
    ]


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

    blacklist_words = [
        "rokote", "rokotus", "korona", "covid", "thl", "maski", "mdna", 
        "foliohattu", "myrkkypiikki", "pakkorokotus", "pandemia", "tartunta",
        "auto", "bensa", "diesel", "sähköauto", "akku", "lataus", "volvo", 
        "polttomoottori", "etuveto", "takaveto", "ajokortti", "kuljettaja",
        "tuulienergia", "aurinkopaneeli",
        "upm", "lakko", "tehdas", "porvari", "duunari", "rikkuri", 
        "sijoittaja", "valtiovalta", "havupuu", "pesonen", "ay-liike", "tes",
        "huora", "seksi", "thaimaa", "lesbo", "homo", "nussia", "perse", 
        "deittailu", "tinder", "seksitykkäys", "huorintekijä", "sateenkaarilippu",
        "jumala", "paavali", "uskonto", "evankeliumi", "lahko", "jeesus", 
        "seurakunta", "alkuräjähdys", "luomiskertomus",
        "hiv", "aids", "kupan", "tippuri"
    ]

    def is_relevant(text):
        if any(bad_word in text for bad_word in blacklist_words):
            return False
            
        has_food = any(food_word in text for food_word in food_words)
        has_health_or_sust = any(word in text for word in health_words + sustainability_words)
        
        return has_food and has_health_or_sust

    all_filtered_dfs = []
    total_original = 0

    print("Aloitetaan datan suodatus ja yhdistäminen uusilla laajoilla sanalistoilla...")

    for filename in input_files:
        input_path = data_dir / filename
        
        if not input_path.exists():
            print(f"\n[VAROITUS] Tiedostoa ei löydy, ohitetaan: {filename}")
            continue

        print(f"\nKäsitellään: {filename}")
        
        # Ladataan data
        df = pd.read_csv(input_path, encoding="utf-8")
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