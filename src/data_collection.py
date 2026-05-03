import pandas as pd
from pathlib import Path
import re

def load_keywords(filepath):
    """Loads keywords from a CSV and returns a clean set."""
    try:
        df = pd.read_csv(filepath, header=None)
        # Convert to string, lowercase, and strip
        keywords = set(df[0].astype(str).str.strip().str.lower().tolist())
        return keywords
    except Exception as e:
        print(f"Error loading keywords: {e}")
        return set()

def parse_vrt_for_project_8(vrt_filepath, keyword_set, output_filepath):
    """
    Parses Kielipankki Vrt format
    Extracts metadata, reconstructs sentences, and filters based on keywords.
    """
    print(f"Starting to parse VRT file: {vrt_filepath}")
    
    # State tracking variables
    current_meta = {}
    current_sentence_words = []
    sentence_has_keyword = False
    matched_keywords_in_sentence = set()
    
    matches_found = 0
    lines_processed = 0

    with open(vrt_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', encoding='utf-8') as f_out:
        
        # Write CSV Header
        f_out.write("thread_id,post_id,timestamp,content,matched_keywords\n")
        
        for line in f_in:
            lines_processed += 1
            if lines_processed % 1000000 == 0:
                print(f"Processed {lines_processed:,} lines... Matches found: {matches_found:,}")
                
            line = line.strip()
            if not line:
                continue
                
            #start of new post extract data
            if line.startswith('<text '):
                current_meta = {}
                # Extract thread_id
                t_match = re.search(r'thread_id="([^"]+)"', line)
                current_meta['thread_id'] = t_match.group(1) if t_match else "unknown"
                
                # Extract post_id
                p_match = re.search(r'msg_id="([^"]+)"', line)
                current_meta['post_id'] = p_match.group(1) if p_match else "unknown"
                
                # Extract datetime
                d_match = re.search(r'datetime="([^"]+)"', line)
                current_meta['timestamp'] = d_match.group(1) if d_match else ""
                
                continue
                
            # Start of a new sentence, resets sentence variables
            elif line.startswith('<sentence'):
                current_sentence_words = []
                sentence_has_keyword = False
                matched_keywords_in_sentence = set()
                continue
                
            #end of sentence evaluate and write to a file.
            elif line.startswith('</sentence>'):
                if sentence_has_keyword and current_meta:
                    # Reconstruct the sentence text
                    full_text = " ".join(current_sentence_words)
                    
                    # Clean the text for CSV export
                    clean_text = full_text.replace('"', '""')
                    matched_str = ", ".join(matched_keywords_in_sentence)
                    
                    # Write to file
                    f_out.write(f'"{current_meta.get("thread_id", "")}","{current_meta.get("post_id", "")}","{current_meta.get("timestamp", "")}","{clean_text}","{matched_str}"\n')
                    matches_found += 1
                continue
                
            # Ignore xml tagz
            elif line.startswith('<'):
                continue
                
           #process word rows
            else:
                parts = line.split('\t')
                
                # The actual word
                word = parts[0].strip()
                current_sentence_words.append(word)
                
                # To check against our keywords
                if len(parts) >= 3:
                    lemma = parts[2].strip().lower()
                    
                    # Check if word is in our keyword
                    if lemma in keyword_set:
                        sentence_has_keyword = True
                        matched_keywords_in_sentence.add(lemma)

    print(f"\nParsing Complete!")
    print(f"Total lines processed: {lines_processed:,}")
    print(f"Total matching sentences found: {matches_found:,}")
    print(f"Data saved to: {output_filepath}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    #path to keywords
    keyword_path = BASE_DIR / "data" / "finnish_health_sustainability_words-v2.csv"
    #path to vrt file
    vrt_path = BASE_DIR / "data" / "s24_2023.vrt" 
    
    # output path
    output_path = BASE_DIR / "data" / "suomi24_filtered_data_s24_2023.csv"
    
    if not keyword_path.exists():
        print(f"ERROR: Keyword file not found: {keyword_path}")
    elif not vrt_path.exists():
        print(f"ERROR: VRT file not found: {vrt_path}")
    else:
        keywords = load_keywords(keyword_path)
        print(f"Loaded {len(keywords)} keywords.")
        
        if keywords:
            parse_vrt_for_project_8(vrt_path, keywords, output_path)