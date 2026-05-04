# SNA-coursework-2026

Course work for Introduction to Social Network Analysis, University of Oulu 2026.

The project analyses Finnish food and health discussions from the Suomi24 corpus
(Kielipankki VRT format) and applies social network analysis techniques to the data
(Project 8).

## Requirements

- Python 3.12.3
- Install dependencies: `pip install -r requirements.txt`
- Optional for FinBERT sentiment analysis: `pip install torch transformers tqdm`
- Optional for GUI image display: `pip install Pillow`

## Workflow

### 1. Data collection

Place Kielipankki `.vrt` files anywhere under the project root (they are found
automatically).  Place a keyword CSV (one keyword per row) in the project root or
`data/` directory.

```
python src/data_collection.py
```

Output: `data/suomi24_filtered_data_<filename>.csv` for each VRT file.

### 2. Optional: strict food filter

Applies stricter food + health word rules and a blacklist.

```
python src/filter_food_data.py
```

Output: `data/suomi24_STRICT_food_data.csv`

### 3. Optional: FinBERT sentiment analysis

Runs `fergusq/finbert-finnsentiment` on the strict food filter output.
Requires PyTorch; uses GPU if available.

```
python src/sentiment_analysis.py
```

Output: `data/suomi24_sentiment_FINAL_results.csv`

### 4. Main analysis pipeline (Steps 5-13)

```
python src/main.py [--threshold T] [--max-threads N] [--min-posts P] [--output-dir DIR]
```

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `0.15` | Cosine similarity threshold for thread network edges |
| `--max-threads` | `5000` | Maximum number of threads in the similarity network |
| `--min-posts` | `2` | Minimum posts per user to include in user network |
| `--output-dir` | `outputs/` | Directory for plots and reports |

Progress is shown live in the terminal.  Detailed output is written to
`<output-dir>/reports/run.log`.

Results are written to:
- `<output-dir>/plots/`  - PNG visualisations
- `<output-dir>/reports/` - CSV and JSON summaries

### 5. Interactive viewer and control panel (GUI)

Requires `Pillow` for image display.

```
python src/app.py
```

The GUI has:
- **DATA MANAGEMENT** (page 1): data status, pipeline step buttons (data collection,
  strict filter, FinBERT sentiment), single config runner, overnight batch scheduler
  with shortest-job-first ordering and OOM retry.
- **Plot pages** (pages 2-12): all Step 5-13 visualisations, browsable with arrow keys
  or the navigation buttons.  Use the config dropdown to switch between results from
  different parameter combinations.
- **Network stats** (last page): full `network_stats.json` displayed as text.

## Source files

| File | Description |
|---|---|
| `src/data_collection.py` | VRT parser, keyword matching, CSV output |
| `src/filter_food_data.py` | Strict food + health secondary filter |
| `src/sentiment_analysis.py` | FinBERT sentiment analysis (GPU-accelerated) |
| `src/sentiment.py` | Keyword-based Finnish sentiment (fast fallback) |
| `src/network_builder.py` | Steps 5-7: network construction |
| `src/network_analysis.py` | Steps 8-13: statistics, centrality, communities, k-core |
| `src/visualization.py` | Plot generation (saved as PNG) |
| `src/progress.py` | In-place two-section terminal progress display |
| `src/main.py` | Main pipeline, CLI entry point |
| `src/app.py` | TkInter GUI viewer and control panel |

## Directory layout

```
data/
  keywords.csv                          keyword list (one per row)
  suomi24_filtered_data_s24_<year>.csv  VRT parser output (one per VRT file)
  suomi24_STRICT_food_data.csv          strict filter output (optional)
  suomi24_sentiment_FINAL_results.csv   FinBERT output (optional)

outputs/
  t<T>_n<N>_p<P>/                       one directory per config run
    plots/                              PNG visualizations
    reports/                            CSVs, network_stats.json, run.log

suomi24-2021-2023-vrt/
  data/
    s24_2021.vrt                        Kielipankki corpus files
    s24_2022.vrt
    s24_2023.vrt
```
