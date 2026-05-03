import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _print_progress(filename: str, pct: float, lines: int, matches: int,
                    elapsed: float, eta: float | None) -> None:
    bar_width = 25
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    eta_str = f"~{_format_duration(eta)}" if eta is not None else "–"
    content = (
        f"[{filename}]  {bar}  {pct:5.1f}%"
        f"  |  Lines: {lines:>12,}"
        f"  |  Matches: {matches:>8,}"
        f"  |  Elapsed: {_format_duration(elapsed)}"
        f"  |  ETA: {eta_str}"
    )
    width = shutil.get_terminal_size(fallback=(120, 24)).columns - 1
    sys.stderr.write(f"\r{content[:width]}\033[K")
    sys.stderr.flush()

def load_keywords(filepath):
    """Loads keywords from a CSV and returns a clean set."""
    try:
        df = pd.read_csv(filepath, header=None)
        return set(df[0].astype(str).str.strip().str.lower().tolist())
    except Exception as e:
        print(f"Error loading keywords: {e}")
        return set()


# ── VRT parsing helpers ───────────────────────────────────────────────────────

def _attr(line: str, pattern: str, fallback: str = "") -> str:
    m = re.search(pattern, line)
    return m.group(1) if m else fallback


def _parse_text_meta(line: str) -> dict:
    """Extract post metadata from a <text ...> VRT tag."""
    u_match = (re.search(r'nick="([^"]+)"', line)
                or re.search(r'author_id="([^"]+)"', line)
                or re.search(r'author="([^"]+)"', line))
    return {
        "thread_id": _attr(line, r'thread_id="([^"]+)"', "unknown"),
        "post_id":   _attr(line, r'msg_id="([^"]+)"', "unknown"),
        "user_id":   u_match.group(1) if u_match else "unknown",
        "section":   _attr(line, r'section_id="([^"]+)"'),
        "timestamp": _attr(line, r'datetime="([^"]+)"'),
    }


def _write_sentence(f_out, meta: dict, words: list, matched: set) -> None:
    """Write one matching sentence as a CSV row."""
    clean_text = " ".join(words).replace('"', '""')
    f_out.write(
        f'"{meta["thread_id"]}","{meta["post_id"]}",'
        f'"{meta["user_id"]}","{meta["section"]}",'
        f'"{meta["timestamp"]}","{clean_text}","{", ".join(matched)}"\n'
    )


def _process_token(line: str, keyword_set: set, words: list, matched: set) -> None:
    """Append word to sentence and record lemma if it is a keyword."""
    parts = line.split('\t')
    words.append(parts[0].strip())
    if len(parts) >= 3:
        lemma = parts[2].strip().lower()
        if lemma in keyword_set:
            matched.add(lemma)


def _handle_sentence_end(f_out, meta: dict, words: list, matched: set) -> int:
    """Write sentence to CSV if it contains keywords; return 1 if written."""
    if matched and meta:
        _write_sentence(f_out, meta, words, matched)
        return 1
    return 0


def _maybe_report_progress(f_in, file_size: int, filename: str,
                            lines: int, matches: int,
                            start_time: float, last_time: float,
                            interval: int) -> float:
    """Print progress bar if interval has elapsed; return updated last_time."""
    now = time.time()
    if now - last_time < interval:
        return last_time
    bytes_read = f_in.tell()
    elapsed = now - start_time
    speed = bytes_read / elapsed if elapsed > 0 else 1
    _print_progress(
        filename,
        pct=100.0 * bytes_read / file_size,
        lines=lines,
        matches=matches,
        elapsed=elapsed,
        eta=(file_size - bytes_read) / speed,
    )
    return now


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_vrt_for_project_8(vrt_filepath, keyword_set, output_filepath):
    """Parse a Kielipankki VRT file and write keyword-matching sentences to CSV."""
    vrt_filepath = Path(vrt_filepath)
    file_size = os.path.getsize(vrt_filepath)
    filename = vrt_filepath.name
    print(f"\nParsing: {filename}  ({file_size / 1e9:.1f} GB)")
    print(f"Output : {output_filepath}")

    current_meta: dict = {}
    sentence_words: list = []
    matched_keywords: set = set()

    lines_processed = 0
    matches_found = 0
    start_time = time.time()
    last_progress_time = start_time

    with open(vrt_filepath, 'r', encoding='utf-8') as f_in, \
            open(output_filepath, 'w', encoding='utf-8') as f_out:

        f_out.write("thread_id,post_id,user_id,section,timestamp,content,matched_keywords\n")

        for raw in iter(f_in.readline, ''):
            lines_processed += 1
            last_progress_time = _maybe_report_progress(
                f_in, file_size, filename,
                lines_processed, matches_found,
                start_time, last_progress_time, interval=30,
            )

            line = raw.strip()
            if not line:
                continue

            if line.startswith('<text '):
                current_meta = _parse_text_meta(line)
            elif line.startswith('<sentence'):
                sentence_words, matched_keywords = [], set()
            elif line.startswith('</sentence>'):
                matches_found += _handle_sentence_end(f_out, current_meta, sentence_words, matched_keywords)
            elif not line.startswith('<'):
                _process_token(line, keyword_set, sentence_words, matched_keywords)

    sys.stderr.write("\n")
    print(f"Done.  Lines: {lines_processed:,}  |  Matches: {matches_found:,}")
    print(f"Saved: {output_filepath}")

_SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules"}


def _find_files(base: Path, pattern: str) -> list[Path]:
    """Recursively find files matching glob pattern, skipping common non-data dirs."""
    results = []
    for p in base.rglob(pattern):
        if not any(skip in p.parts for skip in _SKIP_DIRS):
            results.append(p)
    return sorted(results)


def _pick_keyword_file(base: Path) -> Path | None:
    """Return the best keyword CSV: prefer names with 'keyword'/'sana'/'word', else any CSV."""
    all_csv = _find_files(base, "*.csv")
    # Exclude output files produced by this script
    candidates = [p for p in all_csv if "filtered_data" not in p.name]
    if not candidates:
        return None
    priority = [p for p in candidates if any(k in p.name.lower() for k in ("keyword", "sana", "word"))]
    return priority[0] if priority else candidates[0]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_DIR / "data"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Locate keyword file ────────────────────────────────────────────────
    keyword_path = _pick_keyword_file(BASE_DIR)
    if keyword_path is None:
        print("ERROR: No keyword CSV found anywhere under the project directory.")
        raise SystemExit(1)
    print(f"Keywords : {keyword_path.relative_to(BASE_DIR)}")

    keywords = load_keywords(keyword_path)
    print(f"Loaded {len(keywords)} keywords.")
    if not keywords:
        raise SystemExit(1)

    # ── Locate VRT files ───────────────────────────────────────────────────
    vrt_files = _find_files(BASE_DIR, "*.vrt")
    if not vrt_files:
        print("ERROR: No .vrt files found anywhere under the project directory.")
        raise SystemExit(1)

    print(f"\nFound {len(vrt_files)} VRT file(s):")
    for v in vrt_files:
        print(f"  {v.relative_to(BASE_DIR)}")

    # ── Process each VRT file ─────────────────────────────────────────────
    for vrt_path in vrt_files:
        output_path = OUTPUT_DIR / f"suomi24_filtered_data_{vrt_path.stem}.csv"
        print(f"\n{'─' * 60}")
        parse_vrt_for_project_8(vrt_path, keywords, output_path)