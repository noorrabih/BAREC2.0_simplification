#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAREC-19 Readability Predictor
------------------------------
Scores Arabic sentences on the 19-level BAREC scale with
CAMeL-Lab/readability-arabertv2-d3tok-reg. That model expects D3Tok input,
so the raw text is first preprocessed exactly as in
CAMeL-Lab/barec_analyzer scripts/preprocess.py:
    arclean char map -> simple_word_tokenize(split_digits=True)
    -> CAMeLBERT MSA unfactored disambiguator (top-1) -> dediac'd d3tok

Reads a CSV, scores one or more text columns, and writes the same CSV back
out with four extra columns per scored column <col>:
    <col>_pred_d3tok        -- the D3Tok input fed to the model (one line per
                                sentence), for inspection
    <col>_pred_barec19_all  -- the BAREC-19 level of each sentence, newline-joined
    <col>_pred_barec19      -- the hardest (max) level across those sentences
    <col>_pred_barec5       -- <col>_pred_barec19 mapped to its BAREC-5 band

A cell may hold several sentences on separate lines (e.g. a split
simplification in results.csv); each line is scored on its own and the cell
gets the max, since the text is only as readable as its hardest sentence.
Empty cells (e.g. failed requests) are left blank.

Requires the camel_tools env (camel_tools + transformers + torch) and
`camel_data -i disambig-bert-unfactored-msa`.

Usage:
    srun -p cscc-cpu-p -q cscc-cpu-qos --cpus-per-task 16 --mem=40G \
    --exclude=cn-04 python barec19_predict.py \
    --input /home/nour.rabih/BAREC_simplify/gemini_v2_batch2_3x_cached/results.csv \
     --text-col "original sentence" "simplified sentence" --output /home/nour.rabih/BAREC_simplify/gemini_v2_batch2_3x_cached/results_barec19.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from generate_prompt_v1 import BANDS

MODEL_NAME = "CAMeL-Lab/readability-arabertv2-d3tok-reg"
ARCLEAN_MAP = Path(__file__).resolve().parent / "arclean_map.json"
# The LDC-derived analyzer used in the BAREC paper (almor-s31), as installed
# by camel_data. Falls back to the disambiguator's own default if missing.
DEFAULT_DB = Path.home() / ".camel_tools/data/morphology_db/calima-msa-s31/morphology.db"


def barec19_to_barec5(level):
    for band, (lo, hi) in BANDS.items():
        if lo <= level <= hi:
            return band
    return None


# ---------------------------------------------------------------------------
# Preprocessing (mirrors barec_analyzer scripts/preprocess.py, D3Tok variant)
# ---------------------------------------------------------------------------
def load_disambiguator(db_path):
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

    bert = BERTUnfactoredDisambiguator.pretrained(model_name="msa", pretrained_cache=False, top=1)
    if db_path:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer

        db = MorphologyDB(str(db_path), "a")
        bert._analyzer = Analyzer(db, cache_size=100000, backoff="ADD_PROP")
    return bert


def to_d3tok(sentences, bert):
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.utils.transliterate import Transliterator
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.utils.dediac import dediac_ar

    arclean = Transliterator(CharMapper.mapper_from_json(str(ARCLEAN_MAP)))

    def clean(text):
        out = arclean.transliterate(text)
        return re.sub(r"(?<=\B)ى(?=\B)", "ي", out)

    tokenized = [simple_word_tokenize(clean(s), split_digits=True) for s in sentences]
    disambiguated = bert.disambiguate_sentences(tokenized)
    return [
        " ".join(
            dediac_ar(d.analyses[0][1]["d3tok"]).replace("_+", " +").replace("+_", "+ ")
            for d in sent
        )
        for sent in disambiguated
    ]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_levels(d3tok_sentences, device, batch_size):
    from transformers import pipeline

    readability = pipeline("text-classification", model=MODEL_NAME, device=device)
    results = readability(d3tok_sentences, function_to_apply="none",
                          batch_size=batch_size, truncation=True)
    return [max(round(r["score"] + 0.5), 1) for r in results]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV containing the text column to score")
    ap.add_argument("--text-col", nargs="+", default=["simplified sentence"],
                    help="Column(s) to score, e.g. --text-col source_sentence simplified_sentence")
    ap.add_argument("--output", required=True, help="Where to write the CSV with prediction columns added")
    ap.add_argument("--db", default=str(DEFAULT_DB) if DEFAULT_DB.is_file() else None,
                    help=f"Morphology DB for the analyzer (default: {DEFAULT_DB} if present, "
                         "else camel_tools' built-in default)")
    ap.add_argument("--device", type=int, default=-1, help="GPU index for the readability model; -1 = CPU")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in args.text_col if c not in fieldnames]
    if missing:
        print(f"Column(s) {missing} not in {args.input}. Columns: {fieldnames}", file=sys.stderr)
        sys.exit(1)

    # Flatten every non-empty line of every cell of every scored column into
    # one list, remembering which (column, row) each line came from, so the
    # disambiguator and model each load and run once over everything.
    flat, owner = [], []
    for col in args.text_col:
        for i, row in enumerate(rows):
            for line in (row[col] or "").split("\n"):
                if line.strip():
                    flat.append(line.strip())
                    owner.append((col, i))
    print(f"Scoring {len(flat)} sentence(s) from {len(rows)} row(s) in column(s) {args.text_col}.")
    print(f"Morphology DB: {args.db or 'camel_tools default'}")

    bert = load_disambiguator(args.db)
    d3tok = to_d3tok(flat, bert)
    levels = predict_levels(d3tok, args.device, args.batch_size)

    per_cell = {(col, i): ([], []) for col in args.text_col for i in range(len(rows))}
    for key, d, lvl in zip(owner, d3tok, levels):
        per_cell[key][0].append(d)
        per_cell[key][1].append(lvl)

    new_cols = []
    for col in args.text_col:
        new_cols += [f"{col}_pred_d3tok", f"{col}_pred_barec19_all",
                     f"{col}_pred_barec19", f"{col}_pred_barec5"]
        for i, row in enumerate(rows):
            ds, lvls = per_cell[(col, i)]
            row[f"{col}_pred_d3tok"] = "\n".join(ds)
            row[f"{col}_pred_barec19_all"] = "\n".join(map(str, lvls))
            row[f"{col}_pred_barec19"] = max(lvls) if lvls else ""
            row[f"{col}_pred_barec5"] = barec19_to_barec5(max(lvls)) if lvls else ""

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + [c for c in new_cols if c not in fieldnames])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
