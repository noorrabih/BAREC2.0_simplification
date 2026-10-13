#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAREC Simplification System-Prompt Generator
----------------------------------------------
Reads the BAREC codes/guidelines CSV and produces a ready-to-paste Gemini
system prompt for simplifying an Arabic sentence from a source band down to
a target (easier) band.

Usage:
    python3 generate_prompt.py --csv codes.csv \
        --source-band 3 --out system_prompt.txt

"""
import csv
import argparse
import sys

BANDS = {
    1: (1, 7),
    2: (8, 11),
    3: (12, 13),
    4: (14, 15),
    5: (16, 19),
}

CATEGORY_NAMES = {
    "O": "Orthographic",
    "W": "Sentence Length (Word Count)",
    "M": "Morphology (Verb Conjugation, Pronouns, Prefixes & Suffixes)",
    "S": "Syntax (Sentence Structure)",
    "V": "Vocabulary (Lexicon)",
    "C": "Content & Symbolism (Conceptual Abstraction)",
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            desc, code, level = row[0].strip(), row[1].strip(), row[2].strip()
            if not code or not level:
                continue
            try:
                lvl = int(level)
            except ValueError:
                continue
            rows.append({"desc": desc, "code": code, "level": lvl, "cat": code[0]})
    return rows


def features_up_to(rows, max_level):
    """All features whose level is <= max_level (cumulative), grouped by category."""
    grouped = {}
    for r in rows:
        if r["level"] <= max_level:
            grouped.setdefault(r["cat"], []).append(r)
    for cat in grouped:
        grouped[cat].sort(key=lambda r: r["level"])
    return grouped


def features_between(rows, low_level, high_level):
    """Features strictly above low_level and up to high_level (the 'must remove' band)."""
    grouped = {}
    for r in rows:
        if low_level < r["level"] <= high_level:
            grouped.setdefault(r["cat"], []).append(r)
    for cat in grouped:
        grouped[cat].sort(key=lambda r: r["level"])
    return grouped


def format_feature_list(grouped):
    lines = []
    for cat in ["O", "W", "M", "S", "V", "C"]:
        if cat not in grouped:
            continue
        lines.append(f"\n### {CATEGORY_NAMES[cat]}")
        for r in grouped[cat]:
            lines.append(f"- [{r['code']}]: {r['desc']}")
    return "\n".join(lines).strip()


def format_flat_code_list(grouped):
    codes = []
    for cat in ["O", "W", "M", "S", "V", "C"]:
        for r in grouped.get(cat, []):
            codes.append(r["code"])
    return ", ".join(codes)


def build_prompt(rows, source_band, target_band, num_versions=1):
    target_max_level = BANDS[target_band][1]
    source_max_level = BANDS[source_band][1]

    allowed = features_up_to(rows, target_max_level)                       # OK to keep / introduce
    forbidden = features_between(rows, target_max_level, source_max_level)  # must remove

    allowed_text = format_feature_list(allowed)
    forbidden_text = format_feature_list(forbidden)

    if target_band + 1 == source_band:
        forbidden_range_desc = f"band {source_band}"
    else:
        forbidden_range_desc = f"bands {target_band + 1} through {source_band}"

    if num_versions <= 1:
        output_format = f"""# Output format
Respond **in JSON only**, with no explanation or extra text outside the JSON object, in exactly this shape:

{{
  "target_band": {target_band},
  "simplified_sentences": [
    "<first simplified sentence>",
    "<second simplified sentence, if the sentence was split>"
  ],
  "was_split": <true or false>,
  "features_used": ["feature_code_1", "feature_code_2", ...]
}}

- "features_used" must include **every** linguistic feature actually present in the resulting simplified sentence(s), drawn exclusively from the list of allowed codes from the Allowed Features.
- Do not include any code in "features_used" that isn't actually present in the output sentence."""
    else:
        output_format = f"""# Output format
Produce exactly {num_versions} independent simplifications of the sentence. Each version must be a
genuinely different simplification -- different wording, sentence structure, and/or word choices,
not a trivial rephrasing of another version -- while every version independently follows all the
rules above and fully preserves the meaning of the original sentence.

Respond **in JSON only**, with no explanation or extra text outside the JSON object, in exactly this shape:

{{
  "target_band": {target_band},
  "versions": [
    {{
      "simplified_sentences": [
        "<first simplified sentence>",
        "<second simplified sentence, if this version was split>"
      ],
      "was_split": <true or false>,
      "features_used": ["feature_code_1", "feature_code_2", ...]
    }}
    // ... {num_versions} objects total in "versions"
  ]
}}

- Each version's "features_used" must include **every** linguistic feature actually present in that
  version's simplified sentence(s), drawn exclusively from the list of allowed codes from the Allowed
  Features, and must not include a code that isn't actually present in that version's output."""

    prompt = f"""You are a specialized tool for simplifying Arabic sentences according to the BAREC readability standard, organized into 5 bands of increasing complexity.

# Task
You will be given a single Arabic sentence written at a higher complexity band (source band: {source_band}). Your job is to rewrite it so it is appropriate for the target band: {target_band} (an easier band), while fully preserving the meaning and information in the original sentence, do not omit or add any idea.

# Core rules
- You must NOT use any linguistic feature from the "Forbidden Features" list below (features belonging to the original sentence's band and anything above the target band).
- Every linguistic feature present in the simplified sentence must belong to the "Allowed Features" list below (features of the target band and everything easier than it).
- If the full meaning cannot be conveyed in a single sentence without using a forbidden feature, you are permitted to split the sentence into shorter, simpler sentences, as long as no resulting sentence exceeds the target band's constraints.
- Maintain grammatically and morphologically correct Modern Standard Arabic throughout.
- Don't over-simplify: use the richest allowed features that keep the sentence natural and fluent, but never exceed the target band's limits.

# Forbidden Features (belong to {forbidden_range_desc}, must NOT appear in the output)
{forbidden_text if forbidden_text else "No additional features exist in this range."}

# Allowed Features (band {target_band}, may appear in the output)
{allowed_text}

{output_format}
"""
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--source-band", type=int, required=True, choices=BANDS.keys())
    # ap.add_argument("--target-band", type=int, required=True, choices=BANDS.keys())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.csv)

    # if args.target_band >= args.source_band:
    #     print("target_band must be lower (easier) than source_band", file=sys.stderr)
    #     sys.exit(1)

    prompt = build_prompt(rows, args.source_band, args.source_band - 1)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Wrote system prompt to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
