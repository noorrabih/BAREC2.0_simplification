#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAREC Simplification System-Prompt Generator (v2)
---------------------------------------------------
three feature tiers (Must-Have / Allowed / Forbidden) plus an
explicit numeric "Length Constraints" section (word count), derived
automatically from the W-codes in the guidelines CSV. (A syllable-count
constraint was dropped: the O-codes in this CSV only bound syllable count
up through level 7, which already falls inside band 1's own range, so the
derived syllable bound was unlimited for every band -- not useful.)

Feature-tier scoping:
    Must-Have = features whose level falls inside the target band's own
                range (i.e. new to this band, not inherited from an easier
                band). At least one must appear in the output.
    Allowed   = features from every band easier than the target band.
    Forbidden = features from bands harder than the target band, up to and
                including the source band.

Word counts are pulled out of the W rows entirely (they no longer appear as
bullet features, since Length Constraints already encodes them numerically):
    MAX words = the cumulative value unlocked by the target band's top level.
    MIN words = one more than the value unlocked by the previous (easier)
                band's top level (1 for band 1).

Usage:
    # One prompt, to a single file (or stdout without --out):
    python3 generate_prompt_v2.py --csv codes.csv --source-band 2 \
        --shots-csv valid_shots.csv --out system_prompt_v2.txt

    # One file per band pair (prompt_<src>_to_<tgt>.txt), for gemini_batch.py
    # --prompts-dir. Omit --source-band to write every pair:
    python generate_prompt_v2.py --csv codes.csv \
        --shots-csv shots.csv --num-examples 4 --num-versions 3 --out-dir prompts/
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from generate_prompt_v1 import (
    BANDS,
    load_rows,
    features_up_to,
    features_between,
    format_feature_list,
)

AR2EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _arabic_digits(text):
    return [int(d.translate(AR2EN_DIGITS)) for d in re.findall(r"[٠-٩]+", text)]


def parse_w_row(desc):
    """Returns (max_words, unlimited) encoded by a single W-category row."""
    digits = _arabic_digits(desc)
    if digits:
        return digits[0], False
    return None, True  # "لا حد لعدد الكلمات..." -- no limit


def build_count_tables(rows):
    """Returns (word_limits, count_codes).
    word_limits: {level: (value_or_None, unlimited_bool)}
    count_codes: set of feature codes that are pure word-counts and should
    be excluded from the Must-Have/Allowed/Forbidden bullet lists."""
    word_limits = {}
    count_codes = set()

    for r in rows:
        if r["cat"] == "W":
            word_limits[r["level"]] = parse_w_row(r["desc"])
            count_codes.add(r["code"])

    return word_limits, count_codes


def _cumulative_value(limits, level):
    """(value_or_None, unlimited) unlocked by the given level, taken from
    the highest limits-key <= level. (None, False) if nothing applies yet."""
    applicable = [lv for lv in limits if lv <= level]
    if not applicable:
        return None, False
    return limits[max(applicable)]


def all_length_constraints(word_limits):
    """Length constraints for every band, computed in order so that a band's
    MIN never drops below an easier band's MIN -- once a band's MAX turns
    out unlimited (no further finite bound exists in the CSV), harder bands
    simply inherit the same floor instead of resetting to 1."""
    results = {}
    min_words = 1

    for band in sorted(BANDS):
        target_max = BANDS[band][1]

        max_words, words_unlimited = _cumulative_value(word_limits, target_max)
        if words_unlimited:
            max_words = None

        results[band] = (min_words, max_words)

        # Floor for the NEXT (harder) band: bump only if this band ended
        # with a real finite cap; otherwise carry the current floor forward.
        min_words = max_words + 1 if max_words is not None else min_words

    return results


def length_constraints(word_limits, target_band):
    return all_length_constraints(word_limits)[target_band]


def format_length_section(max_words):
    """None when there's no fixed maximum -- the whole Length Constraints
    section is then omitted from the prompt instead of stating a no-op."""
    if max_words is None:
        return None
    return f"* Each simplified sentence must contain **no more than {max_words} words**. This limit is strict."


def load_shots(shots_csv, source_band, n):
    """Loads up to n (source, simplified) example pairs from valid_shots.csv.

    Matching rule: the shot's own original sentence must have been at
    `source_band` (its BAREC-5 column) -- i.e. a peer of the sentence
    currently being simplified, not the sentence's *target* band. (All rows
    in valid_shots.csv step down exactly one band, so this also lands each
    shot's simplification at source_band - 1.)

    The n picks are also deduplicated by Original ID against each other, so
    two shots that are just different (Gem vs GPT) simplifications of the
    same original sentence never both get used as the two examples.

    Returns [] if the file is missing, unreadable, or has no matching rows.
    """
    if not shots_csv:
        return []
    try:
        with open(shots_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            seen_original_ids = set()
            examples = []
            for row in reader:
                try:
                    band = int((row.get("BAREC-5") or "").strip())
                except ValueError:
                    continue
                if band != source_band:
                    continue
                original_id = (row.get("Original ID") or "").strip()
                if not original_id:
                    continue
                if original_id in seen_original_ids:
                    continue
                source = (row.get("source_sentence") or "").strip()
                simplified = (row.get("simplified_sentence") or "").strip()
                if not source or not simplified:
                    continue
                examples.append((source, simplified))
                seen_original_ids.add(original_id)
                if len(examples) >= n:
                    break
            return examples
    except (FileNotFoundError, OSError):
        print(f"Note: shots CSV '{shots_csv}' not found; proceeding with no Examples section.", file=sys.stderr)
        return []


def format_examples(shots, n):
    """None when there are no real shots -- the whole Examples section is
    then omitted from the prompt instead of padding with fake placeholders."""
    if not shots:
        return None

    blocks = []
    for i, (source, simplified) in enumerate(shots[:n], start=1):
        blocks.append(
            f'## Example {i}\n\nInput:\n"{source}"\n\nOutput:\n"{simplified}"'
        )
    return "\n\n".join(blocks)


def build_output_format(num_versions):
    if num_versions <= 1:
        return """Respond in JSON only, with no explanation or additional text:

{
"simplified_sentences": [
"<simplified sentence>"
],
"was_split": false
}"""

    return f"""Produce exactly {num_versions} independent simplifications of the sentence. Each version must be a
genuinely different simplification -- different wording, sentence structure, and/or word choices,
not a trivial rephrasing of another version -- while every version independently follows all the
rules above and fully preserves the meaning of the original sentence.

Respond in JSON only, with no explanation or additional text:

{{
"versions": [
{{
"simplified_sentences": [
"<simplified sentence>"
],
"was_split": false
}}
// ... {num_versions} objects total in "versions"
]
}}"""


def build_prompt(rows, source_band, target_band, num_versions=1, shots_csv=None, num_examples=2):
    target_min, target_max = BANDS[target_band]
    max_level = max(r["level"] for r in rows)

    word_limits, count_codes = build_count_tables(rows)
    filtered_rows = [r for r in rows if r["code"] not in count_codes]

    must_have = features_between(filtered_rows, target_min - 1, target_max)
    allowed = features_up_to(filtered_rows, target_min - 1)
    # Forbidden = every feature harder than the target band, all the way up
    # to the hardest level in the guidelines -- not just up through the
    # source band -- so the model never reaches for something even more
    # complex than the original sentence either.
    forbidden = features_between(filtered_rows, target_max, max_level)

    must_have_text = format_feature_list(must_have) or "No additional features exist in this range."
    allowed_text = format_feature_list(allowed)
    forbidden_text = format_feature_list(forbidden) or "No additional features exist in this range."

    min_words, max_words = length_constraints(word_limits, target_band)
    if min_words > 1:
        min_words_bullet = f"- [MIN-WORDS]: Sentence contains at least {min_words} words."
        must_have_text = f"{must_have_text}\n\n### Length\n{min_words_bullet}"

    allowed_section = (
        f"The simplified sentence may contain any linguistic features from the Allowed Features list below "
        f"as long as it contains at least one feature from the must-have set.\n"
        f"\n## Allowed Features\n"
        f"{allowed_text}\n"
        if allowed_text else ""
    )

    length_text = format_length_section(max_words)
    length_section = f"\n## Length Constraints\n\n{length_text}\n" if length_text else ""

    shots = load_shots(shots_csv, source_band, num_examples)
    examples_text = format_examples(shots, num_examples)
    examples_section = f"\n# Examples\n\n{examples_text}\n" if examples_text else ""

    output_format = build_output_format(num_versions)

    prompt = f"""You are a specialist in simplifying Arabic sentences according to specified linguistic constraints.

# Task

You will be given one Arabic sentence. Rewrite it into a simpler version while fully preserving the meaning and all information in the original sentence.

Do not omit, add, infer, or change any idea.

# Simplification Requirements

The simplified output must satisfy all of the following conditions:

The simplified sentence must **not contain any** linguistic feature from the Forbidden Features list below.

## Forbidden Features

{forbidden_text}

The simplified sentence must contain **at least one** linguistic feature from the Must-Have Features list below.

## Must-Have Features

{must_have_text}

{allowed_section}{length_section}
# Core Rules

* Preserve the complete meaning and information of the original sentence.
* Do not introduce information that is not explicitly present in the original.
* Do not remove any information from the original.
* Maintain grammatically and morphologically correct Modern Standard Arabic.
* Prefer natural and fluent wording.
* Simplify only as much as necessary to satisfy the specified constraints.
* If the full meaning cannot be expressed naturally in one sentence while satisfying the constraints, you may split it into multiple shorter sentences.
* If the sentence is split, the combined sentences must preserve exactly the information contained in the original.


{examples_section}
# Output Format

{output_format}
"""
    return prompt


def prompt_filename(source_band, target_band):
    """File name used for each band pair's prompt under --out-dir; also what
    gemini_batch.py --prompts-dir looks for."""
    return f"prompt_{source_band}_to_{target_band}.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--source-band", type=int, default=None, choices=BANDS.keys(),
                    help="Required with --out / stdout. With --out-dir, omit to write every band pair.")
    ap.add_argument("--shots-csv", default=None, help="valid_shots.csv-shaped file (columns: BAREC-5, Original ID, الجملة الأصلية, التبسيط)")
    ap.add_argument("--num-examples", type=int, default=2)
    ap.add_argument("--num-versions", type=int, default=1)
    ap.add_argument("--out", default=None, help="Write a single prompt to this file")
    ap.add_argument("--out-dir", default=None,
                    help="Write one prompt per band pair to this directory as prompt_<src>_to_<tgt>.txt")
    args = ap.parse_args()

    if args.source_band is not None:
        source_bands = [args.source_band]
    elif args.out_dir:
        source_bands = sorted(b for b in BANDS if b > 1)
    else:
        ap.error("--source-band is required unless --out-dir is given")

    if any(b - 1 < 1 for b in source_bands):
        print("source_band must be > 1 (there is no band easier than 1)", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(args.csv)
    prompts = {
        (b, b - 1): build_prompt(
            rows,
            b,
            b - 1,
            num_versions=args.num_versions,
            shots_csv=args.shots_csv,
            num_examples=args.num_examples,
        )
        for b in source_bands
    }

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for (source_band, target_band), prompt in prompts.items():
            path = out_dir / prompt_filename(source_band, target_band)
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Wrote system prompt to {path}")
        return

    (prompt,) = prompts.values()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Wrote system prompt to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
