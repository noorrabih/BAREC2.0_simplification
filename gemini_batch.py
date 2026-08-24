"""
BAREC Batch Simplification Runner
----------------------------------
Reads a CSV of (id, sentence, source_band) rows, builds the
correct
BAREC system prompt for each band pair (via generate_prompt.py), submits
everything as ONE Gemini Batch API job (async, ~50% cheaper than the
interactive API), polls until it finishes, then writes:

  1. results.csv       -- one row per input sentence with the simplified
                           output, features used, token counts, and cost
  2. cost_report.txt    -- a plain-text summary you can hand to someone else

Requires: pip install google-genai
Requires an API key: set GEMINI_API_KEY (or pass --api-key)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python gemini_batch.py \
        --input sentences.csv \
        --codes-csv codes.csv \
        --output-dir results/ \
        --model gemini-3.1-pro-preview

Input CSV columns (header required): id, sentence, readability band
    id               -- unique identifier for the row; used as the batch
                         request/response key so you can match results back
                         to your own records
    sentence         -- the Arabic sentence to simplify
    readability band -- integer 1-5 (the band the sentence is currently at,
                         i.e. "BAREC-5" in the output). The target band is
                         always this minus 1 -- simplification is always one
                         band down.

Optional input columns, passed straight through to the output if present:
    barec_19      -- the sentence's original BAREC-19 (1-19) level, if you
                      already have it annotated
    gold_reasons  -- your own ground-truth feature/reason annotation for
                      the original sentence, if you have one

Output (results.csv / results.json) columns:
    ID, original sentence, BAREC-19, BAREC-5, gold reasons,
    simplified sentence, predicted reasons, simplified level

--------------------------------------------------------------------------
COST TRACKING
--------------------------------------------------------------------------
Pricing is hardcoded below from Google's published Gemini API Batch pricing
as of August 2026 (see PRICING dict). Prices change — if your bill doesn't
match, check https://ai.google.dev/gemini-api/docs/pricing and update the
PRICING dict, or pass --price-input / --price-output to override for a
single run.

Each request's cost is computed from the `usageMetadata` block Gemini
returns per response:
    prompt_tokens  = usageMetadata.promptTokenCount
    output_tokens  = usageMetadata.candidatesTokenCount
                      + usageMetadata.thoughtsTokenCount  (thinking tokens
                        are billed at the output rate, see Gemini docs)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Reuse the prompt-building logic from generate_prompt.py (same folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_prompt import load_rows, build_prompt

# ---------------------------------------------------------------------------
# Pricing (USD per 1,000,000 tokens) -- Gemini Batch API, as of Aug 2026.
# "long" tier applies only when a single request's prompt exceeds the
# threshold (200K tokens) -- irrelevant for short sentences, included for
# completeness/correctness.
# Source: https://ai.google.dev/gemini-api/docs/pricing (verify before relying on this for real invoices)
# ---------------------------------------------------------------------------
PRICING = {
    # Flagship reasoning model, newest generation, still in preview as of Aug 2026.
    "gemini-3.1-pro-preview": {
        "threshold_tokens": 200_000,
        "input_std": 1.00,
        "output_std": 6.00,
        "input_long": 2.00,
        "output_long": 9.00,
    },
    # Most recently GA-released model overall (Aug 13, 2026) -- Flash tier,
    # optimized for coding/agentic speed rather than deep reasoning.
    "gemini-3.7-flash": {
        "threshold_tokens": None,
        "input_std": 0.75,
        "output_std": 3.75,
        "input_long": None,
        "output_long": None,
    },
    "gemini-2.5-pro": {
        "threshold_tokens": 200_000,
        "input_std": 0.625,
        "output_std": 5.00,
        "input_long": 1.25,
        "output_long": 7.50,
    },
    "gemini-2.5-flash": {
        "threshold_tokens": None,
        "input_std": 0.15,
        "output_std": 1.25,
        "input_long": None,
        "output_long": None,
    },
}


def compute_cost(model, prompt_tokens, output_tokens):
    rates = PRICING.get(model)
    if rates is None:
        return None  # unknown model, can't price it
    threshold = rates["threshold_tokens"]
    if threshold and prompt_tokens > threshold and rates["input_long"] is not None:
        in_rate, out_rate = rates["input_long"], rates["output_long"]
    else:
        in_rate, out_rate = rates["input_std"], rates["output_std"]
    return (prompt_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


# ---------------------------------------------------------------------------
# Step 1: read the input sentences CSV
# ---------------------------------------------------------------------------
def _normalize_header(h):
    return h.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


# ---------------------------------------------------------------------------
# Gold annotations join (e.g. annotation_reasoning_codes_RC.csv)
# ---------------------------------------------------------------------------
# Category code columns in the annotation file -> the Arabic description
# column that goes with each one (used to format "CODE: description").
_ANNOTATION_CODE_COLS = {
    "Orthography": "تهجئة/إملاء",
    "word count": "عدد الكلمات",
    "Morphology": "تصريف واشتقاق",
    "syntactic": "تراكيب نحوية",
    "vocab": "مفردات",
    "content": "فكرة / محتوى",
}


def load_annotations(path):
    """
    Loads a gold-annotation CSV shaped like annotation_reasoning_codes_RC.csv:
    one row per sentence, an `ID` column, a `Readability_Level_19` column,
    and up to 6 category columns (Orthography, word count, Morphology,
    syntactic, vocab, content) where non-empty cells hold the gold reason
    code(s) for that sentence.

    Returns two dicts for lookup, keyed by ID and by exact sentence text:
        by_id[id_str]       -> {"barec_19": "...", "gold_reasons": "..."}
        by_sentence[text]   -> same
    """
    by_id = {}
    by_sentence = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = {"ID", "Sentence", "Readability_Level_19"} - fieldnames
        if missing:
            print(f"Annotations CSV is missing expected columns: {missing}", file=sys.stderr)
            sys.exit(1)
        available_code_cols = [c for c in _ANNOTATION_CODE_COLS if c in fieldnames]

        for row in reader:
            ann_id = row.get("ID", "").strip()
            sentence = row.get("Sentence", "").strip()
            barec_19 = row.get("Readability_Level_19", "").strip()

            reasons = []
            for code_col in available_code_cols:
                code = row.get(code_col, "").strip()
                if not code:
                    continue
                desc_col = _ANNOTATION_CODE_COLS[code_col]
                desc = row.get(desc_col, "").strip()
                reasons.append(f"{code}: {desc}" if desc else code)
            gold_reasons = " | ".join(reasons)

            entry = {"barec_19": barec_19, "gold_reasons": gold_reasons}
            if ann_id:
                by_id[ann_id] = entry
            if sentence:
                by_sentence[sentence] = entry
    return by_id, by_sentence


def load_input_rows(path):
    rows = []
    seen_ids = set()
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        norm_to_actual = {_normalize_header(h): h for h in fieldnames}

        required_norm = {"id", "sentence", "readabilityband"}
        missing = required_norm - set(norm_to_actual.keys())
        if missing:
            print(f"Input CSV is missing required columns: {missing}", file=sys.stderr)
            sys.exit(1)

        # Optional passthrough columns -- accept a few common header spellings
        barec19_col = norm_to_actual.get("barec19")
        gold_reasons_col = norm_to_actual.get("goldreasons")

        id_col = norm_to_actual["id"]
        sentence_col = norm_to_actual["sentence"]
        source_col = norm_to_actual["readabilityband"]

        for i, row in enumerate(reader):
            sentence = row[sentence_col].strip()
            if not sentence:
                continue
            row_id = row[id_col].strip()
            if not row_id:
                print(f"Row {i}: empty id -- skipping", file=sys.stderr)
                continue
            if row_id in seen_ids:
                print(f"Row {i}: duplicate id {row_id!r} -- skipping "
                      f"(ids must be unique, they're used as the batch response key)", file=sys.stderr)
                continue
            try:
                source_band = int(row[source_col])
                target_band = source_band - 1
            except ValueError:
                print(f"Row {i} (id={row_id}): source_band must be an integer, got "
                      f"{row[source_col]!r} -- skipping", file=sys.stderr)
                continue
            if not (1 <= target_band < source_band <= 5):
                print(f"Row {i} (id={row_id}): target_band ({target_band}) must be less than "
                      f"source_band ({source_band}), both within 1-5 -- skipping", file=sys.stderr)
                continue
            seen_ids.add(row_id)
            rows.append({
                "row_id": row_id,
                "sentence": sentence,
                "source_band": source_band,
                "target_band": target_band,
                "barec_19": row.get(barec19_col, "").strip() if barec19_col else "",
                "gold_reasons": row.get(gold_reasons_col, "").strip() if gold_reasons_col else "",
            })
    return rows


def apply_annotations(rows, by_id, by_sentence):
    """Fill barec_19/gold_reasons from the annotations file wherever the
    input CSV didn't already supply them -- match on id first, fall back to
    exact sentence text."""
    n_matched_id = 0
    n_matched_sentence = 0
    n_unmatched = 0
    for row in rows:
        if row["barec_19"] and row["gold_reasons"]:
            continue  # input CSV already had both, don't overwrite
        entry = by_id.get(row["row_id"])
        if entry:
            n_matched_id += 1
        else:
            entry = by_sentence.get(row["sentence"])
            if entry:
                n_matched_sentence += 1
            else:
                n_unmatched += 1
                continue
        if not row["barec_19"]:
            row["barec_19"] = entry["barec_19"]
        if not row["gold_reasons"]:
            row["gold_reasons"] = entry["gold_reasons"]
    print(f"Annotations join: {n_matched_id} matched by id, "
          f"{n_matched_sentence} matched by sentence text, "
          f"{n_unmatched} unmatched (left blank).")


# Google explicitly recommends NOT overriding temperature for the Gemini 3.x
# family (includes gemini-3.1-pro-preview, gemini-3.7-flash, etc.) -- low
# temperature can cause looping/quality regressions on these models. Rely on
# the system instruction for determinism instead, as Google recommends.
# For older models (gemini-2.5-*) a lower temperature is fine and sensible
# for this rule-constrained task.
def supports_temperature_override(model):
    return not model.startswith("gemini-3")


# ---------------------------------------------------------------------------
# Step 2: build one JSONL batch-input file, caching the system prompt per
# (source_band, target_band) pair so we don't regenerate it per row.
# ---------------------------------------------------------------------------
def build_batch_jsonl(rows, codes_csv_path, jsonl_path, model, temperature=None, num_versions=1):
    guideline_rows = load_rows(codes_csv_path)
    prompt_cache = {}

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            pair = (row["source_band"], row["target_band"])
            if pair not in prompt_cache:
                prompt_cache[pair] = build_prompt(guideline_rows, pair[0], pair[1], num_versions=num_versions)
            system_prompt = prompt_cache[pair]

            generation_config = {"response_mime_type": "application/json"}
            if temperature is not None and supports_temperature_override(model):
                generation_config["temperature"] = temperature

            request_obj = {
                "key": row["row_id"],
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": row["sentence"]}],
                    }],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "generation_config": generation_config,
                },
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    return prompt_cache


# ---------------------------------------------------------------------------
# Step 3-5: submit, poll, download
# ---------------------------------------------------------------------------
def run_batch_job(client, model, jsonl_path, display_name, poll_interval):
    from google.genai import types

    print(f"Uploading {jsonl_path} ...")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    print(f"Uploaded: {uploaded_file.name}")

    job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    print(f"Created batch job: {job.name}")

    completed_states = {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    }

    batch_job = client.batches.get(name=job.name)
    while batch_job.state.name not in completed_states:
        print(f"  status: {batch_job.state.name} -- waiting {poll_interval}s ...")
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=job.name)

    print(f"Job finished: {batch_job.state.name}")
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Batch did not succeed. Error: {getattr(batch_job, 'error', None)}", file=sys.stderr)
        sys.exit(1)

    result_file_name = batch_job.dest.file_name
    print(f"Downloading results from {result_file_name} ...")
    content_bytes = client.files.download(file=result_file_name)
    return content_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# Step 6: parse results, merge with input rows, compute cost, write outputs
# ---------------------------------------------------------------------------
def parse_and_report(rows_by_id, jsonl_text, model, output_dir, codes_csv_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guideline_rows = load_rows(codes_csv_path)
    code_to_desc = {r["code"]: r["desc"] for r in guideline_rows}

    results = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    n_errors = 0

    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("key")
        row = rows_by_id.get(key, {})

        out_row = {
            "row_id": key,
            "sentence": row.get("sentence", ""),
            "source_band": row.get("source_band", ""),
            "target_band": row.get("target_band", ""),
            "barec_19": row.get("barec_19", ""),
            "gold_reasons": row.get("gold_reasons", ""),
            "simplified_sentences": "",
            "was_split": "",
            "features_used": "",
            "prompt_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "error": "",
        }

        if "error" in obj and obj["error"]:
            out_row["error"] = str(obj["error"])
            n_errors += 1
            results.append(out_row)
            continue

        response = obj.get("response", {})
        usage = response.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        # thinking tokens are billed at the output rate -- include them
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)

        cost = compute_cost(model, prompt_tokens, output_tokens)

        total_prompt_tokens += prompt_tokens
        total_output_tokens += output_tokens
        if cost is not None:
            total_cost += cost

        # Extract the model's text output (should be the JSON object we asked for)
        model_text = ""
        try:
            parts = response["candidates"][0]["content"]["parts"]
            model_text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            pass

        def _fmt_features(features_list):
            return "\n".join(f"{feat}: {code_to_desc.get(feat, 'Unknown')}" for feat in features_list)

        try:
            parsed = json.loads(model_text)
            raw_versions = parsed.get("versions")
            if raw_versions is None:
                # single-version schema (num_versions=1 runs)
                raw_versions = [{
                    "simplified_sentences": parsed.get("simplified_sentences", []),
                    "was_split": parsed.get("was_split", ""),
                    "features_used": parsed.get("features_used", []),
                }]

            version_records = []
            for v in raw_versions:
                simplified_list = v.get("simplified_sentences", [])
                features_list = v.get("features_used", [])
                version_records.append({
                    "simplified_sentences": "\n".join(simplified_list),
                    "was_split": v.get("was_split", ""),
                    "features_used": _fmt_features(features_list),
                    "_simplified_sentences_list": simplified_list,
                    "_features_list": features_list,
                })

            # Keep top-level fields pointing at the first version for
            # backwards compatibility with single-version consumers.
            out_row["simplified_sentences"] = version_records[0]["simplified_sentences"]
            out_row["was_split"] = version_records[0]["was_split"]
            out_row["features_used"] = version_records[0]["features_used"]
            out_row["_simplified_sentences_list"] = version_records[0]["_simplified_sentences_list"]
            out_row["_features_list"] = version_records[0]["_features_list"]
            out_row["_versions"] = version_records

        except (json.JSONDecodeError, TypeError):
            out_row["error"] = f"Could not parse model output as JSON: {model_text[:200]}"
            n_errors += 1
            out_row["_versions"] = []
            out_row["_simplified_sentences_list"] = []
            out_row["_features_list"] = []

        out_row["prompt_tokens"] = prompt_tokens
        out_row["output_tokens"] = output_tokens
        out_row["cost_usd"] = round(cost, 6) if cost is not None else ""

        results.append(out_row)

    # A row may carry more than one candidate simplification ("versions").
    # Emit one output row per version, stacked under each other, tagged with
    # a "version" column (1-indexed). Rows that errored out (no versions)
    # still get a single row with version=1 and the error message.
    def _base_fields(r):
        return {
            "ID": r["row_id"],
            "original sentence": r["sentence"],
            "BAREC-19": r["barec_19"],
            "BAREC-5": r["source_band"],
            "gold reasons": r["gold_reasons"],
            "simplified level": r["target_band"],
            "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": r["cost_usd"],
            "error": r["error"],
        }

    def _expand(r, flatten):
        versions = r.get("_versions", [])
        base = _base_fields(r)
        if not versions:
            rec = dict(base)
            rec["version"] = 1
            rec["simplified sentence"] = "" if flatten else []
            rec["predicted reasons"] = "" if flatten else []
            rec["was_split"] = ""
            yield rec
            return
        for i, v in enumerate(versions):
            rec = dict(base)
            rec["version"] = i + 1
            if flatten:
                rec["simplified sentence"] = v["simplified_sentences"]
                rec["predicted reasons"] = v["features_used"]
            else:
                rec["simplified sentence"] = v["_simplified_sentences_list"]
                rec["predicted reasons"] = v["_features_list"]
            rec["was_split"] = v["was_split"]
            yield rec

    fieldnames = ["ID", "version", "original sentence", "BAREC-19", "BAREC-5", "gold reasons",
                  "simplified sentence", "predicted reasons", "was_split",
                  "simplified level", "prompt_tokens", "output_tokens", "cost_usd", "error"]

    # --- write results.csv ---
    results_path = output_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for rec in _expand(r, flatten=True):
                writer.writerow(rec)

    # --- write results.json (same schema, structured/not flattened) ---
    json_path = output_dir / "results.json"
    json_results = [rec for r in results for rec in _expand(r, flatten=False)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    # --- write cost_report.txt ---
    n_ok = len(results) - n_errors
    report_path = output_dir / "cost_report.txt"
    rates = PRICING.get(model, {})
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BAREC Batch Simplification -- Cost Report\n")
        f.write("=" * 45 + "\n")
        f.write(f"Model:                {model}\n")
        f.write(f"Mode:                 Batch API (async, 50% off interactive rate)\n")
        f.write(f"Rows submitted:       {len(results)}\n")
        f.write(f"Rows succeeded:       {n_ok}\n")
        f.write(f"Rows failed:          {n_errors}\n")
        f.write("-" * 45 + "\n")
        f.write(f"Total prompt tokens:  {total_prompt_tokens:,}\n")
        f.write(f"Total output tokens:  {total_output_tokens:,} (includes thinking tokens)\n")
        f.write(f"Rate (input/output):  ${rates.get('input_std', '?')}/M -- ${rates.get('output_std', '?')}/M\n")
        f.write("-" * 45 + "\n")
        f.write(f"TOTAL COST:           ${total_cost:.4f}\n")
        if n_ok:
            f.write(f"Avg. cost / sentence: ${total_cost / n_ok:.6f}\n")
        f.write("=" * 45 + "\n")
        f.write("Note: pricing is hardcoded from Google's published rates and may be\n")
        f.write("stale. Cross-check against your Google Cloud / AI Studio billing\n")
        f.write("dashboard for the authoritative number.\n")

    print(f"\nWrote {results_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(f"\nTotal cost: ${total_cost:.4f}  ({n_ok} succeeded, {n_errors} failed)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with columns: id, sentence, source_band, target_band")
    ap.add_argument("--codes-csv", required=True, help="Path to the BAREC codes/guidelines csv file")
    ap.add_argument("--annotations-csv", default=None,
                     help="Optional gold-annotation CSV (e.g. annotation_reasoning_codes_RC.csv) "
                          "to auto-fill BAREC-19 and gold reasons -- matched by id first, "
                          "falling back to exact sentence text")
    ap.add_argument("--output-dir", default="batch_output", help="Where to write results.csv and cost_report.txt")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--api-key", default=None, help="Defaults to GEMINI_API_KEY / GOOGLE_API_KEY env var")
    ap.add_argument("--poll-interval", type=int, default=30, help="Seconds between job-status checks")
    ap.add_argument("--display-name", default="barec-simplification-batch")
    ap.add_argument("--price-input", type=float, default=None, help="Override $/M input tokens for this run")
    ap.add_argument("--price-output", type=float, default=None, help="Override $/M output tokens for this run")
    ap.add_argument("--temperature", type=float, default=0.3,
                     help="Sampling temperature. Ignored for Gemini 3.x models "
                          "(gemini-3.*) per Google's guidance -- those models "
                          "should be left at their default and steered via the "
                          "system instruction instead.")
    ap.add_argument("--dry-run", action="store_true", help="Build the JSONL and print stats, but don't call the API")
    ap.add_argument("--num-versions", type=int, default=1,
                     help="Number of independent simplified versions (each with its own "
                          "reasons/features) to request per sentence. Default 1.")
    args = ap.parse_args()

    if args.price_input is not None or args.price_output is not None:
        PRICING.setdefault(args.model, {"threshold_tokens": None, "input_long": None, "output_long": None})
        if args.price_input is not None:
            PRICING[args.model]["input_std"] = args.price_input
        if args.price_output is not None:
            PRICING[args.model]["output_std"] = args.price_output

    rows = load_input_rows(args.input)
    if not rows:
        print("No valid rows found in input CSV.", file=sys.stderr)
        sys.exit(1)

    if args.annotations_csv:
        by_id, by_sentence = load_annotations(args.annotations_csv)
        apply_annotations(rows, by_id, by_sentence)
    rows_by_id = {r["row_id"]: r for r in rows}
    print(f"Loaded {len(rows)} sentences across "
          f"{len({(r['source_band'], r['target_band']) for r in rows})} band pair(s).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "batch_requests.jsonl"
    build_batch_jsonl(rows, args.codes_csv, jsonl_path, args.model, num_versions=args.num_versions)
    print(f"Wrote batch request file: {jsonl_path}")

    if args.dry_run:
        print("Dry run: skipping API call. Inspect the JSONL file above before submitting for real.")
        return

    from google import genai
    client = genai.Client(api_key=args.api_key) if args.api_key else genai.Client()

    result_text = run_batch_job(client, args.model, jsonl_path, args.display_name, args.poll_interval)

    raw_results_path = output_dir / "batch_results_raw.jsonl"
    with open(raw_results_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Wrote raw batch results: {raw_results_path}")

    parse_and_report(rows_by_id, result_text, args.model, output_dir, args.codes_csv)


if __name__ == "__main__":
    main()