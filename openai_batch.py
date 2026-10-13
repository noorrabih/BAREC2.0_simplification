"""
BAREC Batch Simplification Runner (OpenAI / gpt-5.6-terra)
------------------------------------------------------------
Reads a CSV of (id, sentence, source_band) rows, builds the correct
BAREC system prompt for each band pair (via generate_prompt.py), submits
everything as ONE OpenAI Batch API job (async, 50% cheaper than the
interactive API), polls until it finishes, then writes:

  1. results.csv       -- one row per input sentence with the simplified
                           output, features used, token counts, and cost
  2. results.json       -- same data, structured (not flattened)
  3. cost_report.txt    -- a plain-text summary you can hand to someone else

Requires: pip install openai
Requires an API key: set OPENAI_API_KEY (or pass --api-key)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python openai_batch.py \
        --input sentences.csv \
        --codes-csv codes.csv \
        --output-dir results/ \
        --model gpt-5.6-terra

Input CSV columns (header required): id, sentence, readability band
    id               -- unique identifier for the row; used as the batch
                         request/response key (custom_id) so you can match
                         results back to your own records
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
MODEL NOTES (gpt-5.6 family)
--------------------------------------------------------------------------
gpt-5.6-terra is the mid-tier model in OpenAI's GPT-5.6 family (Sol =
flagship, Terra = balanced, Luna = cheapest/fastest). It's a reasoning
model: instead of a `temperature` knob, you steer it with
`reasoning.effort` (none / low / medium / high / xhigh / max). Reasoning
tokens are billed as output tokens even though the reasoning text itself
isn't returned by default -- see COST TRACKING below.

--------------------------------------------------------------------------
COST TRACKING
--------------------------------------------------------------------------
Pricing is hardcoded below from OpenAI's published API pricing as of
August 2026 (see PRICING dict), already reflecting the July 30, 2026 price
cut and the Batch API's 50% discount. Prices change -- if your bill
doesn't match, check https://openai.com/api/pricing and update the
PRICING dict, or pass --price-input / --price-output to override for a
single run.

Each request's cost is computed from the `usage` block the Responses API
returns per response:
    prompt_tokens  = usage.input_tokens
    output_tokens  = usage.output_tokens   (this ALREADY includes
                      reasoning tokens -- OpenAI folds
                      usage.output_tokens_details.reasoning_tokens into
                      output_tokens, unlike some other providers that
                      report them separately)
Long-context requests (prompt > 272K tokens) bill at a higher per-token
rate; irrelevant for short sentences but included for completeness.
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
# Pricing (USD per 1,000,000 tokens) -- OpenAI Batch API, as of Aug 2026.
# Batch rates are already the 50%-off async rate (this script always uses
# the Batch API, so these are the numbers that matter).
# "long" tier applies only when a single request's prompt exceeds the
# threshold (272K tokens) -- irrelevant for short sentences, included for
# completeness/correctness.
# Source: https://openai.com/api/pricing (verify before relying on this for real invoices)
# ---------------------------------------------------------------------------
PRICING = {
    # Flagship reasoning model, newest generation, GA July 9, 2026.
    "gpt-5.6-sol": {
        "threshold_tokens": 272_000,
        "input_std": 2.50,
        "output_std": 15.00,
        "input_long": 5.00,
        "output_long": 22.50,
    },
    # Mid-tier / balanced model -- what this script uses by default.
    # Standard rate reflects the July 30, 2026 price cut ($2.00/$12.00
    # interactive -> $1.00/$6.00 on Batch).
    "gpt-5.6-terra": {
        "threshold_tokens": 272_000,
        "input_std": 1.00,
        "output_std": 6.00,
        "input_long": 2.00,
        "output_long": 9.00,
    },
    # Cheapest / fastest tier -- reflects the July 30, 2026 80% price cut.
    "gpt-5.6-luna": {
        "threshold_tokens": 272_000,
        "input_std": 0.10,
        "output_std": 0.60,
        "input_long": 0.20,
        "output_long": 0.90,
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
                      f"(ids must be unique, they're used as the batch custom_id)", file=sys.stderr)
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


# gpt-5.6 models are reasoning models steered via reasoning.effort rather
# than temperature. OpenAI's guidance: use "medium" as a balanced default,
# "low"/"none" for latency-sensitive workloads, "high"/"xhigh"/"max" when
# you need deeper reasoning. We do NOT send a temperature param at all for
# these models (unlike the older non-reasoning models where it's fine).
_VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


def is_reasoning_model(model):
    return model.startswith("gpt-5.6") or model.startswith("gpt-5.5") or model.startswith("gpt-5.4") \
        or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")


# Responses API structured-output schema for the BAREC "versions" output.
# Using text.format: {"type": "json_schema", ...} instead of the older
# json_object mode. json_schema mode validates the *shape* of the model's
# output directly and does NOT require the literal word "json" anywhere in
# the request. json_object mode does require it -- and specifically checks
# the `input` array, not `instructions` -- which is easy to trip on when
# your system prompt (which does say "JSON") is passed as `instructions`
# rather than as part of `input`, as it is here.
_BAREC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "target_band": {"type": "integer"},
        "versions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "simplified_sentences": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "was_split": {"type": "boolean"},
                    "features_used": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["simplified_sentences", "was_split", "features_used"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["target_band", "versions"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Step 2: build one JSONL batch-input file, caching the system prompt per
# (source_band, target_band) pair so we don't regenerate it per row.
# ---------------------------------------------------------------------------
def build_batch_jsonl(rows, codes_csv_path, jsonl_path, model, reasoning_effort=None, num_versions=1):
    guideline_rows = load_rows(codes_csv_path)
    prompt_cache = {}

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            pair = (row["source_band"], row["target_band"])
            if pair not in prompt_cache:
                prompt_cache[pair] = build_prompt(guideline_rows, pair[0], pair[1], num_versions=num_versions)
            system_prompt = prompt_cache[pair]

            body = {
                "model": model,
                "input": [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": row["sentence"]}],
                }],
                "instructions": system_prompt,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "barec_simplification",
                        "schema": _BAREC_JSON_SCHEMA,
                        "strict": True,
                    }
                },
            }
            if reasoning_effort and is_reasoning_model(model):
                body["reasoning"] = {"effort": reasoning_effort}

            request_obj = {
                "custom_id": row["row_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    return prompt_cache


# ---------------------------------------------------------------------------
# Step 3-5: submit, poll, download
# ---------------------------------------------------------------------------
def run_batch_job(client, jsonl_path, poll_interval, metadata_description):
    print(f"Uploading {jsonl_path} ...")
    with open(jsonl_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"Uploaded: {uploaded_file.id}")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": metadata_description},
    )
    print(f"Create:d batch job {batch.id}")

    terminal_states = {"completed", "failed", "cancelled", "expired"}

    batch = client.batches.retrieve(batch.id)
    while batch.status not in terminal_states:
        print(f"  status: {batch.status} -- waiting {poll_interval}s ...")
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)

    print(f"Job finished: {batch.status}")
    if batch.status != "completed":
        print(f"Batch did not succeed. Errors: {getattr(batch, 'errors', None)}", file=sys.stderr)
        sys.exit(1)

    # A batch can finish with status "completed" and STILL have no
    # output_file_id -- that happens when every individual request in the
    # batch failed (e.g. bad model name, malformed body, org not
    # allowed to use the model). In that case the per-request errors are
    # in error_file_id instead. Surface them instead of crashing on None.
    if not batch.output_file_id:
        if batch.error_file_id:
            print("No output file -- every request in the batch appears to have failed. "
                  "Fetching the error file instead ...", file=sys.stderr)
            error_content = client.files.content(batch.error_file_id)
            error_text = error_content.text
            print(error_text[:4000], file=sys.stderr)
            error_path = Path(jsonl_path).parent / "batch_errors_raw.jsonl"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(error_text)
            print(f"Wrote raw error file: {error_path}", file=sys.stderr)
        else:
            print(f"No output file and no error file. Full batch object: {batch}", file=sys.stderr)
        print(f"Check request counts -- batch.request_counts: {getattr(batch, 'request_counts', None)}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Downloading results from output file {batch.output_file_id} ...")
    content = client.files.content(batch.output_file_id)
    return content.text


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
    total_reasoning_tokens = 0
    total_cost = 0.0
    n_errors = 0

    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("custom_id")
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
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "error": "",
        }

        # Batch-level per-request failure (e.g. rate limit, bad request) ends
        # up in the top-level "error" field with a null "response".
        top_level_error = obj.get("error")
        response_obj = obj.get("response")
        if top_level_error or response_obj is None:
            out_row["error"] = str(top_level_error) if top_level_error else "No response object returned"
            n_errors += 1
            out_row["_versions"] = []
            out_row["_simplified_sentences_list"] = []
            out_row["_features_list"] = []
            results.append(out_row)
            continue

        status_code = response_obj.get("status_code")
        body = response_obj.get("body", {})

        if status_code != 200:
            # request-level failure: the model call itself errored
            err = body.get("error") if isinstance(body, dict) else body
            out_row["error"] = f"HTTP {status_code}: {err}"
            n_errors += 1
            out_row["_versions"] = []
            out_row["_simplified_sentences_list"] = []
            out_row["_features_list"] = []
            results.append(out_row)
            continue

        usage = body.get("usage", {}) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        # NOTE: unlike some other providers, OpenAI's output_tokens already
        # INCLUDES reasoning tokens -- don't add reasoning_tokens again or
        # you'll double-count.
        output_tokens = usage.get("output_tokens", 0)
        reasoning_tokens = (usage.get("output_tokens_details", {}) or {}).get("reasoning_tokens", 0)

        cost = compute_cost(model, prompt_tokens, output_tokens)

        total_prompt_tokens += prompt_tokens
        total_output_tokens += output_tokens
        total_reasoning_tokens += reasoning_tokens
        if cost is not None:
            total_cost += cost

        # Extract the model's text output (should be the JSON object we asked for).
        # body["output"] is a list of items -- reasoning items, then a
        # "message" item whose content includes an "output_text" part.
        model_text = ""
        try:
            for item in body.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            model_text += part.get("text", "")
        except (KeyError, TypeError, AttributeError):
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
        out_row["reasoning_tokens"] = reasoning_tokens
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
            "reasoning_tokens": r["reasoning_tokens"],
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
                  "simplified level", "prompt_tokens", "output_tokens", "reasoning_tokens",
                  "cost_usd", "error"]

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
        f.write(f"Total output tokens:  {total_output_tokens:,} (includes {total_reasoning_tokens:,} reasoning tokens)\n")
        f.write(f"Rate (input/output):  ${rates.get('input_std', '?')}/M -- ${rates.get('output_std', '?')}/M (Batch rate)\n")
        f.write("-" * 45 + "\n")
        f.write(f"TOTAL COST:           ${total_cost:.4f}\n")
        if n_ok:
            f.write(f"Avg. cost / sentence: ${total_cost / n_ok:.6f}\n")
        f.write("=" * 45 + "\n")
        f.write("Note: pricing is hardcoded from OpenAI's published rates and may be\n")
        f.write("stale. Cross-check against your OpenAI platform billing dashboard for\n")
        f.write("the authoritative number.\n")

    print(f"\nWrote {results_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(f"\nTotal cost: ${total_cost:.4f}  ({n_ok} succeeded, {n_errors} failed)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with columns: id, sentence, readability band")
    ap.add_argument("--codes-csv", required=True, help="Path to the BAREC codes/guidelines csv file")
    ap.add_argument("--annotations-csv", default=None,
                     help="Optional gold-annotation CSV (e.g. annotation_reasoning_codes_RC.csv) "
                          "to auto-fill BAREC-19 and gold reasons -- matched by id first, "
                          "falling back to exact sentence text")
    ap.add_argument("--output-dir", default="batch_output", help="Where to write results.csv and cost_report.txt")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--api-key", default=None, help="Defaults to OPENAI_API_KEY env var")
    ap.add_argument("--poll-interval", type=int, default=30, help="Seconds between job-status checks")
    ap.add_argument("--display-name", default="barec-simplification-batch",
                     help="Stored as batch metadata.description for your own record-keeping")
    ap.add_argument("--price-input", type=float, default=None, help="Override $/M input tokens for this run")
    ap.add_argument("--price-output", type=float, default=None, help="Override $/M output tokens for this run")
    ap.add_argument("--reasoning-effort", default="medium", choices=sorted(_VALID_REASONING_EFFORTS),
                     help="Reasoning effort for gpt-5.6/o-series reasoning models. "
                          "'medium' is OpenAI's recommended balanced default; use 'low' or "
                          "'none' for latency-sensitive runs, 'high'/'xhigh'/'max' for harder cases.")
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
    build_batch_jsonl(rows, args.codes_csv, jsonl_path, args.model,
                       reasoning_effort=args.reasoning_effort, num_versions=args.num_versions)
    print(f"Wrote batch request file: {jsonl_path}")

    if args.dry_run:
        print("Dry run: skipping API call. Inspect the JSONL file above before submitting for real.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key) if args.api_key else OpenAI()

    result_text = run_batch_job(client, jsonl_path, args.poll_interval, args.display_name)

    raw_results_path = output_dir / "batch_results_raw.jsonl"
    with open(raw_results_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Wrote raw batch results: {raw_results_path}")

    parse_and_report(rows_by_id, result_text, args.model, output_dir, args.codes_csv)


if __name__ == "__main__":
    main()