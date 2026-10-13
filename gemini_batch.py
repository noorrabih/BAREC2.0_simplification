"""
BAREC Batch Simplification Runner
----------------------------------
Reads a CSV of (id, sentence, source_band) rows, picks the prebuilt
BAREC system prompt for each band pair (written beforehand by
generate_prompt_v2.py --out-dir), submits
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
    # 1. Build the prompts once (one file per band pair):
    python generate_prompt_v2.py --csv codes.csv \
        --shots-csv valid_shots.csv --out-dir prompts/

    # 2. Build and submit the batch using those prompts:
    python gemini_batch.py \
        --input sentences_batch2.csv \
        --prompts-dir prompts/ \
        --codes-csv codes.csv \
        --output-dir gemini_v2_batch2_3x/ \
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
    cached_tokens  = usageMetadata.cachedContentTokenCount  (a subset of
                      promptTokenCount, billed at the cached rate instead)

--------------------------------------------------------------------------
CONTEXT CACHING
--------------------------------------------------------------------------
There are only a handful of distinct system prompts (one per band pair), so
by default each one is uploaded once as an explicit Gemini cache
(client.caches.create) and every batch request references it via
`cached_content` instead of repeating the system prompt inline. Caches are
kept alive by extending their TTL while the job is polled, and deleted when
the run ends. Pass --no-cache to send the system prompt inline instead.

Explicit caches have a per-model minimum size (4,096 tokens for
gemini-3.x, 2,048 for gemini-2.5-*). A prompt below the minimum can't be
cached -- that band pair falls back to an inline system prompt, with a
warning.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Prompt file naming is shared with generate_prompt_v2.py (same folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_prompt_v2 import load_rows, prompt_filename

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
        "input_cached": 0.20, # $0.20, prompts <= 200k tokens
        "cache_storage_per_hour": 4.50, # $4.50 / 1,000,000 tokens per hour (storage price)
    },
    
    # Most recently GA-released model overall (Aug 13, 2026) -- Flash tier,
    # optimized for coding/agentic speed rather than deep reasoning.
    "gemini-3.7-flash": {
        "threshold_tokens": None,
        "input_std": 0.75,
        "output_std": 3.75,
        "input_long": None,
        "output_long": None,
        "input_cached": None,  # unknown -- pass --price-cached
        "cache_storage_per_hour": None,
    },
    "gemini-2.5-pro": {
        "threshold_tokens": 200_000,
        "input_std": 0.625,
        "output_std": 5.00,
        "input_long": 1.25,
        "output_long": 7.50,
        "input_cached": 0.125,
        "cache_storage_per_hour": 4.50,
    },
    "gemini-2.5-flash": {
        "threshold_tokens": None,
        "input_std": 0.15,
        "output_std": 1.25,
        "input_long": None,
        "output_long": None,
        "input_cached": 0.03,
        "cache_storage_per_hour": 1.00,
    },
}


def compute_cost(model, prompt_tokens, output_tokens, cached_tokens=0):
    """cached_tokens is the cached part of prompt_tokens (not extra to it).
    Cached tokens are billed at the standard context-caching rate (no batch
    discount); if that rate is unknown they're priced as normal input."""
    rates = PRICING.get(model)
    if rates is None:
        return None  # unknown model, can't price it
    threshold = rates["threshold_tokens"]
    if threshold and prompt_tokens > threshold and rates["input_long"] is not None:
        in_rate, out_rate = rates["input_long"], rates["output_long"]
    else:
        in_rate, out_rate = rates["input_std"], rates["output_std"]
    cached_rate = rates.get("input_cached")
    if cached_rate is None:
        cached_rate = in_rate
    uncached_tokens = prompt_tokens - cached_tokens
    return ((uncached_tokens / 1_000_000) * in_rate
            + (cached_tokens / 1_000_000) * cached_rate
            + (output_tokens / 1_000_000) * out_rate)


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
# Step 2: build one JSONL batch-input file, using the prebuilt system prompt
# for each row's (source_band, target_band) pair from --prompts-dir.
# ---------------------------------------------------------------------------
def load_prompts(prompts_dir, pairs):
    """Reads prompt_<src>_to_<tgt>.txt for every pair; exits listing any
    that are missing rather than submitting a partial batch."""
    prompts, missing = {}, []
    for pair in sorted(pairs):
        path = Path(prompts_dir) / prompt_filename(*pair)
        if not path.is_file():
            missing.append(str(path))
            continue
        prompts[pair] = path.read_text(encoding="utf-8")
    if missing:
        print("Missing prompt file(s) -- build them with generate_prompt_v2.py --out-dir:\n  "
              + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)
    return prompts


def build_batch_jsonl(rows, prompts, jsonl_path, model, temperature=None, cache_names=None):
    """cache_names maps band pair -> cachedContents/... name. Pairs with a
    cache reference it (the system prompt lives in the cache -- Gemini
    rejects a request that sets both); the rest send the prompt inline."""
    cache_names = cache_names or {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            pair = (row["source_band"], row["target_band"])

            generation_config = {"response_mime_type": "application/json"}
            if temperature is not None and supports_temperature_override(model):
                generation_config["temperature"] = temperature

            request = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": row["sentence"]}],
                }],
                "generation_config": generation_config,
            }
            if pair in cache_names:
                request["cached_content"] = cache_names[pair]
            else:
                request["system_instruction"] = {"parts": [{"text": prompts[pair]}]}

            request_obj = {"key": row["row_id"], "request": request}
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Context caching: one explicit cache per band pair's system prompt
# ---------------------------------------------------------------------------
class PromptCaches:
    """Creates one cache per system prompt, extends their TTL while the batch
    runs (so a short TTL is enough no matter how long the job queues), and
    deletes them at the end. Tracks token counts/lifetime for the storage
    cost estimate."""

    def __init__(self, client, model, ttl_seconds):
        self.client = client
        self.model = model
        self.ttl_seconds = ttl_seconds
        self.names = {}         # pair -> cachedContents/...
        self.token_counts = {}  # pair -> tokens held in the cache
        self.created_at = None
        self.deleted_at = None
        self._last_extend = None

    def create(self, prompts, display_name):
        from google.genai import types

        for pair, text in sorted(prompts.items()):
            try:
                cache = self.client.caches.create(
                    model=self.model,
                    config=types.CreateCachedContentConfig(
                        display_name=f"{display_name}-{pair[0]}to{pair[1]}",
                        system_instruction=text,
                        ttl=f"{self.ttl_seconds}s",
                    ),
                )
            except Exception as e:
                print(f"  could not cache prompt {pair[0]}->{pair[1]} ({e}); "
                      f"sending it inline instead", file=sys.stderr)
                continue
            self.names[pair] = cache.name
            tokens = getattr(cache.usage_metadata, "total_token_count", None) or 0
            self.token_counts[pair] = tokens
            print(f"  cached prompt {pair[0]}->{pair[1]}: {cache.name} ({tokens:,} tokens)")
        if self.names:
            self.created_at = self._last_extend = time.time()
        return self.names

    def keep_alive(self):
        """Push expiry out by another full TTL once half of it has elapsed."""
        if not self.names or time.time() - self._last_extend < self.ttl_seconds / 2:
            return
        from google.genai import types

        for name in self.names.values():
            self.client.caches.update(
                name=name, config=types.UpdateCachedContentConfig(ttl=f"{self.ttl_seconds}s"))
        self._last_extend = time.time()
        print(f"  extended cache TTL by {self.ttl_seconds}s")

    def delete(self):
        for name in self.names.values():
            try:
                self.client.caches.delete(name=name)
            except Exception as e:
                print(f"  could not delete cache {name}: {e} (it expires on its own)",
                      file=sys.stderr)
        if self.names:
            self.deleted_at = time.time()
            print(f"Deleted {len(self.names)} cache(s).")

    def storage_cost(self):
        """Estimated storage charge: tokens x hours alive x hourly rate."""
        rate = PRICING.get(self.model, {}).get("cache_storage_per_hour")
        if not self.names or rate is None:
            return None
        hours = ((self.deleted_at or time.time()) - self.created_at) / 3600
        return sum(self.token_counts.values()) / 1_000_000 * hours * rate


# ---------------------------------------------------------------------------
# Step 3-5: submit, poll, download
# ---------------------------------------------------------------------------
def run_batch_job(client, model, jsonl_path, display_name, poll_interval, caches=None):
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
        if caches is not None:
            caches.keep_alive()
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
def parse_and_report(rows_by_id, jsonl_text, model, output_dir, codes_csv_path,
                     cache_storage_cost=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guideline_rows = load_rows(codes_csv_path)
    code_to_desc = {r["code"]: r["desc"] for r in guideline_rows}

    results = []
    total_prompt_tokens = 0
    total_cached_tokens = 0
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
            "cached_tokens": 0,
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
        cached_tokens = usage.get("cachedContentTokenCount", 0)
        # thinking tokens are billed at the output rate -- include them
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)

        cost = compute_cost(model, prompt_tokens, output_tokens, cached_tokens)

        total_prompt_tokens += prompt_tokens
        total_cached_tokens += cached_tokens
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
        out_row["cached_tokens"] = cached_tokens
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
            "cached_tokens": r["cached_tokens"],
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
                  "simplified level", "prompt_tokens", "cached_tokens", "output_tokens",
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
    request_cost = total_cost
    if cache_storage_cost is not None:
        total_cost += cache_storage_cost
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
        f.write(f"  of which cached:    {total_cached_tokens:,}\n")
        f.write(f"Total output tokens:  {total_output_tokens:,} (includes thinking tokens)\n")
        f.write(f"Rate (input/output):  ${rates.get('input_std', '?')}/M -- ${rates.get('output_std', '?')}/M\n")
        f.write(f"Rate (cached input):  ${rates.get('input_cached', '?')}/M\n")
        f.write("-" * 45 + "\n")
        f.write(f"Request cost:         ${request_cost:.4f}\n")
        if cache_storage_cost is not None:
            f.write(f"Cache storage (est.): ${cache_storage_cost:.4f}\n")
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
    ap.add_argument("--prompts-dir", required=True,
                     help="Directory of prebuilt prompt_<src>_to_<tgt>.txt files "
                          "(from generate_prompt_v2.py --out-dir)")
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
    ap.add_argument("--price-cached", type=float, default=None, help="Override $/M cached input tokens for this run")
    ap.add_argument("--no-cache", action="store_true",
                     help="Send each system prompt inline instead of via an explicit context cache")
    ap.add_argument("--cache-ttl", type=int, default=3600,
                     help="Cache TTL in seconds; extended automatically while the job runs")
    ap.add_argument("--temperature", type=float, default=0.3,
                     help="Sampling temperature. Ignored for Gemini 3.x models "
                          "(gemini-3.*) per Google's guidance -- those models "
                          "should be left at their default and steered via the "
                          "system instruction instead.")
    ap.add_argument("--dry-run", action="store_true", help="Build the JSONL and print stats, but don't call the API")
    args = ap.parse_args()

    if args.price_input is not None or args.price_output is not None or args.price_cached is not None:
        PRICING.setdefault(args.model, {"threshold_tokens": None, "input_long": None, "output_long": None})
        if args.price_input is not None:
            PRICING[args.model]["input_std"] = args.price_input
        if args.price_output is not None:
            PRICING[args.model]["output_std"] = args.price_output
        if args.price_cached is not None:
            PRICING[args.model]["input_cached"] = args.price_cached

    rows = load_input_rows(args.input)
    if not rows:
        print("No valid rows found in input CSV.", file=sys.stderr)
        sys.exit(1)

    if args.annotations_csv:
        by_id, by_sentence = load_annotations(args.annotations_csv)
        apply_annotations(rows, by_id, by_sentence)
    rows_by_id = {r["row_id"]: r for r in rows}
    pairs = {(r["source_band"], r["target_band"]) for r in rows}
    print(f"Loaded {len(rows)} sentences across {len(pairs)} band pair(s).")
    prompts = load_prompts(args.prompts_dir, pairs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "batch_requests.jsonl"

    if args.dry_run:
        build_batch_jsonl(rows, prompts, jsonl_path, args.model)
        print(f"Wrote batch request file: {jsonl_path}")
        print("Dry run: skipping API call (and cache creation -- prompts are inline). "
              "Inspect the JSONL file above before submitting for real.")
        return

    from google import genai
    client = genai.Client(api_key=args.api_key) if args.api_key else genai.Client()

    caches = None
    cache_names = {}
    if not args.no_cache:
        print(f"Creating context caches for {len(prompts)} system prompt(s) ...")
        caches = PromptCaches(client, args.model, args.cache_ttl)
        cache_names = caches.create(prompts, args.display_name)

    try:
        build_batch_jsonl(rows, prompts, jsonl_path, args.model, cache_names=cache_names)
        print(f"Wrote batch request file: {jsonl_path} "
              f"({len(cache_names)}/{len(prompts)} prompt(s) via cache)")
        result_text = run_batch_job(client, args.model, jsonl_path, args.display_name,
                                    args.poll_interval, caches)
    finally:
        if caches is not None:
            caches.delete()

    raw_results_path = output_dir / "batch_results_raw.jsonl"
    with open(raw_results_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Wrote raw batch results: {raw_results_path}")

    parse_and_report(rows_by_id, result_text, args.model, output_dir, args.codes_csv,
                     caches.storage_cost() if caches is not None else None)


if __name__ == "__main__":
    main()