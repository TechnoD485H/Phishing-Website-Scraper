"""
gpt_phishing_analyzer.py

Separate OpenAI-based phishing analyzer designed to work alongside
phishing_scraper.py without modifying it.

Modes:
    1) Analyze one URL:
       python gpt_phishing_analyzer.py https://example.com

    2) Analyze an existing labeled dataset:
       python gpt_phishing_analyzer.py --dataset labeled_dataset.csv

    3) Analyze a batch of URLs from a text file:
       python gpt_phishing_analyzer.py --urls urls.txt

The analyzer imports analyze_url() from phishing_scraper.py, so it uses
the same feature extraction and rule-based risk score as the existing
program.

Environment:
    OPENAI_API_KEY=your_api_key_here

Optional:
    OPENAI_MODEL=gpt-5.6-luna

Install:
    pip install openai python-dotenv pandas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Import the existing scraper. This intentionally leaves the scraper itself
# unchanged and reuses its analyze_url() function.
try:
    from phishing_scraper import analyze_url
except ImportError as exc:
    print(
        "ERROR: Could not import phishing_scraper.py.\n"
        "Make sure gpt_phishing_analyzer.py is in the same directory as "
        "phishing_scraper.py."
    )
    raise SystemExit(1) from exc


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_OUTPUT = "gpt_phishing_results.csv"

# These are the same ML/risk features used by train_model.py, plus the
# human-readable rule-based information that helps GPT interpret them.
FEATURE_COLUMNS = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "has_at_symbol",
    "uses_https",
    "has_ip_as_domain",
    "on_free_hosting",
    "has_suspicious_keyword",
    "subdomain_length",
    "subdomain_entropy",
    "num_forms",
    "has_password_field",
    "form_posts_externally",
    "fetch_succeeded",
    "domain_age_known",
    "brand_edit_distance",
    "impersonates_brand",
]

# Keep the request deliberately structured and compact. The model is asked
# to classify the evidence supplied by the scraper; it is not told that the
# rule-based score is ground truth.
SYSTEM_PROMPT = """
You are a defensive cybersecurity classifier. Your task is to assess whether
a website is likely legitimate or phishing using structured evidence produced
by a URL/page scraper.

This is a classification aid, not proof of maliciousness. Do not claim that
a feature alone proves phishing. HTTPS, passwords, forms, free hosting, and
keywords can all occur on legitimate sites. Consider the combination of
signals and the limitations of the evidence.

Return ONLY valid JSON matching the requested schema.

Required fields:
- classification: exactly "phishing", "legitimate", or "uncertain"
- confidence: integer from 0 to 100
- risk_level: exactly "low", "medium", "high", or "critical"
- reasons: array of 1 to 5 concise strings
- recommendation: exactly "allow", "review", or "block"
- summary: one concise sentence

Use "uncertain" when the supplied evidence is insufficient for a confident
decision. Do not invent WHOIS, DNS, page content, reputation, registration,
or visual information that is not supplied.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["phishing", "legitimate", "uncertain"],
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "recommendation": {
            "type": "string",
            "enum": ["allow", "review", "block"],
        },
        "summary": {"type": "string"},
    },
    "required": [
        "classification",
        "confidence",
        "risk_level",
        "reasons",
        "recommendation",
        "summary",
    ],
}


def make_client() -> OpenAI:
    """Create the OpenAI client from OPENAI_API_KEY."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n\n"
            "Set it as an environment variable or put it in a .env file:\n"
            "OPENAI_API_KEY=your_api_key_here"
        )

    return OpenAI(api_key=api_key)


def get_model() -> str:
    """Allow the model to be changed without editing this file."""
    load_dotenv()
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def clean_value(value: Any) -> Any:
    """Convert pandas/numpy-ish values into JSON-friendly Python values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def build_evidence(features: dict[str, Any]) -> dict[str, Any]:
    """
    Build the compact evidence object sent to GPT.

    We intentionally do not send arbitrary dataframe columns or huge page
    contents. This keeps the request small and makes the model's decision
    based on the same structured signals as the existing program.
    """
    evidence: dict[str, Any] = {
        "url": features.get("url", ""),
        "risk_score": clean_value(features.get("risk_score")),
        "risk_reasons": features.get("risk_reasons", ""),
        "matched_brand": features.get("matched_brand", ""),
    }

    evidence["features"] = {
        column: clean_value(features.get(column))
        for column in FEATURE_COLUMNS
        if column in features
    }

    # Domain age is useful context but is deliberately separate from the
    # feature vector used by the Random Forest.
    evidence["domain_age_days"] = clean_value(
        features.get("domain_age_days")
    )

    return evidence


def classify_with_gpt(
    client: OpenAI,
    features: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Send one analyzed website's structured evidence to the OpenAI API."""
    evidence = build_evidence(features)
    model = model or get_model()

    user_prompt = (
        "Assess this website using ONLY the supplied evidence.\n\n"
        "Website evidence:\n"
        + json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "phishing_assessment",
                "strict": True,
                "schema": JSON_SCHEMA,
            }
        },
    )

    raw = response.output_text.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenAI returned invalid JSON:\n{raw}"
        ) from exc

    # Defensive validation after the API response.
    required = {
        "classification",
        "confidence",
        "risk_level",
        "reasons",
        "recommendation",
        "summary",
    }
    missing = required - set(result)
    if missing:
        raise RuntimeError(
            f"OpenAI response is missing fields: {sorted(missing)}"
        )

    return result


def analyze_one_url(
    client: OpenAI,
    url: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Run the existing scraper first, then ask GPT to assess its evidence.
    """
    print(f"\nAnalyzing URL: {url}")
    print("Running existing scraper/feature extraction...")

    features = analyze_url(url)

    print("Sending structured evidence to OpenAI...")
    gpt_result = classify_with_gpt(client, features, model=model)

    result = dict(features)

    result["gpt_classification"] = gpt_result["classification"]
    result["gpt_confidence"] = gpt_result["confidence"]
    result["gpt_risk_level"] = gpt_result["risk_level"]
    result["gpt_recommendation"] = gpt_result["recommendation"]
    result["gpt_reasons"] = " | ".join(gpt_result["reasons"])
    result["gpt_summary"] = gpt_result["summary"]

    return result


def print_result(result: dict[str, Any]) -> None:
    """Print a readable result for interactive single-URL use."""
    print("\n" + "=" * 70)
    print("GPT PHISHING ANALYSIS")
    print("=" * 70)

    print(f"URL:            {result.get('url', '')}")
    print(f"Rule score:     {result.get('risk_score', 'N/A')}")
    print(
        f"Rule reasons:   "
        f"{result.get('risk_reasons', '') or 'none'}"
    )
    print()
    print(f"Classification: {str(result.get('gpt_classification', '')).upper()}")
    print(f"Confidence:     {result.get('gpt_confidence', '')}%")
    print(f"Risk level:     {str(result.get('gpt_risk_level', '')).upper()}")
    print(f"Recommendation: {str(result.get('gpt_recommendation', '')).upper()}")
    print()
    print("GPT reasons:")
    for reason in str(result.get("gpt_reasons", "")).split(" | "):
        if reason:
            print(f"  - {reason}")

    print()
    print(f"Summary: {result.get('gpt_summary', '')}")
    print("=" * 70)


def analyze_dataset(
    client: OpenAI,
    filename: str,
    output: str,
    model: str | None = None,
    delay: float = 0.0,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Run GPT over rows from an existing labeled_dataset.csv.

    Important: this does NOT refetch the websites. It uses the features already
    saved by phishing_scraper.py, making this mode faster and reproducible.
    """
    df = pd.read_csv(filename)

    if "url" not in df.columns:
        raise ValueError(f"{filename} does not contain a 'url' column.")

    if limit is not None:
        df = df.head(limit).copy()

    rows: list[dict[str, Any]] = []

    print(f"Loaded {len(df)} rows from {filename}")
    print(f"OpenAI model: {model or get_model()}")

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\n[{position}/{len(df)}] {row['url']}")

        features = {
            key: clean_value(value)
            for key, value in row.to_dict().items()
        }

        try:
            gpt_result = classify_with_gpt(
                client,
                features,
                model=model,
            )

            output_row = features.copy()
            output_row["gpt_classification"] = gpt_result["classification"]
            output_row["gpt_confidence"] = gpt_result["confidence"]
            output_row["gpt_risk_level"] = gpt_result["risk_level"]
            output_row["gpt_recommendation"] = gpt_result["recommendation"]
            output_row["gpt_reasons"] = " | ".join(gpt_result["reasons"])
            output_row["gpt_summary"] = gpt_result["summary"]
            output_row["gpt_error"] = ""

            print(
                f"    GPT: {gpt_result['classification']} "
                f"({gpt_result['confidence']}%)"
            )

        except Exception as exc:
            print(f"    ERROR: {exc}")

            output_row = features.copy()
            output_row["gpt_classification"] = ""
            output_row["gpt_confidence"] = ""
            output_row["gpt_risk_level"] = ""
            output_row["gpt_recommendation"] = ""
            output_row["gpt_reasons"] = ""
            output_row["gpt_summary"] = ""
            output_row["gpt_error"] = str(exc)

        rows.append(output_row)

        if delay > 0 and position < len(df):
            time.sleep(delay)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output, index=False)

    print(f"\nSaved GPT results to: {output}")

    if "label" in result_df.columns:
        print_dataset_comparison(result_df)

    return result_df


def print_dataset_comparison(df: pd.DataFrame) -> None:
    """
    Compare GPT's classifications to the labels already present in the CSV.

    This is descriptive only. It is NOT a replacement for a properly
    separated evaluation set.
    """
    valid = df[
        df["gpt_classification"].isin(["phishing", "legitimate"])
        & df["label"].isin([0, 1])
    ].copy()

    if valid.empty:
        print("\nNo comparable labelled GPT results available.")
        return

    predicted = (
        valid["gpt_classification"] == "phishing"
    ).astype(int)
    actual = valid["label"].astype(int)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    print("\n" + "=" * 70)
    print("GPT VS DATASET LABELS")
    print("=" * 70)
    print(f"Comparable rows: {len(valid)}")
    print(f"Accuracy:        {accuracy_score(actual, predicted):.3f}")
    print(f"Precision:       {precision_score(actual, predicted, zero_division=0):.3f}")
    print(f"Recall:          {recall_score(actual, predicted, zero_division=0):.3f}")
    print(f"F1:              {f1_score(actual, predicted, zero_division=0):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predicted,
            target_names=["legit", "phishing"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, cols = GPT prediction):")
    print(
        pd.DataFrame(
            confusion_matrix(actual, predicted),
            index=["actual: legit", "actual: phishing"],
            columns=["pred: legit", "pred: phishing"],
        )
    )


def load_urls_file(filename: str) -> list[str]:
    """Read one URL per line, ignoring blank lines and comments."""
    urls = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            url = line.strip()

            if not url or url.startswith("#"):
                continue

            urls.append(url)

    return urls


def analyze_url_file(
    client: OpenAI,
    filename: str,
    output: str,
    model: str | None = None,
    delay: float = 0.0,
    limit: int | None = None,
) -> pd.DataFrame:
    """Analyze URLs from a text file by using the existing scraper."""
    urls = load_urls_file(filename)

    if limit is not None:
        urls = urls[:limit]

    if not urls:
        raise ValueError(f"No URLs found in {filename}")

    rows = []

    print(f"Loaded {len(urls)} URLs from {filename}")

    for position, url in enumerate(urls, start=1):
        print(f"\n[{position}/{len(urls)}]")

        try:
            result = analyze_one_url(client, url, model=model)
            rows.append(result)
            print_result(result)
        except Exception as exc:
            print(f"ERROR analyzing {url}: {exc}")

        if delay > 0 and position < len(urls):
            time.sleep(delay)

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df.to_csv(output, index=False)
        print(f"\nSaved results to: {output}")

    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a separate OpenAI phishing assessment using the "
            "features produced by phishing_scraper.py."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "url",
        nargs="?",
        help="Analyze a single URL.",
    )

    source.add_argument(
        "--dataset",
        help="Analyze an existing labeled_dataset.csv without refetching URLs.",
    )

    source.add_argument(
        "--urls",
        help="Analyze a text file containing one URL per line.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenAI model. Defaults to OPENAI_MODEL or {DEFAULT_MODEL}.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output filename (default: {DEFAULT_OUTPUT}).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows/URLs.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API requests.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        client = make_client()
        model = args.model or get_model()

        print(f"Using OpenAI model: {model}")

        if args.url:
            result = analyze_one_url(
                client,
                args.url,
                model=model,
            )
            print_result(result)

            # Save the single result too, so it can be inspected later.
            pd.DataFrame([result]).to_csv(args.output, index=False)
            print(f"\nSaved result to: {args.output}")

        elif args.dataset:
            analyze_dataset(
                client,
                filename=args.dataset,
                output=args.output,
                model=model,
                delay=args.delay,
                limit=args.limit,
            )

        elif args.urls:
            analyze_url_file(
                client,
                filename=args.urls,
                output=args.output,
                model=model,
                delay=args.delay,
                limit=args.limit,
            )

        return 0

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())