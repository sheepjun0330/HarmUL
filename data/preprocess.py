import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_REFUSAL = (
    "I cannot help with harmful or illegal requests. "
    "I can help with safety-focused, legal alternatives instead."
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dedupe(records: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for rec in records:
        signature = tuple(rec.get(k, "") for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(rec)
    return out


def _extract_from_harmbench(data: List[Dict[str, Any]], refusal_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    forget: List[Dict[str, Any]] = []
    retain: List[Dict[str, Any]] = []
    forget_alt: List[Dict[str, Any]] = []

    for row in data:
        query = _normalize_text(row.get("query"))
        if not query:
            continue

        responses = row.get("response") or []
        scores = row.get("harmbench_score") or []

        for idx, answer in enumerate(responses):
            answer_text = _normalize_text(answer)
            if not answer_text:
                continue
            score = scores[idx] if idx < len(scores) else None

            base = {
                "question": query,
                "answer": answer_text,
                "source": "harmbench",
                "example_idx": row.get("example_idx"),
                "candidate_idx": idx,
            }

            if score == 1:
                forget.append(base)
                forget_alt.append(
                    {
                        "question": query,
                        "answer": answer_text,
                        "alternate": refusal_text,
                        "source": "harmbench",
                        "example_idx": row.get("example_idx"),
                        "candidate_idx": idx,
                    }
                )
            elif score == 0:
                retain.append(base)

    return forget, retain, forget_alt


def _extract_from_jailbreak_log(data: List[Dict[str, Any]], refusal_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    forget: List[Dict[str, Any]] = []
    retain: List[Dict[str, Any]] = []
    forget_alt: List[Dict[str, Any]] = []

    for row in data:
        question = _normalize_text((row.get("final_jailbreak") or {}).get("prompt"))
        if not question:
            question = _normalize_text(row.get("behavior"))
        if not question:
            continue

        attempts = row.get("attempts") or []
        for attempt in attempts:
            verdict = _normalize_text((attempt.get("adjudication") or {}).get("verdict")).lower()
            answer_text = _normalize_text(attempt.get("last_assistant_answer"))
            if not answer_text:
                continue

            base = {
                "question": question,
                "answer": answer_text,
                "source": "jailbreak_log",
                "attempt": attempt.get("attempt"),
            }

            if verdict == "unsafe":
                forget.append(base)
                forget_alt.append(
                    {
                        "question": question,
                        "answer": answer_text,
                        "alternate": refusal_text,
                        "source": "jailbreak_log",
                        "attempt": attempt.get("attempt"),
                    }
                )
            elif verdict == "safe":
                retain.append(base)

    return forget, retain, forget_alt


def _detect_format(data: Any) -> str:
    if not isinstance(data, list) or not data:
        return "unknown"

    first = data[0]
    if not isinstance(first, dict):
        return "unknown"

    if "query" in first and "response" in first and "harmbench_score" in first:
        return "harmbench"

    if "attempts" in first and ("behavior" in first or "final_jailbreak" in first):
        return "jailbreak_log"

    return "unknown"


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sample(records: List[Dict[str, Any]], max_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_size is None or len(records) <= max_size:
        return records
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    keep = sorted(idxs[:max_size])
    return [records[i] for i in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess jailbreak JSON files into unlearning JSONL datasets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files. Supports Harmbench-style and jailbreak_log-style formats.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/unlearn",
        help="Output directory for forget/retain JSONL files.",
    )
    parser.add_argument(
        "--max_forget",
        type=int,
        default=None,
        help="Optional cap on forget examples.",
    )
    parser.add_argument(
        "--max_retain",
        type=int,
        default=None,
        help="Optional cap on retain examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling.",
    )
    parser.add_argument(
        "--refusal_text",
        default=DEFAULT_REFUSAL,
        help="Alternate safe response used for forget_alt.jsonl.",
    )
    args = parser.parse_args()

    all_forget: List[Dict[str, Any]] = []
    all_retain: List[Dict[str, Any]] = []
    all_forget_alt: List[Dict[str, Any]] = []

    for input_path in args.inputs:
        path = Path(input_path)
        data = _read_json(path)
        data_format = _detect_format(data)

        if data_format == "harmbench":
            forget, retain, forget_alt = _extract_from_harmbench(
                data=data,
                refusal_text=args.refusal_text,
            )
        elif data_format == "jailbreak_log":
            forget, retain, forget_alt = _extract_from_jailbreak_log(
                data=data,
                refusal_text=args.refusal_text,
            )
        else:
            raise ValueError(f"Unsupported JSON format: {path}")

        all_forget.extend(forget)
        all_retain.extend(retain)
        all_forget_alt.extend(forget_alt)

    all_forget = _dedupe(all_forget, keys=("question", "answer"))
    all_retain = _dedupe(all_retain, keys=("question", "answer"))
    all_forget_alt = _dedupe(all_forget_alt, keys=("question", "answer", "alternate"))

    all_forget = _sample(all_forget, args.max_forget, args.seed)
    all_retain = _sample(all_retain, args.max_retain, args.seed)

    out_dir = Path(args.out_dir)
    forget_path = out_dir / "forget.jsonl"
    retain_path = out_dir / "retain.jsonl"
    forget_alt_path = out_dir / "forget_alt.jsonl"

    _write_jsonl(forget_path, all_forget)
    _write_jsonl(retain_path, all_retain)
    _write_jsonl(forget_alt_path, all_forget_alt)

    summary = {
        "inputs": args.inputs,
        "forget_count": len(all_forget),
        "retain_count": len(all_retain),
        "forget_alt_count": len(all_forget_alt),
        "out_dir": str(out_dir),
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
