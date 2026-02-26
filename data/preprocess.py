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


def _group_split_by_question(
    forget: List[Dict[str, Any]],
    retain: List[Dict[str, Any]],
    forget_alt: List[Dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
           List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split by normalized question so that the same prompt does not appear in both train and eval.
    Returns:
      train_forget, train_retain, train_forget_alt,
      eval_forget,  eval_retain,  eval_forget_alt
    """
    if eval_ratio <= 0.0:
        return forget, retain, forget_alt, [], [], []

    rng = random.Random(seed)

    # Collect unique questions per split group
    forget_qs = sorted({r.get("question", "") for r in forget if r.get("question")})
    retain_qs = sorted({r.get("question", "") for r in retain if r.get("question")})

    # Sample eval questions separately for forget/retain to preserve class balance
    def pick_eval_questions(qs: List[str]) -> set:
        if not qs:
            return set()
        n_eval = max(1, int(round(len(qs) * eval_ratio))) if eval_ratio > 0 else 0
        qs_copy = qs[:]
        rng.shuffle(qs_copy)
        return set(qs_copy[:n_eval])

    eval_forget_q = pick_eval_questions(forget_qs)
    eval_retain_q = pick_eval_questions(retain_qs)

    # Split records
    train_forget, eval_forget = [], []
    for r in forget:
        (eval_forget if r.get("question") in eval_forget_q else train_forget).append(r)

    train_retain, eval_retain = [], []
    for r in retain:
        (eval_retain if r.get("question") in eval_retain_q else train_retain).append(r)

    # forget_alt follows forget split (same question+answer)
    train_forget_alt, eval_forget_alt = [], []
    for r in forget_alt:
        (eval_forget_alt if r.get("question") in eval_forget_q else train_forget_alt).append(r)

    return train_forget, train_retain, train_forget_alt, eval_forget, eval_retain, eval_forget_alt


def _extract_from_harmbench(
    data: List[Dict[str, Any]],
    refusal_text: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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

            # Goal A:
            # - forget: harmful/unsafe answers (score==1)
            # - retain: safe answers (score==0) [note: this is "safety retain", not general utility]
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


def _extract_from_jailbreak_log(
    data: List[Dict[str, Any]],
    refusal_text: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Legacy format:
      row: {behavior/final_jailbreak, attempts: [{adjudication:{verdict}, last_assistant_answer, attempt}, ...]}
    """
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
            # IMPORTANT: For Goal A we want the model's actual output to the harmful prompt.
            # In this legacy log, last_assistant_answer is treated as the model response.
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


def _extract_from_attack_records(
    data: List[Dict[str, Any]],
    refusal_text: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    New/flat record format (based on your sample):
      row: {
        attempt: int,
        jailbreak_submission: {prompt: str, rationale: str, ...},
        response: str,
        adjudication: {verdict/raw/categories...},
        ... (other fields)
      }

    Goal A mapping:
      - question := jailbreak_submission.prompt
      - answer   := response (target model output)
      - forget if adjudication.verdict == "unsafe"
      - retain if adjudication.verdict == "safe"  (note: "safety retain")
      - forget_alt: same question+answer, alternate := refusal_text
    """
    forget: List[Dict[str, Any]] = []
    retain: List[Dict[str, Any]] = []
    forget_alt: List[Dict[str, Any]] = []

    for row in data:
        question = _normalize_text((row.get("jailbreak_submission") or {}).get("prompt"))
        if not question:
            # fallback: some logs may store prompt at different key
            question = _normalize_text(row.get("prompt"))
        if not question:
            continue

        answer_text = _normalize_text(row.get("response"))
        if not answer_text:
            continue

        verdict = _normalize_text((row.get("adjudication") or {}).get("verdict")).lower()
        # Some datasets use "raw" only.
        if not verdict:
            verdict = _normalize_text((row.get("adjudication") or {}).get("raw")).lower()

        base = {
            "question": question,
            "answer": answer_text,
            "source": "attack_record",
            "attempt": row.get("attempt"),
        }

        if verdict == "unsafe":
            forget.append(base)
            forget_alt.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "alternate": refusal_text,
                    "source": "attack_record",
                    "attempt": row.get("attempt"),
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

    # flat record heuristic
    if "jailbreak_submission" in first and "response" in first and "adjudication" in first:
        return "attack_record"

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess jailbreak JSON files into unlearning JSONL datasets (Goal A).")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files. Supports Harmbench-style, jailbreak_log-style, and flat attack_record-style formats.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/unlearn",
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--max_forget",
        type=int,
        default=None,
        help="Optional cap on TRAIN forget examples (after eval split).",
    )
    parser.add_argument(
        "--max_retain",
        type=int,
        default=None,
        help="Optional cap on TRAIN retain examples (after eval split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling/splitting.",
    )
    parser.add_argument(
        "--refusal_text",
        default=DEFAULT_REFUSAL,
        help="Alternate safe response used for forget_alt.jsonl.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for eval split, grouped by normalized question (prompt). 0 disables eval outputs.",
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
            forget, retain, forget_alt = _extract_from_harmbench(data=data, refusal_text=args.refusal_text)
        elif data_format == "jailbreak_log":
            forget, retain, forget_alt = _extract_from_jailbreak_log(data=data, refusal_text=args.refusal_text)
        elif data_format == "attack_record":
            forget, retain, forget_alt = _extract_from_attack_records(data=data, refusal_text=args.refusal_text)
        else:
            raise ValueError(f"Unsupported JSON format: {path}")

        all_forget.extend(forget)
        all_retain.extend(retain)
        all_forget_alt.extend(forget_alt)

    # Dedupe before splitting
    all_forget = _dedupe(all_forget, keys=("question", "answer"))
    all_retain = _dedupe(all_retain, keys=("question", "answer"))
    all_forget_alt = _dedupe(all_forget_alt, keys=("question", "answer", "alternate"))

    # Split by question to prevent leakage across train/eval
    train_forget, train_retain, train_forget_alt, eval_forget, eval_retain, eval_forget_alt = _group_split_by_question(
        forget=all_forget,
        retain=all_retain,
        forget_alt=all_forget_alt,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    # Optional caps (train only)
    train_forget = _sample(train_forget, args.max_forget, args.seed)
    train_retain = _sample(train_retain, args.max_retain, args.seed)

    out_dir = Path(args.out_dir)

    # Train outputs
    _write_jsonl(out_dir / "forget.jsonl", train_forget)
    _write_jsonl(out_dir / "retain.jsonl", train_retain)
    _write_jsonl(out_dir / "forget_alt.jsonl", train_forget_alt)

    # Eval outputs (if enabled)
    if args.eval_ratio > 0.0:
        _write_jsonl(out_dir / "eval_forget.jsonl", eval_forget)
        _write_jsonl(out_dir / "eval_retain.jsonl", eval_retain)
        _write_jsonl(out_dir / "eval_forget_alt.jsonl", eval_forget_alt)

    summary = {
        "inputs": args.inputs,
        "eval_ratio": args.eval_ratio,
        "train_forget_count": len(train_forget),
        "train_retain_count": len(train_retain),
        "train_forget_alt_count": len(train_forget_alt),
        "eval_forget_count": len(eval_forget),
        "eval_retain_count": len(eval_retain),
        "eval_forget_alt_count": len(eval_forget_alt),
        "out_dir": str(out_dir),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()