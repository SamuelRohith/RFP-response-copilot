"""
cleaners.py — text cleaning layer for the RFP Response Co-pilot.

Takes the raw text from a LoadedDocument and removes extraction noise
(broken line wraps, stray whitespace, repeated headers/footers) WITHOUT
stripping structural signal (numbering, headings) that later steps rely on.
"""

import re
from collections import Counter


def rejoin_wrapped_lines(text: str) -> str:
    """
    PDF extraction often inserts a newline wherever a line visually wrapped,
    not where the sentence actually ended. Heuristic: if a line does NOT end
    in sentence-ending punctuation (. ? : ;) or a colon, and the next line
    doesn't look like a new bullet/heading/number, join them with a space.

    Deliberately conservative: several line types are excluded from merging
    because merging them destroys real structure, not just wrapped text:
      - numbered/lettered headings ("1. Scope of Work", "SECTION 3 — ...")
      - "label: value" metadata lines ("RFP Number: RHN-ITS-2026-014")
      - short, title-cased lines (likely a title or heading, not prose)
    """
    lines = text.split("\n")
    joined = []
    bullet_or_number = re.compile(r"^\s*([\u2022\-\*]|\d+[\.\)]|[a-zA-Z][\.\)])\s")
    heading_pattern = re.compile(r"^\s*(\d+\.\s+\S|SECTION\s+\d+|##\s+\S|[A-Z][a-zA-Z ]{2,40}:\s*\S)")

    def looks_like_heading_or_metadata(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if heading_pattern.match(stripped):
            return True
        # short line with no terminal punctuation and title-like capitalization
        if len(stripped) <= 70 and not stripped.endswith((".", ",", ";")):
            words = stripped.split()
            if words and sum(1 for w in words if w[:1].isupper()) / len(words) > 0.6:
                return True
        return False

    for line in lines:
        stripped = line.rstrip()
        prev = joined[-1] if joined else ""
        if (
            joined
            and prev
            and not prev.endswith((".", "?", ":", ";", "!"))
            and not bullet_or_number.match(line)
            and stripped
            and not stripped.isupper()
            and not looks_like_heading_or_metadata(prev)
            and not looks_like_heading_or_metadata(stripped)
        ):
            joined[-1] = prev + " " + stripped.lstrip()
        else:
            joined.append(stripped)

    return "\n".join(joined)


def collapse_whitespace(text: str) -> str:
    """Normalize multiple blank lines and trailing/leading spaces per line."""
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces before newline
    text = re.sub(r"\n{3,}", "\n\n", text)           # 3+ blank lines -> 1 blank line
    text = re.sub(r"[ \t]{2,}", " ", text)           # runs of spaces/tabs -> single space
    return text.strip()


def remove_repeated_boilerplate(text: str, min_occurrences: int = 3) -> str:
    """
    Detects short lines (likely headers/footers/page numbers) that repeat
    verbatim across the document and removes all but the first occurrence.
    Only targets SHORT lines to avoid accidentally deleting real repeated
    content like a requirement that legitimately appears twice.
    """
    lines = text.split("\n")
    line_counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) <= 80)

    boilerplate = {
        line for line, count in line_counts.items()
        if count >= min_occurrences
    }

    cleaned_lines = []
    seen_once = set()
    for line in lines:
        stripped = line.strip()
        if stripped in boilerplate:
            if stripped not in seen_once:
                cleaned_lines.append(line)
                seen_once.add(stripped)
            # else: skip repeated boilerplate occurrence
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_text(text: str) -> str:
    """Full cleaning pipeline, applied in a deliberate order."""
    text = remove_repeated_boilerplate(text)
    text = rejoin_wrapped_lines(text)
    text = collapse_whitespace(text)
    return text


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.loaders import load_document

    for test_path in sys.argv[1:]:
        doc = load_document(test_path)
        cleaned = clean_text(doc.text)
        print(f"\n=== {test_path} ===")
        print(f"raw chars: {len(doc.text)} -> cleaned chars: {len(cleaned)}")
        print("--- first 600 chars of cleaned text ---")
        print(cleaned[:600])
