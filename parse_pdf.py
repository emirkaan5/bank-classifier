# filename: cc_pdf_to_csv.py
"""
Usage:
  pip install pdfplumber python-dateutil tqdm
  python cc_pdf_to_csv.py /path/to/statements/ output.csv
  # or
  python cc_pdf_to_csv.py /path/to/one.pdf output.csv
"""

import sys
import re
import csv
import os
from pathlib import Path
from dateutil import parser as dateparser
import pdfplumber
from tqdm import tqdm

# --- Helpers ---------------------------------------------------------------

CURRENCY_RE = re.compile(r'(?<![\d,.-])(-?\$?\d{1,3}(?:,\d{3})*\.\d{2})(?![\d])')
DATE_CANDIDATE_RE = re.compile(
    r'\b((?:\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?)|(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}))\b'
)
POSSIBLE_HEADER_HINTS = re.compile(
    r'(previous\s+balance|new\s+balance|total\s+payment|statement\s+period|credit\s+limit)',
    re.I
)

def normalize_money(s):
    """Convert money string like '$1,234.56' or '-123.45' to float."""
    s = s.replace('$', '').replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return None

def try_parse_date(token, year_hint=None):
    """
    Try to parse a date from a token like '03/12' or '2025-03-12'.
    If year is missing, use year_hint when available.
    """
    token = token.strip()
    try:
        dt = dateparser.parse(token, dayfirst=False, yearfirst=False, default=year_hint)
        return dt.date().isoformat()
    except Exception:
        return None

def detect_year_hint(all_text):
    """
    Try to infer statement year from text (look for 'Statement Period: mm/dd/yyyy - mm/dd/yyyy' etc.).
    """
    year_hits = re.findall(r'(20\d{2})', all_text)
    if year_hits:
        # choose the most common or last
        from collections import Counter
        c = Counter(year_hits)
        year = int(c.most_common(1)[0][0])
        # Use January 1st as default time for dateutil default.
        import datetime as _dt
        return _dt.datetime(year, 1, 1)
    return None

def is_header_line(line):
    # crude filter to skip obvious headers/footers
    if len(line.strip()) < 3:
        return True
    if POSSIBLE_HEADER_HINTS.search(line):
        return True
    if re.search(r'page\s+\d+\s+of\s+\d+', line, re.I):
        return True
    return False

def clean_line(line):
    # Collapse whitespace, keep original for alignment in case-by-case tweaks
    return re.sub(r'\s+', ' ', line).strip()

# --- Core parsing ----------------------------------------------------------

def extract_text_lines(pdf_path):
    """Extract lines of text from a PDF using pdfplumber."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines.extend(text.splitlines())
    return lines

def guess_transactions(lines, year_hint=None):
    """
    Generic transaction extraction:
      - Find lines that contain at least one currency amount.
      - Assume the LAST currency on the line is the transaction amount.
      - If a second-to-last currency exists, treat it as a running balance (optional).
      - Try to grab a date token from the start or anywhere on the line.
      - Everything in between is description.
    Returns list of dicts with keys:
      date, description, amount, balance (optional), type (debit/credit/payment/fee/unknown), source_line
    """
    txns = []
    for raw_line in lines:
        if not raw_line or is_header_line(raw_line):
            continue

        # currency candidates
        monies = list(CURRENCY_RE.finditer(raw_line))
        if not monies:
            continue

        # Heuristic: ignore rows that look like totals only
        if re.search(r'\b(total|new balance|previous balance|payments?\s+and\s+credits)\b', raw_line, re.I):
            continue

        amount_m = monies[-1]
        amount_str = amount_m.group(1)
        amount = normalize_money(amount_str)
        if amount is None:
            continue

        balance = None
        if len(monies) >= 2:
            balance = normalize_money(monies[-2].group(1))

        # Try to find a date
        date = None
        # prefer date near line start
        head = raw_line[:40]
        date_m = DATE_CANDIDATE_RE.search(head) or DATE_CANDIDATE_RE.search(raw_line)
        if date_m:
            date_tok = date_m.group(1)
            date = try_parse_date(date_tok, year_hint=year_hint)

        # Build description: everything except the trailing amount (and balance if present)
        cut_idx = amount_m.start()
        desc_segment = raw_line[:cut_idx].rstrip()
        if balance is not None:
            # remove balance token as well
            bal_m = monies[-2]
            # clip description up to before balance, but keep text between balance and amount as well
            desc_segment = raw_line[:bal_m.start()].rstrip()

        description = clean_line(desc_segment)

        # Classify type: crude rules
        line_l = raw_line.lower()
        if any(k in line_l for k in ['payment', 'autopay', 'thank you']):
            tx_type = 'payment'
            # payments are usually negative to the balance; amounts can appear as -value or positive under "payments"
            if amount > 0:
                amount = -amount
        elif any(k in line_l for k in ['fee', 'interest', 'late']):
            tx_type = 'fee/interest'
        elif any(k in line_l for k in ['credit', 'refund', 'reversal']) and amount > 0:
            tx_type = 'credit'
            amount = -abs(amount)  # credits reduce balance; make them negative for spending/outflow sign convention
        else:
            # Purchases typically positive on the right column; use positive for outflow
            tx_type = 'purchase'
            amount = abs(amount)

        txns.append({
            "date": date,
            "description": description or None,
            "amount": round(amount, 2) if amount is not None else None,
            "balance": round(balance, 2) if balance is not None else None,
            "type": tx_type,
            "source_line": clean_line(raw_line),
        })

    # Post-filter: keep rows that have at least date+amount or description+amount
    filtered = [
        t for t in txns
        if t["amount"] is not None and (t["date"] is not None or t["description"])
    ]
    return filtered

# --- Optional: bank-specific tweak hooks (examples only; off by default) ---

def tweak_for_known_banks(txns, all_text):
    """
    Hook to apply bank-specific fixes (enable & expand as needed).
    Example heuristics are commented for Chase/Amex/CapOne layouts.
    """
    text_l = all_text.lower()

    # Example: Chase sometimes lists "Payments and Credits" as a section:
    # We already negate credits; you could add extra checks here.

    # Example: American Express often uses two date columns (transaction date / posting date).
    # You could parse two dates and pick one; here we keep the detected one.

    # Example: Capital One might separate categories; no change here.

    return txns

# --- Orchestrator ----------------------------------------------------------

def parse_pdf(pdf_path):
    lines = extract_text_lines(pdf_path)
    all_text = "\n".join(lines)
    year_hint = detect_year_hint(all_text)
    txns = guess_transactions(lines, year_hint=year_hint)
    txns = tweak_for_known_banks(txns, all_text)
    # add file metadata
    for t in txns:
        t["source_file"] = str(pdf_path)
    return txns

def iter_pdf_files(pathlike):
    p = Path(pathlike)
    if p.is_file() and p.suffix.lower() == ".pdf":
        yield p
    elif p.is_dir():
        for fp in sorted(p.rglob("*.pdf")):
            yield fp
    else:
        raise FileNotFoundError(f"No PDF found at {p}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python cc_pdf_to_csv.py <pdf_or_folder> <output.csv>")
        sys.exit(1)

    inp, out_csv = sys.argv[1], sys.argv[2]
    all_rows = []
    pdfs = list(iter_pdf_files(inp))

    for pdf_path in tqdm(pdfs, desc="Parsing PDFs"):
        try:
            rows = parse_pdf(pdf_path)
            all_rows.extend(rows)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed on {pdf_path}: {e}\n")

    # Deduplicate obvious duplicates (same file+date+desc+amount)
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r.get("source_file"), r.get("date"), r.get("description"), r.get("amount"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # Write CSV
    fieldnames = ["date", "description", "amount", "balance", "type", "source_file", "source_line"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in deduped:
            w.writerow(r)

    print(f"Saved {len(deduped)} rows to {out_csv}")

if __name__ == "__main__":
    main()

