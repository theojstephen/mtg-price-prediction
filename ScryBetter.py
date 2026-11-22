#!/usr/bin/env python3
"""
Scryfall scanner  --  the fast sibling of mtg_goldfish_scanner.py.

Collects ability text, rarity, USD price, mana value and power straight from
Scryfall's API (no browser, no page-by-page scraping), then writes:

  scryfall_cards.csv          one row per card per run (use --append to build
                              a daily price history)
  scryfall_llm.txt / .jsonl   compact blocks for a local LLM (Qwen etc.)

Setup
  pip install requests

Usage
  python scryfall_scanner.py --sets fin eoe                 # specific sets
  python scryfall_scanner.py --sets fin --append            # add today's prices
  python scryfall_scanner.py --all                          # every paper card
  python scryfall_scanner.py --export-only                  # rebuild LLM files

Sets are Scryfall set CODES (3-5 letters, e.g. fin, eoe), not names. Find them
at https://scryfall.com/sets.

Notes
  * Prices are Scryfall's USD prices (sourced from TCGplayer), refreshed about
    once a day, so running more than once a day gains nothing.
  * Basic lands, tokens, emblems and digital-only cards are skipped.
  * In the LLM files, mana value and power become "high" (4+) or "low" (3-),
    and threshold phrases in the text ("power 4 or greater") are translated the
    same way. Everything else in the text is kept exactly as Scryfall has it.
  * Please keep to Scryfall's API rules: identify yourself in the User-Agent
    below and keep the short delay between requests.
"""
import argparse
import csv
import json
import re
import time
from datetime import date
from pathlib import Path

import requests

API = "https://api.scryfall.com"
HEADERS = {
    "User-Agent": "mtg-price-project/0.1 (personal research; replace-with-your-email)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}
DELAY = 0.12  # Scryfall asks for 50-100 ms between requests
SKIP_LAYOUTS = {"token", "double_faced_token", "emblem", "art_series",
                "vanguard", "scheme", "planar"}
FIELDS = ["scraped_at", "set", "position", "name", "ability_text", "rarity",
          "price_usd", "mana_value", "power", "foil", "url"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def to_int(value):
    """'4' / 4.0 / '' / '*' / None -> int or None."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def short_name(name):
    """Front face of a double-faced or split card."""
    return name.split(" // ")[0].strip()


# --------------------------------------------------------------------------
# Talking to Scryfall
# --------------------------------------------------------------------------
def get_json(url, params=None, retries=3):
    """GET with the required headers, a polite delay, and rate-limit retries.
    Returns None for a 404 (e.g. an unknown set code)."""
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        time.sleep(DELAY)
        if r.status_code == 404:
            return None
        if r.status_code == 429:  # too many requests: back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Scryfall is rate-limiting this client; wait a minute and retry.")


def fetch_set(code):
    """All paper cards of one set (one entry per card name), 175 per request."""
    url = f"{API}/cards/search"
    params = {"q": f"e:{code} game:paper", "unique": "cards", "order": "set"}
    cards = []
    while url:
        data = get_json(url, params)
        if not data:
            return cards
        cards.extend(data.get("data", []))
        url = data.get("next_page") if data.get("has_more") else None
        params = None  # next_page already contains the query
    return cards


def fetch_all():
    """Scryfall's 'oracle cards' bulk file: one printing per unique card."""
    meta = get_json(f"{API}/bulk-data/oracle-cards")
    if not meta:
        raise RuntimeError("Could not find Scryfall's bulk-data listing.")
    print(f"[bulk] downloading {meta['download_uri']} "
          f"(~{meta.get('size', 0) / 1e6:.0f} MB, this can take a minute)...")
    r = requests.get(meta["download_uri"], headers=HEADERS, timeout=600)
    r.raise_for_status()
    return r.json()


def wanted(c):
    if c.get("layout") in SKIP_LAYOUTS:
        return False
    if "paper" not in c.get("games", ["paper"]):
        return False
    if "Basic Land" in (c.get("type_line") or ""):
        return False
    return True


def card_to_row(c, today):
    faces = c.get("card_faces") or [{}]
    text = c.get("oracle_text")
    if text is None:  # double-faced cards keep their text on each face
        text = " // ".join(f.get("oracle_text", "") for f in faces)
    power = c.get("power", faces[0].get("power"))  # front face for DFCs
    return {
        "scraped_at": today,
        "set": c.get("set", ""),
        "position": c.get("collector_number", ""),
        "name": c["name"],
        "ability_text": text,
        "rarity": c.get("rarity", ""),
        "price_usd": (c.get("prices") or {}).get("usd") or "",
        "mana_value": to_int(c.get("cmc")),
        "power": to_int(power),  # None for non-creatures and "*" powers
        "foil": "False",
        "url": c.get("scryfall_uri", ""),
    }


# --------------------------------------------------------------------------
# LLM export: compact, consistent records for a small local model
# --------------------------------------------------------------------------
# (upper price limit in USD, label) -- LLMs read tiers more reliably than decimals
PRICE_TIERS = [(0.50, "bulk"), (2.00, "cheap"), (10.00, "mid"), (50.00, "high")]
HIGH_FROM = 4  # mana value / power of 4 and above -> "high", 3 and below -> "low"


def price_tier(price):
    for limit, label in PRICE_TIERS:
        if price < limit:
            return label
    return "premium"


def level(n):
    return "high" if n >= HIGH_FROM else "low"


# "mana value 4 or greater", "power 2 or less", "mana value 3" ... but not
# "total power 8 or greater" or "base power 1" (those are different ideas)
STAT_RE = re.compile(
    r"(?<!total )(?<!base )\b(mana value|power)\s+(\d+)"
    r"(?:\s+or\s+(greater|more|less|fewer))?\b", re.I)


def translate_stats(text):
    """'power 4 or greater' -> 'high power', 'mana value 3 or less' -> 'low
    mana value'. Only these threshold phrases change; every other number in
    the text (damage, cards drawn, counters, mana symbols) is left alone."""
    def repl(m):
        stat, n = m.group(1), int(m.group(2))
        direction = (m.group(3) or "").lower()
        if direction in ("greater", "more"):
            return f"high {stat}" if n >= HIGH_FROM else m.group(0)
        if direction in ("less", "fewer"):
            return f"low {stat}" if n < HIGH_FROM else m.group(0)
        return f"{level(n)} {stat}"  # exact value, e.g. "with mana value 3"
    return STAT_RE.sub(repl, text)


def normalize_text(name, text, strip_reminder=False, tilde_name=False,
                   stat_words=True):
    """Tidy ability text. By default the wording is kept exactly as scraped;
    only line breaks become ' | '. Optional shortening:
      strip_reminder -> drop reminder text in parentheses
      tilde_name     -> replace the card's own name with ~"""
    text = text or ""
    if strip_reminder:
        text = re.sub(r"\([^)]*\)", "", text)
    if tilde_name:
        full = short_name(name)
        # legendary cards often refer to themselves by the part before the comma
        for variant in sorted({full, full.split(",")[0].strip()}, key=len, reverse=True):
            if len(variant) > 2:
                text = re.sub(rf"\b{re.escape(variant)}\b", "~", text)
    if stat_words:
        text = translate_stats(text)
    text = re.sub(r"\s*\n\s*", " | ", text.strip())  # one line, abilities split by |
    return re.sub(r"\s{2,}", " ", text).strip(" |")


def collector_order(row):
    m = re.match(r"\d+", row.get("position") or "")
    return (row["set"], int(m.group()) if m else 0)


def export_llm(csv_path, out_path, strip_reminder=False, tilde_name=False,
               stat_words=True):
    """Rebuild the LLM files from the whole CSV (latest scrape per card wins)."""
    latest = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("foil") or "").lower() == "true":
                continue
            latest[(row["set"], row["name"])] = row  # newer rows overwrite older

    records, skipped = [], 0
    for row in sorted(latest.values(), key=collector_order):
        try:
            price = float(row["price_usd"])
        except (ValueError, TypeError):
            skipped += 1
            continue
        rarity = (row.get("rarity") or "").strip().lower()
        text = normalize_text(row["name"], row.get("ability_text"),
                              strip_reminder, tilde_name, stat_words)
        if not rarity or not text:  # no rarity, or vanilla/blank text: no signal
            skipped += 1
            continue
        i = len(records) + 1
        tier = price_tier(price)
        name = short_name(row["name"])
        mv, pw = to_int(row.get("mana_value")), to_int(row.get("power"))
        traits = []
        if mv is not None:
            traits.append(f"{level(mv)} mana value" if stat_words else f"mana value {mv}")
        if pw is not None:  # creatures only
            traits.append(f"{level(pw)} power" if stat_words else f"power {pw}")
        header = f"Rarity: {rarity} | Price: ${price:.2f} ({tier})"
        header += "".join(f" | {t}" for t in traits)
        block = f"[{i}] {name}\n{header}\nText: {text}"
        records.append({"id": i, "name": name, "set": row["set"], "rarity": rarity,
                        "price_usd": price, "price_tier": tier, "traits": traits,
                        "text": text, "prompt_block": block})

    out = Path(out_path)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    txt = out.with_suffix(".txt")
    txt.write_text("\n\n".join(r["prompt_block"] for r in records) + "\n",
                   encoding="utf-8")
    print(f"[llm] {len(records)} cards -> {out} and {txt} "
          f"({skipped} skipped: no price, rarity or text)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Scryfall card scanner")
    ap.add_argument("--sets", nargs="*", default=[],
                    help="Scryfall set codes, e.g. fin eoe")
    ap.add_argument("--all", action="store_true",
                    help="every paper card (one printing each) from the bulk file")
    ap.add_argument("--out", default="scryfall_cards.csv")
    ap.add_argument("--append", action="store_true",
                    help="append to the CSV (for daily price-history logging)")
    ap.add_argument("--llm-out", default="scryfall_llm.jsonl",
                    help="LLM-ready output (also writes a .txt next to it); "
                         "empty string disables it")
    ap.add_argument("--export-only", action="store_true",
                    help="skip scanning; rebuild the LLM files from --out")
    ap.add_argument("--strip-reminder", action="store_true",
                    help="LLM export: drop reminder text in parentheses (fewer tokens)")
    ap.add_argument("--tilde-name", action="store_true",
                    help="LLM export: replace each card's own name in its text with ~")
    ap.add_argument("--no-stat-words", action="store_true",
                    help="LLM export: show real numbers instead of high/low for "
                         "mana value and power")
    args = ap.parse_args()
    stat_words = not args.no_stat_words

    if args.export_only:
        if not Path(args.out).exists():
            ap.error(f"{args.out} not found; scan something first")
        export_llm(args.out, args.llm_out or "scryfall_llm.jsonl",
                   args.strip_reminder, args.tilde_name, stat_words)
        return
    if not args.sets and not args.all:
        ap.error("give --sets (set codes) or --all, or use --export-only")

    today = date.today().isoformat()
    rows = []
    if args.all:
        cards = fetch_all()
        rows = [card_to_row(c, today) for c in cards if wanted(c)]
        print(f"[all] {len(rows)} cards kept out of {len(cards)}")
    for code in args.sets:
        code = code.lower()
        cards = fetch_set(code)
        kept = [card_to_row(c, today) for c in cards if wanted(c)]
        if not cards:
            print(f"[set] {code}: nothing found -- check the code at scryfall.com/sets")
        else:
            print(f"[set] {code}: {len(kept)} cards ({len(cards) - len(kept)} skipped)")
        rows.extend(kept)

    if not rows:
        print("No cards collected.")
        return

    path = Path(args.out)
    fields = FIELDS
    appending = args.append and path.exists()
    if appending:
        with path.open(newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), None)
        if existing and existing != fields:
            print("[warn] existing CSV has different columns; new rows follow its "
                  "header. Rename or delete the old file for a clean start.")
            fields = existing
    with path.open("a" if appending else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not appending:
            w.writeheader()
        w.writerows(rows)

    no_price = sum(1 for r in rows if not r["price_usd"])
    no_text = sum(1 for r in rows if not r["ability_text"])
    print(f"\nSaved {len(rows)} cards to {path}")
    print(f"Missing -> price: {no_price}, text: {no_text}")

    if args.llm_out:
        export_llm(args.out, args.llm_out, args.strip_reminder, args.tilde_name,
                   stat_words)


if __name__ == "__main__":
    main()