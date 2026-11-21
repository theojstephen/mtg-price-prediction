#!/usr/bin/env python3
"""
MTGGoldfish scanner  --  collects ability text, rarity and price (USD) per card.

Pipeline
  1. Playwright loads an MTGGoldfish set page (real browser, so JS-rendered
     tables work).
  2. BeautifulSoup parses the rows: card name, rarity, paper price.
  3. Ability text comes from Scryfall (default, fast) or from each MTGGoldfish
     card page (--text-source goldfish, slow: one page load per card).
  4. Rows are written to CSV with a scrape date, so running it daily builds
     the price history needed for the prediction model.

Setup
  pip install playwright beautifulsoup4 requests
  playwright install chromium

Usage
  python mtg_goldfish_scanner.py --sets Aether+Revolt Kaladesh --out cards.csv
  python mtg_goldfish_scanner.py --sets Aether+Revolt --debug        # saves HTML
  python mtg_goldfish_scanner.py --sets Kaladesh --resume            # skip finished sets
  python mtg_goldfish_scanner.py --export-only --out cards.csv       # rebuild LLM files
  (every scrape also writes cards_llm.jsonl and cards_llm.txt for the LLM)
  python mtg_goldfish_scanner.py --sets AER --url-template "https://www.mtggoldfish.com/index/{set}"

IMPORTANT: the CSS/HTML structure of MTGGoldfish was not verified against the
live site when this was written. The row parser is deliberately structure-
agnostic (it looks for rows containing a /price/ link, then finds rarity and
price by pattern), but if a run returns 0 cards use --debug and inspect the
saved HTML in ./debug/ to adjust the selectors.

Be polite: check MTGGoldfish's robots.txt and terms of service, keep the
delay between page loads, and do not try to bypass a block. If the site shows
a Cloudflare challenge this script stops instead of evading it; use Scryfall's
bulk data for prices in that case.
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = "https://www.mtggoldfish.com"
UA = "mtg-price-project/0.1 (personal research; replace-with-your-email)"
DEFAULT_TEMPLATE = BASE + "/sets/{set}"

RARITY_RE = re.compile(r"\b(mythic|rare|uncommon|common)\b", re.I)
PRICE_RE = re.compile(r"\$?\s*([\d,]+\.\d{2})")
BLOCK_MARKERS = ("just a moment", "attention required", "cf-chl", "cf-challenge")
TEXT_SELECTORS = [".card-text", ".card-oracle-text", ".card-text-container",
                  "[class*='oracle']"]


class Blocked(Exception):
    """Raised when the site serves a bot challenge instead of content."""


# --------------------------------------------------------------------------
# Politeness: robots.txt
# --------------------------------------------------------------------------
def load_robots():
    rp = robotparser.RobotFileParser()
    try:
        r = requests.get(BASE + "/robots.txt", headers={"User-Agent": UA}, timeout=15)
        rp.parse(r.text.splitlines())
        return rp
    except Exception as exc:  # network error, block, etc.
        print(f"[warn] could not read robots.txt ({exc}); check it manually.",
              file=sys.stderr)
        return None


def allowed(rp, url):
    return True if rp is None else rp.can_fetch(UA, url)


# --------------------------------------------------------------------------
# Fetching with Playwright
# --------------------------------------------------------------------------
def fetch_html(page, url, wait_selector="table", debug_name=None):
    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    try:
        page.wait_for_selector(wait_selector, timeout=15_000)
    except PWTimeout:
        pass  # parse whatever loaded; the caller handles empty results
    html = page.content()
    if debug_name:
        Path("debug").mkdir(exist_ok=True)
        Path("debug", debug_name).write_text(html, encoding="utf-8")
    low = html.lower()
    if any(marker in low for marker in BLOCK_MARKERS):
        raise Blocked(f"Bot challenge detected at {url}")
    return html


def polite_sleep(lo=3.0, hi=6.0):
    time.sleep(random.uniform(lo, hi))


# --------------------------------------------------------------------------
# Parsing with BeautifulSoup
# --------------------------------------------------------------------------
def find_rarity(tr):
    """Find rarity in a table row: whole-cell text first, then attributes."""
    for td in tr.find_all(["td", "th"]):
        txt = td.get_text(" ", strip=True)
        if txt.lower() in {"common", "uncommon", "rare", "mythic", "mythic rare"}:
            return txt.lower().replace(" rare", "")
    for el in [tr] + tr.find_all(True):
        for key in ("class", "title", "alt", "data-rarity", "data-original-title"):
            val = el.get(key)
            if not val:
                continue
            val = " ".join(val) if isinstance(val, list) else val
            m = RARITY_RE.search(val)
            if m:
                return m.group(1).lower()
    return None


def find_price(tr):
    """First cell that is purely a dollar amount = current price."""
    for td in tr.find_all("td"):
        m = re.fullmatch(r"\$?\s*([\d,]+\.\d{2})", td.get_text(strip=True))
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def parse_listing(html, set_label):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    paper = [t for t in tables
             if "paper" in (" ".join(t.get("class", [])) + (t.get("id") or "")).lower()]
    scope = paper or tables or [soup]

    cards, seen = [], set()
    position = 0  # row order on the page, starting at 1
    for table in scope:
        for tr in table.find_all("tr"):
            link = tr.select_one('a[href*="/price/"]')
            if not link:
                continue
            name = link.get_text(" ", strip=True)
            href = urljoin(BASE, link["href"].split("#")[0])
            foil = "foil" in href.lower() or "foil" in name.lower()
            key = (name, foil, href)
            if not name or key in seen:
                continue
            seen.add(key)
            position += 1
            cards.append({
                "position": position,
                "name": name,
                "rarity": find_rarity(tr),
                "price_usd": find_price(tr),
                "foil": foil,
                "set": set_label,
                "url": href,
                "ability_text": "",
            })
    return cards


def parse_card_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for sel in TEXT_SELECTORS:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text("\n", strip=True)
    return ""


# --------------------------------------------------------------------------
# Ability text sources
# --------------------------------------------------------------------------
def clean_name(name):
    """Drop variant suffixes like '(Showcase)' and use the front face."""
    name = re.sub(r"\s*\(.*?\)", "", name).strip()
    return name.split(" // ")[0].strip()


def scryfall_texts(names):
    """Batch lookup, 75 names per request, as Scryfall allows."""
    headers = {"User-Agent": UA, "Accept": "application/json"}
    uniq = sorted({clean_name(n) for n in names})
    out = {}
    for i in range(0, len(uniq), 75):
        batch = uniq[i:i + 75]
        r = requests.post("https://api.scryfall.com/cards/collection",
                          json={"identifiers": [{"name": n} for n in batch]},
                          headers=headers, timeout=30)
        r.raise_for_status()
        for c in r.json().get("data", []):
            text = c.get("oracle_text")
            if text is None:  # double-faced cards
                text = " // ".join(f.get("oracle_text", "")
                                   for f in c.get("card_faces", []))
            out[c["name"].split(" // ")[0]] = text
        time.sleep(0.2)
    return out


# --------------------------------------------------------------------------
# LLM export: compact, consistent records for a small local model (Qwen etc.)
# --------------------------------------------------------------------------
# (upper price limit in USD, label) -- LLMs read tiers more reliably than decimals
PRICE_TIERS = [(0.50, "bulk"), (2.00, "cheap"), (10.00, "mid"), (50.00, "high")]


def price_tier(price):
    for limit, label in PRICE_TIERS:
        if price < limit:
            return label
    return "premium"


def normalize_text(name, text, strip_reminder=False, tilde_name=False):
    """Tidy ability text. By default the wording is kept exactly as scraped;
    only line breaks become ' | '. Optional shortening:
      strip_reminder -> drop reminder text in parentheses
      tilde_name     -> replace the card's own name with ~"""
    text = text or ""
    if strip_reminder:
        text = re.sub(r"\([^)]*\)", "", text)
    if tilde_name:
        full = clean_name(name)
        # legendary cards often refer to themselves by the part before the comma
        for variant in sorted({full, full.split(",")[0].strip()}, key=len, reverse=True):
            if len(variant) > 2:
                text = re.sub(rf"\b{re.escape(variant)}\b", "~", text)
    text = re.sub(r"\s*\n\s*", " | ", text.strip())  # one line, abilities split by |
    return re.sub(r"\s{2,}", " ", text).strip(" |")


def export_llm(csv_path, out_path, strip_reminder=False, tilde_name=False):
    """Rebuild the LLM files from the whole CSV (latest scrape per card wins)."""
    latest = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("foil") or "").lower() == "true":
                continue  # foil rows duplicate the card with a different price
            latest[(row["set"], row["name"])] = row  # newer rows overwrite older

    def order(row):
        try:
            return (row["set"], int(row.get("position") or 0))
        except ValueError:
            return (row["set"], 0)

    records, skipped = [], 0
    for row in sorted(latest.values(), key=order):
        try:
            price = float(row["price_usd"])
        except (ValueError, TypeError):
            skipped += 1
            continue
        rarity = (row.get("rarity") or "").strip().lower()
        text = normalize_text(row["name"], row.get("ability_text"),
                              strip_reminder, tilde_name)
        if not rarity or not text:  # no rarity, or vanilla/blank text: no signal
            skipped += 1
            continue
        i = len(records) + 1
        tier = price_tier(price)
        name = clean_name(row["name"])
        block = (f"[{i}] {name}\n"
                 f"Rarity: {rarity} | Price: ${price:.2f} ({tier})\n"
                 f"Text: {text}")
        records.append({"id": i, "name": name, "set": row["set"], "rarity": rarity,
                        "price_usd": price, "price_tier": tier, "text": text,
                        "prompt_block": block})

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
    ap = argparse.ArgumentParser(description="MTGGoldfish card scanner")
    ap.add_argument("--sets", nargs="*", default=[],
                    help="set identifiers as used in the URL, e.g. Aether+Revolt")
    ap.add_argument("--llm-out", default="cards_llm.jsonl",
                    help="LLM-ready output (also writes a .txt next to it); "
                         "empty string disables it")
    ap.add_argument("--export-only", action="store_true",
                    help="skip scraping; rebuild the LLM files from --out")
    ap.add_argument("--strip-reminder", action="store_true",
                    help="LLM export: drop reminder text in parentheses (fewer tokens)")
    ap.add_argument("--tilde-name", action="store_true",
                    help="LLM export: replace each card's own name in its text with ~")
    ap.add_argument("--url-template", default=DEFAULT_TEMPLATE,
                    help="listing URL with a {set} placeholder")
    ap.add_argument("--text-source", choices=["scryfall", "goldfish"],
                    default="scryfall")
    ap.add_argument("--out", default="cards.csv")
    ap.add_argument("--append", action="store_true",
                    help="append to the CSV (for daily price-history logging)")
    ap.add_argument("--resume", action="store_true",
                    help="skip sets already present in --out and append the rest")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--debug", action="store_true", help="save fetched HTML to ./debug")
    args = ap.parse_args()

    if args.export_only:
        if not Path(args.out).exists():
            ap.error(f"{args.out} not found; scrape something first")
        export_llm(args.out, args.llm_out or "cards_llm.jsonl",
                   args.strip_reminder, args.tilde_name)
        return
    if not args.sets:
        ap.error("--sets is required (unless you use --export-only)")

    # --resume: skip sets already saved in the output file, then append
    if args.resume:
        args.append = True
        out_path = Path(args.out)
        if out_path.exists():
            with out_path.open(newline="", encoding="utf-8") as f:
                done = {row.get("set") for row in csv.DictReader(f)}
            skipped = [s for s in args.sets if s in done]
            args.sets = [s for s in args.sets if s not in done]
            if skipped:
                print(f"[resume] already in {args.out}, skipping: {', '.join(skipped)}")
        if not args.sets:
            print("[resume] nothing left to do.")
            return

    rp = load_robots()
    all_cards = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(user_agent=UA, locale="en-US")
        page = ctx.new_page()

        try:
            for s in args.sets:
                url = args.url_template.format(set=s)
                if not allowed(rp, url):
                    print(f"[skip] robots.txt disallows {url}")
                    continue
                print(f"[set] {s} -> {url}")
                html = fetch_html(page, url, debug_name=f"set_{s}.html" if args.debug else None)
                cards = parse_listing(html, s)
                print(f"      parsed {len(cards)} rows")
                if not cards:
                    print("      0 rows: run with --debug and check the saved HTML.")
                all_cards.extend(cards)
                polite_sleep()

            if args.text_source == "goldfish":
                for i, c in enumerate(all_cards, 1):
                    if c["foil"] or not allowed(rp, c["url"]):
                        continue
                    print(f"[text] {i}/{len(all_cards)} {c['name']}")
                    try:
                        html = fetch_html(page, c["url"], wait_selector="body")
                        c["ability_text"] = parse_card_text(html)
                    except Blocked:
                        raise
                    except Exception as exc:
                        print(f"      failed: {exc}")
                    polite_sleep()
        except Blocked as exc:
            print(f"\n[stopped] {exc}\nNot attempting to bypass it. Try again later, "
                  "slow down, or use Scryfall bulk data for prices.")
        finally:
            browser.close()

    if args.text_source == "scryfall" and all_cards:
        print("[text] fetching ability text from Scryfall...")
        texts = scryfall_texts([c["name"] for c in all_cards])
        for c in all_cards:
            c["ability_text"] = texts.get(clean_name(c["name"]), "")

    if not all_cards:
        print("No cards collected.")
        return

    today = date.today().isoformat()
    fields = ["scraped_at", "set", "position", "name", "ability_text", "rarity",
              "price_usd", "foil", "url"]
    path = Path(args.out)
    appending = args.append and path.exists()
    if appending:
        # keep the existing header so old and new rows line up
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
        for c in all_cards:
            w.writerow({"scraped_at": today, **c})

    missing_rarity = sum(1 for c in all_cards if not c["rarity"])
    missing_price = sum(1 for c in all_cards if c["price_usd"] is None)
    missing_text = sum(1 for c in all_cards if not c["ability_text"])
    print(f"\nSaved {len(all_cards)} cards to {path}")
    print(f"Missing -> rarity: {missing_rarity}, price: {missing_price}, "
          f"text: {missing_text}")

    if args.llm_out:
        export_llm(args.out, args.llm_out, args.strip_reminder, args.tilde_name)


if __name__ == "__main__":
    main()