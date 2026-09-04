#!/usr/bin/env python3
"""
Récupère les menus de cantine publiés sur le site de Cagnac-les-Mines
(généralement 3 à 4 semaines à l'avance), les fait analyser par l'API
Mistral (vision, Pixtral), et met à jour data/menus.json.

Ce fichier JSON est ensuite lu par le plugin TRMNL via une simple URL
statique (jsDelivr / raw.githubusercontent), sans API ni serveur.
"""

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

LISTING_URL = "https://www.cagnac-les-mines.fr/fr/rb/1771616/ecole-117"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "menus.json")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-2603")

# On ne garde en cache que les semaines qui se terminent après cette date
# (on ne réanalyse jamais une semaine déjà passée depuis plus de quelques jours)
KEEP_SINCE = date.today() - timedelta(days=3)

MONTHS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}

WEEK_RE = re.compile(
    r"Semaine\s*n[°o]\s*(\d+)\s*-\s*du\s+(\d{1,2})(?:er)?"
    r"(?:\s+([A-Za-zéûîôàèç]+))?\s+au\s+(\d{1,2})(?:er)?\s+"
    r"([A-Za-zéûîôàèç]+)\s+(\d{4})",
    re.IGNORECASE,
)

DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]

EXTRACTION_PROMPT = """Tu regardes le menu hebdomadaire d'une cantine scolaire française (image d'un PDF, une semaine, colonnes Lundi à Vendredi).

Chaque plat peut porter un ou plusieurs petits badges :
- "fr" : un petit drapeau français (bleu-blanc-rouge) = "Origine France"
- "label" : une pastille orange/dorée avec une médaille/ruban = "Produit labellisé"
- "bio" : une pastille verte avec le texte "BIO"
- "signature" : une pastille orange avec une toque de chef = "Plat signature"

Renvoie UNIQUEMENT du JSON valide, sans texte autour, avec exactement ce schéma :

{
  "lundi": [{"text": "...", "badges": ["fr"]}, ...],
  "mardi": [...],
  "mercredi": [...],
  "jeudi": [...],
  "vendredi": [...]
}

Règles :
- Un jour sans repas (ex: mercredi, pas d'école) doit être un tableau vide [].
- "text" = le nom du plat tel qu'écrit, sans le badge.
- "badges" = tableau vide [] si aucun badge sur ce plat.
- Garde l'ordre des plats tel qu'affiché (entrée, plat, garniture, fromage, dessert...).
- Si deux plats sont proposés en alternative (ex: précédés d'un "/"), garde-les comme deux entrées séparées dans la liste, dans l'ordre.
"""


def fetch_week_links():
    resp = requests.get(LISTING_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    weeks = []
    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True)
        if "semaine" not in label.lower():
            continue
        m = WEEK_RE.search(label)
        if not m:
            continue
        week_num, d1, mon1, d2, mon2, year = m.groups()
        year = int(year)
        mon2_num = MONTHS_FR.get(mon2.lower())
        mon1_num = MONTHS_FR.get(mon1.lower()) if mon1 else mon2_num
        if mon1_num is None or mon2_num is None:
            continue
        start_year = year - 1 if mon1_num > mon2_num else year
        start = date(start_year, mon1_num, int(d1))
        end = date(year, mon2_num, int(d2))

        href = a["href"]
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.cagnac-les-mines.fr" + href

        weeks.append({
            "week_number": int(week_num),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "pdf_url": href,
        })

    # dédoublonne par numéro de semaine, garde la première occurrence
    seen = set()
    unique = []
    for w in weeks:
        if w["week_number"] in seen:
            continue
        seen.add(w["week_number"])
        unique.append(w)
    return unique


def pdf_to_jpeg_b64(pdf_bytes):
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "menu.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        prefix = os.path.join(tmp, "page")
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", "150", "-f", "1", "-l", "1", pdf_path, prefix],
            check=True,
        )
        jpg_files = sorted(
            fn for fn in os.listdir(tmp) if fn.startswith("page") and fn.endswith(".jpg")
        )
        if not jpg_files:
            raise RuntimeError("pdftoppm n'a produit aucune image")
        img_path = os.path.join(tmp, jpg_files[0])

        from PIL import Image
        im = Image.open(img_path)
        if im.height > im.width:
            im = im.rotate(-90, expand=True)
        out_path = os.path.join(tmp, "final.jpg")
        im.save(out_path, "JPEG", quality=90)

        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")


def extract_menu_with_mistral(image_b64):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 2000,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }],
    }
    resp = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    parsed = json.loads(raw)
    for day in DAYS:
        parsed.setdefault(day, [])
    return parsed


def load_existing():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"updated_at": None, "weeks": []}


def save(data):
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    if not MISTRAL_API_KEY:
        print("MISTRAL_API_KEY manquant", file=sys.stderr)
        sys.exit(1)

    store = load_existing()
    by_week_num = {w["week_number"]: w for w in store["weeks"]}

    found = fetch_week_links()
    changed = False

    for w in found:
        end = date.fromisoformat(w["end_date"])
        if end < KEEP_SINCE:
            continue  # semaine trop ancienne, on ne s'en occupe plus

        cached = by_week_num.get(w["week_number"])
        if cached and cached.get("pdf_url") == w["pdf_url"] and cached.get("days"):
            continue  # déjà extrait, rien de neuf

        print(f"Traitement semaine {w['week_number']} ({w['start_date']} -> {w['end_date']})")
        try:
            pdf_bytes = requests.get(w["pdf_url"], timeout=30).content
            image_b64 = pdf_to_jpeg_b64(pdf_bytes)
            days = extract_menu_with_mistral(image_b64)
        except Exception as exc:  # on ne casse pas tout pour une semaine en erreur
            print(f"  Échec extraction semaine {w['week_number']}: {exc}", file=sys.stderr)
            continue

        by_week_num[w["week_number"]] = {
            "week_number": w["week_number"],
            "start_date": w["start_date"],
            "end_date": w["end_date"],
            "pdf_url": w["pdf_url"],
            "days": days,
        }
        changed = True

    # purge les semaines trop anciennes du cache
    kept = {
        num: w for num, w in by_week_num.items()
        if date.fromisoformat(w["end_date"]) >= KEEP_SINCE
    }
    if len(kept) != len(by_week_num):
        changed = True

    store["weeks"] = sorted(kept.values(), key=lambda w: w["start_date"])

    if changed:
        save(store)
        print("data/menus.json mis à jour")
    else:
        print("Rien de nouveau")


if __name__ == "__main__":
    main()
