import requests
import psycopg2
from bs4 import BeautifulSoup
import time
import re
import hashlib


# Shared session so the anti-bot cookie is reused across all requests
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
})


def _solve_challenge(html):
    """UFCStats serves a JS proof-of-work page instead of real content.
    Reproduce it in Python: find n so sha256(nonce:n) starts with N zeros,
    POST it to /__c to earn the access cookie."""
    nonce_match = re.search(r'nonce\s*=\s*"([0-9a-f]+)"', html)
    tlen_match = re.search(r'new Array\((\d+)\+1\)\.join\(\'0\'\)', html)
    if not nonce_match or not tlen_match:
        return False

    nonce = nonce_match.group(1)
    target = "0" * int(tlen_match.group(1))

    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1

    resp = SESSION.post(
        f"{BASE_URL}/__c",
        data={"nonce": nonce, "n": n},
        timeout=15,
    )
    return 200 <= resp.status_code < 300


def fetch(url, tries=3):
    """GET a URL through the shared session, transparently solving the
    anti-bot challenge (once per session) if it appears."""
    for attempt in range(tries):
        try:
            response = SESSION.get(url, timeout=15)
        except Exception as e:
            print(f"  Request error, retrying ({attempt + 1}/{tries}): {e}")
            time.sleep(3)
            continue

        if "Checking your browser" in response.text:
            if _solve_challenge(response.text):
                continue  # cookie earned, retry the real request
            print("  Failed to solve anti-bot challenge")
            time.sleep(3)
            continue

        return response

    return None


# Database connection


def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ufc_analytics",
        user="postgres",
        password="ufc123",
        port="5432"
    )

# Quick test to make sure we can connect
if __name__ == "__main__":
    try:
        conn = get_connection()
        print("Connected to ufc_analytics successfully!")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")


#  Get list of recent events from ufcstats.com


BASE_URL = "http://ufcstats.com"

def get_event_urls(limit=300):
    print("Fetching event list...")
    events = []
    page = 1

    while len(events) < limit:
        url = f"{BASE_URL}/statistics/events/completed?page={page}"
        response = fetch(url)
        if response is None:
            print(f"  Failed to fetch page {page}, stopping")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("tr.b-statistics__table-row")

        found_any = False
        for row in rows:
            link = row.select_one("a.b-link_style_black")
            date_tag = row.select_one("span.b-statistics__date")
            if link:
                events.append({
                    "name": link.text.strip(),
                    "url":  link["href"],
                    "date": date_tag.text.strip() if date_tag else "January 1, 2024"
                })
                found_any = True
            if len(events) >= limit:
                break

        if not found_any:
            break

        page += 1
        time.sleep(1)

    print(f"Found {len(events)} events")
    return events


#  Get all fight URLs from a single event page


def get_fight_urls(event_url):
    response = fetch(event_url)
    if response is None:
        print(f"  Skipping event after 3 failed attempts")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    fights = []
    rows = soup.select("tr.b-fight-details__table-row")

    for row in rows:
        link = row.get("data-link")
        if link:
            fights.append(link)

    print(f"  Found {len(fights)} fights")
    return fights



# Scrape round-by-round stats from a single fight page


def get_fight_details(fight_url):
    response = fetch(fight_url)
    if response is None:
        print(f"    Skipping fight after 3 failed attempts")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    fight = {}

    # --- Fighter names ---
    fighters = soup.select("a.b-fight-details__person-link")
    if len(fighters) < 2:
        return None
    fight["fighter1"] = fighters[0].text.strip()
    fight["fighter2"] = fighters[1].text.strip()
    fight["fighter1_url"] = fighters[0].get("href")
    fight["fighter2_url"] = fighters[1].get("href")

    # --- Winner ---
    fight["winner"] = None
    statuses = soup.select("i.b-fight-details__person-status")
    for i, status in enumerate(statuses):
        if "style_green" in status.get("class", []) or status.text.strip() == "W":
            fight["winner"] = fight["fighter1"] if i == 0 else fight["fighter2"]
            break

    # --- Win method ---
    method_tag = soup.select_one("i.b-fight-details__text-item_first i[style='font-style: normal']")
    fight["win_method"] = method_tag.text.strip() if method_tag else None

    # --- Round, time, weight class, scheduled rounds ---
    fight["win_round"]        = None
    fight["win_time"]         = None
    fight["weight_class"]     = None
    fight["scheduled_rounds"] = None

    for item in soup.select("i.b-fight-details__text-item"):
        label = item.select_one("i.b-fight-details__label")
        if not label:
            continue
        label_text = label.text.strip()
        full_text  = item.text.replace(label_text, "").strip()
        if "Round:" in label_text:
            try:
                fight["win_round"] = int(full_text)
            except:
                pass
        elif "Time:" in label_text and "Time format:" not in label_text:
            fight["win_time"] = full_text
        elif "Time format:" in label_text:
            # e.g. "5 Rnd (5-5-5-5-5)" or "3 Rnd (5-5-5)" — the leading number
            # is the SCHEDULED rounds (independent of how the fight ended).
            m = re.search(r"(\d+)\s*Rnd", full_text)
            if m:
                fight["scheduled_rounds"] = int(m.group(1))
        elif "Weight class:" in label_text:
            fight["weight_class"] = full_text

    # Weight class fallback from fight title 
    if not fight["weight_class"]:
        title = soup.select_one("i.b-fight-details__fight-title")
        if title:
            fight["weight_class"] = title.text.strip().replace("Bout", "").strip()

    #  Round-by-round stats 
    # The per round table is the FIRST js-fight-table on the page
    fight["rounds"] = []

    def parse_of(val):
        parts = val.strip().split(" of ")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except:
                return 0, 0
        return 0, 0

    def ctrl_to_seconds(val):
        val = val.strip()
        if ":" in val:
            try:
                m, s = val.split(":")
                return int(m) * 60 + int(s)
            except:
                return 0
        return 0

    round_tables = soup.select("table.b-fight-details__table.js-fight-table")
    if not round_tables:
        return fight

    # First table = per-round totals (KD, sig str, total str, TD, sub, ctrl)
    per_round_table = round_tables[0]
    rows = per_round_table.select("tr.b-fight-details__table-row")

    round_num = 0
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if not cols:
            continue

        # Each col has 2 <p> tags — one per fighter
        def get_both(col_index):
            ps = cols[col_index].select("p.b-fight-details__table-text")
            if len(ps) >= 2:
                return ps[0].text.strip(), ps[1].text.strip()
            return "0", "0"

        round_num += 1

        kd1,    kd2    = get_both(1)
        sig1,   sig2   = get_both(2)
        total1, total2 = get_both(4)
        td1,    td2    = get_both(5)
        sub1,   sub2   = get_both(7)
        rev1,   rev2   = get_both(8)
        ctrl1,  ctrl2  = get_both(9)

        sig_l1,   sig_a1   = parse_of(sig1)
        sig_l2,   sig_a2   = parse_of(sig2)
        total_l1, total_a1 = parse_of(total1)
        total_l2, total_a2 = parse_of(total2)
        td_l1,    td_a1    = parse_of(td1)
        td_l2,    td_a2    = parse_of(td2)

        fight["rounds"].append({
            "round_number":            round_num,
            "fighter":                 fight["fighter1"],
            "knockdowns":              int(kd1) if kd1.isdigit() else 0,
            "sig_strikes_landed":      sig_l1,
            "sig_strikes_attempted":   sig_a1,
            "total_strikes_landed":    total_l1,
            "total_strikes_attempted": total_a1,
            "takedowns_landed":        td_l1,
            "takedowns_attempted":     td_a1,
            "submission_attempts":     int(sub1) if sub1.isdigit() else 0,
            "reversals":               int(rev1) if rev1.isdigit() else 0,
            "ctrl_time_seconds":       ctrl_to_seconds(ctrl1),
        })
        fight["rounds"].append({
            "round_number":            round_num,
            "fighter":                 fight["fighter2"],
            "knockdowns":              int(kd2) if kd2.isdigit() else 0,
            "sig_strikes_landed":      sig_l2,
            "sig_strikes_attempted":   sig_a2,
            "total_strikes_landed":    total_l2,
            "total_strikes_attempted": total_a2,
            "takedowns_landed":        td_l2,
            "takedowns_attempted":     td_a2,
            "submission_attempts":     int(sub2) if sub2.isdigit() else 0,
            "reversals":               int(rev2) if rev2.isdigit() else 0,
            "ctrl_time_seconds":       ctrl_to_seconds(ctrl2),
        })

    # --- Second table = significant strikes by target (head / body / leg) ---
    # Columns: 0 Fighter, 1 Sig.str, 2 Sig.str%, 3 Head, 4 Body, 5 Leg, ...
    # Merge these into the round dicts already built from the totals table.
    if len(round_tables) > 1:
        sig_rows = round_tables[1].select("tr.b-fight-details__table-row")
        round_num = 0
        for row in sig_rows:
            cols = row.select("td.b-fight-details__table-col")
            if not cols:
                continue

            def get_both2(col_index):
                ps = cols[col_index].select("p.b-fight-details__table-text")
                if len(ps) >= 2:
                    return ps[0].text.strip(), ps[1].text.strip()
                return "0 of 0", "0 of 0"

            round_num += 1
            head1, head2 = get_both2(3)
            body1, body2 = get_both2(4)
            leg1,  leg2  = get_both2(5)

            targets = {
                fight["fighter1"]: (parse_of(head1), parse_of(body1), parse_of(leg1)),
                fight["fighter2"]: (parse_of(head2), parse_of(body2), parse_of(leg2)),
            }
            for r in fight["rounds"]:
                if r["round_number"] != round_num:
                    continue
                (hl, ha), (bl, ba), (ll, la) = targets[r["fighter"]]
                r["head_strikes_landed"],    r["head_strikes_attempted"] = hl, ha
                r["body_strikes_landed"],    r["body_strikes_attempted"] = bl, ba
                r["leg_strikes_landed"],     r["leg_strikes_attempted"]  = ll, la

    return fight


# Parse fighter physicals (height / reach / stance / DOB) from a profile page


def parse_height_to_inches(text):
    """'5' 9\"' -> 69.0 (inches). Returns None if unparseable."""
    m = re.search(r"(\d+)'\s*(\d+)", text)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return None


def parse_reach_to_inches(text):
    """'74\"' -> 74.0. Returns None if unparseable/missing ('--')."""
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def parse_dob(text):
    """'Jul 14, 1988' -> date. Returns None if missing ('--')."""
    from datetime import datetime
    text = text.strip()
    try:
        return datetime.strptime(text, "%b %d, %Y").date()
    except Exception:
        return None


def get_fighter_profile(fighter_url):
    """Scrape height / reach / stance / DOB from a fighter-details page."""
    response = fetch(fighter_url)
    if response is None:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    profile = {"height_in": None, "reach_in": None, "stance": None, "dob": None}

    for li in soup.select("li.b-list__box-list-item"):
        txt = li.get_text(" ", strip=True)
        if txt.startswith("Height:"):
            profile["height_in"] = parse_height_to_inches(txt)
        elif txt.startswith("Reach:"):
            profile["reach_in"] = parse_reach_to_inches(txt.replace("Reach:", ""))
        elif txt.upper().startswith("STANCE:"):
            stance = txt.split(":", 1)[1].strip()
            profile["stance"] = stance or None
        elif txt.startswith("DOB:"):
            profile["dob"] = parse_dob(txt.split(":", 1)[1])

    return profile


# Insert scraped fight data into PostgreSQL


def upsert_fighter_profile(conn, fighter_id, profile):
    """Fill in a fighter's physicals if we scraped any (never overwrite with NULL)."""
    if not profile or not any(profile.values()):
        return
    cur = conn.cursor()
    cur.execute("""
        UPDATE fighters
        SET height_in = COALESCE(%s, height_in),
            reach_in  = COALESCE(%s, reach_in),
            stance    = COALESCE(%s, stance),
            dob       = COALESCE(%s, dob)
        WHERE fighter_id = %s
    """, (profile.get("height_in"), profile.get("reach_in"),
          profile.get("stance"), profile.get("dob"), fighter_id))
    conn.commit()
    cur.close()


def event_exists(conn, event_name):
    """True if we've already scraped this event (used to skip re-fetching)."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM events WHERE event_name = %s", (event_name,))
    found = cur.fetchone() is not None
    cur.close()
    return found


def insert_fight(conn, event_name, event_date, fight):
    cur = conn.cursor()

    # Parse date string like "March 01, 2025" into a proper date
    from datetime import datetime
    try:
        parsed_date = datetime.strptime(event_date, "%B %d, %Y").date()
    except:
        parsed_date = datetime(2024, 1, 1).date()

    # --- Insert event if it doesn't exist yet ---
    cur.execute("""
        INSERT INTO events (event_name, event_date, card_type)
        VALUES (%s, %s, 'Fight Night')
        ON CONFLICT (event_name) DO NOTHING
        RETURNING event_id
    """, (event_name, parsed_date))
    row = cur.fetchone()
    if row:
        event_id = row[0]
    else:
        cur.execute("SELECT event_id FROM events WHERE event_name = %s", (event_name,))
        event_id = cur.fetchone()[0]

    # --- Insert fighters if they don't exist yet ---
    def get_or_create_fighter(name):
        cur.execute("SELECT fighter_id FROM fighters WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("""
            INSERT INTO fighters (name, weight_class)
            VALUES (%s, %s)
            ON CONFLICT (name) DO NOTHING
            RETURNING fighter_id
        """, (name, fight.get("weight_class") or "Unknown"))
        row = cur.fetchone()
        if row:
            return row[0]
        # Lost a race / already inserted this run — fetch the existing id
        cur.execute("SELECT fighter_id FROM fighters WHERE name = %s", (name,))
        return cur.fetchone()[0]

    f1_id = get_or_create_fighter(fight["fighter1"])
    f2_id = get_or_create_fighter(fight["fighter2"])

    # --- Resolve winner_id ---
    winner_id = None
    if fight.get("winner") == fight["fighter1"]:
        winner_id = f1_id
    elif fight.get("winner") == fight["fighter2"]:
        winner_id = f2_id

    # --- Insert the fight (skip cleanly if it already exists) ---
    cur.execute("""
        INSERT INTO fights (event_id, fighter1_id, fighter2_id, winner_id,
                            weight_class, win_method, win_round, win_time,
                            scheduled_rounds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id, fighter1_id, fighter2_id) DO NOTHING
        RETURNING fight_id
    """, (
        event_id, f1_id, f2_id, winner_id,
        fight.get("weight_class") or "Unknown",
        fight.get("win_method"),
        fight.get("win_round"),
        fight.get("win_time"),
        fight.get("scheduled_rounds") or (5 if fight.get("win_round") == 5 else 3)
    ))
    row = cur.fetchone()
    if row is None:
        print(f"    Skipped (already exists): {fight['fighter1']} vs {fight['fighter2']}")
        conn.commit()
        cur.close()
        return
    fight_id = row[0]

    # --- Insert round stats ---
    for r in fight["rounds"]:
        fighter_id = f1_id if r["fighter"] == fight["fighter1"] else f2_id
        cur.execute("""
            INSERT INTO round_stats (
                fight_id, fighter_id, round_number,
                sig_strikes_landed, sig_strikes_attempted,
                total_strikes_landed, total_strikes_attempted,
                head_strikes_landed, head_strikes_attempted,
                body_strikes_landed, body_strikes_attempted,
                leg_strikes_landed, leg_strikes_attempted,
                takedowns_landed, takedowns_attempted,
                submission_attempts, reversals, ctrl_time_seconds, knockdowns
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            fight_id, fighter_id, r["round_number"],
            r["sig_strikes_landed"],           r["sig_strikes_attempted"],
            r["total_strikes_landed"],         r["total_strikes_attempted"],
            r.get("head_strikes_landed", 0),   r.get("head_strikes_attempted", 0),
            r.get("body_strikes_landed", 0),   r.get("body_strikes_attempted", 0),
            r.get("leg_strikes_landed", 0),    r.get("leg_strikes_attempted", 0),
            r["takedowns_landed"],             r["takedowns_attempted"],
            r["submission_attempts"],          r.get("reversals", 0),
            r["ctrl_time_seconds"],            r["knockdowns"]
        ))

    conn.commit()

    # --- Fill in fighter physicals from their profile pages (once per fighter) ---
    for fid, furl in ((f1_id, fight.get("fighter1_url")),
                      (f2_id, fight.get("fighter2_url"))):
        if not furl:
            continue
        cur.execute("SELECT reach_in, height_in FROM fighters WHERE fighter_id = %s", (fid,))
        reach, height = cur.fetchone()
        if reach is None and height is None:          # not filled in yet
            upsert_fighter_profile(conn, fid, get_fighter_profile(furl))

    cur.close()
    print(f"    Inserted: {fight['fighter1']} vs {fight['fighter2']} | Winner: {fight.get('winner')} | Date: {parsed_date}")


if __name__ == "__main__":
    try:
        conn = get_connection()
        print("Connected to ufc_analytics successfully!")

        events = get_event_urls(limit=300)

        # Events are listed newest-first. Everything we've already scraped sits
        # at the bottom of the list, so once we hit a run of events already in
        # the DB we can stop paging through old history.
        STOP_AFTER_CONSECUTIVE_EXISTING = 5
        consecutive_existing = 0
        new_events = 0

        for e in events:
            if event_exists(conn, e["name"]):
                consecutive_existing += 1
                print(f"\nEvent: {e['name']} — already in DB, skipping")
                if consecutive_existing >= STOP_AFTER_CONSECUTIVE_EXISTING:
                    print("\nReached already scraped events now stopping.")
                    break
                continue

            consecutive_existing = 0
            print(f"\nEvent: {e['name']} — {e['date']}")
            fight_urls = get_fight_urls(e["url"])
            if not fight_urls:
                print("  Skipping event, no fights found (likely upcoming)")
                continue

            inserted_any = False
            for url in fight_urls:
                fight = get_fight_details(url)
                if fight and fight["rounds"]:
                    insert_fight(conn, e["name"], e["date"], fight)
                    inserted_any = True
                time.sleep(1)
            if inserted_any:
                new_events += 1
            time.sleep(2)

        conn.close()
        print(f"\nDone! Scraped {new_events} new event(s).")

    except Exception as e:
        import traceback
        traceback.print_exc()
