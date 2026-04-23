import requests
import psycopg2
from bs4 import BeautifulSoup
import time


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
        try:
            response = requests.get(url, timeout=15)
        except Exception as e:
            print(f"  Timeout on page {page}, stopping: {e}")
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
    for attempt in range(3):
        try:
            response = requests.get(event_url, timeout=15)
            break
        except Exception as e:
            print(f"  Timeout on event page, retrying ({attempt+1}/3)...")
            time.sleep(3)
    else:
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
    for attempt in range(3):
        try:
            response = requests.get(fight_url, timeout=15)
            break
        except Exception as e:
            print(f"    Timeout on fight page, retrying ({attempt+1}/3)...")
            time.sleep(3)
    else:
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

    # --- Round, time, weight class ---
    fight["win_round"]    = None
    fight["win_time"]     = None
    fight["weight_class"] = None

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
        elif "Time:" in label_text:
            fight["win_time"] = full_text
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
            "ctrl_time_seconds":       ctrl_to_seconds(ctrl2),
        })

    return fight


# Insert scraped fight data into PostgreSQL


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
        ON CONFLICT DO NOTHING
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
            RETURNING fighter_id
        """, (name, fight.get("weight_class", "Unknown")))
        return cur.fetchone()[0]

    f1_id = get_or_create_fighter(fight["fighter1"])
    f2_id = get_or_create_fighter(fight["fighter2"])

    # --- Resolve winner_id ---
    winner_id = None
    if fight.get("winner") == fight["fighter1"]:
        winner_id = f1_id
    elif fight.get("winner") == fight["fighter2"]:
        winner_id = f2_id

    # --- Skip if this fight already exists ---
    cur.execute("""
        SELECT fight_id FROM fights
        WHERE fighter1_id = %s AND fighter2_id = %s AND event_id = %s
    """, (f1_id, f2_id, event_id))
    if cur.fetchone():
        print(f"    Skipped (already exists): {fight['fighter1']} vs {fight['fighter2']}")
        cur.close()
        return

    # --- Insert the fight ---
    cur.execute("""
        INSERT INTO fights (event_id, fighter1_id, fighter2_id, winner_id,
                            weight_class, win_method, win_round, win_time,
                            scheduled_rounds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING fight_id
    """, (
        event_id, f1_id, f2_id, winner_id,
        fight.get("weight_class"),
        fight.get("win_method"),
        fight.get("win_round"),
        fight.get("win_time"),
        5 if fight.get("win_round") == 5 else 3
    ))
    fight_id = cur.fetchone()[0]

    # --- Insert round stats ---
    for r in fight["rounds"]:
        fighter_id = f1_id if r["fighter"] == fight["fighter1"] else f2_id
        cur.execute("""
            INSERT INTO round_stats (
                fight_id, fighter_id, round_number,
                sig_strikes_landed, sig_strikes_attempted,
                total_strikes_landed, total_strikes_attempted,
                takedowns_landed, takedowns_attempted,
                submission_attempts, ctrl_time_seconds, knockdowns
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            fight_id, fighter_id, r["round_number"],
            r["sig_strikes_landed"],    r["sig_strikes_attempted"],
            r["total_strikes_landed"],  r["total_strikes_attempted"],
            r["takedowns_landed"],      r["takedowns_attempted"],
            r["submission_attempts"],   r["ctrl_time_seconds"],
            r["knockdowns"]
        ))

    conn.commit()
    cur.close()
    print(f"    Inserted: {fight['fighter1']} vs {fight['fighter2']} | Winner: {fight.get('winner')} | Date: {parsed_date}")


if __name__ == "__main__":
    try:
        conn = get_connection()
        print("Connected to ufc_analytics successfully!")

        events = get_event_urls(limit=300)
        for e in events:
            print(f"\nEvent: {e['name']} — {e['date']}")
            fight_urls = get_fight_urls(e["url"])
            if not fight_urls:
                print("  Skipping event, no fights found")
                continue
            for url in fight_urls:
                fight = get_fight_details(url)
                if fight and fight["rounds"]:
                    insert_fight(conn, e["name"], e["date"], fight)
                time.sleep(1)
            time.sleep(2)

        conn.close()
        print("\nDone!")

    except Exception as e:
        import traceback
        traceback.print_exc()
