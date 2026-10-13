"""Deterministic synthetic EdTech demo data: 12 interconnected tables, 12 months, fictional identities.

Embedded (synthetic) stories the analytics should be able to discover:
  * Seasonality: leads peak Mar-Jun (new academic year), dip in Dec and Sep.
  * Social Ads lead-to-enrollment conversion drops from Aug 2026 (a "why did enrollment decline?" story).
  * Referral / Organic learners are more engaged and retain better than Social Ads learners.
  * Learners whose first class is more than 7 days after enrollment engage less afterwards.
  * Slow support resolution (>72h) is followed by lower engagement (association only).
  * Some learners improve steadily, ~10% decline; topics differ in difficulty.
  * Payments: installments, failed payments with retries, pending dues, refunds on cancellation.
  * Online vs offline (Vedantu-style LIVE online classes vs offline learning centres in 8 cities):
    offline learners pay a centre fee, attend more, use the app less and churn less; online reaches every city
    but drops off more. Patna centre attendance falls from Jul 2026; Hyderabad centre runs near seat capacity.

`make_dirty()` injects controlled data-quality issues for tests only; the default demo stays clean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2026-09-30")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "edtech"
TABLES = ["students", "enrollments", "courses", "learning_activity", "attendance", "assessments", "payments",
          "support_tickets", "marketing_spend", "instructors", "classes", "student_surveys", "centres"]
OPTIONAL_TABLES = {"centres"}  # uploads without offline data are treated as online-only
MODE_COLS = {"students": ["learning_mode", "centre_id"], "classes": ["learning_mode", "centre_id"]}
CENTRES = [  # id, name, city, seat capacity, opened (all synthetic)
    ("VC01", "Bengaluru Centre", "Bengaluru", 160, "2023-06-01"),
    ("VC02", "Delhi NCR Centre", "Delhi NCR", 160, "2023-04-01"),
    ("VC03", "Hyderabad Centre", "Hyderabad", 76, "2024-04-01"),
    ("VC04", "Kolkata Centre", "Kolkata", 110, "2024-01-01"),
    ("VC05", "Pune Centre", "Pune", 110, "2024-06-01"),
    ("VC06", "Patna Centre", "Patna", 100, "2023-09-01"),
    ("VC07", "Lucknow Centre", "Lucknow", 90, "2024-07-01"),
    ("VC08", "Jaipur Centre", "Jaipur", 90, "2025-01-01"),
]
OFFLINE_COURSES = {"C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"}  # school & entrance programmes
CENTRE_FEE = 1.3  # offline list fee = online list price × 1.3
P_OFFLINE = {"Centre Walk-in": 1.0, "School Partnerships": 0.55, "Referral": 0.45, "Organic": 0.3, "Paid Search": 0.25,
             "Webinars": 0.15, "Social Ads": 0.15, "YouTube": 0.12}

COURSES = [  # id, name, subject, level, weeks, list price, popularity
    ("C01", "JEE Main Physics", "Physics", "Advanced", 36, 34999, 1.4),
    ("C02", "JEE Main Mathematics", "Mathematics", "Advanced", 36, 34999, 1.4),
    ("C03", "NEET Biology", "Biology", "Advanced", 36, 32999, 1.3),
    ("C04", "NEET Chemistry", "Chemistry", "Advanced", 36, 32999, 1.1),
    ("C05", "Class 10 Mathematics", "Mathematics", "Intermediate", 24, 14999, 1.2),
    ("C06", "Class 10 Science", "Science", "Intermediate", 24, 14999, 1.0),
    ("C07", "Class 8 Foundation Maths", "Mathematics", "Foundation", 16, 8999, 0.8),
    ("C08", "Class 12 Physics Boards", "Physics", "Intermediate", 20, 12999, 0.8),
    ("C09", "Spoken English", "English", "Foundation", 12, 5999, 0.7),
    ("C10", "Coding for Kids", "Computer Science", "Foundation", 12, 7999, 0.7),
    ("C11", "Olympiad Mathematics", "Mathematics", "Advanced", 16, 9999, 0.5),
    ("C12", "CUET Crash Course", "General Aptitude", "Intermediate", 12, 6999, 0.6),
]
TOPICS = {
    "Physics": ["Mechanics", "Electrostatics", "Optics", "Thermodynamics", "Modern Physics"],
    "Mathematics": ["Algebra", "Calculus", "Geometry", "Trigonometry", "Probability"],
    "Biology": ["Genetics", "Human Physiology", "Ecology", "Cell Biology", "Plant Physiology"],
    "Chemistry": ["Organic", "Inorganic", "Physical", "Chemical Bonding", "Equilibrium"],
    "Science": ["Light", "Electricity", "Chemical Reactions", "Life Processes", "Heredity"],
    "English": ["Grammar", "Vocabulary", "Listening", "Speaking", "Writing"],
    "Computer Science": ["Logic", "Loops", "Functions", "Debugging", "Projects"],
    "General Aptitude": ["Quantitative", "Reasoning", "Verbal", "General Awareness", "Data Interpretation"],
}
TOPIC_DIFFICULTY = [0, -7, -3, -12, -5]  # score offset by topic position
LEVEL_ADJ = {"Foundation": 6, "Intermediate": 0, "Advanced": -7}
# channel: lead share, engagement shift, cost per lead (None = no spend), P(qualified), P(enrol | counselled)
CHANNELS = {
    "Paid Search": (0.20, 0.00, 1500, 0.55, 0.50),
    "Social Ads": (0.25, -0.10, 1000, 0.45, 0.40),
    "YouTube": (0.13, 0.00, 900, 0.50, 0.46),
    "Referral": (0.10, 0.10, None, 0.75, 0.66),
    "Organic": (0.15, 0.04, None, 0.60, 0.52),
    "School Partnerships": (0.09, 0.05, 600, 0.70, 0.56),
    "Webinars": (0.08, 0.03, 700, 0.65, 0.56),
    "Centre Walk-in": (0.05, 0.08, None, 0.80, 0.62),
}
CAMPAIGNS = {"Paid Search": ["Brand Search", "Exam Keywords"], "Social Ads": ["Reels Prospecting", "Retargeting"],
             "YouTube": ["Topper Stories", "Free Masterclass"], "Referral": ["Refer-a-Friend"],
             "Organic": ["SEO / Direct"], "School Partnerships": ["School Tie-ups"],
             "Webinars": ["Weekend Webinar Series"], "Centre Walk-in": ["Centre Walk-in"]}
SEASON = {10: 1.0, 11: 0.9, 12: 0.75, 1: 1.0, 2: 1.1, 3: 1.3, 4: 1.6, 5: 1.5, 6: 1.3, 7: 1.0, 8: 0.9, 9: 0.85}
CITIES = ["Bengaluru", "Delhi NCR", "Mumbai", "Hyderabad", "Kolkata", "Pune", "Chennai", "Jaipur", "Lucknow", "Patna"]
CITY_P = [0.17, 0.17, 0.13, 0.11, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06]
FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vikram", "Ananya", "Arjun", "Nisha", "Karan", "Priya",
         "Siddharth", "Tara", "Neel", "Riya"]
LAST = ["Rao", "Sharma", "Iyer", "Banerjee", "Menon", "Kulkarni", "Singh", "Das", "Pillai", "Joshi", "Bose", "Nair"]
SLA_HOURS = {"Urgent": 2, "High": 4, "Medium": 12, "Low": 24}


def _instructors(rng) -> pd.DataFrame:
    subjects = sorted({c[2] for c in COURSES})
    rows, i = [], 1
    for s in subjects:
        for _ in range(2 if s in ("Physics", "Mathematics", "Chemistry", "Biology") else 1):
            rows.append({"instructor_id": f"I{i:02d}",
                         "instructor_name": f"{FIRST[(i * 5) % 16]} {LAST[(i * 7) % 12]} (fictional)",
                         "subject": s, "experience_band": rng.choice(["0-3 yrs", "3-7 yrs", "7+ yrs"]),
                         "_effect": rng.normal(0, 3)})
            i += 1
    return pd.DataFrame(rows)


def _classes(rng, courses: pd.DataFrame, instructors: pd.DataFrame) -> pd.DataFrame:
    rows, n = [], 1
    days = pd.date_range(START, END + pd.Timedelta(days=60), freq="D")
    for k, c in enumerate(courses.itertuples()):
        pool = instructors[instructors.subject == c.subject].instructor_id.tolist()
        for b in (1, 2):
            inst = pool[(b - 1) % len(pool)]
            weekdays = [(k + b) % 6, (k + b + 3) % 6]
            for d in days[days.weekday.isin(weekdays)]:
                rows.append({"class_id": f"K{n:05d}", "course_id": c.course_id, "batch_id": f"{c.course_id}-B{b}",
                             "instructor_id": inst, "class_date": d + pd.Timedelta(hours=17 + b),
                             "scheduled_minutes": 90 if c.course_level == "Advanced" else 60,
                             "enrolled_capacity": 150, "learning_mode": "Online", "centre_id": None})
                n += 1
        if c.course_id not in OFFLINE_COURSES:
            continue
        for j, (vc, *_rest) in enumerate(CENTRES):  # one classroom batch per course per centre
            inst = pool[j % len(pool)]
            weekdays = [(k + j) % 6, (k + j + 2) % 6]
            for d in days[days.weekday.isin(weekdays)]:
                rows.append({"class_id": f"K{n:05d}", "course_id": c.course_id, "batch_id": f"{c.course_id}-{vc}",
                             "instructor_id": inst, "class_date": d + pd.Timedelta(hours=16),
                             "scheduled_minutes": 120 if c.course_level == "Advanced" else 90,
                             "enrolled_capacity": 40, "learning_mode": "Offline", "centre_id": vc})
                n += 1
    return pd.DataFrame(rows)


def generate(seed: int = SEED, scale: float = 1.0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    courses = pd.DataFrame(COURSES, columns=["course_id", "course_name", "subject", "course_level",
                                             "duration_weeks", "list_price", "_pop"])
    instructors = _instructors(rng)
    classes = _classes(rng, courses, instructors)
    inst_effect = instructors.set_index("instructor_id")["_effect"]
    batch_inst = classes.drop_duplicates("batch_id").set_index("batch_id")["instructor_id"]
    class_by_batch = {b: g[["class_id", "class_date", "scheduled_minutes"]].to_numpy()
                      for b, g in classes.groupby("batch_id")}
    cinfo = courses.set_index("course_id")

    # ---------------- leads / students ----------------
    days = pd.date_range(START, END, freq="D")
    per_day = rng.poisson(28 * scale * np.array([SEASON[d.month] for d in days]) * np.where(days.weekday >= 5, 1.15, 1))
    signup = np.repeat(days.values, per_day)
    n = len(signup)
    ch_names = list(CHANNELS)
    share = np.array([CHANNELS[c][0] for c in ch_names])
    ch = rng.choice(ch_names, n, p=share / share.sum())
    campaign = np.array([rng.choice(CAMPAIGNS[c]) for c in ch])
    pop = courses._pop / courses._pop.sum()
    course = rng.choice(courses.course_id, n, p=pop)
    city = rng.choice(CITIES, n, p=CITY_P)
    centre_of = {cty: vc for vc, _, cty, *_ in CENTRES}
    walk = ch == "Centre Walk-in"
    city[walk] = rng.choice(list(centre_of), walk.sum())
    course[walk & ~np.isin(course, list(OFFLINE_COURSES))] = "C05"  # walk-ins ask about centre programmes
    offline = (np.isin(city, list(centre_of)) & np.isin(course, list(OFFLINE_COURSES))
               & (rng.random(n) < np.array([P_OFFLINE[c] for c in ch])))
    mode = np.where(offline, "Offline", "Online")
    centre = np.where(offline, [centre_of.get(cty, "") for cty in city], None)
    counselor_mult = dict(zip([f"Counselor {i:02d}" for i in range(1, 11)], np.round(rng.uniform(0.8, 1.2, 10), 2)))
    counselor = rng.choice(list(counselor_mult), n)
    signup = pd.to_datetime(signup) + pd.to_timedelta(rng.integers(8, 22, n), unit="h")
    qualified = rng.random(n) < np.array([CHANNELS[c][3] for c in ch])
    counselled = qualified & (rng.random(n) < 0.8)
    counsel_date = signup + pd.to_timedelta(rng.gamma(2, 1.5, n), unit="D")
    p_enrol = np.array([CHANNELS[c][4] for c in ch]) * np.array([counselor_mult[c] for c in counselor])
    p_enrol *= np.where(cinfo.loc[course, "list_price"].to_numpy() > 30000, 0.9, 1.05)
    p_enrol *= np.where(offline, 1.15, 1.0)  # in-person counselling at the centre
    late_social = (ch == "Social Ads") & (signup >= pd.Timestamp("2026-08-01"))
    p_enrol = np.where(late_social, p_enrol * 0.45, p_enrol)  # the embedded decline story
    enrol_date = counsel_date + pd.to_timedelta(rng.gamma(2, 3, n), unit="D")
    enrolled = counselled & (rng.random(n) < np.clip(p_enrol, 0, 0.95)) & (enrol_date <= END)

    students = pd.DataFrame({
        "student_id": [f"S{i:05d}" for i in range(1, n + 1)],
        "signup_date": signup.floor("min"), "city": city,
        "acquisition_channel": ch, "campaign": campaign, "course_id": course, "counselor": counselor,
        "learning_mode": mode, "centre_id": centre,
        "lead_qualified": qualified,
        "counseling_date": pd.Series(counsel_date.floor("min")).where(counselled & (counsel_date <= END)),
        "enrollment_date": pd.Series(enrol_date.normalize()).where(enrolled),
    })

    # ---------------- per-learner simulation ----------------
    act, att, asm, tix, srv, enr, pay = [], [], [], [], [], [], []
    status = np.where(~qualified, "Lead - not qualified",
                      np.where(~counselled, "Lead - lost",
                               np.where(enrol_date > END, "In pipeline", "Lead - lost")))
    status = status.astype(object)
    act_types = np.array(["video_watch", "practice_quiz", "assignment_submit", "notes_read"])
    end_day = END + pd.Timedelta(hours=23, minutes=59)
    eid = pid = tid = sid = aid = 0
    for i in np.flatnonzero(enrolled):
        s_id = students.student_id.iat[i]
        c_id = course[i]
        c = cinfo.loc[c_id]
        off = bool(offline[i])
        e = float(np.clip(rng.beta(4, 3) + CHANNELS[ch[i]][1] + 0.06 * off + rng.normal(0, 0.05), 0.05, 0.98))
        d0 = students.enrollment_date.iat[i]
        batch = f"{c_id}-{centre[i]}" if off else f"{c_id}-B{rng.integers(1, 3)}"
        ieff = inst_effect[batch_inst[batch]]
        lag = rng.exponential((2.0 if off else 3.0) / max(e, 0.2))
        never_start = lag > 40
        if lag > 7:
            e *= 0.72
        growth = rng.normal(0.55 * e, 0.35) if rng.random() > 0.10 else rng.normal(-0.6, 0.2)
        n_weeks = int((END - d0).days // 7) + 1
        dur = int(c.duration_weeks)
        # lifecycle
        cancel = (not never_start and rng.random() < 0.03 + 0.08 * (1 - e) + 0.05 * (lag > 7)) or \
                 (never_start and rng.random() < 0.6)
        cancel_day = int(rng.integers(2, 15))
        hazard = (0.006 + 0.06 * (1 - e) ** 2) * (0.6 if off else 1.0)
        churn_w = int(rng.geometric(hazard))
        active_weeks = [] if (never_start or cancel) else list(range(0, min(churn_w, dur, n_weeks)))
        completed = churn_w >= dur and dur <= n_weeks and not cancel and not never_start
        renew = completed and rng.random() < 0.25 + 0.3 * e
        if renew:
            churn2 = int(rng.geometric(hazard))
            active_weeks += list(range(dur, min(dur + churn2, n_weeks)))
        elif not completed and active_weeks and rng.random() < 0.12:  # reactivation after a gap
            gap = int(rng.integers(3, 9))
            stop = active_weeks[-1] + 1
            active_weeks += list(range(stop + gap, min(stop + gap + int(rng.integers(2, 8)), n_weeks)))
        if cancel and not never_start:
            active_weeks = list(range(0, min(2, n_weeks))) if cancel_day > 7 else [0]
        active_weeks = np.array(sorted(set(active_weeks)), dtype=int)

        # enrollment + payments
        disc = rng.choice([0, 0.1, 0.2, 0.3], p=[0.45, 0.3, 0.17, 0.08])
        list_fee = float(c.list_price) * (CENTRE_FEE if off else 1.0)
        price = float(round(list_fee * (1 - disc), -1))
        cancel_date = d0 + pd.Timedelta(days=cancel_day)
        for k, (edate, is_renew) in enumerate([(d0, False)] + ([(d0 + pd.Timedelta(weeks=dur, days=int(rng.integers(0, 10))), True)] if renew else [])):
            if edate > END:
                continue
            eid += 1
            e_id = f"E{eid:06d}"
            refund = 0.0
            e_status = "Active"
            if not is_renew:
                if cancel:
                    e_status = "Cancelled"
                elif completed:
                    e_status = "Completed"
            plan = 1 if rng.random() < 0.6 else 3
            inst_amt = round(price / plan, 2)
            paid = 0.0
            for j in range(plan):
                due = edate + pd.DateOffset(months=j)
                if due > END or (e_status == "Cancelled" and due > cancel_date):
                    break
                pid += 1
                if rng.random() < 0.08:  # failed attempt, maybe retried
                    pay.append((f"P{pid:06d}", s_id, e_id, due, inst_amt, "Failed", 0.0))
                    pid += 1
                    retry = due + pd.Timedelta(days=3)
                    ok = rng.random() < 0.7
                    st_ = "Success" if ok else ("Pending" if retry > END - pd.Timedelta(days=20) else "Failed")
                    if retry <= END:
                        pay.append((f"P{pid:06d}", s_id, e_id, retry, inst_amt, st_, 0.0))
                        paid += inst_amt if ok else 0
                else:
                    pay.append((f"P{pid:06d}", s_id, e_id, due, inst_amt, "Success", 0.0))
                    paid += inst_amt
            if e_status == "Cancelled" and paid > 0:
                refund = round(paid if cancel_day <= 7 else paid * 0.5, 2)
                rdate = cancel_date + pd.Timedelta(days=3)
                if rdate <= END:
                    pid += 1
                    pay.append((f"P{pid:06d}", s_id, e_id, rdate, 0.0, "Refunded", refund))
                else:
                    refund = 0.0
            enr.append((e_id, s_id, c_id, batch, edate, e_status, round(list_fee - price, 2), price, refund,
                        is_renew, cancel_date if e_status == "Cancelled" else pd.NaT,
                        edate + pd.Timedelta(weeks=dur) if e_status == "Completed" else pd.NaT))

        if len(active_weeks) == 0:
            status[i] = "Cancelled" if cancel else "Inactive (inferred)"
            continue
        first_class = d0 + pd.Timedelta(days=lag)
        start_w = int(lag // 7)
        active_weeks = active_weeks[active_weeks >= start_w] if len(active_weeks[active_weeks >= start_w]) else active_weeks[:1]
        ew = np.clip(e * (1 - 0.01 * active_weeks) + rng.normal(0, 0.08, len(active_weeks)), 0.02, 1)

        # support tickets (reduce later engagement when slow)
        for w, lvl in zip(active_weeks, ew):
            wk_start = d0 + pd.Timedelta(weeks=int(w))
            season = 1.4 if wk_start.month in (4, 5, 6) else 1.0
            if rng.random() < (0.035 + 0.04 * (1 - lvl)) * season * (0.7 if off else 1.0):
                tid += 1
                if off:
                    cat = rng.choice(["Payment", "Class Schedule"] if w < 3 else
                                     ["Content Doubt", "Centre Facilities", "Class Schedule", "Payment"])
                else:
                    cat = rng.choice(["Payment", "Login / Access", "Class Schedule"] if w < 3 else
                                     ["Content Doubt", "Technical Issue", "Class Schedule", "Payment", "Login / Access"])
                prio = rng.choice(list(SLA_HOURS), p=[0.08, 0.22, 0.45, 0.25])
                created = wk_start + pd.Timedelta(hours=float(rng.uniform(0, 7 * 24)))
                if created > end_day:
                    continue
                frt = float(rng.gamma(1.6, SLA_HOURS[prio] * 0.6))
                res_h = float(rng.gamma(2, 14) * (2.2 if cat == "Technical Issue" else 1))
                resolved = created + pd.Timedelta(hours=res_h)
                st_ = "Resolved" if resolved <= end_day else "Open"
                reopened = st_ == "Resolved" and rng.random() < 0.08
                csat = float(np.clip(round(5.2 - res_h / 30 + rng.normal(0, 0.7)), 1, 5)) if st_ == "Resolved" and rng.random() < 0.6 else np.nan
                tix.append((f"T{tid:06d}", s_id, c_id, created.floor("min"), (created + pd.Timedelta(hours=frt)).floor("min"),
                            resolved.floor("min") if st_ == "Resolved" else pd.NaT, cat, prio,
                            "Reopened" if reopened else st_, csat))
                if res_h > 72:
                    ew[active_weeks > w] *= 0.85

        # learning activity
        sessions = rng.poisson(0.4 + 3 * ew if off else 0.8 + 6 * ew)  # offline learners use the app less
        ts = pd.DatetimeIndex([])
        wk = np.repeat(active_weeks, sessions)
        if len(wk):
            dow = rng.choice(7, len(wk), p=[0.13, 0.13, 0.13, 0.13, 0.11, 0.17, 0.20])
            hour = rng.choice(24, len(wk), p=_HOUR_P)
            ts = d0 + pd.to_timedelta(wk * 7 + dow, unit="D") + pd.to_timedelta(hour, unit="h") + \
                pd.to_timedelta(rng.integers(0, 60, len(wk)), unit="m")
            keep = ts <= end_day
            ts, wk_k = ts[keep], wk[keep]
            typ = rng.choice(act_types, len(ts), p=[0.25, 0.45, 0.15, 0.15] if off else [0.45, 0.3, 0.1, 0.15])
            dur_m = np.where(typ == "video_watch", rng.gamma(3, 9, len(ts)),
                             np.where(typ == "practice_quiz", rng.gamma(2, 8, len(ts)), rng.gamma(2, 6, len(ts))))
            for t_, ty, dm in zip(ts, typ, dur_m.round(1)):
                aid += 1
                act.append((f"A{aid:07d}", s_id, c_id, t_, ty, dm, f"{c_id}-{ty[:2].upper()}{int(rng.integers(1, 120)):03d}"))

        # attendance on scheduled classes: from first class until 2 weeks after last active week
        last_active = d0 + pd.Timedelta(weeks=int(active_weeks.max()) + 1)
        cls = class_by_batch[batch]
        dates = cls[:, 1].astype("datetime64[ns]")
        mask = (dates >= np.datetime64(first_class.normalize())) & (dates <= np.datetime64(min(last_active + pd.Timedelta(weeks=2), end_day)))
        wk_of = ((dates[mask] - np.datetime64(d0)) / np.timedelta64(7, "D")).astype(int)
        lvl_map = dict(zip(active_weeks, ew))
        p_att = np.array([0.15 + 0.8 * lvl_map[w] if w in lvl_map else 0.03 for w in wk_of])
        p_att = np.clip(p_att + ieff / 40 + 0.10 * off, 0, 0.98)
        if off and centre[i] == "VC06":  # Patna centre: attendance slump from Jul 2026
            p_att = np.where(dates[mask] >= np.datetime64("2026-07-01"), p_att * 0.65, p_att)
        attended = rng.random(mask.sum()) < p_att
        for (k_id, k_date, k_min), a in zip(cls[mask], attended):
            att.append((s_id, c_id, k_id, k_date, bool(a), round(float(k_min) * rng.uniform(0.6, 1.0), 0) if a else 0.0))

        # assessments every ~2 weeks while active
        topics = TOPICS[c.subject]
        mx = 100 if c.course_level != "Foundation" else 50
        for n_t, w in enumerate(active_weeks[::2]):
            if rng.random() > 0.85:
                continue
            ti = n_t % 5
            base = 42 + 28 * ew[np.searchsorted(active_weeks, w)] + TOPIC_DIFFICULTY[ti] + LEVEL_ADJ[c.course_level]
            pct = float(np.clip(base + growth * w + ieff + rng.normal(0, 8), 2, 100))
            adate = d0 + pd.Timedelta(weeks=int(w), days=int(rng.integers(0, 7)))
            if adate > END:
                continue
            q = int(rng.integers(20, 31))
            attempted = int(max(1, round(q * rng.uniform(0.75, 1.0))))
            correct = int(round(attempted * pct / 100))
            asm.append((s_id, c_id, topics[ti], adate, round(pct * mx / 100, 1), mx, attempted, correct))

        # surveys
        for w_s in (4, dur):
            if w_s in set(active_weeks) and rng.random() < 0.5:
                sid += 1
                sat = int(np.clip(round(2 + 3 * e + rng.normal(0, 0.6)), 1, 5))
                srv.append((f"V{sid:06d}", s_id, d0 + pd.Timedelta(weeks=w_s), sat,
                            int(np.clip(round(3 + 7 * e + rng.normal(0, 1.4)), 0, 10))))
        went = [k_date for (_, k_date, _), a in zip(cls[mask], attended) if a]
        last = max([ts.max()] if len(wk) and len(ts) else [first_class], default=first_class)
        last = max(last, max(went)) if went else last
        status[i] = ("Cancelled" if cancel else "Active" if (END - last).days <= 30 else
                     "Completed" if completed else "Inactive (inferred)")

    students["student_status"] = status
    enrollments = pd.DataFrame(enr, columns=["enrollment_id", "student_id", "course_id", "batch_id", "enrollment_date",
                                             "enrollment_status", "discount", "price", "refund_amount", "is_renewal",
                                             "cancellation_date", "completion_date"])
    activity = pd.DataFrame(act, columns=["activity_id", "student_id", "course_id", "activity_date", "activity_type",
                                          "duration_minutes", "content_id"])
    attendance = pd.DataFrame(att, columns=["student_id", "course_id", "class_id", "class_date", "attended",
                                            "class_duration_minutes"])
    attendance.insert(0, "attendance_id", [f"AT{i:07d}" for i in range(1, len(attendance) + 1)])
    assessments = pd.DataFrame(asm, columns=["student_id", "course_id", "topic", "assessment_date", "score",
                                             "max_score", "questions_attempted", "correct_answers"])
    assessments.insert(0, "assessment_id", [f"X{i:06d}" for i in range(1, len(assessments) + 1)])
    payments = pd.DataFrame(pay, columns=["payment_id", "student_id", "enrollment_id", "payment_date", "amount",
                                          "payment_status", "refund_amount"])
    tickets = pd.DataFrame(tix, columns=["ticket_id", "student_id", "course_id", "created_at", "first_response_at",
                                         "resolved_at", "issue_category", "priority", "status", "csat_score"])
    surveys = pd.DataFrame(srv, columns=["survey_id", "student_id", "survey_date", "satisfaction_score",
                                         "recommendation_score"])
    surveys = surveys[surveys.survey_date <= END]

    # marketing spend derived from lead volumes (no spend for Referral / Organic)
    leads = students.assign(date=students.signup_date.dt.normalize()).groupby(
        ["date", "acquisition_channel", "campaign"]).size().reset_index(name="leads")
    leads = leads[leads.acquisition_channel.map(lambda c: CHANNELS[c][2] is not None)]
    cpl = leads.acquisition_channel.map(lambda c: CHANNELS[c][2]).astype(float)
    leads["spend"] = (leads.leads * cpl * rng.uniform(0.8, 1.25, len(leads))).round(0)
    leads.loc[(leads.acquisition_channel == "Social Ads") & (leads.date >= "2026-08-01"), "spend"] *= 1.2
    leads["clicks"] = (leads.leads / rng.uniform(0.04, 0.09, len(leads))).round(0).astype(int)
    leads["impressions"] = (leads.clicks / rng.uniform(0.008, 0.025, len(leads))).round(0).astype(int)
    marketing = leads.rename(columns={"acquisition_channel": "channel"})[
        ["date", "channel", "campaign", "spend", "impressions", "clicks", "leads"]]

    return {
        "students": students, "enrollments": enrollments, "courses": courses.drop(columns="_pop"),
        "learning_activity": activity, "attendance": attendance, "assessments": assessments, "payments": payments,
        "support_tickets": tickets, "marketing_spend": marketing.reset_index(drop=True),
        "instructors": instructors.drop(columns="_effect"), "classes": classes, "student_surveys": surveys,
        "centres": pd.DataFrame([(a, b, c, d, pd.Timestamp(o)) for a, b, c, d, o in CENTRES],
                                columns=["centre_id", "centre_name", "city", "seat_capacity", "opened_date"]),
    }


_HOUR_P = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 1.2, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.5, 5.0, 7.0, 9.0,
                    11.0, 12.0, 11.0, 8.0, 5.0, 2.0])
_HOUR_P = _HOUR_P / _HOUR_P.sum()

DATE_COLS = {"students": ["signup_date", "counseling_date", "enrollment_date"],
             "enrollments": ["enrollment_date", "cancellation_date", "completion_date"],
             "learning_activity": ["activity_date"], "attendance": ["class_date"], "assessments": ["assessment_date"],
             "payments": ["payment_date"], "support_tickets": ["created_at", "first_response_at", "resolved_at"],
             "marketing_spend": ["date"], "classes": ["class_date"], "student_surveys": ["survey_date"],
             "centres": ["opened_date"]}


def ensure_modes(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Uploaded data without offline fields is treated as online-only (no centres)."""
    t = dict(tables)
    for name, cols in MODE_COLS.items():
        if name in t and "learning_mode" not in t[name].columns:
            t[name] = t[name].assign(learning_mode="Online", centre_id=None)
    if "centres" not in t:
        t["centres"] = pd.DataFrame(columns=["centre_id", "centre_name", "city", "seat_capacity", "opened_date"])
    return t


def write_all(seed: int = SEED, folder: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    folder.mkdir(parents=True, exist_ok=True)
    tables = generate(seed)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)
    return tables


def load_all(folder: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    if not all((folder / f"{t}.csv").exists() for t in TABLES):
        write_all(folder=folder)
    out = {}
    for t in TABLES:
        df = pd.read_csv(folder / f"{t}.csv")
        for c in DATE_COLS.get(t, []):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        out[t] = df
    return out


def make_dirty(tables: dict[str, pd.DataFrame], seed: int = 1) -> dict[str, pd.DataFrame]:
    """TEST FIXTURE ONLY: inject controlled data-quality problems into a copy of the tables."""
    rng = np.random.default_rng(seed)
    t = {k: v.copy() for k, v in tables.items()}
    a = t["assessments"]
    idx = rng.choice(len(a), 5, replace=False)
    a.loc[idx, "score"] = a.loc[idx, "max_score"] + 20  # impossible score
    p = t["payments"]
    p.loc[rng.choice(len(p), 3, replace=False), "amount"] = -500.0  # negative payment
    e = t["enrollments"]
    t["enrollments"] = pd.concat([e, e.head(4)], ignore_index=True)  # duplicate primary keys
    act = t["learning_activity"]
    act.loc[rng.choice(len(act), 6, replace=False), "student_id"] = "S99999"  # orphan foreign key
    s = t["students"]
    s.loc[rng.choice(s.index[s.enrollment_date.notna()], 4, replace=False), "enrollment_date"] = pd.Timestamp("2025-01-01")
    tk = t["support_tickets"]
    r = tk.index[tk.resolved_at.notna()][:3]
    tk.loc[r, "resolved_at"] = tk.loc[r, "created_at"] - pd.Timedelta(hours=5)  # resolved before created
    s.loc[rng.choice(len(s), 30, replace=False), "city"] = np.nan
    return t


if __name__ == "__main__":
    import time
    t0 = time.time()
    tabs = write_all()
    print({k: len(v) for k, v in tabs.items()}, f"{time.time() - t0:.1f}s")
