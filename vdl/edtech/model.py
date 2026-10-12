"""Filtered view over the 12 EdTech tables. Every page receives the same Model, so global filters apply
consistently. Segment filters (course, subject, cohort, channel, status) restrict the student population;
the date window is applied by metric/analytics functions so previous-period comparisons remain possible."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .generate import END, START


@dataclass(frozen=True)
class Filters:
    start: pd.Timestamp = START
    end: pd.Timestamp = END
    courses: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    cohorts: tuple[str, ...] = ()  # enrollment month "YYYY-MM"
    channels: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [f"{self.start:%d %b %Y} – {self.end:%d %b %Y}"]
        for name, v in (("course", self.courses), ("subject", self.subjects), ("cohort", self.cohorts),
                        ("channel", self.channels), ("status", self.statuses)):
            if v:
                parts.append(f"{name}: {', '.join(v)}")
        return " · ".join(parts)


@dataclass
class Model:
    tables: dict[str, pd.DataFrame]
    f: Filters = field(default_factory=Filters)

    def __post_init__(self) -> None:
        t = self.tables
        self.data_start, self.data_end = START, END
        self._events = None
        self.courses = t["courses"]
        s = t["students"].merge(self.courses[["course_id", "course_name", "subject"]], on="course_id", how="left")
        s["cohort"] = s.enrollment_date.dt.strftime("%Y-%m")
        f = self.f
        if f.courses:
            s = s[s.course_name.isin(f.courses) | s.course_id.isin(f.courses)]
        if f.subjects:
            s = s[s.subject.isin(f.subjects)]
        if f.cohorts:
            s = s[s.cohort.isin(f.cohorts)]
        if f.channels:
            s = s[s.acquisition_channel.isin(f.channels)]
        if f.statuses:
            s = s[s.student_status.isin(f.statuses)]
        self.students = s
        ids = set(s.student_id)
        keep = lambda name: t[name][t[name].student_id.isin(ids)]  # noqa: E731
        self.enrollments = keep("enrollments")
        self.activity = keep("learning_activity")
        self.attendance = keep("attendance")
        self.assessments = keep("assessments").assign(
            pct=lambda d: (d.score / d.max_score * 100).round(2))
        self.payments = keep("payments")
        self.tickets = keep("support_tickets")
        self.surveys = keep("student_surveys")
        course_ids = set(s.course_id) if (f.courses or f.subjects) else set(self.courses.course_id)
        self.classes = t["classes"][t["classes"].course_id.isin(course_ids)]
        self.instructors = t["instructors"]
        mk = t["marketing_spend"]
        self.marketing = mk[mk.channel.isin(f.channels)] if f.channels else mk
        self.enrolled = s[s.enrollment_date.notna()]

    # ---------- helpers ----------
    @property
    def start(self) -> pd.Timestamp:
        return self.f.start

    @property
    def end(self) -> pd.Timestamp:
        return min(self.f.end, self.data_end)

    def empty(self) -> bool:
        return self.enrolled.empty

    @staticmethod
    def window(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
        end_incl = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        return df[(df[col] >= pd.Timestamp(start)) & (df[col] <= end_incl)]

    def events(self) -> pd.DataFrame:
        """Unified learner activity events (platform activity + attended live classes)."""
        if getattr(self, "_events", None) is not None:
            return self._events
        a = self.activity[["student_id", "activity_date"]].rename(columns={"activity_date": "ts"})
        b = self.attendance.loc[self.attendance.attended, ["student_id", "class_date"]].rename(columns={"class_date": "ts"})
        self._events = pd.concat([a, b], ignore_index=True)
        return self._events

    def prev_window(self, start=None, end=None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        start, end = pd.Timestamp(start or self.start), pd.Timestamp(end or self.end)
        length = end - start + pd.Timedelta(days=1)
        p_start, p_end = start - length, start - pd.Timedelta(days=1)
        return None if p_start < self.data_start else (p_start, p_end)

    def course_name(self) -> dict[str, str]:
        return dict(zip(self.courses.course_id, self.courses.course_name))
