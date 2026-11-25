"""The measured history line shown beside an alert (Telegram and the web alert detail).

It is history, not a forecast, and it was measured on the rise of real pumps only, so a sell-phase
alert goes without it (`history_line` gives None).
"""

from typing import Any

# source: docs/ROADMAP.md, "ผลเฟส 0" (REAL real_t001, budget 2, 5 seeds): 88.8 % [83.8, 93.4] of
# caught real pumps were below the alert price 1 h later, and the median highest price within that
# hour (max_up_1h) was +15.5 %: the two are always shown together. History, not a forecast.
HISTORY_TH = (
    "ข้อมูลในอดีต (pump จริงที่ระบบจับได้ คู่ BTC 2018–2021): 1 ชม. หลังเตือน "
    "ราคาต่ำกว่าตอนเตือน 89% ของครั้ง แต่ในชั่วโมงนั้นราคายังขึ้นต่อได้ (มัธยฐาน +15%)"
)
HISTORY_SOURCE_TH = "ที่มา: ผลเฟส 0 ใน docs/ROADMAP.md"


def history_line(alert: Any) -> str | None:
    """`HISTORY_TH` for a pump-phase or phase-less alert, None for a sell-phase one."""
    return None if getattr(alert, "phase", None) == "sell" else HISTORY_TH
