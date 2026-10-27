"""Ethics guard: synthetic attacks may only be injected into replay / sim, never live data."""

ALLOWED_MODES = frozenset({"replay", "sim"})


class EthicsError(RuntimeError):
    """Red-Team injection was requested in a mode where it is not allowed."""


def assert_injection_allowed(mode: str) -> None:
    """Raise `EthicsError` unless `mode` is exactly "replay" or "sim"."""
    if mode not in ALLOWED_MODES:
        raise EthicsError(
            f"ห้ามฉีดข้อมูลโจมตีจำลองในโหมด {mode!r} (อนุญาตเฉพาะ replay / sim เท่านั้น) / "
            f"Red-Team injection is forbidden in mode {mode!r}: "
            "synthetic attacks are allowed only in 'replay' or 'sim', never on live data."
        )
