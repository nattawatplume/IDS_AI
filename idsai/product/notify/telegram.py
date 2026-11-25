"""Telegram alerts: account linking over long polling, and one Thai message per new alert.

`TELEGRAM_BOT_TOKEN` turns it on; without it `TelegramService.from_env` gives None and nothing here
runs. The token is part of every Bot API URL, so it must never reach a log line, an exception text
or an API answer: `TelegramClient` turns every failure into a `TelegramError` that carries only the
method name and the HTTP status (raised `from None`, so no chained httpx error carries the URL),
and `RedactFilter` scrubs it from the `httpx` / `httpcore` request logs.

Linking: `POST /api/me/telegram/link` issues a one-time code (`LinkCodes`, memory only, 10 min);
the user presses Start on `t.me/<bot>?start=<code>` and the poller stores their private `chat_id`
as the server-only pref `tg_chat_id`. `/stop` in the chat unlinks it.

Notifying: only an `Alert` with status "open" counts; recipients are the users with a chat whose
`watch` pref (absent: every symbol) has the symbol; at most one message per (chat, symbol) per
`RATE_S`. Sends run as their own tasks, so a slow Telegram never holds up the bus. The text is plain
(no parse mode) and states that it is a statistical anomaly, not investment advice, and accuses no
one. Nothing here sends orders or asks for exchange keys.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx

from idsai.contracts.alert import Alert
from idsai.data.store import TELEGRAM_CHAT_PREF
from idsai.product.explainer import HISTORY_TH, explain, history_line

log = logging.getLogger(__name__)

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
API_BASE_ENV = "TELEGRAM_API_BASE"
PUBLIC_URL_ENV = "IDSAI_PUBLIC_URL"
DEFAULT_API_BASE = "https://api.telegram.org"
PREF_KEY = TELEGRAM_CHAT_PREF  # server-only: set here, never accepted or returned by the API
CODE_TTL_S = 600
RATE_S = 600  # at most one message per (chat_id, symbol) in this many seconds
POLL_TIMEOUT_S = 25
BACKOFF_MAX_S = 60
MAX_SENDS = 8  # sends in flight at once
BANGKOK = timezone(timedelta(hours=7), "Asia/Bangkok")  # no DST in Thailand

HEAD_TH = "⚠️ ตรวจพบความผิดปกติทางสถิติ"
DISCLAIMER_TH = "เป็นข้อมูลความผิดปกติทางสถิติ ไม่ใช่คำแนะนำการลงทุน และไม่ได้กล่าวหาผู้ใด"
LINKED_TH = "เชื่อมบัญชี {name} แล้ว พิมพ์ /stop เพื่อยกเลิก"
EXPIRED_TH = "ลิงก์นี้หมดอายุหรือใช้ไปแล้ว กลับไปกด «เชื่อม Telegram» บนเว็บอีกครั้ง"
STOPPED_TH = "ยกเลิกการแจ้งเตือนแล้ว เชื่อมใหม่ได้จากเว็บ"
NOT_LINKED_TH = "แชตนี้ยังไม่ได้เชื่อมกับบัญชีใด"
LINK_FAILED_TH = "เชื่อมไม่สำเร็จ ลองกดเชื่อมใหม่บนเว็บ"
TIME_TH = "(เวลาไทย)"
LOCAL_ID = "local"  # the built-in user of auth-off mode (idsai.product.api.app.LOCAL_USER)


class TelegramError(Exception):
    """A Bot API call failed. Its text is the method and the HTTP status only, never the URL."""

    def __init__(self, method: str, status: int | None = None, retry_after: int | None = None):
        self.method = method
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"telegram {method} failed (HTTP {status if status else 'no answer'})")


class RedactFilter(logging.Filter):
    """Replaces the token in a log record's text (httpx logs every request URL at INFO)."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:  # a broken record: leave it to logging's own error path
            return True
        if self.token and self.token in text:
            record.msg = text.replace(self.token, "<redacted>")
            record.args = ()
        return True


_REDACT_LOGGERS = ("httpx", "httpcore")


def install_redaction(token: str) -> RedactFilter:
    f = RedactFilter(token)
    for name in _REDACT_LOGGERS:
        logging.getLogger(name).addFilter(f)
    return f


def remove_redaction(f: RedactFilter) -> None:
    for name in _REDACT_LOGGERS:
        logging.getLogger(name).removeFilter(f)


class TelegramClient:
    """Thin async wrapper over the three Bot API methods this feature uses."""

    def __init__(
        self,
        token: str,
        base: str = DEFAULT_API_BASE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not token:
            raise ValueError("a bot token is required")
        self._http = httpx.AsyncClient(
            base_url=f"{base.rstrip('/')}/bot{token}",
            transport=transport,
            timeout=httpx.Timeout(POLL_TIMEOUT_S + 10, connect=10),
        )

    async def _call(self, method: str, **params: Any) -> Any:
        try:
            r = await self._http.post(f"/{method}", json=params)
        except (httpx.HTTPError, OSError):  # the exception text holds the URL: drop it
            raise TelegramError(method) from None
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code == 200 and isinstance(body, dict) and body.get("ok") is True:
            return body.get("result")
        retry_after = None
        if isinstance(body, dict) and isinstance(body.get("parameters"), dict):
            ra = body["parameters"].get("retry_after")
            if isinstance(ra, int) and ra > 0:
                retry_after = ra
        raise TelegramError(method, r.status_code, retry_after)

    async def get_me(self) -> dict:
        res = await self._call("getMe")
        if not isinstance(res, dict):
            raise TelegramError("getMe", 200)
        return res

    async def get_updates(self, offset: int | None, timeout: int = POLL_TIMEOUT_S) -> list[dict]:
        params: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        res = await self._call("getUpdates", **params)
        return [u for u in res if isinstance(u, dict)] if isinstance(res, list) else []

    async def send_message(self, chat_id: int, text: str) -> None:
        await self._call("sendMessage", chat_id=chat_id, text=text, disable_web_page_preview=True)

    async def aclose(self) -> None:
        await self._http.aclose()


class LinkCodes:
    """One-time link codes in memory: one per user, a new one replaces the old, `ttl_s` life."""

    def __init__(self, ttl_s: float = CODE_TTL_S, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self._clock = clock
        self._by_code: dict[str, tuple[str, float]] = {}
        self._by_user: dict[str, str] = {}

    def issue(self, user_id: str) -> str:
        old = self._by_user.pop(user_id, None)
        if old is not None:
            self._by_code.pop(old, None)
        self._purge()
        code = secrets.token_urlsafe(24)  # 32 chars of A-Za-z0-9_-: Telegram's start-param set
        self._by_code[code] = (user_id, self._clock() + self.ttl_s)
        self._by_user[user_id] = code
        return code

    def take(self, code: str) -> str | None:
        """The user of `code` if it is valid and unexpired; the code is used up either way."""
        hit = self._by_code.pop(code, None)
        if hit is None:
            return None
        user_id, expires = hit
        if self._by_user.get(user_id) == code:
            del self._by_user[user_id]
        return user_id if self._clock() < expires else None

    def _purge(self) -> None:
        now = self._clock()
        for code, (uid, exp) in list(self._by_code.items()):
            if exp <= now:
                del self._by_code[code]
                if self._by_user.get(uid) == code:
                    del self._by_user[uid]


def fmt_bangkok(ts_us: int) -> str:
    return (
        datetime.fromtimestamp(ts_us / 1e6, UTC).astimezone(BANGKOK).strftime("%H:%M:%S %d/%m/%Y")
    )


def message_text(alert: Alert, public_url: str | None = None) -> str:
    """Plain Thai text of one alert. The explainer line names the technique (e.g. MM-T006 ทุบราคา)
    and the phase (M4, "ช่วง: กำลังพุ่ง" / "ช่วง: กำลังเทขาย"). `HISTORY_TH` was measured on
    alerts in the rise of real pumps only, so a sell-phase alert goes without it."""
    lines = [
        f"{HEAD_TH} · {alert.symbol}",
        f"{fmt_bangkok(alert.ts_event_us)} {TIME_TH}",
        explain(alert).text_th,
        *(() if history_line(alert) is None else (HISTORY_TH,)),
        DISCLAIMER_TH,
    ]
    if public_url:
        lines.append(public_url)
    return "\n".join(lines)


def watches(prefs: Mapping[str, Any], symbol: str) -> bool:
    """Absent `watch`: every symbol; a list: only those (empty: none)."""
    w = prefs.get("watch")
    if w is None:
        return True
    return isinstance(w, list) and symbol in w


class TelegramService:
    """The poller (linking) and the notifier (alerts) over one `TelegramClient`.

    `store` is the app `Store` (`get_prefs`, `set_prefs`, `list_prefs`, `get_user`, `list_users`,
    `add_audit`). With `auth_enabled` only enabled accounts in `users` get messages; with auth off
    only the built-in `local` user does. Everything runs on the app's event loop."""

    def __init__(
        self,
        client: TelegramClient,
        store: Any,
        public_url: str | None = None,
        *,
        token: str = "",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        local_name: str = LOCAL_ID,
        auth_enabled: bool = False,
    ) -> None:
        self.client = client
        self.store = store
        self.public_url = public_url or None
        self.codes = LinkCodes(clock=clock)
        self.bot_username: str | None = None
        self._token = token
        self._clock = clock
        self._sleep = sleep
        self._local_name = local_name
        self.auth_enabled = auth_enabled
        self._offset: int | None = None
        self._last_sent: dict[tuple[int, str], float] = {}
        self._sends: set[asyncio.Task] = set()
        self._send_slots = asyncio.Semaphore(MAX_SENDS)
        self._tasks: list[asyncio.Task] = []
        self._filter: RedactFilter | None = install_redaction(token) if token else None

    @classmethod
    def from_env(
        cls,
        store: Any,
        env: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        **kw: Any,
    ) -> TelegramService | None:
        env = os.environ if env is None else env
        token = (env.get(TOKEN_ENV) or "").strip()
        if not token:
            return None
        base = (env.get(API_BASE_ENV) or "").strip() or DEFAULT_API_BASE
        client = TelegramClient(token, base, transport=transport)
        public = (env.get(PUBLIC_URL_ENV) or "").strip() or None
        return cls(client, store, public, token=token, **kw)

    # ---- links ----

    def link_url(self, user_id: str) -> dict | None:
        """`{url, expires_s}` for `user_id`, or None while the bot's name is not known yet."""
        if not self.bot_username:
            return None
        code = self.codes.issue(user_id)
        return {"url": f"https://t.me/{self.bot_username}?start={code}", "expires_s": CODE_TTL_S}

    def linked(self, prefs: Mapping[str, Any]) -> bool:
        return PREF_KEY in prefs

    def _link(self, user_id: str, chat_id: int) -> None:
        prefs = self.store.get_prefs(user_id)
        prefs[PREF_KEY] = chat_id
        self.store.set_prefs(user_id, prefs)

    def unlink_user(self, user_id: str) -> bool:
        prefs = self.store.get_prefs(user_id)
        if PREF_KEY not in prefs:
            return False
        del prefs[PREF_KEY]
        self.store.set_prefs(user_id, prefs)
        return True

    def unlink_chat(self, chat_id: int, action: str = "telegram_stop") -> int:
        """Unlink every user bound to `chat_id`; each one gets an audit row `action`."""
        n = 0
        for user_id, prefs in self.store.list_prefs().items():
            if prefs.get(PREF_KEY) == chat_id:
                del prefs[PREF_KEY]
                self.store.set_prefs(user_id, prefs)
                self._audit(user_id, action)
                n += 1
        return n

    def _username(self, user_id: str) -> str:
        """The account's username (validated at creation), never its free-text display name."""
        user = self.store.get_user(user_id)
        if user is None:
            return self._local_name
        return user.get("username") or self._local_name

    def _audit(self, user_id: str, action: str) -> None:
        try:
            now = time.time_ns() // 1000
            self.store.add_audit(now, user_id, self._username(user_id), action, "", {})
        except Exception as e:  # the audit log must not break linking or sending
            log.warning("telegram: audit not written (%s)", type(e).__name__)

    # ---- poller ----

    async def handle_update(self, update: Mapping[str, Any]) -> None:
        msg = update.get("message")
        if not isinstance(msg, dict):
            return
        chat = msg.get("chat")
        if not isinstance(chat, dict) or chat.get("type") != "private":
            return  # groups and channels never link and never get answers
        chat_id = chat.get("id")
        text = msg.get("text")
        if not isinstance(chat_id, int) or isinstance(chat_id, bool) or not isinstance(text, str):
            return
        parts = text.strip().split(maxsplit=1)
        if not parts:
            return
        cmd = parts[0].split("@", 1)[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/start":
            user_id = self.codes.take(arg) if arg else None
            if user_id is None:
                await self._reply(chat_id, EXPIRED_TH)
                return
            try:
                self._link(user_id, chat_id)
                name = self._username(user_id)
            except Exception as e:  # e.g. the account store is down; the code stays used up
                log.warning("telegram: link not stored (%s)", type(e).__name__)
                await self._reply(chat_id, LINK_FAILED_TH)
                return
            self._audit(user_id, "telegram_link")
            await self._reply(chat_id, LINKED_TH.format(name=name))
        elif cmd == "/stop":
            n = self.unlink_chat(chat_id)
            await self._reply(chat_id, STOPPED_TH if n else NOT_LINKED_TH)

    async def _reply(self, chat_id: int, text: str) -> None:
        try:
            await self.client.send_message(chat_id, text)
        except TelegramError as e:
            log.warning("telegram: %s", e)

    async def poll_once(self) -> None:
        if self.bot_username is None:
            me = await self.client.get_me()
            name = me.get("username")
            if not isinstance(name, str) or not name:
                raise TelegramError("getMe", 200)
            self.bot_username = name
        updates = await self.client.get_updates(self._offset, POLL_TIMEOUT_S)
        for u in updates:
            uid = u.get("update_id")
            if isinstance(uid, int):
                self._offset = max(self._offset or 0, uid + 1)
            try:
                await self.handle_update(u)
            except Exception as e:  # one bad update must not stop the poller
                log.warning("telegram: update not handled (%s)", type(e).__name__)

    async def poll_forever(self) -> None:
        delay = 1.0
        while True:
            try:
                await self.poll_once()
                delay = 1.0
                await asyncio.sleep(0)  # a server that answers at once must not starve the loop
                continue
            except TelegramError as e:
                log.warning("telegram: %s", e)
                if e.status == 429 and e.retry_after:
                    await self._sleep(e.retry_after)
                    continue
            except Exception as e:  # anything else: type only, the text may hold the URL
                log.warning("telegram: poll failed (%s)", type(e).__name__)
            await self._sleep(delay)
            delay = min(delay * 2, BACKOFF_MAX_S)

    # ---- notifier ----

    def _may_receive(self, user_id: str, users: Mapping[str, Mapping[str, Any]]) -> bool:
        if user_id == LOCAL_ID:
            return not self.auth_enabled
        user = users.get(user_id)
        return user is not None and not user.get("disabled")

    def recipients(self, symbol: str) -> list[int]:
        users = {u["user_id"]: u for u in self.store.list_users()}
        out: list[int] = []
        for user_id, prefs in self.store.list_prefs().items():
            chat = prefs.get(PREF_KEY)
            if (
                isinstance(chat, int)
                and chat not in out
                and watches(prefs, symbol)
                and self._may_receive(user_id, users)
            ):
                out.append(chat)
        return out

    def notify(self, alert: Any) -> list[asyncio.Task]:
        """Start one send per due recipient of `alert`; anything but an open Alert is ignored."""
        if not isinstance(alert, Alert) or alert.status != "open":
            return []
        text = message_text(alert, self.public_url)
        now = self._clock()
        started = []
        for chat in self.recipients(alert.symbol):
            key = (chat, alert.symbol)
            last = self._last_sent.get(key)
            if last is not None and now - last < RATE_S:
                continue
            self._last_sent[key] = now
            t = asyncio.create_task(self._send(chat, text))
            self._sends.add(t)
            t.add_done_callback(self._sends.discard)
            started.append(t)
        return started

    async def _send(self, chat_id: int, text: str) -> None:
        async with self._send_slots:
            for _ in range(2):
                try:
                    await self.client.send_message(chat_id, text)
                    return
                except TelegramError as e:
                    log.warning("telegram: %s", e)
                    if e.status == 403:  # the user blocked the bot: stop writing to them
                        try:
                            self.unlink_chat(chat_id, "telegram_blocked")
                        except Exception as err:  # a store error must not escape the task
                            log.warning("telegram: unlink failed (%s)", type(err).__name__)
                        return
                    if e.status == 429 and e.retry_after:
                        await self._sleep(e.retry_after)
                        continue
                    return
                except Exception as e:
                    log.warning("telegram: send failed (%s)", type(e).__name__)
                    return

    async def consume(self, queue: asyncio.Queue) -> None:
        while True:
            m = await queue.get()
            try:
                self.notify(m)
            except Exception as e:  # e.g. the account store is down: the next alert may work
                log.warning("telegram: notify failed (%s)", type(e).__name__)

    # ---- lifecycle ----

    def start(self, bus: Any | None) -> None:
        """Start the poller, and the notifier when there is a bus (`bus.subscribe("alerts")`)."""
        self._tasks.append(asyncio.create_task(self.poll_forever()))
        if bus is not None:
            q = bus.subscribe("alerts")
            self._bus, self._queue = bus, q
            self._tasks.append(asyncio.create_task(self.consume(q)))

    async def stop(self) -> None:
        tasks = [*self._tasks, *self._sends]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        bus = getattr(self, "_bus", None)
        if bus is not None:
            bus.unsubscribe("alerts", self._queue)
            self._bus = None
        with contextlib.suppress(Exception):
            await self.client.aclose()
        if self._filter is not None:
            remove_redaction(self._filter)
            self._filter = None
