# Telegram alerts + watchlist (ROADMAP Phase 1.1–1.3)

Goal: the system reaches the user. When a new alert opens on a coin the user watches, a Thai message arrives in their Telegram within seconds. Nothing sends orders, nothing asks for exchange keys, and nothing gives investment advice.

## Enablement
- Env `TELEGRAM_BOT_TOKEN` turns it on. If unset or empty, the feature is fully off: no task, no network, and the UI hides it.
  - The token is never logged, printed, put in an error message, or returned by any API. Error logs show the method name and the HTTP status only.
- Env `TELEGRAM_API_BASE` (default `https://api.telegram.org`) exists only for tests or a local fake.
- Env `IDSAI_PUBLIC_URL` (optional) is the site link added to messages.
- The user creates the bot with @BotFather and sets the token on Render themselves. Add the key to `render.yaml` as `sync: false`.

## Transport: long polling (no webhook)
- A background asyncio task in the API process calls `getMe` once to learn the bot username, then loops on `getUpdates(timeout=25, offset)`.
- There is no public webhook endpoint, so there is no webhook secret to protect.
- On errors it backs off: 1, 2, 4 … up to 60 s. On HTTP 429 it honours `retry_after`.
- It uses the existing `httpx` dependency (AsyncClient). The task stops cleanly on app shutdown.

## Linking an account
1. A signed-in user does `POST /api/me/telegram/link` (same-origin, Signed).
   - The server creates a random one-time code: `secrets.token_urlsafe`, ≤ 64 characters, the Telegram start-param charset.
   - It is kept in memory only, with a 10-minute TTL. One code per user; a new request replaces the old one.
   - It returns `{"url": "https://t.me/<bot>?start=<code>", "expires_s": 600}`.
2. The user opens the link and presses Start. The poller receives `/start <code>` in a private chat.
   - **Valid, unexpired code:** store `chat_id` against that user and delete the code. Reply in Thai: "เชื่อมบัญชี <username> แล้ว พิมพ์ /stop เพื่อยกเลิก". It shows the username (checked by `check_username`), never the free-text display name.
   - **Storing the link fails** (e.g. the account store is down): reply "เชื่อมไม่สำเร็จ ลองกดเชื่อมใหม่บนเว็บ". The code stays used up.
   - **Invalid or expired code:** reply in Thai saying it has expired, and to press link again on the site.
   - Messages that are not from a private chat are ignored.
3. Unlinking:
   - `/stop` in the chat unlinks every user bound to that chat_id.
   - `DELETE /api/me/telegram` unlinks the current user.
4. Storage: a server-only pref key `tg_chat_id` in the existing `user_prefs` (Postgres in prod), set and removed only by this code.
   - `PrefsIn` (extra = forbid) must not accept it.
   - It must be stripped from every API response that returns prefs: `/api/me`, the login response and `PUT /api/me/prefs`. Use one helper, `public_prefs()`.
   - Instead, `/api/me` returns `"telegram": {"enabled": bool, "linked": bool}`.
5. When auth is off (the local user), the feature works the same way for `local`.

## Watchlist
- A new `PrefsIn` field `watch: list[str] | None` (max 50). Each entry must be in `state.symbols`, otherwise 422. Null removes it.
- If it is absent, the user watches every symbol. An empty list means notify nothing.
- UI: in the existing user/settings area, a Thai section "แจ้งเตือน Telegram" has:
  - a link/unlink button with status
  - checkboxes per symbol for the watchlist, saved through the existing `savePrefs`
  - when not enabled, the section is hidden

## Notifying
- `TelegramNotifier` subscribes to the bus topic `alerts` and acts only on `Alert` objects with `status == "open"`. It ignores dict events such as `alerts_reset` and `replay_restarted`, and ignores `updated`/`closed`.
- Recipients are the users with `tg_chat_id` whose watchlist contains `alert.symbol`, read from the account store.
  - Only enabled accounts that exist in `users` count. The built-in `local` row counts only when auth is off.
  - Disabling an account (`Store.set_disabled`, so both the admin route and the CLI) removes its `tg_chat_id`. Re-enabling it needs a new link.
- Rate limit: at most 1 message per (chat_id, symbol) per 600 s, in memory. Each send runs off the bus loop, so a slow Telegram call never blocks the pipeline.
- Message (Thai, plain text, no Markdown parse mode, so there are no escaping bugs):
  ```
  ⚠️ ตรวจพบความผิดปกติทางสถิติ · <SYMBOL>
  <time in Asia/Bangkok, HH:MM:SS DD/MM/YYYY> (เวลาไทย)
  <explanation text from idsai.product.explainer.template.explain(alert)>
  ข้อมูลในอดีต (pump จริงที่ระบบจับได้ คู่ BTC 2018–2021): 1 ชม. หลังเตือน ราคาต่ำกว่าตอนเตือน 89% ของครั้ง แต่ในชั่วโมงนั้นราคายังขึ้นต่อได้ (มัธยฐาน +15%)
  เป็นข้อมูลความผิดปกติทางสถิติ ไม่ใช่คำแนะนำการลงทุน และไม่ได้กล่าวหาผู้ใด
  <IDSAI_PUBLIC_URL if set>
  ```
  The historical line is a module constant with its source noted in a comment (`docs/ROADMAP.md`, Phase 0 results). Never call F a probability.
- A send failure for one chat is logged without the token and does not affect the other chats.
- If Telegram answers 403 (the user blocked the bot), unlink that chat. A store error while unlinking is logged by type only and never escapes the send task.
- The header time is labelled "(เวลาไทย)", because the explainer text inside the message gives the time in UTC.
- The historical line keeps both numbers from `docs/ROADMAP.md` (Phase 0): 88.8 % below the alert price after 1 h, and a median highest price within that hour (`max_up_1h`) of +15.5 %. They are always shown together.

## Audit
- Route side: `telegram_link_code` (a link was issued) and `telegram_unlink` (`DELETE`, only when a link existed).
- Bot side, through the same `store.add_audit`: `telegram_link` (/start), `telegram_stop` (/stop) and `telegram_blocked` (a 403 unlink), one row per affected user.
- The Activity filter lists these actions.

## Code layout
- New `idsai/product/notify/__init__.py` and `idsai/product/notify/telegram.py`:
  - `TelegramClient` (thin httpx wrapper: `get_me`, `get_updates`, `send_message`)
  - `LinkCodes`
  - `TelegramService`, which owns the poller and the notifier, built from env
- Wire it into app startup and shutdown where the app already starts background work. See `create_app` and how `serve` builds the state in `idsai/cli.py`.
  - Replay mode works too, so an inject demo produces a Telegram message.
- `idsai/product/api/app.py`: the routes, `public_prefs`, `PrefsIn.watch`, and `telegram` in `/api/me`.
- Web: `app.js`, `hud.js` and `index.html` as needed. The style matches the existing HUD.

## Tests (no real network, never the real data/)
- Use `httpx.MockTransport` for every Telegram call.
- Cover:
  - the link flow: valid, expired, wrong code, not a private chat
  - `/stop`
  - `DELETE` unlink
  - `tg_chat_id` never appears in any prefs response, and `PrefsIn` rejects it
  - watch validation
  - the notifier sends only on `open` for watched symbols
  - the rate limit
  - a 403 auto-unlinks
  - a 429 retry_after is honoured
  - the token never appears in logs (caplog)
  - feature off when the env var is missing
  - the message contains the disclaimer and no "ควรซื้อ/ขาย"

## Out of scope
- Discord and email.
- Per-user quiet hours.
- Changing the web alert card (separate step).
