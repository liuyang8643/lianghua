---
name: qmt-launch
description: Start or verify the local QMT client for WBR by using `trading/qmt.py`. Use when `xtquant` or `xtdata` calls fail because QMT is not running, when a task needs live QMT market/trading access, or before validating code paths that depend on a local `XtMiniQmt.exe` session.
---

# QMT Launch

Use this skill to bring up the local QMT GUI process and confirm the repo can talk to it.

## Workflow

1. Confirm `configs/env.py` exists and `QMT_ROOT_DIR` points at one or more candidate QMT `bin.x64` directories.
2. Start QMT from the repository root:

```powershell
uv run python trading/qmt.py
```

`trading/qmt.py` returns the existing `XtMiniQmt.exe` process if QMT is already running. When it launches a new instance, it copies `_linkMini` to `linkMini` under the selected QMT directory and then starts `XtMiniQmt.exe linkMini`.

3. Verify the process exists:

```powershell
Get-Process XtMiniQmt -ErrorAction SilentlyContinue
```

or:

```powershell
uv run python -c "from trading.qmt import get_qmt_process; p = get_qmt_process(); print(None if p is None else p.pid)"
```

4. Verify `xtquant` connectivity before deeper analysis. Prefer a low-level probe first:

```powershell
uv run python -c "from xtquant import xtdata; c = xtdata.get_client(); print('connected' if c and c.is_connected() else 'disconnected')"
```

If a task needs actual market data, run a second probe after QMT is up:

```powershell
uv run python -c "from xtquant import xtdata; print(len(xtdata.get_sector_list()))"
```

5. If connectivity still fails, inspect these conditions before changing repository code:

- QMT is open and logged in.
- One of the configured `QMT_ROOT_DIR` entries is valid.
- `XtMiniQmt.exe` remains alive after launch.
- The current Python environment matches the project dependencies and can import `pandas`, `numpy`, and `pyarrow` cleanly.

## Notes

- Run commands from the repository root so `configs.env` resolves correctly.
- Use `uv run` instead of the system Python to stay on the project interpreter and dependency set.
- Do not kill an existing QMT process unless the user explicitly asks for it; the running GUI session may be in active use.
