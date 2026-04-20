"""
test_tradingview.py — Step-by-step TradingView workflow test.

Steps
-----
  1. Launch / Focus TradingView
  2. Type ticker letter-by-letter  (opens TV search overlay)
  3. Wait for dropdown              (pixel-delta detection)
  4. Press Enter to load chart
  5. Alt+W to add to watchlist

Usage
-----
  python test_tradingview.py              # prompts for ticker
  python test_tradingview.py AAPL         # pre-fills ticker
"""

import sys
import time
import ctypes
import subprocess
import os
import numpy as np
import pyautogui

# ── Win32 constants ───────────────────────────────────────────────────────────
_user32      = ctypes.windll.user32
SW_RESTORE   = 9
WM_KEYDOWN   = 0x0100
WM_KEYUP     = 0x0101
VK_RETURN    = 0x0D

# ── TradingView identity ──────────────────────────────────────────────────────
TV_TITLE_FRAGMENT = "TradingView"     # substring that appears in the window title
TV_FOLDER   = r"C:\Program Files\WindowsApps\TradingView.Desktop_3.0.0.7652_x64__n534cwy3pjxzj"
TV_EXE_NAME = "TradingView.exe"
TV_AUMID    = "TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop"

LAUNCH_TIMEOUT   = 30     # seconds to wait for window after launch
FOCUS_SETTLE     = 2.5    # seconds to let app load before typing
DROPDOWN_TIMEOUT = 5.0    # seconds to wait for search dropdown
DROPDOWN_THRESH  = 3.0    # mean pixel delta → "dropdown appeared"
CHART_SETTLE     = 2.0    # seconds after Enter before Alt+W


# ─────────────────────────────────────────────────────────────────────────────
# WIN32 WINDOW DETECTION  (bypasses pygetwindow — works with Store/UWP apps)
# ─────────────────────────────────────────────────────────────────────────────

# Use c_size_t for HWND (pointer-sized on both 32/64-bit), c_int return so
# the callback is well-defined on 64-bit Windows.
_EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t)


def _enum_windows(fragment: str, require_visible: bool = False) -> list[tuple[int, str]]:
    """Return list of (hwnd, title) for windows whose title contains fragment."""
    results = []
    fragment_lower = fragment.lower()

    def callback(hwnd, _):
        if require_visible and not _user32.IsWindowVisible(hwnd):
            return 1
        length = _user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return 1
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if not fragment_lower or fragment_lower in title.lower():
            results.append((hwnd, title))
        return 1   # must return non-zero to keep enumerating

    _user32.EnumWindows(_EnumWindowsProc(callback), 0)
    return results


def find_tv_hwnd() -> tuple[int, str] | None:
    """Return (hwnd, title) for the TradingView window, or None.
    Tries visible windows first; falls back to all windows so UWP frames are caught."""
    for require_visible in (True, False):
        hits = _enum_windows(TV_TITLE_FRAGMENT, require_visible=require_visible)
        if hits:
            return hits[0]
    return None


def focus_hwnd(hwnd: int) -> bool:
    """Bring TradingView to the foreground and click its center to give genuine input focus."""
    try:
        # Print window class so we know exactly what we're focusing
        cls_buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, cls_buf, 256)
        print(f"  Window class: {cls_buf.value}")

        _user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.2)
        _user32.SetForegroundWindow(hwnd)
        _user32.BringWindowToTop(hwnd)
        time.sleep(0.3)

        # Use pygetwindow to get coordinates in pyautogui's logical pixel space,
        # then click the center — this gives genuine OS input focus.
        import pygetwindow as gw
        wins = gw.getWindowsWithTitle(TV_TITLE_FRAGMENT)
        if wins:
            w = wins[0]
            cx, cy = w.left + w.width // 2, w.top + w.height // 2
            print(f"  Clicking center ({cx}, {cy}) via pygetwindow coords")
            pyautogui.click(cx, cy)
            time.sleep(0.4)
        else:
            print(f"  ⚠️  pygetwindow couldn't find TV — skipping click")

        fg = _user32.GetForegroundWindow()
        print(f"  GetForegroundWindow: {fg}  (expected {hwnd})")
        return True
    except Exception as e:
        print(f"  ⚠️  Focus error: {e}")
        return False


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Return (left, top, width, height) for a given HWND."""
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
    r = RECT()
    _user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def _try_powershell() -> bool:
    print(f"     [1] PowerShell Start-Process '{TV_AUMID}'")
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f'Start-Process "shell:appsFolder\\{TV_AUMID}"'],
            timeout=8, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
        print(f"         stderr: {r.stderr.strip()}")
        return False
    except Exception as e:
        print(f"         error: {e}")
        return False


def _try_explorer() -> bool:
    print(f"     [2] explorer.exe shell:appsFolder\\{TV_AUMID}")
    try:
        subprocess.Popen(["explorer.exe", f"shell:appsFolder\\{TV_AUMID}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"         error: {e}")
        return False


def _try_direct_exe() -> bool:
    exe = os.path.join(TV_FOLDER, TV_EXE_NAME)
    print(f"     [3] Direct exe: {exe}")
    if not os.path.isfile(exe):
        print(f"         not found")
        return False
    try:
        subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"         error: {e}")
        return False


def _try_cmd_start() -> bool:
    print(f"     [4] cmd /c start shell:appsFolder\\{TV_AUMID}")
    try:
        subprocess.Popen(f'start "" "shell:appsFolder\\{TV_AUMID}"',
                         shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"         error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LAUNCH / FOCUS
# ─────────────────────────────────────────────────────────────────────────────

def step_launch() -> int | None:
    """Open and focus TradingView. Returns HWND on success, None on failure."""
    print("\n── Step 1: Launch / Focus ───────────────────────────────────────")

    hit = find_tv_hwnd()
    if hit:
        hwnd, title = hit
        print(f"  ✅ Already open: \"{title}\"  (HWND {hwnd})")
        ok = focus_hwnd(hwnd)
        print(f"  {'✅ Focused' if ok else '❌ Focus failed'}")
        return hwnd if ok else None

    print(f"  🚀 Not open — trying launch strategies:")
    for strategy in (_try_powershell, _try_explorer, _try_direct_exe, _try_cmd_start):
        if strategy():
            break
    else:
        print(f"  ❌ All strategies failed.")
        return None

    # Poll for up to LAUNCH_TIMEOUT seconds; fall back to manual confirm
    print(f"  ⏳ Waiting for window", end="", flush=True)
    deadline = time.time() + LAUNCH_TIMEOUT
    while time.time() < deadline:
        hit = find_tv_hwnd()
        if hit:
            hwnd, title = hit
            print(f" ✅  \"{title}\"  (HWND {hwnd})")
            print(f"  ⏳ Letting app settle ({FOCUS_SETTLE}s)...")
            time.sleep(FOCUS_SETTLE)
            focus_hwnd(hwnd)
            return hwnd
        print(".", end="", flush=True)
        time.sleep(0.5)

    # Detection failed — dump titles to help diagnose, then let user continue manually
    print(f"\n  ⚠️  Could not detect TradingView window automatically.")
    print(f"  🔍 All windows with a title right now:")
    for hwnd, title in _enum_windows(""):
        print(f"      [{hwnd}] \"{title}\"")

    input("\n  👉 Focus TradingView manually, then press Enter to continue...")
    # Return a sentinel (-1) so the calling steps know there's no HWND for rect queries
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TYPE TICKER  +  STEP 3 — WAIT FOR DROPDOWN
# ─────────────────────────────────────────────────────────────────────────────

def step_type_and_wait(hwnd: int, ticker: str) -> bool:
    print(f"\n── Step 2: Open search + type '{ticker}' ────────────────────────")

    # Re-detect the live window — TradingView may have swapped HWNDs after loading
    hit = find_tv_hwnd()
    if hit:
        hwnd, title = hit
        print(f"  Live HWND: {hwnd}  \"{title}\"")
        focus_hwnd(hwnd)
        time.sleep(0.3)
    else:
        print(f"  ⚠️  Could not re-detect window — using original HWND {hwnd}")

    # Open symbol search explicitly with Ctrl+K (works regardless of TV internal focus)
    print(f"  Sending Ctrl+K to open symbol search...")
    pyautogui.hotkey("ctrl", "k")
    time.sleep(0.5)   # wait for search overlay to open

    # Capture baseline AFTER search opens, BEFORE typing the ticker
    left, top, width, height = get_window_rect(hwnd)
    margin = 10
    region = (left + margin, top + 40, width - margin * 2, min(300, height // 3))
    try:
        baseline = np.array(pyautogui.screenshot(region=region), dtype=np.int16)
    except Exception:
        baseline = None

    # Type each letter into the open search field
    print(f"  Typing: ", end="", flush=True)
    for letter in ticker:
        print(letter, end="", flush=True)
        pyautogui.press(letter.lower())
        time.sleep(0.12)
    print()

    # ── Step 3: wait for dropdown results ────────────────────────────────────
    print(f"\n── Step 3: Wait for search results ─────────────────────────────")
    detected = False
    if baseline is not None:
        print(f"  Polling for pixel change", end="", flush=True)
        deadline = time.time() + DROPDOWN_TIMEOUT
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                current = np.array(pyautogui.screenshot(region=region), dtype=np.int16)
                delta = np.abs(current - baseline).mean()
                if delta >= DROPDOWN_THRESH:
                    detected = True
                    print(f" ✅ detected (Δ={delta:.1f})")
                    break
            except Exception:
                break
            print(".", end="", flush=True)
        if not detected:
            print(f" ⏱️  timeout — will Enter anyway")
    else:
        print(f"  ⚠️  No baseline — using fixed wait")
        time.sleep(1.5)

    time.sleep(0.2 if detected else 0.5)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CONFIRM  (Enter)
# ─────────────────────────────────────────────────────────────────────────────

def step_confirm() -> bool:
    print(f"\n── Step 4: Confirm selection (Enter) ────────────────────────────")
    print(f"  (The first result in the dropdown will be selected)")
    pyautogui.press("enter")
    print(f"  ↵ Enter sent")
    print(f"  ⏳ Waiting {CHART_SETTLE}s for chart to load...")
    time.sleep(CHART_SETTLE)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ADD TO WATCHLIST  (Alt+W)
# ─────────────────────────────────────────────────────────────────────────────

def step_add_watchlist() -> bool:
    print(f"\n── Step 5: Add to watchlist (Alt+W) ─────────────────────────────")
    print(f"  (The ticker showing in the chart will be added to the watchlist)")
    pyautogui.hotkey("alt", "w")
    print(f"  ✅ Alt+W sent")
    print(f"\n  👀 Check: did a watchlist popup/confirmation appear in TradingView?")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else None

    print("=" * 58)
    print("  TradingView Workflow Test")
    print("=" * 58)

    if cli_ticker:
        ticker = cli_ticker
    else:
        raw = input("\nTicker (e.g. AAPL): ").strip().upper()
        ticker = raw if raw else "AAPL"

    if not (2 <= len(ticker) <= 5 and ticker.isalpha()):
        print(f"⚠️  '{ticker}' doesn't look like a valid ticker.")
        sys.exit(1)

    print(f"\n⚡ Running in 3 seconds...")
    time.sleep(3)

    hwnd = step_launch()
    if hwnd is None:
        print("\n❌ Stopped at Step 1.")
        sys.exit(1)

    time.sleep(0.5)   # let the click-focus settle
    step_type_and_wait(hwnd, ticker)
    step_confirm()
    step_add_watchlist()

    print("\n" + "=" * 58)
    print(f"  Done — verify {ticker} in TradingView watchlist")
    print("=" * 58)
