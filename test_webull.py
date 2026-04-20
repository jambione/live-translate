"""
test_webull.py — Test opening and adding a ticker to Webull Desktop watchlist.

Operates independently of LIVE_MODE in workflows.py.
  dry  — prints the steps, no keystrokes sent
  live — opens/focuses Webull, types the ticker, hits Enter

Usage
-----
  python test_webull.py              # prompts for ticker + mode
  python test_webull.py IXHL         # pre-fills ticker, prompts for mode
  python test_webull.py IXHL dry     # immediate dry run
  python test_webull.py IXHL live    # immediate live run (3s countdown)
"""

import sys
import time
import ctypes
import subprocess
import numpy as np
import pyautogui
import pygetwindow as gw

from workflows import WB_WINDOW, WB_LAUNCH

# How long to wait for the dropdown to appear after typing (seconds)
DROPDOWN_TIMEOUT = 4.0
# Pixel-change threshold — how much the screen region must change to count as "dropdown appeared"
DROPDOWN_THRESHOLD = 3.0

# ========================= WINDOW HELPERS =========================

def find_webull() -> object | None:
    """Return the first Webull window object, or None."""
    try:
        wins = gw.getWindowsWithTitle(WB_WINDOW)
        return wins[0] if wins else None
    except Exception:
        return None


def focus_webull(win) -> bool:
    """
    Bring the Webull window to the foreground using ctypes directly.
    Skips pygetwindow activate() which misreads Win32 error code 0 as failure.
    """
    try:
        hwnd = win._hWnd
        ctypes.windll.user32.ShowWindow(hwnd, 9)   # SW_RESTORE
        time.sleep(0.15)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        ctypes.windll.user32.BringWindowToTop(hwnd)
        time.sleep(0.35)
        return True
    except Exception as e:
        print(f"  ⚠️  Focus error: {e}")
        return False


def open_and_focus_webull():
    """
    Focus Webull if open, otherwise launch it and wait for the window.
    Returns the window object on success, None on failure.
    """
    win = find_webull()

    if win:
        print(f"  ✅ Webull already open: \"{win.title}\"")
        print(f"     Focusing...")
        return win if focus_webull(win) else None

    # Not open — launch it
    print(f"  🚀 Webull not found — launching from:")
    print(f"     {WB_LAUNCH}")
    try:
        subprocess.Popen([WB_LAUNCH])
    except Exception as e:
        print(f"  ❌ Launch failed: {e}")
        print(f"     Check WB_LAUNCH path in workflows.py")
        return None

    # Poll for the window to appear
    print(f"  ⏳ Waiting for Webull to open", end="", flush=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        win = find_webull()
        if win:
            print(" ✅")
            time.sleep(1.5)   # let the UI finish loading
            focus_webull(win)
            return win
        print(".", end="", flush=True)
        time.sleep(0.5)

    print("\n  ❌ Webull did not appear within 30s")
    return None


# ========================= PREFLIGHT =========================

def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def preflight():
    section("Preflight — Webull window detection")
    win = find_webull()
    if win:
        print(f"  ✅ Found   : \"{win.title}\"")
        print(f"  Position  : ({win.left}, {win.top})")
        print(f"  Size      : {win.width} × {win.height}")
        print(f"  Minimized : {win.isMinimized}")
    else:
        print(f"  ℹ️  Webull Desktop is not currently open.")
        print(f"     Will be launched from: {WB_LAUNCH}")


# ========================= DROPDOWN DETECTION =========================

def wait_for_dropdown(win, timeout: float = DROPDOWN_TIMEOUT) -> bool:
    """
    Poll the lower portion of the Webull window for pixel changes that indicate
    the search dropdown has appeared. Takes a baseline screenshot just before
    typing starts, then compares every 100ms until the mean pixel delta exceeds
    DROPDOWN_THRESHOLD or the timeout is reached.

    Returns True if the dropdown was detected, False on timeout.
    """
    # Watch a horizontal band just below the top of the window — where the
    # search field and its dropdown sit.
    margin = 10
    region = (
        win.left  + margin,
        win.top   + 40,             # skip title bar
        win.width - margin * 2,
        min(300, win.height // 3),  # top third of the window
    )

    try:
        baseline = np.array(pyautogui.screenshot(region=region), dtype=np.int16)
    except Exception:
        return False  # can't capture — fall through to timeout behaviour

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.1)
        try:
            current = np.array(pyautogui.screenshot(region=region), dtype=np.int16)
            delta   = np.abs(current - baseline).mean()
            if delta >= DROPDOWN_THRESHOLD:
                return True
        except Exception:
            pass

    return False


# ========================= TEST =========================

def run_add_wb(ticker: str, dry: bool) -> bool:
    section(f"Test: ADD_WB  ({'DRY RUN' if dry else 'LIVE'})  →  {ticker}")

    if dry:
        print(f"  Steps that would run:")
        print(f"    1. Open / focus Webull Desktop")
        print(f"    2. Ctrl+2  → Stocks tab")
        for i, ch in enumerate(ticker, 3):
            print(f"    {i}. press '{ch.lower()}'")
        print(f"    {len(ticker) + 3}. Wait for dropdown to appear")
        print(f"    {len(ticker) + 4}. press 'enter'")
        print(f"\n  [DRY RUN] No actions taken.")
        return True

    # ── Live ─────────────────────────────────────────────────────────────────
    win = open_and_focus_webull()
    if not win:
        return False

    hwnd = win._hWnd

    print(f"  Sending Ctrl+2 (Stocks tab) → pyautogui (Webull is focused)...")
    from workflows import _wb_post_char, _wb_post_enter
    pyautogui.hotkey("ctrl", "2")
    time.sleep(0.5)

    # Capture baseline BEFORE typing so we can detect the dropdown appearing
    margin = 10
    region = (win.left + margin, win.top + 40, win.width - margin * 2, min(300, win.height // 3))
    try:
        baseline = np.array(pyautogui.screenshot(region=region), dtype=np.int16)
    except Exception:
        baseline = None

    print(f"  Typing '{ticker}' (direct to HWND): ", end="", flush=True)
    for letter in ticker:
        print(letter, end="", flush=True)
        _wb_post_char(hwnd, letter)
        time.sleep(0.08)
    print()

    # Wait for dropdown
    print(f"  Waiting for dropdown", end="", flush=True)
    detected = False
    if baseline is not None:
        deadline = time.time() + DROPDOWN_TIMEOUT
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                current = np.array(pyautogui.screenshot(region=region), dtype=np.int16)
                if np.abs(current - baseline).mean() >= DROPDOWN_THRESHOLD:
                    detected = True
                    break
            except Exception:
                break
            print(".", end="", flush=True)

    if detected:
        print(f" ✅ dropdown detected")
        time.sleep(0.15)
    else:
        print(f" ⏱️  timeout — sending Enter anyway")

    _wb_post_enter(hwnd)
    time.sleep(0.3)

    print(f"\n  ✅ Done — check Webull watchlist for {ticker}")
    return True


# ========================= MAIN =========================

if __name__ == "__main__":
    cli_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else None
    cli_mode   = sys.argv[2].lower() if len(sys.argv) > 2 else None

    print("=" * 55)
    print("  Webull Watchlist Add — Test Script")
    print("=" * 55)

    preflight()

    # Mode selection
    if cli_mode == "live":
        dry = False
    elif cli_mode == "dry":
        dry = True
    else:
        print("\nSelect mode:")
        print("  d / dry  — show steps only, no keystrokes")
        print("  l / live — open Webull and type the ticker")
        mode_input = input("Mode [d/l]: ").strip().lower()
        dry = mode_input not in ("l", "live")

    # Test loop
    while True:
        if cli_ticker:
            ticker = cli_ticker
            cli_ticker = None
        else:
            raw = input("\nTicker to test (or 'q' to quit): ").strip().upper()
            if raw in ("Q", "QUIT", "EXIT", ""):
                break
            ticker = raw

        if not (2 <= len(ticker) <= 5 and ticker.isalpha()):
            print(f"  ⚠️  '{ticker}' doesn't look like a valid ticker (2–5 letters).")
            continue

        if not dry:
            print(f"\n  ⚡ Running in 3 seconds...")
            time.sleep(3)

        run_add_wb(ticker, dry)

        again = input("\n  Test another ticker? [y/n]: ").strip().lower()
        if again not in ("y", "yes"):
            break

    print("\nDone.")
