"""
test_workflows.py — Interactive test suite for all workflow actions.

Modes
-----
  dry   : Just prints what would happen — no windows focused, no keystrokes sent.
  live  : Actually focuses windows and fires keystrokes. Apps must be open.

Usage
-----
  python test_workflows.py          # prompts for mode
  python test_workflows.py dry      # always dry-run
  python test_workflows.py live     # always live
"""

import sys
import time
import pyperclip
import pygetwindow as gw

from workflows import (
    TV_WINDOW, WB_WINDOW,
    find_window_title,
    workflow_add_tv,
    workflow_buy,
    workflow_sell_all,
)

# ========================= HELPERS =========================

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"

results: list[tuple[str, str]] = []


def record(name: str, ok: bool | None):
    if ok is None:
        results.append((name, SKIP))
    elif ok:
        results.append((name, PASS))
    else:
        results.append((name, FAIL))


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def confirm(prompt: str) -> bool:
    ans = input(f"{prompt} [y/n]: ").strip().lower()
    return ans in ("y", "yes", "")


# ========================= PREFLIGHT CHECKS =========================

def check_windows() -> dict[str, str | None]:
    """Detect which target windows are currently open."""
    section("Preflight — window detection")

    tv = find_window_title(TV_WINDOW)
    wb = find_window_title(WB_WINDOW)

    if tv:
        print(f"  📺 TradingView found  : \"{tv}\"")
    else:
        print(f"  📺 TradingView        : NOT FOUND ('{TV_WINDOW}' not open)")

    if wb:
        print(f"  📊 Webull Desktop found: \"{wb}\"")
    else:
        print(f"  📊 Webull Desktop     : NOT FOUND ('{WB_WINDOW}' not open)")

    return {"tv": tv, "wb": wb}


def check_clipboard():
    section("Preflight — clipboard")
    try:
        pyperclip.copy("TEST_CLIP")
        readback = pyperclip.paste()
        ok = readback == "TEST_CLIP"
        print(f"  Clipboard write/read : {'OK' if ok else 'FAILED'}")
        record("clipboard read/write", ok)
    except Exception as e:
        print(f"  Clipboard error: {e}")
        record("clipboard read/write", False)


# ========================= INDIVIDUAL WORKFLOW TESTS =========================

def test_add_tv(ticker: str, dry: bool):
    section(f"Test: ADD_TV  ({ticker})")
    pyperclip.copy(ticker)
    ok = workflow_add_tv(ticker, dry_run=dry)
    if not dry:
        time.sleep(0.6)
        ok = ok  # window success already reported inside workflow
    record(f"ADD_TV ({ticker})", ok if not dry else None)



def test_buy(ticker: str, dry: bool):
    section(f"Test: BUY  ({ticker})")
    pyperclip.copy(ticker)
    ok = workflow_buy(ticker, dry_run=dry)
    if not dry:
        time.sleep(0.6)
    record(f"BUY ({ticker})", ok if not dry else None)


def test_sell_all(ticker: str, dry: bool):
    section(f"Test: SELL ALL  ({ticker})")
    pyperclip.copy(ticker)
    ok = workflow_sell_all(ticker, dry_run=dry)
    if not dry:
        time.sleep(0.6)
    record(f"SELL ALL ({ticker})", ok if not dry else None)


def test_watch(ticker: str, dry: bool):
    """WATCH fires ADD_TV (TradingView watchlist)."""
    section(f"Test: WATCH  ({ticker})")
    pyperclip.copy(ticker)
    ok = workflow_add_tv(ticker, dry_run=dry)
    record(f"WATCH ({ticker})", ok if not dry else None)


# ========================= SUMMARY =========================

def print_summary():
    section("Results")
    max_len = max(len(name) for name, _ in results) if results else 0
    passed = sum(1 for _, s in results if s == PASS)
    failed = sum(1 for _, s in results if s == FAIL)
    skipped = sum(1 for _, s in results if s == SKIP)

    for name, status in results:
        print(f"  {name:<{max_len}}  {status}")

    print(f"\n  {passed} passed  |  {failed} failed  |  {skipped} skipped (dry run)")
    if failed:
        print("\n  ⚠️  Check that target apps are open and window titles match:")
        print(f"       TV_WINDOW = \"{TV_WINDOW}\"")
        print(f"       WB_WINDOW = \"{WB_WINDOW}\"")
        print("     Run find_windows.py to see exact titles.")


# ========================= MENU =========================

MENU = """
  1  WATCH    — TradingView watchlist add
  2  BUY      — Webull Shift+B
  3  SELL ALL — Webull Shift+A
  4  Run ALL  — all 3 tests back-to-back
  q  Quit
"""

DEFAULT_TICKER = "AAPL"


def run_menu(dry: bool, windows: dict):
    mode_label = "DRY RUN" if dry else "LIVE"
    print(f"\n🧪  Workflow Test Suite  [{mode_label}]")
    print(f"    Default ticker: {DEFAULT_TICKER}")
    print(MENU)

    while True:
        choice = input("Choice (1-6 / q): ").strip().lower()

        if choice in ("q", "quit", "exit"):
            break

        # Optionally override ticker
        ticker_input = input(f"  Ticker [Enter = {DEFAULT_TICKER}]: ").strip().upper()
        ticker = ticker_input if ticker_input else DEFAULT_TICKER

        if not dry:
            print(f"\n  ⚡ Running LIVE in 2 seconds — switch to target app if needed…")
            time.sleep(2)

        if choice == "1":
            test_watch(ticker, dry)
        elif choice == "2":
            test_buy(ticker, dry)
        elif choice == "3":
            test_sell_all(ticker, dry)
        elif choice == "4":
            test_watch(ticker, dry)
            time.sleep(1.0)
            test_buy(ticker, dry)
            time.sleep(1.0)
            test_sell_all(ticker, dry)
        else:
            print("  Invalid choice.")
            continue

        print()

    print_summary()


# ========================= ENTRY POINT =========================

if __name__ == "__main__":
    # Determine mode from CLI arg or prompt
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if arg == "dry":
        dry = True
    elif arg == "live":
        dry = False
    else:
        print("Select mode:")
        print("  d / dry  — print actions only, no keystrokes")
        print("  l / live — actually focus windows and fire keys")
        mode_input = input("Mode [d/l]: ").strip().lower()
        dry = mode_input not in ("l", "live")

    windows = check_windows()
    check_clipboard()

    if not dry and not windows["tv"] and not windows["wb"]:
        print("\n⚠️  Neither TradingView nor Webull Desktop is open.")
        print("   Open the apps first, or switch to dry-run mode.")
        if not confirm("Continue anyway?"):
            sys.exit(0)

    run_menu(dry, windows)
