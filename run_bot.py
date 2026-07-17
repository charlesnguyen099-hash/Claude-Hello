"""
Auto-update runner — chay tren may local nhu server
- Moi 60 giay kiem tra git co commit moi khong
- Neu co -> git reset --hard (khong bi block boi local changes) -> restart bot
- API keys doc tu environment variables, khong bao gio thay doi
- Khong can lam gi them sau khi chay script nay
"""

import subprocess
import sys
import time
import os
import signal

BRANCH    = "claude/bybit-futures-auto-trading-6qmy79"
BOT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bybit_bot")
CHECK_SEC = 60
bot_process = None


def git_current_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def git_remote_commit() -> str:
    subprocess.run(["git", "fetch", "origin", BRANCH], capture_output=True)
    r = subprocess.run(["git", "rev-parse", f"origin/{BRANCH}"], capture_output=True, text=True)
    return r.stdout.strip()


def git_update():
    # reset --hard dam bao khong bi block boi local changes
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"])
    print(f"[RUNNER] Reset to origin/{BRANCH}")


def start_bot():
    global bot_process
    # Inherit env tu process cha -> API keys, BYBIT_TESTNET duoc giu nguyen
    print("[RUNNER] Starting bot...")
    bot_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=BOT_DIR,
        env=os.environ.copy()
    )
    print(f"[RUNNER] Bot started (PID={bot_process.pid})")


def stop_bot():
    global bot_process
    if bot_process and bot_process.poll() is None:
        print(f"[RUNNER] Stopping bot (PID={bot_process.pid})...")
        bot_process.terminate()
        try:
            bot_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bot_process.kill()
        bot_process = None
        print("[RUNNER] Bot stopped.")


def handle_exit(sig, frame):
    print("\n[RUNNER] Shutting down...")
    stop_bot()
    sys.exit(0)


signal.signal(signal.SIGINT,  handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


if __name__ == "__main__":
    print(f"[RUNNER] Branch: {BRANCH}")
    print(f"[RUNNER] Check interval: {CHECK_SEC}s")
    print(f"[RUNNER] API key set: {'YES' if os.environ.get('BYBIT_API_KEY') else 'NO — set BYBIT_API_KEY truoc khi chay'}")

    # Lay code moi nhat ngay khi khoi dong (tranh chay code cu)
    git_update()
    current = git_current_commit()
    print(f"[RUNNER] Starting at commit: {current[:7]}")
    start_bot()

    while True:
        time.sleep(CHECK_SEC)

        # Bot crash -> restart ngay
        if bot_process and bot_process.poll() is not None:
            print("[RUNNER] Bot exited unexpectedly — restarting...")
            git_update()
            current = git_current_commit()
            start_bot()
            continue

        # Kiem tra commit moi tren remote
        try:
            remote = git_remote_commit()
            if remote and remote != current:
                print(f"[RUNNER] New commit: {current[:7]} -> {remote[:7]} — updating...")
                stop_bot()
                git_update()
                current = git_current_commit()
                start_bot()
            else:
                print(f"[RUNNER] Up to date. Commit: {current[:7]}")
        except Exception as e:
            print(f"[RUNNER] Git check error: {e}")
