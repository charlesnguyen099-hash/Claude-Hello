"""
Auto-update runner — chay tren may local nhu server
- Moi 60 giay kiem tra git co commit moi khong
- Neu co -> git pull -> restart bot tu dong
- Khong can lam gi them sau khi chay script nay
"""

import subprocess
import sys
import time
import os
import signal

BRANCH      = "claude/bybit-futures-auto-trading-6qmy79"
BOT_DIR     = os.path.join(os.path.dirname(__file__), "bybit_bot")
CHECK_SEC   = 60   # kiem tra moi 60 giay
bot_process = None


def git_current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def git_remote_commit() -> str:
    subprocess.run(
        ["git", "fetch", "origin", BRANCH],
        capture_output=True
    )
    result = subprocess.run(
        ["git", "rev-parse", f"origin/{BRANCH}"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def git_pull():
    subprocess.run(["git", "pull", "origin", BRANCH])


def start_bot():
    global bot_process
    print("[RUNNER] Starting bot...")
    bot_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=BOT_DIR
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
    print(f"[RUNNER] Auto-update runner started. Branch: {BRANCH}")
    print(f"[RUNNER] Check interval: {CHECK_SEC}s")

    current = git_current_commit()
    start_bot()

    while True:
        time.sleep(CHECK_SEC)

        # Kiem tra bot con song khong
        if bot_process and bot_process.poll() is not None:
            print("[RUNNER] Bot exited unexpectedly — restarting...")
            start_bot()
            continue

        # Kiem tra commit moi
        try:
            remote = git_remote_commit()
            if remote and remote != current:
                print(f"[RUNNER] New commit detected: {current[:7]} -> {remote[:7]}")
                stop_bot()
                git_pull()
                current = git_current_commit()
                start_bot()
            else:
                print(f"[RUNNER] No update. Commit: {current[:7]}")
        except Exception as e:
            print(f"[RUNNER] Git check failed: {e}")
