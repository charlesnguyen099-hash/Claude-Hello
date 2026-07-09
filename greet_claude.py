#!/usr/bin/env python3
"""Send a daily morning greeting to Claude at 6am."""

import anthropic
from datetime import datetime


def greet_claude():
    client = anthropic.Anthropic()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[
            {"role": "user", "content": f"Hello! Chào buổi sáng Claude! Hôm nay là {now}. Chúc một ngày tốt lành!"}
        ]
    )

    print(f"[{now}] Sent greeting to Claude.")
    print(f"Claude replied: {message.content[0].text}")


if __name__ == "__main__":
    greet_claude()
