import os

import requests


def send_telegram_message(message: str, cfg: dict) -> bool:
    """Sends a message via the Telegram Bot API. Token/chat ID are read from
    environment variables (names set in config.yaml) — never hardcoded.
    Get a bot token from @BotFather and your chat ID from @userinfobot on
    Telegram, then export STOCK_TOOL_TELEGRAM_BOT_TOKEN and
    STOCK_TOOL_TELEGRAM_CHAT_ID before running the screener. Returns False
    (rather than raising) when credentials are missing, so the screener can
    still run and print results without them.
    """
    bot_token = os.environ.get(cfg["telegram"]["bot_token_env"])
    chat_id = os.environ.get(cfg["telegram"]["chat_id_env"])

    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=15)
    return resp.ok
