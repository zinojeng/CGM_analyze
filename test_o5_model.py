"""簡單的 OpenAI 模型測試工具。

用法：
    python test_o5_model.py --model gpt-4o-mini --prompt "Hi"

預設會從環境變數 OPENAI_API_KEY 讀取金鑰，也可透過 --api-key 傳入。
"""

import argparse
import os
import sys
from typing import Optional

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenAI chat completion model")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="要測試的模型名稱 (預設: gpt-4o-mini)"
    )
    parser.add_argument(
        "--prompt",
        default="請回覆：測試成功",
        help="傳給模型的訊息內容"
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="直接傳入 OpenAI API Key，若未提供將讀取環境變數 OPENAI_API_KEY"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="回應的最大 tokens 數"
    )
    return parser.parse_args()


def get_api_key(cli_key: Optional[str]) -> str:
    api_key = cli_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[錯誤] 請透過 --api-key 或環境變數 OPENAI_API_KEY 提供 OpenAI 金鑰")
    return api_key


def _call_chat_completion(client: OpenAI, model: str, messages, max_tokens: int):
    """呼叫 Chat Completions API，兼容新舊參數名稱。"""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
    except Exception as exc:  # pylint: disable=broad-except
        msg = str(exc)
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0,
            )
        raise


def main() -> None:
    args = parse_args()
    api_key = get_api_key(args.api_key)

    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    try:
        response = _call_chat_completion(client, args.model, messages, args.max_tokens)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"調用失敗：{exc}")
        sys.exit(1)

    choice = response.choices[0]

    print("=== 測試資訊 ===")
    print(f"模型：{args.model}")
    print(f"提示訊息：{args.prompt}")
    print("--- 回應 ---")
    print(choice.message.content)

    usage = getattr(response, "usage", None)
    if usage:
        print("--- Token 使用量 ---")
        print(f"prompt_tokens = {usage.prompt_tokens}")
        print(f"completion_tokens = {usage.completion_tokens}")
        print(f"total_tokens = {usage.total_tokens}")


if __name__ == "__main__":
    main()
