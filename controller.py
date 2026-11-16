"""The small HM controller interface, with raw local-server request accounting."""

import asyncio
import datetime
import json
import math
import os
import time
import urllib.request
from pathlib import Path


def sampling_parameters(value=None):
    """Validate explicit vLLM sampling; old frozen plans retain HM/GPT-era defaults."""
    legacy = {"temperature": 0.2, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
              "presence_penalty": 0.0, "repetition_penalty": 1.0}
    if value is None:
        return legacy
    if not isinstance(value, dict) or set(value) != set(legacy):
        raise ValueError("sampling must declare exactly temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values()) or not (
        value["temperature"] >= 0 and 0 < value["top_p"] <= 1
        and type(value["top_k"]) is int and value["top_k"] >= 0
        and 0 <= value["min_p"] <= 1 and -2 <= value["presence_penalty"] <= 2
        and value["repetition_penalty"] > 0
    ):
        raise ValueError("sampling contains an invalid type, nonfinite value or out-of-range parameter")
    return dict(value)


def plain(value):
    """Serialize simulator arrays and events without importing its numerical runtime."""
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def record(path, value):
    """Append and flush one inspectable record; partial episodes keep their evidence."""
    with Path(path).open("a") as stream:
        stream.write(json.dumps(plain(value), ensure_ascii=False) + "\n")


def save(path, value):
    """Atomically publish a JSON manifest on the same filesystem."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Controller:
    """HM's async_batch_prompt contract; seeds, failures and every retry remain visible."""

    def __init__(self, url, model, output, thinking, seed, max_tokens=4000, max_requests=256,
                 thinking_token_budget=None, sampling=None):
        if not url.startswith("http://127.0.0.1:"):
            raise ValueError("Hardware accounting requires a server on the same allocated node")
        if thinking_token_budget is not None and (
            type(thinking_token_budget) is not int or not 0 < thinking_token_budget < max_tokens
        ):
            raise ValueError("thinking_token_budget must be a positive integer below max_tokens")
        self.url, self.model, self.output = url.rstrip("/"), model, Path(output)
        self.thinking, self.seed = thinking, seed
        self.max_tokens, self.max_requests, self.calls = max_tokens, max_requests, 0
        self.thinking_token_budget = thinking_token_budget if thinking else None
        self.sampling = sampling_parameters(sampling)
        self.total_inference_cost = 0  # Upstream display compatibility; no monetary price is inferred.

    async def request(self, payload):
        """Use the standard-library HTTP client without unrecorded automatic retries."""
        request = urllib.request.Request(
            self.url + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )

        def read():
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)

        return await asyncio.to_thread(read)

    async def complete(self, system, user, temperature=None, json_output=False):
        """Return one completion while retaining native usage and full response JSON."""
        if self.calls >= self.max_requests:
            raise RuntimeError("Episode request limit reached; preserve this episode as failed")
        index, self.calls = self.calls, self.calls + 1
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload = {
            **self.sampling,
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.sampling["temperature"] if temperature is None else temperature,
            "seed": self.seed + index,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        if self.thinking_token_budget is not None:
            payload["thinking_token_budget"] = self.thinking_token_budget
        row = {
            "call": index,
            "messages": messages,
            "model": self.model,
            "thinking": self.thinking,
            "max_tokens": self.max_tokens,
            "thinking_token_budget": self.thinking_token_budget,
            "temperature": payload["temperature"],
            "top_p": payload["top_p"],
            "sampling": {key: payload[key] for key in self.sampling},
            "sampling_seed": self.seed + index,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "backend_batch_start_monotonic": time.monotonic(),
            "status": "failed",
        }
        try:
            response = await self.request(payload)
            row["response"] = response
            usage = response.get("usage") or {}
            row.update(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                returned_model=response.get("model"),
                reasoning_tokens_reported=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            )
            choice = response["choices"][0]
            row["finish_reason"] = choice.get("finish_reason")
            if row["finish_reason"] == "length":
                row["failure_kind"] = "output_token_limit"
                raise RuntimeError(
                    f"Completion truncated at max_tokens={self.max_tokens}; "
                    f"reasoning_tokens_reported={row['reasoning_tokens_reported']}. No complete final answer."
                )
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                row["failure_kind"] = "empty_final_answer"
                raise ValueError("Server returned no final answer")
            row["status"] = "complete"
            return answer
        except BaseException as error:
            row["error"] = repr(error)
            raise
        finally:
            row["backend_batch_end_monotonic"] = time.monotonic()
            record(self.output / "requests.jsonl", row)

    async def async_batch_prompt(self, expertise, messages, temperature=None):
        """Keep upstream hypothesis batching and its response order."""
        return await asyncio.gather(
            *(self.complete(expertise, message, temperature) for message in messages)
        )
