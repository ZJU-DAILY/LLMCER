import os
import threading
from functools import partial, wraps
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


# Single shared OpenAI client. Reads OPENAI_API_KEY from the environment as
# usual; additionally honours OPENAI_BASE_URL so that any OpenAI-compatible
# endpoint (a relay, a proxy, Azure, vLLM, etc.) can be used without code
# changes. Leave OPENAI_BASE_URL unset to talk to api.openai.com directly.
_BASE_URL = os.environ.get("OPENAI_BASE_URL") or None
_OPENAI_CLIENT = OpenAI(base_url=_BASE_URL) if _BASE_URL else OpenAI()


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def openai_chat_complete(
    messages,
    model,
    client=_OPENAI_CLIENT,
    **kwargs,
):
    response = client.chat.completions.create(messages=messages, model=model, **kwargs)
    if response.choices is None:
        raise ValueError(f"Error response: {response}")
    return response


class Seq2SeqWrapper:
    def __init__(
        self,
        model_name: str = "flan-t5-xxl",
        model_dir: Path = Path("models/hf_models"),
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self._model = self._tokenizer = None
        self.initialize()

    def initialize(self):
        global torch
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.nn.CrossEntropyLoss = partial(torch.nn.CrossEntropyLoss, reduction="none")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_dir / self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / self.model_name
        )

    def generate(self, source: str, **kwargs) -> str:
        input_ids = self.tokenizer(source, return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids, return_dict_in_generate=True, **kwargs
            )
        target = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return target

    def cal_log_probs(self, sources: list[str], targets: list[str]) -> list[float]:
        inputs = self.tokenizer(
            text=sources,
            text_target=targets,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True)
            log_probs = (-outputs.loss).view(inputs.labels.size(0), -1).mean(dim=1)
        return log_probs.tolist()

    @property
    def model(self):
        if self._model is not None:
            return self._model

        self.initialize()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        self.initialize()
        return self._tokenizer


class ChatWrapper:
    def __init__(
        self,
        model_name: str = "Mistral-7B-Instruct-v0.1",
        model_dir: Path = Path("models/hf_models"),
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self._model = self._tokenizer = self._pipeline = None
        self.initialize()

    def initialize(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir / self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / self.model_name
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._pipeline = pipeline(
            task="text-generation", model=self._model, tokenizer=self._tokenizer
        )

    def generate(self, source: str, **kwargs) -> str:
        messages = [{"role": "user", "content": source}]
        target = self.pipeline(messages, **kwargs)[0]["generated_text"][-1]["content"]
        return target

    def cal_log_probs(self, sources: list[str], targets: list[str]) -> list[float]:
        raise NotImplementedError

    @property
    def model(self):
        if self._model is not None:
            return self._model

        self.initialize()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        self.initialize()
        return self._tokenizer

    @property
    def pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        self.initialize()
        return self._pipeline


class APICostCalculator:
    # fmt: off
    _model_cost_per_1m_tokens = {
        # https://openai.com/api/pricing/
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-3.5-turbo-0125": {"prompt": 0.5, "completion": 1.5},
        "gpt-3.5-turbo-instruct": {"prompt": 1.5, "completion": 2.0},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        # gpt-5.4-mini official OpenAI pricing: $0.75 / $4.50 per 1M tokens.
        "gpt-5.4-mini": {"prompt": 0.75, "completion": 4.50},
        "gpt-4o": {"prompt": 5, "completion": 15},
        "gpt-4o-2024-05-13": {"prompt": 5, "completion": 15},
        "gpt-4-turbo": {"prompt": 10, "completion": 30},
        "gpt-4-turbo-2024-04-09": {"prompt": 10, "completion": 30},
        "gpt-4": {"prompt": 30, "completion": 60},
        # https://platform.openai.com/docs/deprecations/
        "gpt-3.5-turbo-0301": {"prompt": 1.5, "completion": 2.0},
        "gpt-3.5-turbo-0613": {"prompt": 1.5, "completion": 2.0},
        "gpt-3.5-turbo-16k-0613": {"prompt": 3, "completion": 4.0},
        "gpt-3.5-turbo-1106": {"prompt": 1.0, "completion": 2.0},
    }
    # fmt: on

    def __init__(self, model_name: str = "gpt-4o-mini"):
        if model_name not in self._model_cost_per_1m_tokens:
            raise ValueError(f"Unknown model name: {model_name}")
        self._model_name = model_name
        self._cost = 0
        # Also accumulate token / call counts so the harness can report
        # LLMCER-style efficiency stats (API Calls / in-out Tokens). Guarded by a
        # lock because thread_map drives this concurrently.
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._n_calls = 0
        self._lock = threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            if response.usage is None:
                return response
            cost = (
                self._model_cost_per_1m_tokens[self._model_name]["prompt"]
                * response.usage.prompt_tokens
                + self._model_cost_per_1m_tokens[self._model_name]["completion"]
                * response.usage.completion_tokens
            ) / 1000000.0
            with self._lock:
                self._cost += cost
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._n_calls += 1
            return response

        return wrapper

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value: int):
        self._cost = value

    @property
    def prompt_tokens(self):
        return self._prompt_tokens

    @property
    def completion_tokens(self):
        return self._completion_tokens

    @property
    def n_calls(self):
        return self._n_calls
