import re
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.utils import APICostCalculator, openai_chat_complete


class Selecting:
    template = Template(
        """Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.

Given entity record:
{{ anchor }}

Candidate records:{% for candidate in candidates %}
[{{ loop.index }}] {{ candidate }}{% endfor %}
"""
    )

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        template: Template = template,
    ):
        self.model = model_name
        self.template = template

        self.api_cost_decorator = APICostCalculator(model_name=model_name)
        cache = Cache(f"results/diskcache/selecting_{model_name}")
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )
        # A non-cached fallback for retrying malformed gateway responses.
        self.chat_complete_uncached = self.api_cost_decorator(openai_chat_complete)

    def _parse_response(self, response, n_candidates: int):
        content = getattr(getattr(response, "choices", [None])[0], "message", None)
        content = getattr(content, "content", None)
        if not content:
            return [False] * n_candidates

        idx = re.search(r"\[(\d+)\]", content.strip())
        preds = [False] * n_candidates
        if idx:
            idx = int(idx.group(1))
            if 1 <= idx <= n_candidates:
                preds[idx - 1] = True
        return preds

    def __call__(self, instance) -> list[bool]:
        # NOTE: temperature=0.0 / max_tokens / logprobs were REMOVED for the
        # packyapi gateway's gpt-5.4-mini reasoning model: it rejects
        # temperature=0 and max_tokens (HTTP 400) and never returns logprobs
        # (Selecting only reads the "[N]" text, so logprobs were unused anyway).
        # reasoning_effort='none' gives the lowest-variance, cheapest decoding,
        # matching how LLMCER drives this same gateway. Faithful selecting logic
        # (prompt + "[N]" parse) is unchanged.
        kwargs = {"seed": 42}
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            kwargs["reasoning_effort"] = "none"
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 3
        messages = [
            {
                "role": "user",
                "content": self.template.render(
                    anchor=instance["anchor"],
                    candidates=instance["candidates"],
                ),
            }
        ]

        n_candidates = len(instance["candidates"])
        # Single API call per anchor (faithful to ComEM's original selecting).
        # An earlier 3-attempt retry was added for the packyapi gateway's
        # intermittent empty responses; on the official OpenAI endpoint a "[0]"
        # reply means "genuinely no match" and re-querying wastes 2 extra API
        # calls per anchor with no signal gain.
        try:
            response = self.chat_complete(
                messages=messages, model=self.model, **kwargs,
            )
            return self._parse_response(response, n_candidates)
        except Exception as exc:
            print(f"[warn] selecting fallback to no-match: "
                  f"{type(exc).__name__}: {str(exc)[:150]}")
            return [False] * n_candidates

    @property
    def cost(self):
        return self.api_cost_decorator.cost

    @cost.setter
    def cost(self, value: int):
        self.api_cost_decorator.cost = value


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    selector = Selecting()
    for file in dataset_files:
        dataset = file.stem
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)

        groupby = list(
            df.groupby("id_left")[["record_left", "record_right", "label"]]
            .apply(lambda x: x.to_dict("list"))
            .to_dict()
            .items()
        )
        instances = [
            {
                "anchor": v["record_left"][0],
                "candidates": v["record_right"],
                "labels": v["label"],
            }
            for _, v in groupby
        ]

        preds_lst = thread_map(
            selector,
            instances,
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {selector.cost:.2f}")

        results[dataset] = classification_report(
            labels[: len(preds)], preds, output_dict=True
        )["True"]
        results[dataset].pop("support")
        for k, v in results[dataset].items():
            results[dataset][k] = v * 100

    results["mean"] = {
        "precision": sum(v["precision"] for v in results.values()) / len(results),
        "recall": sum(v["recall"] for v in results.values()) / len(results),
        "f1-score": sum(v["f1-score"] for v in results.values()) / len(results),
    }
    df = pd.DataFrame.from_dict(results, orient="index")
    print(df)
    print(df.to_csv(float_format="%.2f", index=False))
    print(f"{selector.cost:.2f}")
