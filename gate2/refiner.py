"""Transformers adapter for the local AgenticASR clean-window Refiner."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
)


class TransformersRefiner:
    def __init__(self, model_path: Path, *, max_new_tokens: int = 128) -> None:
        import torch
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        started = time.perf_counter()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_path / "tokenizer.json"),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="</s>",
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.chat_template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            use_safetensors=True,
        )
        configured_eos = self.model.generation_config.eos_token_id
        if isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.tokenizer.convert_ids_to_tokens(im_end_token_id) != "<|im_end|>":
            im_end_token_id = None
        self.stop_token_ids = list(dict.fromkeys([
            *(configured_eos or []),
            self.tokenizer.eos_token_id,
            im_end_token_id,
        ]))
        self.stop_token_ids = [
            token_id for token_id in self.stop_token_ids
            if isinstance(token_id, int) and 0 <= token_id < len(self.tokenizer)
        ]
        self.load_sec = time.perf_counter() - started

    def generation_metadata(self, continuation: Any) -> dict[str, Any]:
        token_ids = continuation.tolist()
        stop_index = next(
            (index for index, token_id in enumerate(token_ids) if token_id in self.stop_token_ids),
            None,
        )
        stop_token_id = token_ids[stop_index] if stop_index is not None else None
        effective_ids = token_ids[: stop_index + 1] if stop_index is not None else token_ids
        return {
            "finish_reason": "stop" if stop_token_id is not None else "length",
            "stop_token_id": stop_token_id,
            "generated_tokens": len(effective_ids),
            "tail_token_ids": effective_ids[-16:],
            "tail_text_with_special_tokens": self.tokenizer.decode(
                effective_ids[-16:], skip_special_tokens=False
            ),
        }

    @staticmethod
    def build_system_prompt(read_only_prefix: str, trusted_memory: list[str]) -> str:
        additions = []
        if read_only_prefix:
            additions.append(f"只读前文（不得输出）：{read_only_prefix}")
        if trusted_memory:
            additions.append("已确认可信实体：" + "；".join(trusted_memory))
        return SYSTEM_PROMPT + (("\n" + "\n".join(additions)) if additions else "")

    @staticmethod
    def has_key_suffix(raw: str) -> bool:
        return re.search(r"\s*<KEY>\[[^\]]*(?:\]\s*)?$", raw.strip()) is not None

    @staticmethod
    def has_incomplete_key_suffix(raw: str) -> bool:
        return re.search(r"\s*<KEY>\[[^\]]*$", raw.strip()) is not None

    @staticmethod
    def clean_generated(raw: str) -> tuple[str, list[str]]:
        value = raw.strip()
        value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
        keys: list[str] = []
        match = re.search(r"\s*<KEY>\[([^\]]*)\]\s*$", value)
        if match:
            keys = [item.strip() for item in re.split(r"[,，、]", match.group(1)) if item.strip()]
            value = value[: match.start()].strip()
        else:
            # Generation can hit max_new_tokens while repeating an audit suffix.
            # An unterminated <KEY>[...] is metadata, never transcript content.
            incomplete_key = re.search(r"\s*<KEY>\[[^\]]*$", value)
            if incomplete_key:
                value = value[: incomplete_key.start()].strip()
        value = re.sub(r"^assistant\s*[:：]?\s*", "", value, flags=re.IGNORECASE)
        value = value.strip("`\n ")
        if value.startswith("“") and value.endswith("”"):
            value = value[1:-1].strip()
        return value, keys

    @staticmethod
    def generation_inputs(inputs: Any) -> dict[str, Any]:
        """Keep only tensor inputs accepted by causal-LM generation.

        Some fast tokenizers emit ``token_type_ids`` from a chat template even
        though decoder-only models such as CPM do not accept that argument.
        """
        allowed = ("input_ids", "attention_mask")
        return {key: inputs[key] for key in allowed if key in inputs}

    def rewrite(self, *, read_only_prefix: str, active_source_window: str, trusted_memory: list[str]) -> dict[str, Any]:
        system_prompt = self.build_system_prompt(read_only_prefix, trusted_memory)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": active_source_window},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in self.generation_inputs(inputs).items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.stop_token_ids,
            )
        inference_sec = time.perf_counter() - started
        continuation = generated[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        clean_text, keys = self.clean_generated(raw)
        generation = self.generation_metadata(continuation)
        return {
            "system_prompt": system_prompt,
            "user_text": active_source_window,
            "raw_text": raw,
            "text": clean_text,
            "keys": keys,
            **generation,
            "key_suffix_present": self.has_key_suffix(raw),
            "incomplete_key_suffix": self.has_incomplete_key_suffix(raw),
            "inference_sec": inference_sec,
        }

    def rewrite_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        conversations = []
        system_prompts = []
        for request in requests:
            system_prompt = self.build_system_prompt(
                str(request.get("read_only_prefix", "")), list(request.get("trusted_memory", []))
            )
            system_prompts.append(system_prompt)
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(request["active_source_window"])},
                ]
            )
        inputs = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in self.generation_inputs(inputs).items()
        }
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.stop_token_ids,
            )
        elapsed = time.perf_counter() - started
        input_width = inputs["input_ids"].shape[1]
        continuations = generated[:, input_width:]
        raw_texts = self.tokenizer.batch_decode(continuations, skip_special_tokens=True)
        output = []
        for request, system_prompt, raw, continuation in zip(requests, system_prompts, raw_texts, continuations):
            clean_text, keys = self.clean_generated(raw)
            generation = self.generation_metadata(continuation)
            output.append(
                {
                    "system_prompt": system_prompt,
                    "user_text": str(request["active_source_window"]),
                    "raw_text": raw.strip(),
                    "text": clean_text,
                    "keys": keys,
                    **generation,
                    "key_suffix_present": self.has_key_suffix(raw),
                    "incomplete_key_suffix": self.has_incomplete_key_suffix(raw),
                    "batch_inference_sec": elapsed,
                    "amortized_inference_sec": elapsed / len(requests),
                }
            )
        return output
