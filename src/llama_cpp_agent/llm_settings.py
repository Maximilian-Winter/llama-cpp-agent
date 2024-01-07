import json
from dataclasses import dataclass
from typing import List


@dataclass
class LlamaLLMGenerationSettings:
    max_tokens: int = 0
    temperature: float = 0.35
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.05
    typical_p: float = 1.0
    repeat_penalty: float = 1.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    tfs_z: float = 1.0
    stop_sequences: List[str] = None
    stream: bool = True
    print_output: bool = True
    # k_last_messages: int = 0

    def save(self, file_path: str):
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaLLMGenerationSettings":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaLLMGenerationSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaLLMGenerationSettings":
        return LlamaLLMGenerationSettings(**settings)

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class LlamaLLMSettings:
    model_path: str
    n_gpu_layers: int = 0
    f16_kv: bool = True
    offload_kqv: bool = True
    use_mlock: bool = False
    embedding: bool = False
    n_threads: int = None
    n_batch: int = 512
    n_ctx: int = 512
    last_n_tokens_size: int = 64
    verbose: bool = False
    seed: int = -1

    def save(self, file_path: str):
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaLLMSettings":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaLLMSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaLLMSettings":
        return LlamaLLMSettings(**settings)

    def as_dict(self) -> dict:
        return self.__dict__
