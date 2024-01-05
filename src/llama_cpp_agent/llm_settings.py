import json
from dataclasses import dataclass


@dataclass
class LlamaLLMSettings:
    model_path: str
    n_gpu_layers: int
    f16_kv: bool
    offload_kqv: bool
    use_mlock: bool
    embedding: bool
    n_threads: int
    n_batch: int
    n_ctx: int
    last_n_tokens_size: int
    verbose: bool
    seed: int

    def save(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'r') as file:
            loaded_messages = json.load(file)
            return LlamaLLMSettings(**loaded_messages)

    def as_dict(self):
        return self.__dict__
