import json
from dataclasses import dataclass
from typing import List


@dataclass
class LlamaLLMGenerationSettings:
    """
    Data class representing generation settings for the Llama language model.

    Attributes:
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Controls the randomness of the generated output (higher values increase randomness).
        top_k (int): Controls the diversity of the generated output by limiting the top-k tokens considered.
        top_p (float): Controls the diversity of the generated output by limiting the cumulative probability of tokens.
        min_p (float): Minimum probability threshold for token selection.
        typical_p (float): Typical probability used for token selection.
        repeat_penalty (float): Penalty for repeating the same token in the output.
        mirostat_mode (int): Mode for using Mirostat, if enabled.
        mirostat_tau (float): Mirostat hyperparameter tau.
        mirostat_eta (float): Mirostat hyperparameter eta.
        tfs_z (float): TFS Z hyperparameter.
        stop_sequences (List[str]): List of stop sequences to indicate the end of generation.
        stream (bool): If True, generates output as a stream (partial responses); if False, generates complete responses.
        print_output (bool): If True, prints the generated output.
    """

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

    def save(self, file_path: str):
        """
        Save the LlamaLLMGenerationSettings object to a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaLLMGenerationSettings":
        """
        Load LlamaLLMGenerationSettings from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            LlamaLLMGenerationSettings: Loaded LlamaLLMGenerationSettings object.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaLLMGenerationSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaLLMGenerationSettings":
        """
        Create LlamaLLMGenerationSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing LlamaLLMGenerationSettings attributes.

        Returns:
            LlamaLLMGenerationSettings: Created LlamaLLMGenerationSettings object.
        """
        return LlamaLLMGenerationSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert LlamaLLMGenerationSettings to a dictionary.

        Returns:
            dict: Dictionary representation of the object.
        """
        return self.__dict__


@dataclass
class LlamaLLMSettings:
    """
    Data class representing settings for the Llama language model.

    Attributes:
        model_path (str): The path to the Llama language model.
        n_gpu_layers (int): Number of GPU layers.
        f16_kv (bool): If True, uses float16 for key and value tensors in self-attention layers.
        offload_kqv (bool): If True, offloads key, query, and value tensors to CPU in self-attention layers.
        use_mlock (bool): If True, uses mlock to lock model weights in RAM.
        embedding (bool): If True, enables ability to get embeddings from the model.
        n_threads (int): Number of threads to use (None for automatic).
        n_batch (int): Batch size.
        n_ctx (int): Context size.
        last_n_tokens_size (int): Size of the buffer for last n tokens.
        verbose (bool): If True, enables verbose mode.
        seed (int): Random seed for reproducibility.
    """

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
        """
        Save the LlamaLLMSettings object to a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaLLMSettings":
        """
        Load LlamaLLMSettings from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            LlamaLLMSettings: Loaded LlamaLLMSettings object.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaLLMSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaLLMSettings":
        """
        Create LlamaLLMSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing LlamaLLMSettings attributes.

        Returns:
            LlamaLLMSettings: Created LlamaLLMSettings object.
        """
        return LlamaLLMSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert LlamaLLMSettings to a dictionary.

        Returns:
            dict: Dictionary representation of the object.
        """
        return self.__dict__
