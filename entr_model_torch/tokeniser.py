import os
import urllib.request
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import json
from transformers import PreTrainedTokenizerFast

# Download tokenizer.json file


def download_tokenizer(tokenizer_url: str = "https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct/resolve/main/tokenizer.json", 
                      tokenizer_path: str = "tokenizer.json") -> str:
    """
    Downloads the tokenizer file if it doesn't exist locally.
    
    Args:
        tokenizer_url (str): URL to download the tokenizer from
        tokenizer_path (str): Local path to save the tokenizer
        
    Returns:
        str: Path to the tokenizer file
    """
    if not os.path.exists(tokenizer_path):
        print("Downloading tokenizer.json...")
        urllib.request.urlretrieve(tokenizer_url, tokenizer_path)
        print("Download complete.")
    else:
        print("tokenizer.json already exists.")
    
    return tokenizer_path

logger = getLogger(__name__)

# The following constants remain unchanged
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using a tokenizer.json file.
    """

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 17

    def __init__(self, tokenizer_path: str):
        """
        Initializes the Tokenizer with a tokenizer.json file.

        Args:
            tokenizer_path (str): The path to the tokenizer.json file.
        """
        assert os.path.isfile(tokenizer_path), tokenizer_path

        self.model = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        special_tokens = [
            '<|endoftext|>',
            '<|im_start|>',
            '<|im_end|>',
            "<repo_name>",
            "<reponame>",
            "<file_sep>",
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<jupyter_script>",
            "<empty_output>"
        ]

        self.special_tokens = {token: self.model.convert_tokens_to_ids(token) for token in special_tokens}

        self.n_words: int = self.model.vocab_size
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens['<|im_start|>']
        self.eos_id: int = self.special_tokens['<|im_end|>']
        self.eot_id: int = self.special_tokens['<|im_start|>']
        self.eom_id: int = self.special_tokens['<|im_end|>']
        self.python_tag_id = self.special_tokens['<jupyter_code>']
        self.pad_id: int = self.special_tokens['<|endoftext|>']
        self.stop_tokens = [
            self.special_tokens['<|im_start|>'],
            self.special_tokens['<|im_end|>'],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal['all'], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal['all'], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.
        """
        if allowed_special is None:
            allowed_special = set()
        assert isinstance(s, str)

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(self.model.encode(substr, add_special_tokens=False))
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.model.decode(t)

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]