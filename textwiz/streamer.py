from transformers import TextIteratorStreamer, AutoTokenizer

from .constants import SENTENCEPIECE_CHARACTER

class TextContinuationStreamer(TextIteratorStreamer):
    """Same as `TextIteratorStreamer`, but add the first space that does not get added by default during 
    continuation of prompts for models using Llama tokenizer (Llama2, Vicuna,...).
    """

    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = False, timeout: float | None = None,
                 **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.counter = 0


    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        add_space = False
        if self.counter == 0:
            first_token = self.tokenizer.convert_ids_to_tokens(self.token_cache[0])
            add_space = first_token.startswith(SENTENCEPIECE_CHARACTER)
            

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        if add_space:
            printable_text = ' ' + printable_text

        self.on_finalized_text(printable_text)
        self.counter += 1