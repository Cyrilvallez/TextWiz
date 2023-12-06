import re
from enum import Enum

import torch
import numpy as np
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from .code_parser import CodeParser

# If we reach one of these patterns, it means that the model has finished generating the solution as a 
# function and continues useless generation (basically stop words used in the Codex/HumanEval 
# paper: https://arxiv.org/pdf/2107.03374.pdf). Should only be used when the prompt is a python function definition.
CODE_STOP_PATTERNS = (
    '\nclass',
    '\ndef',
    '\n#',
    '\nif',
    '\nprint',
    '\n@'
)

# Extended code stopping patterns. This is mostly useful for chat models which often output code blocks 
# starting with ">>>" to show examples
EXTENDED_CODE_STOP_PATTERNS = CODE_STOP_PATTERNS + (
    '\n>>>',
)

# Pattern to match out of indentation, i.e. anything that does not start by a space
OUT_OF_INDENTATION_REGEX = re.compile(r'^\S|\n\S')

# Pattern to match out of assignment, i.e. anything after a newline that is not immediately at the start of the string
# and that it not followed by a space or )} character (some long assignment may be on multiple lines, but in this
# case there are spaces between newlines and continuation of the assignment and they end by either `)` or `}`)
# OUT_OF_ASSIGNMENT_REGEX = re.compile(r'^(\n*+)(.*?)(?:(\n+\s*[)])|(\n+\s*})|(?:\n+\S))', re.DOTALL)
OUT_OF_ASSIGNMENT_REGEX = re.compile(r'^(\n*+)(.*?)(?:(\n+[)])|(\n+})|(?:\n+\S))', re.DOTALL)


class StoppingType(Enum):
    """Convenient way to define a stopping pattern in a clear and concise manner, and the stopping criteria
    and post-processing associated to it.
    """

    PYTHON_HUMAN_EVAL = CODE_STOP_PATTERNS
    PYTHON_HUMAN_EVAL_EXTENDED = EXTENDED_CODE_STOP_PATTERNS
    OUT_OF_INDENTATION = OUT_OF_INDENTATION_REGEX
    OUT_OF_ASSIGNMENT = OUT_OF_ASSIGNMENT_REGEX

    def create_stopping_criteria(self, prompt_ids_length: int, tokenizer: PreTrainedTokenizerBase,
                                 extra_eos_tokens: list[str] | None,
                                 parser: CodeParser | None = None) -> StoppingCriteria:
        """Create the stopping criteria associated with the current enum member.

        Parameters
        ----------
        prompt_ids_length : int
            Length of the input ids prompt.
        tokenizer : PreTrainedTokenizerBase
            The tokenizer to use to decode the sequences.
        extra_eos_tokens : list[str] | None
            List of extra eos tokens.
        parser : CodeParser | None, optional
            A parser to extract code from generated sequences. The `stopping_patterns` will be applied on the
            parsed sequences. This should be used with caution, as it was designed only for chat models that
            embed code in their output in natural language. The default is None, i.e. no parsing.

        Returns
        -------
        StoppingCriteria
            The stopping criteria.
        """
        
        if isinstance(self.value, list) or isinstance(self.value, tuple):
            return TextPatternStopping(prompt_ids_length, tokenizer, self.value, extra_eos_tokens, parser)
        elif isinstance(self.value, re.Pattern):
            return RegexPatternStopping(prompt_ids_length, tokenizer, self.value, extra_eos_tokens, parser)
        else:
            raise TypeError('Value for the enumeration member is not supported.')
        
    
    def post_process_sequences(self, prompt_truncated_generated_sequences: list[str]) -> list[str]:
        """Post process the sequences according to the current enum member.

        Parameters
        ----------
        prompt_truncated_generated_sequences : list[str]
            Decoded PROMPT-TRUNCATED outputs of a model. Passing the full decoded outputs may induce errors in the logic.

        Returns
        -------
        list[str]
            The post-processed outputs.
        """

        if isinstance(self.value, list) or isinstance(self.value, tuple):
            return post_process_stopping_patterns(prompt_truncated_generated_sequences, self.value)
        elif isinstance(self.value, re.Pattern):
            return post_process_regex_pattern(prompt_truncated_generated_sequences, self.value)
        else:
            raise TypeError('Value for the enumeration member is not supported.')





class TextPatternStopping(StoppingCriteria):
    """Stop generation upon meeting any of the `stopping_patterns` or `extra_eos_tokens`.
    """

    def __init__(self, prompt_ids_length: int, tokenizer: PreTrainedTokenizerBase,
                 stopping_patterns: list[str] | tuple[str] | None, extra_eos_tokens: list[str] | None = None,
                 parser: CodeParser | None = None):

        super().__init__()
        self.prompt_ids_length = prompt_ids_length
        self.tokenizer = tokenizer
        self.parser = parser
        self.stopping_patterns = tuple() if stopping_patterns is None else tuple(stopping_patterns)
        self.extra_eos_tokens = tuple() if extra_eos_tokens is None else tuple(extra_eos_tokens)
        self.all_patterns = self.stopping_patterns + self.extra_eos_tokens

        if len(self.all_patterns) == 0:
            raise ValueError('You did not provide any patterns or extra eos tokens upon which to stop generation.')
    

    def check_patterns(self, generated_sequences: list[str], patterns: tuple[str]) -> list[bool]:
        """Check if there is at least one of the `patterns` in each of the `generated_sequences`.

        Parameters
        ----------
        generated_sequences : list[str]
            Decoded outputs of the models.
        patterns : tuple[str]
            Patterns to check for.

        Returns
        -------
        list[bool]
            Whether each sequence is finished or not.
        """

        done_sequences = []

        for sequence in generated_sequences:
            done = any([pattern in sequence for pattern in patterns])
            done_sequences.append(done)

        return done_sequences


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Return `True` if all sequences are finished being generated (i.e. there is at least one stopping
        pattern or eos in each sequence). Unfortunately, this cannot return a list of boolean to inform
        the generation function which sequences are done or not, and append <pad-token> to the finished
        sequences.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Outputs ids of the model.
        scores : torch.FloatTensor
            Scores.

        Returns
        -------
        bool
            `True` if all sequences are done being generated, `False` otherwise.
        """

        outputs = input_ids[:, self.prompt_ids_length:]
        generated_sequences = self.tokenizer.batch_decode(outputs)
        
        # If we don't use a parser, just check against all patterns
        if self.parser is None:
            done_sequences = self.check_patterns(generated_sequences, self.all_patterns)
            return all(done_sequences)
        # Else first check the eos in the full sequences, then parse and check for the other patterns
        else:
            done_with_eos = self.check_patterns(generated_sequences, self.extra_eos_tokens)
            parsed_sequences = [self.parser(sequence) for sequence in generated_sequences]
            done_with_patterns = self.check_patterns(parsed_sequences, self.stopping_patterns)
            return all(np.logical_or(done_with_eos, done_with_patterns))
        


class RegexPatternStopping(StoppingCriteria):
    """Stop generation if we detect a match with a given regex pattern.
    """

    def __init__(self, prompt_ids_length: int, tokenizer: PreTrainedTokenizerBase, regex: re.Pattern | str | None,
                 extra_eos_tokens: list[str] | None = None, parser: CodeParser | None = None):

        super().__init__()
        self.prompt_ids_length = prompt_ids_length
        self.tokenizer = tokenizer
        if isinstance(regex, str):
            self.regex = re.compile(regex)
        else:
            self.regex = regex
        self.extra_eos_tokens = extra_eos_tokens
        self.parser = parser

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        outputs = input_ids[:, self.prompt_ids_length:]
        generated_sequences = self.tokenizer.batch_decode(outputs)

        done_sequences = []

        for sequence in generated_sequences:
            # Check for extra eos
            done_eos = any([pattern in sequence for pattern in self.extra_eos_tokens])
            
            if self.regex is not None:
                if self.parser is not None:
                    sequence = self.parser(sequence)
                done_regex = self.regex.search(sequence) is not None
                done_sequences.append(any([done_eos, done_regex]))
            else:
                done_sequences.append(done_eos)

        return all(done_sequences)
    




def create_stopping_criteria(prompt_ids_length: int, tokenizer: PreTrainedTokenizerBase,
                             stopping_patterns: StoppingType | list[str] | tuple[str] | re.Pattern | str | None,
                             extra_eos_tokens: list[str] | None,
                             parser: CodeParser | None = None) -> StoppingCriteriaList | None:
    """Create the correct stopping criteria depending on the value and type of `stopping_patterns`.

    Parameters
    ----------
    prompt_ids_length : int
        Length of the input ids prompt.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use to decode the sequences.
    stopping_patterns : StoppingType | list[str] | tuple[str] | re.Pattern | str | None
        The type of early stopping to use. This should be an instance of the `StoppingType` enum, or eventually
        a list or tuple of str, in which case the iterable will be passed to a `TextPatternStopping` instance. It can
        also be a re.Pattern or str, which is interpreted as a regex and is passed to a `RegexPatternStopping` instance.
        If `None`, only the `extra_eos_tokens` will be used for early stopping.
    extra_eos_tokens : list[str] | None
        List of extra eos tokens.
    parser : CodeParser | None, optional
        A parser to extract code from generated sequences. The `stopping_patterns` will be applied on the
        parsed sequences. This should be used with caution, as it was designed only for chat models that
        embed code in their output in natural language. The default is None, i.e. no parsing.

    Returns
    -------
    StoppingCriteriaList | None
        The stopping criteria.
    """
    
    if isinstance(stopping_patterns, StoppingType):
        criteria = stopping_patterns.create_stopping_criteria(prompt_ids_length, tokenizer, extra_eos_tokens, parser)
        
    elif isinstance(stopping_patterns, list) or isinstance(stopping_patterns, tuple):
        criteria = TextPatternStopping(prompt_ids_length, tokenizer, stopping_patterns, extra_eos_tokens, parser)

    elif isinstance(stopping_patterns, re.Pattern) or isinstance(stopping_patterns, str):
        criteria = RegexPatternStopping(prompt_ids_length, tokenizer, stopping_patterns, extra_eos_tokens, parser)

    elif stopping_patterns is None:
        if len(extra_eos_tokens) == 0:
            criteria = None
        else:
            criteria = TextPatternStopping(prompt_ids_length, tokenizer, None, extra_eos_tokens, parser)

    else:
        raise TypeError('Cannot infer stopping criteria based on the type of stopping patterns.')
    
    criteria_list = StoppingCriteriaList([criteria]) if criteria is not None else None
    
    return criteria_list



def post_process_stopping_patterns(prompt_truncated_generated_sequences: list[str],
                                   stopping_patterns: list[str] | tuple[str] | None) -> list[str]:
    """Post-process the outputs of a model to truncate according to a list of patterns upon which we stop
    generation (this is needed because the StoppingCriteria cannot immediately stop the generation of each
    sequence upon meeting a pattern in the case of more than 1 `num_return_sequences`).

    Parameters
    ----------
    prompt_truncated_generated_sequences : list[str]
        Decoded PROMPT-TRUNCATED outputs of a model. Passing the full decoded outputs may induce errors in the logic.
    stopping_patterns : list[str] | tuple[tr] | None,
        The list of patterns to use to stop generation.

    Returns
    -------
    list[str]
        The truncated outputs to meet the criteria of the stopping patterns.
    """

    # If there are no stopping patterns
    if stopping_patterns is None or len(stopping_patterns) == 0:
        return prompt_truncated_generated_sequences

    generated_sequences_curated = []
    
    for sequence in prompt_truncated_generated_sequences:
        
        stop_index = len(sequence)

        # Scan the sequence for each pattern, and return the minimum index such that none of the patterns are
        # in the sequence
        for pattern in stopping_patterns:
            index = sequence.find(pattern)
            if index != -1:
                stop_index = min(stop_index, index)

        curated_sequence = sequence[0:stop_index]
        generated_sequences_curated.append(curated_sequence)

    return generated_sequences_curated



def post_process_regex_pattern(prompt_truncated_generated_sequences: list[str], regex: re.Pattern | str | None) -> list[str]:
    """Post-process the outputs of a model to truncate according to a regex pattern (this is needed because
    the StoppingCriteria cannot immediately stop the generation of each sequence upon meeting a pattern in the
    case of more than 1 `num_return_sequences`). If the `regex` contains one or more groups, the output sequence is
    the content of all the groups. If it does not contain any group, the output is the sequence up to the index of
    the match.

    Parameters
    ----------
    prompt_truncated_generated_sequences : list[str]
        Decoded PROMPT-TRUNCATED outputs of a model. Passing the full decoded outputs may induce errors in the logic.
    regex : re.Pattern | str | None
        Pattern to match.

    Returns
    -------
    list[str]
        The truncated outputs to meet the criteria of out of indentation.
    """

    # If there are no regex
    if regex is None:
        return prompt_truncated_generated_sequences
        
    if isinstance(regex, str):
        regex = re.compile(regex)
    
    generated_sequences_curated = []
    
    for sequence in prompt_truncated_generated_sequences:

        match = regex.search(sequence)
        if match is not None:
            if regex.groups > 0:
                curated = ''.join(group for group in match.groups() if group is not None)
            else:
                curated = sequence[0:match.start()]
            generated_sequences_curated.append(curated)
        else:
            generated_sequences_curated.append(sequence)

    return generated_sequences_curated



def post_process_extra_eos_tokens(prompt_truncated_outputs: torch.Tensor, pad_token_id: int,
                                  extra_eos_tokens_ids: list[int] | None) -> torch.Tensor:
    """Process the outputs of a model to convert all tokens that were generated after an extra eos to 
    `pad_token_id`. This way, everything after the extra eos will be ignored when calling
    tokenizer.batch_decode(..., skip_special_tokens=True) later.

    NOTE: if the original tokenizer.eos_token is found at some point in the generated sequence, all
    subsequent tokens are set to tokenizer.pad_token automatically so we don't need to add tokenizer.eos_token
    to extra_eos_tokens.

    Parameters
    ----------
    prompt_truncated_outputs : torch.Tensor
        The PROMPT-TRUNCATED output of a model. Passing the full outputs may induce errors in the logic.
    pad_token_id : int
        The id of the pad token.
    extra_eos_tokens_ids : list[int] | None
        The list of extra eos tokens ids.

    Returns
    -------
    torch.Tensor
        The modified output.
    """

    # If there are no extra eos tokens
    if extra_eos_tokens_ids is None or len(extra_eos_tokens_ids) == 0:
        return prompt_truncated_outputs
    
    outputs = prompt_truncated_outputs.clone().detach()

    for i, sequence_ids in enumerate(prompt_truncated_outputs):

        stop_index = len(sequence_ids)

        # Scan the sequence for each eos, and set all subsequent ids to pad_token_id
        for eos_ids in extra_eos_tokens_ids:
            nonzero = torch.nonzero(sequence_ids == eos_ids)
            if len(nonzero) != 0:
                stop_index = min(stop_index, int(nonzero[0][0]))

        # Everything after the first extra eos is set to pad_token
        outputs[i, stop_index:] = pad_token_id

    return outputs



def post_process_sequences(prompt_truncated_outputs: torch.Tensor, tokenizer: PreTrainedTokenizerBase,
                           stopping_patterns: StoppingType | list[str] | tuple[str] | re.Pattern | str | None,
                           extra_eos_tokens: list[str] | None, parser: CodeParser | None = None) -> list[str]:
    """Apply all steps of post-processing to the prompt-truncated outputs of a model.

    Parameters
    ----------
    prompt_truncated_outputs : torch.Tensor
        The PROMPT-TRUNCATED output of a model. Passing the full outputs may induce errors in the logic.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used by the model.
    stopping_patterns : StoppingType | list[str] | tuple[str] | re.Pattern | str | None
        The type of early stopping to use. This should be an instance of the `StoppingType` enum, or eventually
        a list or tuple of str, in which case the iterable will be passed to a `TextPatternStopping` instance. It can
        also be a re.Pattern or str, which is interpreted as a regex and is passed to a `RegexPatternStopping` instance.
        If `None`, only the `extra_eos_tokens` will be used for early stopping.
    extra_eos_tokens : list[str] | None
        List of extra eos tokens.
    parser : CodeParser | None, optional
        A parser to extract code from generated sequences. The `stopping_patterns` will be applied on the
        parsed sequences. This should be used with caution, as it was designed only for chat models that
        embed code in their output in natural language. The default is None, i.e. no parsing.

    Returns
    -------
    list[str]
        The post-processed generated sequences.
    """
    
    # Return None if extra_eos_tokens is None
    extra_eos_tokens_ids = tokenizer.convert_tokens_to_ids(extra_eos_tokens)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Check if we find any of the extra eos tokens. We first look for extra eos in this way so that we
    # can later use tokenizer.batch_decode(..., skip_special_tokens=True), i.e. easily remove all the 
    # special tokens
    processed_outputs = post_process_extra_eos_tokens(prompt_truncated_outputs, pad_token_id, extra_eos_tokens_ids)

    # Decode sequences
    prompt_truncated_sequences = tokenizer.batch_decode(processed_outputs, skip_special_tokens=True)

    # Parse sequences
    if parser is not None:
        prompt_truncated_sequences = [parser(sequence) for sequence in prompt_truncated_sequences]

    # Truncate according to stopping pattern
    if isinstance(stopping_patterns, StoppingType):
        final_sequences = stopping_patterns.post_process_sequences(prompt_truncated_sequences)
    elif isinstance(stopping_patterns, list) or isinstance(stopping_patterns, tuple):
        final_sequences = post_process_stopping_patterns(prompt_truncated_sequences, stopping_patterns)
    elif isinstance(stopping_patterns, re.Pattern) or isinstance(stopping_patterns, str):
        final_sequences = post_process_regex_pattern(prompt_truncated_sequences, stopping_patterns)
    elif stopping_patterns is None:
        final_sequences = prompt_truncated_sequences
    else:
        raise TypeError('Cannot post process outputs based on the type of stopping patterns.')

    return final_sequences

