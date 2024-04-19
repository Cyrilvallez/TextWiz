import logging
import contextlib
import io

class LoggingFilter(logging.Filter):
    """Used to remove messages or warnings issued by the `logging` library.
    """

    def __init__(self, patterns: list[str] | str):

        super().__init__()
        self.patterns = [patterns] if type(patterns) == str else patterns

    def filter(self, record):
        return not any(pattern in record.getMessage() for pattern in self.patterns)
    

# This is displayed whenever we convert to better-transformer -> we never train so useless
BETTER_TRANSFORMER_WARNING = ('The BetterTransformer implementation does not support padding during training, '
                              'as the fused kernels do not support attention masks. Beware that passing padded '
                              'batched data during training may result in unexpected outputs.')
# This is due to dialo-gpt -> we never pad inputs so useless
PADDING_SIDE_WARNING = ("A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer.")
# This is due to code-llama tokenizers for which they added special tokens for easy infilling mode
ADDED_TOKENS_WARNING = ("Special tokens have been added in the vocabulary, make sure the associated word "
                        "embeddings are fine-tuned or trained.")
# This is due to codegen25 tokenizers at load time
UNK_TOKEN_WARNING = "Using unk_token, but it is not set yet."
# Flash attention 2 warning
FLASH_ATTENTION_2_WARNING = ("You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. "
                             "Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.")


optimum_logger = logging.getLogger('optimum.bettertransformer.transformation')
optimum_logger.addFilter(LoggingFilter(BETTER_TRANSFORMER_WARNING))

generation_logger = logging.getLogger('transformers.generation.utils')
generation_logger.addFilter(LoggingFilter(PADDING_SIDE_WARNING))

tokenization_logger = logging.getLogger('transformers.tokenization_utils_base')
tokenization_logger.addFilter(LoggingFilter([ADDED_TOKENS_WARNING, UNK_TOKEN_WARNING]))

modeling_logger = logging.getLogger('transformers.modeling_utils')
modeling_logger.addFilter(LoggingFilter(FLASH_ATTENTION_2_WARNING))


# warnings.filterwarnings(action='ignore', message=better_transformer_warning)


BITSANDBYTES_WELCOME = '\n' + '='*35 + 'BUG REPORT' + '='*35 + '\n' + \
                        ('Welcome to bitsandbytes. For bug reports, please run\n\npython -m bitsandbytes\n\n'
                        ' and submit this information together with your error trace to: '
                        'https://github.com/TimDettmers/bitsandbytes/issues') + \
                        '\n' + '='*80 + '\n'
BITSANDBYTES_SETUPS = (
    'CUDA SETUP: CUDA runtime path found:',
    'CUDA SETUP: Highest compute capability among GPUs detected:',
    'CUDA SETUP: Detected CUDA version',
    'CUDA SETUP: Loading binary',
    )

@contextlib.contextmanager
def swallow_bitsandbytes_prints():
    """Remove all prints generated by bitsandbytes's (correct) initialization (they clutter the output stream,
    especially when doing multiprocessing).
    Note: Since bitsandbytes version 0.41, this is not necessary anymore as the prints were removed 
    (see https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/cextension.py#L42)
    """
    with contextlib.redirect_stdout(io.StringIO()) as f:
        yield
    all_prints = f.getvalue()
    # Remove all bitsandbytes setup output
    if BITSANDBYTES_WELCOME in all_prints:
        to_keep, after = all_prints.split(BITSANDBYTES_WELCOME, 1)
        lines = after.splitlines()

        # first line after the welcome is just a path
        lines_to_keep = []
        for i, line in enumerate(lines[1:]):
            if i < len(BITSANDBYTES_SETUPS):
                assert line.startswith(BITSANDBYTES_SETUPS[i]), 'The bitsandbytes print format is not as expected'
            else:
                lines_to_keep.append(line)
        
        to_keep += '\n'.join(lines_to_keep)

        # reprint without the bitsandbytes block
        if to_keep != '':
            print(to_keep)

    else:
        if all_prints != '':
            # The last newline is an artifact of print() adding newline thus remove it
            if all_prints.endswith('\n'):
                print(all_prints.rsplit('\n', 1)[0])
            else:
                print(all_prints)
    


        