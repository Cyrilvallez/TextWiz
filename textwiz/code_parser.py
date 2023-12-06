import re
from abc import ABC, abstractmethod

from . import utils


class CodeParser(ABC):
    """Abstract base class for code parsers."""

    @abstractmethod
    def parse(self, s: str) -> list[str] | None:
        """Parse a string `s`, and return all code blocks within `s`. This function try to parse `s`
        with methods of increasing complexity. As soon, as one method produces at least one match, we consider
        the parsing as succesful and return that match. Return `None` if there are no matches.

        Parameters
        ----------
        s : str
            String to parse.

        Returns
        -------
        list[str] | None
            List containing all code blocks.
        """
        return NotImplementedError('Abstract method.')


    def concatenate(self, code_blocks: list[str] | None) -> str:
        """Concatenate multiple code blocks into a single code block.

        Parameters
        ----------
        code_blocks : list[str]
            The strings representing the code blocks

        Returns
        -------
        str
            Single code block.
        """
            
        if code_blocks is None:
            return ''
        return '\n\n'.join(code_blocks)
    

    def full_parse(self, s: str) -> str:
        """Parse all Python code contained in `s`, and concatenate it. Return an empty string if no code
        was detected.

        Parameters
        ----------
        s : str
            String to parse.

        Returns
        -------
        str
            The truncated code output.
        """

        blocks = self.parse(s)
        block = self.concatenate(blocks)
        return block
    

    @utils.copy_docstring_and_signature(full_parse)
    def __call__(self, *args, **kwargs):
        return self.full_parse(*args, **kwargs)



# Python keywords suseptible to start a line
PYTHON_START_KEYWORDS = [
    'from ',
    'import ',
    'def ',
    '#',
    'class ',
    '@',
    'if ',
    'for ',
    'while ',
    'with ',
    'print',
]

# Patterns that represent a Python line
PYTHON_PATTERNS = [
    # matches a line containing a "=" symbol, without containing a newline before it
    r'(?:[^\n]+)=',
]

# Regex containing all pythons markers as a non-capturing OR (chained with pipes |) group
PYTHON_MARKERS = '|'.join(map(re.escape, PYTHON_START_KEYWORDS)) + '|' + '|'.join(PYTHON_PATTERNS)
PYTHON_GROUP = r'(?:' + PYTHON_MARKERS + r')'

# matches anything between start of string or newline followed by a PYTHON_MARKER, and newline followed
# by text (not space) that is not a PYTHON_MARKER, or end of string
# \nPYTHON_MARKER (blabla) \nNON-PYTHON_MARKER
# NOTE: standalone statements such as "foo.append(1)" which are not indented (inside a block such as if, for etc)
# will not be recognized
PYTHON_CODE_REGEX = r'(?:^|\n)(' + PYTHON_GROUP + r'.*?)' + r'(?:$|(?:\n(?!' + PYTHON_GROUP + r')\S))'

# Regexes suseptible to capture python code. Starting with the easiest and simplest form of blocks to
# parse, and increasing in complexity (and thus possible parsing errors)
# NOTE: Every regex needs to be searched with the flag re.DOTALL
PYTHON_CODE_REGEXES = [
    # Regexes that match the usual markdown python syntax with 3 backticks
    r'```python(?: )*\n(.*?)(?:$|\n```)',
    r'```(?: )*\n(.*?)(?:$|\n```)',
    PYTHON_CODE_REGEX,
]



class PythonParser(CodeParser):
    """Parser that extracts Python code from strings. 

    NOTE: It is not perfect and should only be used for precise cases! It is not robust against all possible
    weird corner-cases. For example, the following patterns are NOT parsed correctly:

    - text containing code blocks formatted in different ways (e.g. one block with triple backticks, one without)
    - text containing triple backticks formatted code blocks, themselves containing triple backticks
    - standalone statements such as `foo.append(1)` or `foo.get_result()` which are not inside triple backticks or
    indented code blocks

    These kind of patterns should be extremely rare in practice in our use-case but might exist. There could also
    be other patterns or edge-cases uncorrectly handled by the following parser.
    """

    def __init__(self):
        self.python_start_keywords = PYTHON_START_KEYWORDS
        self.python_patterns = PYTHON_PATTERNS
        self.python_group = PYTHON_GROUP
        self.code_regexes = PYTHON_CODE_REGEXES
    

    def parse(self, s: str) -> list[str] | None:

        out = []

        # Check if the beginning of the string is an indented block (if it is, we assume it's code)
        start = re.match(r'^\n?(?:    )+', s)
        if start is not None:
            match = re.search(r'^(.*?)(?:$|(?:\n(?!' + self.python_group + r')\S))', s, re.DOTALL)
            first_block = match.group(1)
            # remove that part of the initial string
            s = s.replace(first_block, '', 1)
            out.append(first_block.rstrip())

        for regex in self.code_regexes:
            matches = re.findall(regex, s, re.DOTALL)
            if len(matches) > 0:
                # remove trailing newlines
                return out + [x.rstrip() for x in matches]
        
        out = None if len(out) == 0 else out

        return out
    



# Some simple tests (because parsing code is very prone to errors)
# Note that some known issues (see class docstring) are not tested for

_TEST_INPUTS = [
    """Here's a Python implementation of the function:
```python
def foo(bar):
    print(bar)
```""",

    """Here's a Python implementation of the function:
```python
def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """    print('test)
    return True""",

    """
    return True
foobar
def test():
    print('test')
should not be code
    """,

    """    print('bar')
foobar
```python
def foo(bar):
    print('bar')
```""",

    # Without backticks
    """Here's a Python implementation of the function:

def parse_music(music_string: str) -> List[int]:
    \"\"\"Parses a music string in the special ASCII format and returns a list of note durations.\"\"\"
    note_durations = []
    current_duration = 0
    for char in music_string:
        if char == "o":
            current_duration = 4
        elif char == "o|":
            current_duration = 2
        elif char == ".|":
            current_duration = 1
        else:
            raise ValueError(f"Invalid character in music string: {char}")
        note_durations.append(current_duration)
    return note_durations
    
Here's an example usage:

music_string = "o o|.| o| o|.|.|.|.| o o"
note_durations = parse_music(music_string)
print(note_durations) # Output: [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]""",
]


_EXPECTED_OUTPUTS = [
    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """    print('test)
    return True""",

    """
    return True

def test():
    print('test')""",

    """    print('bar')

def foo(bar):
    print('bar')""",

    """def parse_music(music_string: str) -> List[int]:
    \"\"\"Parses a music string in the special ASCII format and returns a list of note durations.\"\"\"
    note_durations = []
    current_duration = 0
    for char in music_string:
        if char == "o":
            current_duration = 4
        elif char == "o|":
            current_duration = 2
        elif char == ".|":
            current_duration = 1
        else:
            raise ValueError(f"Invalid character in music string: {char}")
        note_durations.append(current_duration)
    return note_durations

music_string = "o o|.| o| o|.|.|.|.| o o"
note_durations = parse_music(music_string)
print(note_durations) # Output: [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]""",

]


def _run_tests():

    parser = PythonParser()

    for i, (input, output) in enumerate(zip(_TEST_INPUTS, _EXPECTED_OUTPUTS)):
        assert parser(input) == output, f'test {i} failed'

    print('All tests completed succesfully')



if __name__ == '__main__':

    _run_tests()