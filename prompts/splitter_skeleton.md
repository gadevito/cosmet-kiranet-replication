<!-- Sentence Splitter: structure of the system prompt (SYSTEM_SPLIT_UC in prompt.py of the CosMet
     repository) and of the user prompt (USER_SPLIT). The instruction and the example format are
     those of the original study; the example blocks are replaced by placeholders. -->

# System prompt

Split all the following sentences, numbering the new sentences so that each new sentence is grammatically correct.

Example 1:
<<T1 of the original study: 16 English example sentences, each followed by its expected split
  ("Split:" + numbered sentences); examples separated by "###"; see prompt.py in the CosMet repository>>
...
Example 16:
...

###

Example 17:
<<COMPANY EXAMPLES 1-14 -- withheld (Italian sentences from the 2024 calibration set, each followed by
  its expected split, same format as Examples 1-16)>>
...
Example 30:
...

###

# User prompt

{use_case_text}

Split:
