<!-- COSMIC Analyzer: structure of the system prompt (SYSTEM_COSMIC_MEASURE in prompt.py of the CosMet
     repository) and of the user prompt (USER_COSMIC_MEASURE). The system prompt consists of worked
     examples only; the output format (the "measurement" block) is defined by the examples and is the
     same in every example. The example blocks are replaced by placeholders. -->

# System prompt

FP "<name of the functional process>":
<numbered steps of the functional process, as produced by the Sentence Splitter>

FP "<name of the functional process>" measurement:
Triggering Event: <...>
Functional User: <...>
Sub-Processes:
1. <step 1>
    FU (<functional user>) – DG (<data group>) – OOI (<object of interest>) – Entry | Exit | Read | Write.
2. <step 2>
    No data movement.
...

###

<<T2 of the original study: 7 English worked use cases in the format above (3 from C-Reg, 4 engineered
  for batch / external exchanges), separated by "###"; see prompt.py in the CosMet repository>>

###

<<COMPANY EXAMPLES 1-5 -- withheld (Italian functional processes from the 2024 calibration set with their
  accepted manual COSMIC analysis, same format; names of functional users, data groups and objects of
  interest in Italian, movement types and field names in English)>>

###

# User prompt

FP "{uc_name}":
{split_output}

FP: "{uc_name}" measurement:
