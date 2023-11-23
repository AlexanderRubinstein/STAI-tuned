# STAI-tuned
Utility code from STAI (https://scalabletrustworthyai.github.io/)

## 
1. Lists can be added to a GoogleSheets cell as follows: `[<element_1> <element_2> ... <element_n>]`., i.e., a `[`, followed by a space-separated list of elements, followed by a `]`. Please note that you don't need double quotes around strings. For example, the list `[elem1 elem2 ... elemn]` will be parsed in Python as `["elem1", "elem2", ..., "elemn"]`.
1. You can set GoogleSheets columns to `--` if you don't want to use a certain config item for a certain row. These cells will be ignored while parsing the row.