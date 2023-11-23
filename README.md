# STAI-tuned
Utility code from STAI (https://scalabletrustworthyai.github.io/)


## Settings 

You can edit `settings.py` in order to tailor the tool to your own preferences.

| Setting | Purpose |
|----------|----------|
| `EMPTY_STRING` | You can set GoogleSheets columns to the value of `EMPTY_STRING` if you don't want to use a certain config item for a certain row. These cells will be ignored while parsing the row. |
| `NESTED_CONFIG_KEY_SEPARATOR` | If you have values in your config file which are dictionaries themselves, e.g.: `{"model": {"name": "resnet20", "num_classes": 10}}`, you need to flatten out the keys, e.g.: `delta:model.name1`, `delta:model.num_classes`, provided that `NESTED_CONFIG_KEY_SEPARATOR` is set to `.`. Feel free to set your own desired separator, e.g.: `/`.  |


## Fine-Grained Guidelines
1. Lists can be added to a GoogleSheets cell as follows: `[<element_1> <element_2> ... <element_n>]`., i.e., a `[`, followed by a space-separated list of elements, followed by a `]`. Please note that you don't need double quotes around strings. For example, the list `[elem1 elem2 ... elemn]` will be parsed in Python as `["elem1", "elem2", ..., "elemn"]`.
