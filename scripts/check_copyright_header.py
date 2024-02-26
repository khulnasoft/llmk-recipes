# Copyright (c) Khulnasoft Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llmk 2 Community License Agreement.

import re
from pathlib import Path

WORK_DIR = Path(__file__).parents[1]
PATTERN = "(Khulnasoft Platforms, Inc. and affiliates)|(Khulnasoft, Inc(\.|,)? and its affiliates)|([0-9]{4}-present(\.|,)? Khulnasoft)|([0-9]{4}(\.|,)? Khulnasoft)"

HEADER = """# Copyright (c) Khulnasoft Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llmk 2 Community License Agreement.\n\n"""

#Files in black list must be relative to main repo folder
BLACKLIST = ["eval/open_llm_leaderboard/hellaswag_utils.py"]

if __name__ == "__main__":
    for ext in ["*.py", "*.sh"]:
        for file in WORK_DIR.rglob(ext):
            normalized = file.relative_to(WORK_DIR)
            if normalized.as_posix() in BLACKLIST:
                continue
            
            text = file.read_text()
            if not re.search(PATTERN, text):
                text = HEADER + text
                file.write_text(text)
        