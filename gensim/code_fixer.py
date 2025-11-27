import os
from typing import Optional, Tuple

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from gensim.utils import (
    add_to_txt,
    extract_code,
    generate_feedback,
)


def _load_prompt_template(cfg) -> str:
    """Load the repair prompt from disk or fall back to a default template."""
    candidate_paths = [
        os.path.join("prompts", cfg.get("prompt_folder", ""), "cliport_prompt_code_repair.txt"),
        os.path.join("prompts", "cliport_prompt_code_repair.txt"),
    ]

    for path in candidate_paths:
        if path and os.path.exists(path):
            with open(path, "r") as handle:
                return handle.read()

    # Fallback prompt if no file exists.
    return (
        "You are an expert CLIport task engineer. The task name is TASK_NAME_TEMPLATE.\n"
        "Task description: TASK_DESCRIPTION_TEMPLATE\n"
        "Attempt AUTO_FIX_ATTEMPT_TEMPLATE failed with the following error:\n"
        "ERROR_LOG_TEMPLATE\n\n"
        "Fix the implementation below and return ONLY the corrected code wrapped in a python code block:\n"
        "CURRENT_CODE_TEMPLATE"
    )


def attempt_code_repair(
    cfg,
    task_metadata: dict,
    current_code: str,
    error_log: str,
    attempt_idx: int,
    interaction_log,
) -> Optional[Tuple[str, str]]:
    """Ask the LLM to repair the current code snippet using the provided traceback."""
    prompt_template = _load_prompt_template(cfg)
    prompt = prompt_template

    task_name = task_metadata.get("task-name", "unknown-task")
    task_desc = task_metadata.get("task-description", "No description provided.")
    prompt = prompt.replace("TASK_NAME_TEMPLATE", task_name)
    prompt = prompt.replace("TASK_DESCRIPTION_TEMPLATE", task_desc)
    prompt = prompt.replace("CURRENT_CODE_TEMPLATE", current_code.strip())
    prompt = prompt.replace("ERROR_LOG_TEMPLATE", error_log.strip())
    prompt = prompt.replace("AUTO_FIX_ATTEMPT_TEMPLATE", str(attempt_idx + 1))

    temperature = cfg.get("gpt_temperature", 0.0)
    response = generate_feedback(
        prompt,
        temperature=temperature,
        interaction_txt=interaction_log,
    )

    repaired_code, class_name = extract_code(response)
    if not repaired_code:
        print("Auto-fix failed to return code.")
        return None

    add_to_txt(
        interaction_log,
        f"[AUTO-FIX ATTEMPT {attempt_idx + 1}] Updated code for {task_name}",
        with_print=True,
    )
    print(
        highlight(
            f"[AUTO-FIX ATTEMPT {attempt_idx + 1}] Generated class: {class_name}",
            PythonLexer(),
            TerminalFormatter(),
        )
    )
    return repaired_code, class_name

