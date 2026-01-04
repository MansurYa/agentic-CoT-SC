"""
Централизованное хранилище системных промптов для всех узлов.

Промпты загружаются из текстовых файлов в директории prompts/en/.
Это позволяет легко редактировать промпты без изменения кода.
"""

import os

# Путь к директории с промптами
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "en")


def _load_prompt(filename: str) -> str:
    """
    Загружает промпт из текстового файла.

    :param filename: Имя файла в директории prompts/en/
    :return: Содержимое промпта
    """
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


# ========================================
# СИСТЕМНЫЕ ПРОМПТЫ ДЛЯ ВСЕХ ЭТАПОВ
# ========================================

# Этап 1: Генератор (создание HTML/JS решений)
PROMPT_GENERATOR = _load_prompt("generator_system_prompt.txt")

# Этап 2: Верификатор (анализ кода и визуала)
PROMPT_VERIFIER = _load_prompt("verifier_system_prompt.txt")

# Этап 3: Судья (выбор лучшего решения)
PROMPT_JUDGE = _load_prompt("judge_system_prompt.txt")

# Этап 4: Синтезатор (создание финальной версии)
PROMPT_SYNTHESIZER = _load_prompt("synthesizer_system_prompt.txt")


# ========================================
# ВСПОМОГАТЕЛЬНЫЕ ПРОМПТЫ
# ========================================

# Добавочный текст для промптов с JSON требованием
JSON_INSTRUCTION_SUFFIX = "\n\nIMPORTANT: Output MUST be valid JSON. No markdown, no explanations."

# Заметка о скриншоте для non-vision моделейG
NO_VISION_NOTE = "\n\n[SYSTEM NOTE: Screenshot was available but not provided due to model limitations.]"
