"""
Узел-синтезатор: создает финальное решение на основе победителя и советов судьи.

Этот модуль реализует финальный этап - сборку итогового решения.
Берёт код победителя и улучшает его, применяя рекомендации судьи
и устраняя найденные недостатки.
"""

# Стандартные библиотеки
import re
import logging
from typing import Dict, Any

# Модули текущего проекта
from src.core.llm_client import LLMClient
from src.domain.state import AgenticState
from config.prompts import PROMPT_SYNTHESIZER

logger = logging.getLogger(__name__)

# Регулярное выражение для извлечения HTML кода из ответа модели
HTML_REGEX = re.compile(
    r"```html\s*(.*?)```|(\s*<!DOCTYPE html>.*?</html>)",
    re.IGNORECASE | re.DOTALL
)


async def node_synthesizer(state: AgenticState) -> Dict[str, Any]:
    """
    Создаёт финальное улучшенное решение.

    Берёт код победителя, выбранного судьёй, и применяет к нему
    рекомендации по улучшению. Использует мощную LLM модель для
    финальной полировки кода, устранения найденных недостатков
    и интеграции лучших идей из других решений.

    :param state: Глобальное состояние с результатами судейства
    :return: Словарь с ключом 'final_html_code' - финальным решением
    """
    attempts = state["attempts"]
    idx = state.get("winner_candidate_index", -1)
    decision = state.get("judge_feedback")
    config = state["config"]
    task = state["user_task"]

    # Защита от выхода за границы
    if not attempts or idx < 0 or idx >= len(attempts):
        logger.error("❌ Synthesizer: No valid winner found.")
        return {"final_html_code": "<!-- ERROR: No valid solution generated -->"}

    winner = attempts[idx]

    # Если победитель пуст (fallback), отдаем ошибку
    if not winner.get("html_content"):
        logger.error("❌ Synthesizer: Winner has no code.")
        return {"final_html_code": "<!-- ERROR: Winner solution is empty -->"}

    logger.info(f"🏗️ Synthesizing final build based on {winner['model_name_human']}...")

    # === КРИТИЧЕСКИ ВАЖНО: Формируем ПОЛНЫЙ контекст ДЛЯ ВСЕХ КАНДИДАТОВ ===
    # Синтезатор должен видеть ВСЁ, что было сгенерировано всеми моделями
    all_candidates_data = _build_full_synthesis_context(attempts, idx, task)

    # Формируем промпт для синтезатора с ПОЛНЫМИ данными
    user_msg = (
        f"{'='*100}\n"
        f"ORIGINAL TASK:\n"
        f"{'='*100}\n"
        f"{task}\n\n"
        f"{all_candidates_data}\n\n"
        f"{'='*100}\n"
        f"JUDGE'S DECISION\n"
        f"{'='*100}\n"
        f"Winner: Candidate #{idx} ({winner['model_name_human']})\n"
        f"Reasoning: {decision['reasoning'] if decision else 'N/A'}\n\n"
        f"Synthesis Advice (CRITICAL - follow these instructions carefully):\n"
        f"{decision['synthesis_advice'] if decision else 'None'}\n\n"
        f"{'='*100}\n"
        f"YOUR TASK\n"
        f"{'='*100}\n"
        f"You are the winning model. Create the GOLDEN ARTIFACT by synthesizing\n"
        f"the best elements from ALL candidates while avoiding their mistakes.\n"
        f"Follow the Judge's synthesis advice. Output complete HTML code in <thought> + ```html block.\n"
    )

    client = LLMClient(api_key=config.get("system", {}).get("api_key"))

    try:
        # КРИТИЧЕСКИ ВАЖНО: используем модель ПОБЕДИТЕЛЯ, а не fallback!
        synth_conf = config.get("synthesizer", {})
        # Берем модель победителя (model_config_id из winner)
        winner_model_id = winner.get("model_config_id")
        # Если по какой-то причине не можем определить, используем fallback
        model_id = winner_model_id if winner_model_id else synth_conf.get("fallback_model_id", "openai/gpt-4o")

        logger.info(f"🤖 Using model {model_id} for synthesis")

        response = await client.get_completion(
            system_prompt=PROMPT_SYNTHESIZER,
            user_prompt=user_msg,
            model_id=model_id,
            temperature=synth_conf.get("temperature", 0.0),
            max_tokens=synth_conf.get("max_tokens", 8000),
            supports_vision=False
        )

        raw = response["content"]

        # Парсинг
        match = HTML_REGEX.search(raw)
        if match:
            final_code = match.group(1) or match.group(2)
            final_code = final_code.strip()
        else:
            # Если не нашли тегов, но текст есть - считаем это кодом (риск, но лучше чем ничего)
            if "<html" in raw.lower():
                final_code = raw
            else:
                logger.warning("⚠️ Synthesizer output doesn't look like HTML, using winner's code as-is")
                final_code = winner["html_content"]  # Fallback на оригинал

        logger.info(f"✅ Final code synthesized ({len(final_code)} chars)")

    except Exception as e:
        logger.error(f"❌ Synthesizer failed: {e}")
        final_code = winner["html_content"]  # Fallback: возвращаем код победителя без изменений
        logger.info("⚠️ Using winner's code without modifications due to synthesis error")

    return {"final_html_code": final_code}


def _build_full_synthesis_context(attempts: list, winner_idx: int, task: str) -> str:
    """
    Строит ПОЛНЫЙ контекст для синтезатора со ВСЕМИ данными ВСЕХ кандидатов.

    КРИТИЧЕСКИ ВАЖНО: Передаем абсолютно ВСЁ:
    - FULL_LLM_RESPONSE (полный вывод с <thought> блоками)
    - COMPLETE_CODE (весь HTML/CSS/JS)
    - EXECUTION_STATUS и ПОЛНЫЕ логи
    - SCREENSHOT (base64 - если модель поддерживает vision)
    - ПОЛНУЮ верификацию (весь текст критики + scores)

    :param attempts: Список всех SolutionAttempt
    :param winner_idx: Индекс победителя (для пометки)
    :param task: Оригинальная задача
    :return: Гигантская строка со ВСЕМИ данными для всех кандидатов
    """
    blocks = []

    for i, att in enumerate(attempts):
        is_winner = " ⭐ WINNER ⭐" if i == winner_idx else ""
        model_name = att.get("model_name_human", "Unknown")
        status = att.get("status", "unknown")

        # ПОЛНЫЙ вывод LLM
        full_llm_output = att.get("raw_llm_output", "N/A - Raw output not captured")

        # ПОЛНЫЙ код
        complete_code = att.get("html_content", "N/A - No code generated")

        # ПОЛНЫЕ логи
        logs = att.get("execution_logs", [])
        logs_str = "\n".join(logs) if logs else "No console logs"

        # ПОЛНАЯ верификация
        verif_data = att.get("verification") or {}
        critique_full = verif_data.get("critique_text", "N/A - Verification not performed")
        score_logic = verif_data.get("score_logic", 0)
        score_visual = verif_data.get("score_visual", 0)
        found_bugs = verif_data.get("found_bugs", [])

        # Screenshot (если есть)
        has_screenshot = att.get("screenshot_base64")
        screenshot_note = "Screenshot attached" if has_screenshot else "No screenshot"

        block = (
            f"\n{'='*100}\n"
            f"CANDIDATE #{i}{is_winner} | MODEL: {model_name}\n"
            f"{'='*100}\n\n"
            f"--- STATUS ---\n"
            f"Execution Status: {status}\n"
            f"{screenshot_note}\n\n"
            f"--- FULL LLM RESPONSE (original output with <thought> blocks) ---\n"
            f"{full_llm_output}\n\n"
            f"--- COMPLETE CODE (entire HTML/CSS/JS as executed) ---\n"
            f"{complete_code}\n\n"
            f"--- EXECUTION LOGS (complete browser console output) ---\n"
            f"{logs_str}\n\n"
            f"--- VERIFIER CRITIQUE (complete QA analysis) ---\n"
            f"Logic Score: {score_logic}/10\n"
            f"Visual Score: {score_visual}/10\n"
            f"Full Critique:\n{critique_full}\n"
            f"Found Bugs: {', '.join(found_bugs) if found_bugs else 'None reported'}\n"
        )
        blocks.append(block)

    return "\n".join(blocks)
