"""
Узел-судья: анализирует все решения и выбирает лучшее.

Этот модуль реализует критически важный этап алгоритма CoT-SC -
сравнительный анализ всех сгенерированных решений и выбор победителя.
Содержит исправление логики маппинга локальных индексов в глобальные.
"""

# Стандартные библиотеки
import logging
from typing import Dict, Any, List, Tuple

# Модули текущего проекта
from src.core.llm_client import LLMClient
from src.domain.state import AgenticState, JudgeDecision, SolutionAttempt
from config.prompts import PROMPT_JUDGE

logger = logging.getLogger(__name__)


async def node_judge(state: AgenticState) -> Dict[str, Any]:
    """
    Анализирует все решения и выбирает лучшее.

    Принимает результаты всех параллельных веток (генерация + исполнение +
    верификация), сравнивает их между собой и выбирает победителя.
    Содержит критическое исправление: маппинг локальных индексов валидных
    кандидатов обратно в глобальные индексы списка attempts.

    :param state: Глобальное состояние графа со всеми результатами
    :return: Словарь с решением судьи и индексом победителя
    """
    attempts = state["attempts"]
    task = state["user_task"]
    config = state["config"]

    # --- FIX: СОХРАНЯЕМ ОРИГИНАЛЬНЫЕ ИНДЕКСЫ ---
    # Мы создаем список кортежей: (original_index, attempt_object)
    # Это позволит нам потом понять, на кого реально указывает Судья.
    valid_candidates: List[Tuple[int, SolutionAttempt]] = [
        (i, a) for i, a in enumerate(attempts)
        if a.get("html_content") and len(a["html_content"]) > 50
        # Пропускаем совсем пустые или короткие заглушки
    ]

    if not valid_candidates:
        logger.error("❌ Judge: No valid candidates generated.")
        return {
            "winner_candidate_index": -1,
            "judge_feedback": JudgeDecision(
                best_model_name="None",
                best_attempt_idx=-1,
                reasoning="All attempts failed generation.",
                synthesis_advice=""
            )
        }

    # Формирование контекста для LLM (безопасно)
    context_str = _build_candidates_context(valid_candidates)

    client = LLMClient(api_key=config.get("system", {}).get("api_key"))
    user_message = (
        f"ORIGINAL TASK: {task}\n\n"
        f"=== CANDIDATE ANALYSIS ===\n"
        f"{context_str}\n\n"
        f"INSTRUCTIONS:\n"
        f"Compare candidates 0 to {len(valid_candidates) - 1}.\n"
        f"Select the 'best_attempt_idx' (local index from the list above).\n"
        f"Provide 'reasoning' and 'synthesis_advice' for the final merge.\n"
        f"Output JSON."
    )

    try:
        judge_conf = config.get("judge", {})
        response = await client.get_json_completion(
            system_prompt=PROMPT_JUDGE,
            user_prompt=user_message,
            model_id=judge_conf.get("model_id", "anthropic/claude-3.5-sonnet"),
            temperature=judge_conf.get("temperature", 0.0),
            max_tokens=judge_conf.get("max_tokens", 2000)
        )

        data = response["parsed_content"]

        # Получаем локальный индекс (в рамках valid_candidates)
        local_idx = int(data.get("best_attempt_idx", 0))
        local_idx = max(0, min(local_idx, len(valid_candidates) - 1))

        # --- FIX: МАППИМ ОБРАТНО В ГЛОБАЛЬНЫЙ ИНДЕКС ---
        # Судья выбрал 0-го из валидных, а в глобальном списке это может быть 5-й
        global_idx = valid_candidates[local_idx][0]
        winner_name = valid_candidates[local_idx][1]["model_name_human"]

        decision = JudgeDecision(
            best_model_name=winner_name,
            best_attempt_idx=global_idx,
            reasoning=data.get("reasoning", "No reasoning provided"),
            synthesis_advice=data.get("synthesis_advice", "No specific advice")
        )

        logger.info(f"⚖️ Judge Winner: #{global_idx} ({winner_name})")
        logger.info(f"Reasoning: {decision['reasoning'][:200]}...")

    except Exception as e:
        logger.error(f"🔥 Judge Crashed: {e}")
        # Fallback: берем первый попавшийся валидный вариант
        fallback_global_idx = valid_candidates[0][0]
        decision = JudgeDecision(
            best_model_name=valid_candidates[0][1]["model_name_human"],
            best_attempt_idx=fallback_global_idx,
            reasoning=f"System Error in Judge Node: {str(e)}",
            synthesis_advice=""
        )

    return {
        "judge_feedback": decision,
        "winner_candidate_index": decision["best_attempt_idx"]
    }


def _build_candidates_context(candidates: List[Tuple[int, SolutionAttempt]]) -> str:
    """
    Формирует ПОЛНОЕ текстовое представление всех кандидатов для LLM.

    КРИТИЧЕСКИ ВАЖНО: Передаем ВСЕ данные каждого кандидата согласно новому промпту:
    - FULL_LLM_OUTPUT (полный ответ с <thought> блоками)
    - COMPLETE_CODE (весь HTML/CSS/JS)
    - EXECUTION_STATUS и ПОЛНЫЕ логи
    - ПОЛНУЮ верификацию (весь текст критики)
    - Screenshot доступность (сам screenshot передается отдельно через vision API)

    :param candidates: Список кортежей (глобальный_индекс, решение)
    :return: Полная форматированная строка с ВСЕМИ данными кандидатов
    """
    blocks = []
    for local_idx, (global_idx, att) in enumerate(candidates):
        # 1. Извлекаем все данные безопасно
        model_name = att.get("model_name_human", "Unknown")
        status = att.get("status", "unknown")

        # ПОЛНЫЙ вывод LLM (с рассуждениями)
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

        # Screenshot info (сам скриншот судья получит через vision API если поддерживается)
        has_screenshot = "Yes" if att.get("screenshot_base64") else "No"

        # 2. Формируем блок с ПОЛНОЙ информацией
        block = (
            f"\n{'='*80}\n"
            f"CANDIDATE #{local_idx} | MODEL: {model_name}\n"
            f"{'='*80}\n\n"
            f"--- EXECUTION STATUS ---\n"
            f"Status: {status}\n"
            f"Screenshot Available: {has_screenshot}\n\n"
            f"--- FULL LLM RESPONSE (including <thought> blocks) ---\n"
            f"{full_llm_output}\n\n"
            f"--- COMPLETE CODE (entire HTML/CSS/JS) ---\n"
            f"{complete_code}\n\n"
            f"--- EXECUTION LOGS (complete browser console) ---\n"
            f"{logs_str}\n\n"
            f"--- VERIFIER CRITIQUE (complete QA analysis) ---\n"
            f"Logic Score: {score_logic}/10\n"
            f"Visual Score: {score_visual}/10\n"
            f"Critique: {critique_full}\n"
            f"Found Bugs: {', '.join(found_bugs) if found_bugs else 'None'}\n"
        )
        blocks.append(block)

    return "\n".join(blocks)
