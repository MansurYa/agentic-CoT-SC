"""
Узел-верификатор: анализирует код и скриншот, ищет ошибки.

Этот модуль отвечает за критический анализ сгенерированного кода.
Использует LLM с vision capabilities для проверки как логики кода,
так и визуального результата на скриншоте.
"""

# Стандартные библиотеки
import logging
from typing import Dict, Any, List

# Модули текущего проекта
from src.core.llm_client import LLMClient
from src.domain.state import SolutionAttempt, VerificationResult
from config.prompts import PROMPT_VERIFIER

logger = logging.getLogger(__name__)


async def node_verifier(state: Dict[str, Any]) -> Dict[str, List[SolutionAttempt]]:
    """
    Проводит верификацию сгенерированного кода и визуального результата.

    Анализирует исходный код, логи консоли браузера и скриншот
    (если доступен). Использует LLM для поиска синтаксических ошибок,
    логических проблем и визуальных артефактов.

    :param state: Локальный стейт ветки с результатом исполнения
    :return: Словарь с обновлённым attempt, содержащим результаты верификации
    """

    # 1. Извлекаем текущую попытку (она одна в этой ветке)
    current_attempt = state["attempts"][-1]
    config = state.get("config", {})

    # Если генерация провалилась, верифицировать нечего
    if not current_attempt["html_content"] or current_attempt["status"] == "failed":
        logger.info(f"⏭️ Skipping verification for {current_attempt['model_name_human']} (no code)")
        return {"attempts": [current_attempt]}

    logger.info(f"🧐 Verifying {current_attempt['model_name_human']}...")

    # 2. Формируем контекст для критика (согласно новому промпту)
    logs_str = "\n".join(current_attempt["execution_logs"])  # Передаем ВСЕ логи
    has_screenshot = bool(current_attempt["screenshot_base64"])

    # КРИТИЧЕСКИ ВАЖНО: передаем ПОЛНЫЙ ответ LLM (с <thought> блоками)
    full_llm_response = current_attempt.get("raw_llm_output", "N/A - Not captured")
    parsed_code = current_attempt.get("html_content", "N/A")

    user_msg = (
        f"USER_TASK:\n{state.get('user_task', 'N/A')}\n\n"
        f"=== FULL_LLM_RESPONSE (with <thought> blocks) ===\n"
        f"{full_llm_response}\n\n"
        f"=== PARSED_CODE (extracted HTML/JS/CSS) ===\n"
        f"{parsed_code}\n\n"
        f"=== EXECUTION_LOGS (Browser Console) ===\n"
        f"{logs_str}\n\n"
        f"EXECUTION STATUS: {current_attempt['status']}\n\n"
        "Note: Screenshot is attached separately as image (if available).\n"
        "Analyze according to your investigation protocol and return strict JSON."
    )

    client = LLMClient(api_key=config.get("system", {}).get("api_key"))

    # 3. Определяем, использовать ли Vision
    verifier_conf = config.get("verifier", {})
    verifier_model = verifier_conf.get("model_id", "openai/gpt-4o")
    use_vision = verifier_conf.get("use_vision_if_available", True) and has_screenshot

    try:
        response = await client.get_json_completion(
            system_prompt=PROMPT_VERIFIER,
            user_prompt=user_msg,
            model_id=verifier_model,
            temperature=verifier_conf.get("temperature", 0.2),
            max_tokens=verifier_conf.get("max_tokens", 2000),
            image_base64=current_attempt["screenshot_base64"] if use_vision else None,
            supports_vision=use_vision
        )

        data = response["parsed_content"]

        # 4. Обновляем объект попытки
        verification = VerificationResult(
            score_logic=int(data.get("score_logic", 0)),
            score_visual=int(data.get("score_visual", 0)),
            critique_text=data.get("critique_text", "No critique"),
            found_bugs=data.get("found_bugs", [])
        )

        current_attempt["verification"] = verification
        current_attempt["status"] = "verified"  # Маркируем как проверенный

        logger.info(f"✅ Verification complete for {current_attempt['model_name_human']}: "
                   f"Logic={verification['score_logic']}/10, Visual={verification['score_visual']}/10")

    except Exception as e:
        logger.error(f"❌ Verifier failed for {current_attempt['model_name_human']}: {e}")
        # Не роняем процесс, просто пишем, что верификация не удалась
        current_attempt["verification"] = VerificationResult(
            score_logic=0, score_visual=0,
            critique_text=f"Verification process failed: {str(e)}",
            found_bugs=["Verifier Crash"]
        )
        current_attempt["status"] = "verified"  # Всё равно помечаем как обработанный

    return {"attempts": [current_attempt]}
