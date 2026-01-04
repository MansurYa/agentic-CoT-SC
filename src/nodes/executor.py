"""
Узел-исполнитель: запускает HTML код в браузере и делает скриншот.

Этот модуль отвечает за исполнение сгенерированного HTML/JS кода
в изолированной среде (headless browser через Playwright).
Собирает логи консоли и создаёт скриншот результата.
"""

# Стандартные библиотеки
import logging
from typing import Dict, Any, List

# Модули текущего проекта
from src.core.sandbox import HTMLSandbox
from src.domain.state import SolutionAttempt

logger = logging.getLogger(__name__)


async def node_executor(state: Dict[str, Any]) -> Dict[str, List[SolutionAttempt]]:
    """
    Запускает HTML код в headless браузере и делает скриншот.

    Принимает сгенерированный код из предыдущего узла, загружает его
    в изолированный браузер через Playwright, ожидает полной загрузки
    всех ресурсов, собирает логи консоли и создаёт скриншот.

    :param state: Локальный стейт ветки с полем 'attempts'
    :return: Словарь с обновлённым attempt, содержащим скриншот и логи
    """

    # Берем последний attempt из этой ветки
    current_attempt = state["attempts"][-1]

    # Получаем настройки sandbox из конфига (если прокинут)
    config = state.get("config", {})

    if current_attempt["status"] == "failed" or not current_attempt["html_content"]:
        logger.info(f"⏭️ Skipping execution for {current_attempt['model_name_human']} (no code)")
        return {
            "attempts": [current_attempt],
            "config": config,
            "user_task": state.get("user_task"),
            "user_task_original": state.get("user_task_original")
        }

    logger.info(f"▶️ Executing code from {current_attempt['model_name_human']}...")

    sandbox_conf = config.get("sandbox", {})

    sandbox = HTMLSandbox(
        headless=sandbox_conf.get("headless", True),
        viewport_size=sandbox_conf.get("viewport", {"width": 1280, "height": 720})
    )

    # Запуск
    result = await sandbox.run_html(
        current_attempt["html_content"],
        timeout_ms=sandbox_conf.get("timeout_ms", 15000)
    )

    # Обновляем attempt (TypedDict мутабелен в рантайме Python)
    current_attempt["screenshot_base64"] = result["screenshot_base64"]
    current_attempt["execution_logs"] = result["logs"]

    if result["success"]:
        current_attempt["status"] = "executed"
        current_attempt["error_message"] = None
        logger.info(f"✅ Execution successful for {current_attempt['model_name_human']}")
    else:
        # Если плейрайт упал, но код есть - статус executed_failed,
        # чтобы верификатор все равно посмотрел код
        current_attempt["status"] = "executed_failed"

        # Если ошибка была в таймауте, пишем это явно
        if result["error_message"] and "Timeout" in result["error_message"]:
            current_attempt["error_message"] = "TIMEOUT: Code took too long to render"
        else:
            current_attempt["error_message"] = f"Runtime Error: {result.get('error_message')}"

        logger.warning(f"⚠️ Execution failed for {current_attempt['model_name_human']}: {current_attempt['error_message']}")

    return {
        "attempts": [current_attempt],
        "config": config,
        "user_task": state.get("user_task"),
        "user_task_original": state.get("user_task_original")
    }
