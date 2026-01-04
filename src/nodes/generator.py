"""
Узел-генератор: превращает текстовую задачу в HTML/JS код.

Этот модуль отвечает за параллельную генерацию решений задачи
различными LLM моделями. Каждый генератор работает независимо,
создавая self-contained HTML файл.
"""

# Стандартные библиотеки
import re
import logging
import uuid
from typing import Dict, Any, List

# Модули текущего проекта
from src.core.llm_client import LLMClient
from src.domain.state import SolutionAttempt, UsageStats
from config.prompts import PROMPT_GENERATOR

logger = logging.getLogger(__name__)

# Регулярное выражение для извлечения HTML кода из ответа модели
# 1. Ищет содержимое внутри ```html ... ```
# 2. ИЛИ ищет контент между <!DOCTYPE html> и </html>
HTML_REGEX = re.compile(
    r"```html\s*(.*?)```|(\s*<!DOCTYPE html>.*?</html>)",
    re.IGNORECASE | re.DOTALL
)


async def node_generator(state: Dict[str, Any]) -> Dict[str, List[SolutionAttempt]]:
    """
    Генерирует HTML/JS решение задачи с помощью LLM модели.

    Узел запускается параллельно для каждой модели из конфигурации.
    Принимает задачу пользователя, отправляет её в LLM и извлекает
    сгенерированный HTML код из ответа модели.

    :param state: Локальный стейт воркера с полями 'user_task' и 'model_config'
    :return: Словарь с ключом 'attempts', содержащий список из одного SolutionAttempt
    """

    # 1. Распаковка payload (Input Validation)
    task = state.get("user_task")
    conf = state.get("model_config")
    global_config = state.get("config")
    user_task_original = state.get("user_task_original")

    if not task or not conf or not global_config:
        raise ValueError(f"Generator received invalid state: {list(state.keys())}")

    logger.info(f"🤖 Generating with {conf['name']}...")
    
    api_key = global_config.get("system", {}).get("api_key")
    client = LLMClient(api_key=api_key)

    try:
        # 2. Вызов LLM
        response = await client.get_completion(
            system_prompt=PROMPT_GENERATOR,
            user_prompt=task,
            model_id=conf["model_id"],
            temperature=conf["temperature"],
            max_tokens=conf.get("max_tokens", 4000),
            supports_vision=False  # Генерация всегда текстовая
        )

        raw_content = response["content"]  # ПОЛНЫЙ ответ LLM (с <thought> блоками)
        usage = response["usage"]

        # 3. Парсинг кода (Robust Parsing)
        match = HTML_REGEX.search(raw_content)
        if match:
            # group(1) - это то, что внутри ```html```
            # group(2) - это то, что внутри <!DOCTYPE>...</html>
            html_code = match.group(1) or match.group(2)
            html_code = html_code.strip()
            status = "generated"
            err = None
        else:
            # Fallback: Если модель вернула код без оберток, но он похож на HTML
            if "<html" in raw_content.lower() and "</html>" in raw_content.lower():
                html_code = raw_content.strip()
                status = "generated"
                err = None
            else:
                html_code = None
                status = "failed"
                err = "HTML tags not found in response"
                logger.warning(f"⚠️ {conf['name']} output format mismatch.")

    except Exception as e:
        logger.error(f"❌ Generator {conf['name']} crashed: {e}")
        raw_content = None
        html_code = None
        status = "failed"
        err = str(e)
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # 4. Создание объекта SolutionAttempt
    attempt = SolutionAttempt(
        attempt_id=str(uuid.uuid4()),
        model_config_id=conf["model_id"],
        model_name_human=conf["name"],
        status=status,
        raw_llm_output=raw_content,  # ПОЛНЫЙ ответ модели (с рассуждениями)
        html_content=html_code,       # Только спаршенный код (для исполнения)
        error_message=err,
        screenshot_base64=None,
        execution_logs=[],
        verification=None,  # Пока пусто
        usage=usage
    )

    # Возвращаем список, чтобы operator.add в глобальном стейте добавил его
    return {
        "attempts": [attempt],
        "config": global_config,
        "user_task": task,
        "user_task_original": user_task_original
    }
