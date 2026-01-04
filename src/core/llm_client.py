"""
Универсальный клиент для работы с OpenRouter API.

Инкапсулирует логику:
- Повторных попыток при сетевых ошибках
- Автоматического fallback для моделей без JSON режима
- Безопасной работы с Vision API
- Подсчёта токенов и формирования статистики
"""

# Стандартные библиотеки
import os
import json
import logging
import re
from typing import Optional, Dict, Any, List, Literal

# Сторонние библиотеки
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Модули текущего проекта
from src.domain.state import UsageStats

# Константы
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROJECT_GITHUB_URL = "https://github.com/MansurYa/Agentic-CoT-SC"
PROJECT_NAME = "Agentic-CoT-SC"

# Настройки повторных попыток
MAX_RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT_SECONDS = 2
RETRY_MAX_WAIT_SECONDS = 10

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Асинхронный клиент для работы с LLM моделями через OpenRouter.

    Основные возможности:
    - Автоматические повторные попытки при сетевых сбоях
    - Fallback на текстовый режим для моделей без JSON поддержки
    - Защита от отправки изображений non-vision моделям
    - Подсчёт токенов и статистики использования
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализирует клиент OpenRouter.

        :param api_key: API ключ, переданный из конфигурации.
        :raises ValueError: Если API ключ не предоставлен.
        """
        if not api_key:
            raise ValueError("API key not provided to LLMClient! Check config.")

        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": PROJECT_GITHUB_URL,
                "X-Title": PROJECT_NAME,
            }
        )

    @staticmethod
    def _clean_markdown_json(text: str) -> str:
        """
        Очищает JSON от markdown оборток.

        Некоторые модели возвращают JSON в формате ```json ... ```,
        этот метод извлекает чистый JSON.
        """
        pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        return match.group(1).strip() if match else text.strip()

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT_SECONDS, max=RETRY_MAX_WAIT_SECONDS),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        reraise=True
    )
    async def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        image_base64: Optional[str] = None,
        supports_vision: bool = False,
        response_format: Literal["text", "json_object"] = "text"
    ) -> Dict[str, Any]:
        """
        Получает ответ от LLM модели через OpenRouter API.

        :param system_prompt: Системный промпт (роль модели)
        :param user_prompt: Текст запроса пользователя
        :param model_id: ID модели на OpenRouter (например, 'anthropic/claude-3.5-sonnet')
        :param temperature: Температура генерации (0.0-1.0)
        :param max_tokens: Максимальное количество токенов ответа
        :param image_base64: Изображение в base64 (опционально, для vision моделей)
        :param supports_vision: Поддерживает ли модель изображения
        :param response_format: Формат ответа ("text" или "json_object")
        :return: Словарь с ключами "content" (текст ответа) и "usage" (статистика)
        """

        messages = [{"role": "system", "content": system_prompt}]
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        # Добавляем картинку ТОЛЬКО если она есть И модель её поддерживает
        if image_base64:
            if supports_vision:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "high"  # Просим высокую детализацию для поиска багов
                    }
                })
            else:
                logger.warning(f"Model {model_id} does not support vision. Image dropped.")
                user_content[0]["text"] += "\n\n[SYSTEM NOTE: Screenshot was available but your model architecture does not support vision input.]"

        messages.append({"role": "user", "content": user_content})

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": response_format}
            )
            return self._extract_result(response)

        except BadRequestError as e:
            # ИСПРАВЛЕНИЕ ОШИБКИ JSON:
            # Если мы просили JSON, но модель/провайдер это не поддерживает (400 Bad Request),
            # пробуем откатиться на text mode.
            if response_format == "json_object":
                logger.warning(f"⚠️ {model_id} rejected JSON mode. Retrying as text.")
                return await self.get_completion(
                    system_prompt=system_prompt + "\n\nIMPORTANT: OUTPUT MUST BE VALID JSON. NO MARKDOWN.",
                    user_prompt=user_prompt,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_base64=image_base64,
                    supports_vision=supports_vision,
                    response_format="text"  # Откат на текст
                )
            raise e

    def _extract_result(self, response) -> Dict[str, Any]:
        """Вспомогательный метод для извлечения данных и статистики."""
        usage_raw = response.usage
        usage_stats: UsageStats = {
            "input_tokens": usage_raw.prompt_tokens if usage_raw else 0,
            "output_tokens": usage_raw.completion_tokens if usage_raw else 0,
            "total_tokens": usage_raw.total_tokens if usage_raw else 0
        }
        return {
            "content": response.choices[0].message.content,
            "usage": usage_stats
        }

    async def get_json_completion(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Обертка для гарантированного получения JSON.
        Пытается распарсить ответ, если это не валидный JSON, бросает ошибку
        (которую можно поймать выше и сделать ретрай).
        """
        kwargs['response_format'] = "json_object"

        # Для JSON режима важно упомянуть 'JSON' в промпте, иначе API OpenAI ругается
        sys_prompt = kwargs.get('system_prompt', '')
        if "json" not in sys_prompt.lower():
            kwargs['system_prompt'] = sys_prompt + "\n\nIMPORTANT: Output strictly in JSON format."

        result = await self.get_completion(*args, **kwargs)

        clean_text = self._clean_markdown_json(result["content"])

        try:
            parsed_json = json.loads(clean_text)
            # Возвращаем уже объект Python, а не строку
            result["parsed_content"] = parsed_json
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from {kwargs.get('model_id')}")
            logger.error(f"Content preview: {clean_text[:100]}...")
            # Здесь можно добавить логику 'Repair', но пока просто роняем
            raise ValueError(f"Model output was not valid JSON: {clean_text[:100]}...")
