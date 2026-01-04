"""
Изолированная среда для запуска HTML/JS кода через Playwright.

Основные возможности:
- Запуск HTML в headless браузере (Chromium)
- Создание скриншотов после полной загрузки
- Перехват логов консоли браузера
- Обработка ошибок выполнения JavaScript
"""

# Стандартные библиотеки
import base64
import asyncio
import logging
from typing import TypedDict, List, Optional

# Сторонние библиотеки
from playwright.async_api import async_playwright

# Константы для настройки браузера
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 720
DEFAULT_SCREENSHOT_QUALITY = 80
ERROR_SCREENSHOT_QUALITY = 60
DEFAULT_DEVICE_SCALE = 1.0
RENDER_DELAY_SECONDS = 2  # Пауза для отрисовки canvas/анимаций после networkidle

# Аргументы запуска Chromium
CHROMIUM_LAUNCH_ARGS = ['--no-sandbox']  # Необходимо для Docker окружения

logger = logging.getLogger(__name__)


class ExecutionResult(TypedDict):
    """
    Результат выполнения HTML кода в браузере.

    :param success: Успешно ли выполнен код
    :param screenshot_base64: Скриншот страницы в base64 (если удалось)
    :param logs: Логи консоли браузера
    :param error_message: Сообщение об ошибке (если произошла)
    """
    success: bool
    screenshot_base64: Optional[str]
    logs: List[str]
    error_message: Optional[str]


class HTMLSandbox:
    """
    Песочница для безопасного выполнения HTML/JS кода.

    Запускает код в изолированном headless браузере, собирает логи,
    делает скриншот и возвращает результат.
    """

    def __init__(self, headless: bool = True, viewport_size: Optional[dict] = None):
        """
        Инициализирует песочницу.

        :param headless: Запускать браузер в headless режиме
        :param viewport_size: Размер viewport (по умолчанию 1280x720)
        """
        if viewport_size is None:
            viewport_size = {
                "width": DEFAULT_VIEWPORT_WIDTH,
                "height": DEFAULT_VIEWPORT_HEIGHT
            }
        self.headless = headless
        self.viewport = viewport_size

    async def run_html(self, html_content: str, timeout_ms: int = 10000) -> ExecutionResult:
        """
        Запускает HTML код в изолированном браузере и создаёт скриншот.

        :param html_content: Строка с полным HTML кодом
        :param timeout_ms: Максимальное время ожидания загрузки и рендера (мс)
        :return: Результат выполнения с логами и скриншотом
        """
        logs: List[str] = []
        screenshot_b64: Optional[str] = None
        error_msg: Optional[str] = None
        is_success = False

        async with async_playwright() as p:
            # Запускаем браузер (Chromium)
            browser = await p.chromium.launch(headless=self.headless, args=CHROMIUM_LAUNCH_ARGS)

            # Создаем контекст (изолированная сессия, как инкогнито)
            context = await browser.new_context(
                viewport=self.viewport,
                device_scale_factor=DEFAULT_DEVICE_SCALE
            )

            page = await context.new_page()

            # --- 1. Настройка перехватчиков (Hooks) ---

            # Ловим console.log, console.error и т.д.
            page.on("console", lambda msg: logs.append(f"[{msg.type.upper()}] {msg.text}"))

            # Ловим неотловленные исключения JS (critical!)
            page.on("pageerror", lambda exc: logs.append(f"[JS_EXCEPTION] {exc}"))

            try:
                # --- 2. Загрузка контента ---
                # wait_until="networkidle" означает "жди пока сетевая активность не утихнет"
                # Это важно для подгрузки CDN библиотек (Three.js, etc.)
                await page.set_content(
                    html_content,
                    wait_until="networkidle",
                    timeout=timeout_ms
                )

                # Дополнительная пауза для отрисовки анимаций/canvas
                # Иногда networkidle срабатывает, но canvas начинает рисоваться через 100мс
                await asyncio.sleep(RENDER_DELAY_SECONDS)

                # --- 3. Скриншот ---
                screenshot_bytes = await page.screenshot(
                    type="jpeg",
                    quality=DEFAULT_SCREENSHOT_QUALITY,
                    full_page=False  # Нам нужен только вьюпорт
                )

                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                is_success = True

            except Exception as e:
                # Если упало по таймауту или ошибка парсинга HTML
                error_msg = str(e)
                logger.warning(f"Playwright execution failed: {error_msg}")

                # Попытаемся сделать скриншот даже при ошибке (чтобы видеть состояние)
                try:
                    if not screenshot_b64:
                        screenshot_bytes = await page.screenshot(type="jpeg", quality=ERROR_SCREENSHOT_QUALITY)
                        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                except:
                    pass  # Если совсем всё плохо (браузер крашнулся), то скриншота не будет

            finally:
                await browser.close()

        return {
            "success": is_success,
            "screenshot_base64": screenshot_b64,
            "logs": logs,
            "error_message": error_msg
        }
