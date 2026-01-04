"""
Модель данных для Agentic-CoT-SC.

Определяет структуру состояния, передаваемого между узлами графа.
Все TypedDict классы используются для строгой типизации данных в пайплайне.
"""

# Стандартные библиотеки
import operator
from typing import TypedDict, List, Annotated, Dict, Optional, Any, Literal


class UsageStats(TypedDict):
    """
    Статистика потребления токенов для одного запроса к LLM.

    Используется для отслеживания затрат и формирования отчётов.
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int


class VerificationResult(TypedDict):
    """
    Результат работы верификатора - критика и оценка решения.

    :param score_logic: Оценка логики кода (1-10)
    :param score_visual: Оценка визуального качества (1-10)
    :param critique_text: Подробная текстовая критика
    :param found_bugs: Список найденных проблем и ошибок
    """
    score_logic: int
    score_visual: int
    critique_text: str
    found_bugs: List[str]


class JudgeDecision(TypedDict):
    """
    Решение судьи о выборе лучшего решения из всех кандидатов.

    :param best_model_name: Имя победившей модели
    :param best_attempt_idx: Глобальный индекс лучшего решения
    :param reasoning: Обоснование выбора
    :param synthesis_advice: Рекомендации для финального синтеза
    """
    best_model_name: str
    best_attempt_idx: int
    reasoning: str
    synthesis_advice: str


class SolutionAttempt(TypedDict):
    """
    Полный паспорт одного решения задачи.

    Объект последовательно наполняется данными на каждом этапе пайплайна:
    Generator -> Executor -> Verifier.
    """
    # Идентификация
    attempt_id: str
    model_config_id: str
    model_name_human: str

    # Результат генерации
    status: Literal["generated", "executed", "executed_failed", "verified", "failed"]
    raw_llm_output: Optional[str]  # ПОЛНЫЙ ответ модели (с <thought> блоками и всем текстом)
    html_content: Optional[str]     # Спаршенный HTML код (только для исполнения)
    error_message: Optional[str]

    # Результат исполнения (Playwright)
    screenshot_base64: Optional[str]
    execution_logs: List[str]

    # Результат верификации
    verification: Optional[VerificationResult]

    # Метрики использования
    usage: UsageStats


class AgenticState(TypedDict):
    """
    Глобальное состояние графа LangGraph.

    Передаётся между всеми узлами графа и аккумулирует результаты работы.
    Критически важно поле 'attempts' с аннотацией operator.add для Map-Reduce.
    """
    # Входные данные
    user_task: str
    config: Dict[str, Any]

    # Аккумулятор результатов (Map-Reduce)
    # operator.add обеспечивает слияние списков из параллельных веток
    attempts: Annotated[List[SolutionAttempt], operator.add]

    # Решение судьи
    judge_feedback: Optional[JudgeDecision]

    # Финальный результат
    final_html_code: Optional[str]
    final_report_path: Optional[str]
    winner_candidate_index: Optional[int]
