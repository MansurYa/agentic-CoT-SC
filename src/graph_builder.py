"""
Сборка LangGraph с использованием Map-Reduce паттерна.

Этот модуль содержит критическую логику построения графа вычислений
для мультиагентной системы. Реализует паттерн Map-Reduce для
параллельной генерации и последовательного анализа решений.

Основные компоненты:
- Диспетчер (dispatcher) - распределяет задачи по воркерам
- Подграф воркера - цепочка Gen -> Exec -> Verif
- Главный граф - оркестрация всех узлов
"""

# Стандартные библиотеки
import logging
from typing import Any, Dict, List

# Сторонние библиотеки
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Модули текущего проекта
from src.domain.state import AgenticState
from src.nodes.generator import node_generator
from src.nodes.executor import node_executor
from src.nodes.verifier import node_verifier
from src.nodes.judge import node_judge
from src.nodes.synthesizer import node_synthesizer

logger = logging.getLogger(__name__)


# --- ФУНКЦИЯ ДИСПЕТЧЕРА (FAN-OUT) ---

def dispatcher(state: AgenticState) -> List[Send]:
    """
    Диспетчер для распределения задач по параллельным воркерам.

    Создаёт N команд Send для запуска подграфов обработки.
    Каждый подграф получает свою модель из конфигурации и
    выполняется независимо от других (Map-фаза алгоритма).

    :param state: Глобальное состояние графа с конфигурацией
    :return: Список команд Send для параллельного выполнения
    """
    generators_conf = state["config"]["generators"]
    task = state["user_task"]
    config = state["config"]

    logger.info(f"📤 Dispatching {len(generators_conf)} parallel workers...")

    # Возвращаем список команд Send.
    # Каждая команда запускает цепочку 'worker_chain' с уникальным аргументом.
    return [
        Send("worker_chain", {
            "user_task": task,
            "model_config": conf,
            "config": config,  # Прокидываем весь конфиг для доступа в узлах
            "attempts": []  # Инициализируем пустой список для reducer'а
        })
        for conf in generators_conf
    ]


# --- ПОСТРОЕНИЕ ОСНОВНОГО ГРАФА ---

def build_graph():
    """
    Собирает и компилирует граф выполнения для Agentic-CoT-SC.

    Создаёт сложный граф вычислений, состоящий из:
    1. Подграфа воркера (Generator -> Executor -> Verifier)
    2. Главного графа с диспетчеризацией и синтезом

    Схема выполнения:
    START -> Dispatcher (Map) -> [N × Worker Chain] -> Judge (Reduce) -> Synthesizer -> END

    :return: Скомпилированный граф LangGraph, готовый к выполнению
    """

    # Создаем главный граф
    workflow = StateGraph(AgenticState)

    # --- ПОДГРАФ (WORKER CHAIN) ---
    # Это линейная цепочка: Gen -> Exec -> Verif
    # LangGraph позволяет добавлять узлы как функции напрямую

    # Создаем отдельный подграф для воркера
    worker_graph = StateGraph(dict)  # Используем простой dict для локального стейта

    worker_graph.add_node("generator", node_generator)
    worker_graph.add_node("executor", node_executor)
    worker_graph.add_node("verifier", node_verifier)

    # Связи внутри воркера (линейная цепочка)
    worker_graph.add_edge(START, "generator")
    worker_graph.add_edge("generator", "executor")
    worker_graph.add_edge("executor", "verifier")
    worker_graph.add_edge("verifier", END)

    # Компилируем подграф
    worker_chain = worker_graph.compile()

    # --- ГЛАВНЫЙ ГРАФ ---

    # Добавляем скомпилированный воркер как узел
    workflow.add_node("worker_chain", worker_chain)

    # Добавляем узлы главного уровня
    workflow.add_node("judge", node_judge)
    workflow.add_node("synthesizer", node_synthesizer)

    # --- СВЯЗИ ГЛАВНОГО ГРАФА ---

    # 1. START -> Dispatcher (который возвращает список Send)
    # conditional_edges позволяет вернуть динамический список переходов
    workflow.add_conditional_edges(
        START,
        dispatcher,
        # Список возможных целевых узлов (в данном случае только worker_chain)
        ["worker_chain"]
    )

    # 2. Worker Chain -> Judge
    # LangGraph автоматически ждет завершения ВСЕХ параллельных веток Send
    # перед переходом к следующему узлу
    workflow.add_edge("worker_chain", "judge")

    # 3. Judge -> Synthesizer -> END
    workflow.add_edge("judge", "synthesizer")
    workflow.add_edge("synthesizer", END)

    # Компилируем граф
    compiled_graph = workflow.compile()

    logger.info("✅ Graph compiled successfully")

    return compiled_graph
