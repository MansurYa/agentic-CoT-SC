"""
Главная точка входа для Agentic-CoT-SC.

Этот модуль запускает мультиагентную систему генерации и верификации кода
на основе алгоритма Chain-of-Thought Self-Consistency. Управляет CLI интерфейсом,
загружает конфигурацию, инициализирует LangGraph и сохраняет результаты.
"""

# Стандартные библиотеки
import asyncio
import argparse
import os
import sys
import traceback

# Сторонние библиотеки
from colorama import Fore, Style

# Модули текущего проекта
from src.graph_builder import build_graph
from src.utils import setup_logging, load_config, save_experiment_results

async def main():
    """
    Основная функция запуска Agentic-CoT-SC.

    Выполняет полный цикл работы системы:
    1. Парсинг аргументов командной строки (задача, путь к конфигу)
    2. Загрузка и валидация конфигурационного файла
    3. Настройка системы логирования
    4. Проверка наличия API ключей
    5. Инициализация и компиляция графа LangGraph
    6. Запуск параллельной генерации и верификации решений
    7. Сохранение результатов эксперимента в структурированную папку

    :raises SystemExit: При отсутствии конфигурации или API ключа
    :raises Exception: При критических ошибках во время выполнения графа
    """

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Agentic-CoT-SC: Multi-Agent Code Generation with Self-Consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --task "Create a particle system with mouse interaction"
  python main.py --task "Animate a solar system using Three.js" --config config/custom.yaml
        """
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Описание задачи для HTML/JS генерации"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/agents_config.yaml",
        help="Путь к конфигурационному файлу (по умолчанию: config/agents_config.yaml)"
    )
    args = parser.parse_args()

    # Загрузка конфигурации
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"{Fore.RED}❌ Ошибка: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # Настройка логирования
    setup_logging(config["system"]["log_level"])

    # Проверка API ключа в конфиге
    api_key = config.get("system", {}).get("api_key")
    if not api_key or "your-key-here" in api_key:
        print(f"{Fore.RED}❌ Ошибка: API-ключ не найден или является плейсхолдером в {args.config}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Откройте файл {args.config} и вставьте ваш ключ в поле 'api_key'.{Style.RESET_ALL}")
        sys.exit(1)

    # Красивый вывод старта
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🚀 Запуск Agentic-CoT-SC{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    print(f"{Fore.GREEN}📋 Задача:{Style.RESET_ALL} {args.task}")
    print(f"{Fore.GREEN}🤖 Генераторов:{Style.RESET_ALL} {len(config['generators'])}")
    print(f"{Fore.GREEN}⚙️  Конфиг:{Style.RESET_ALL} {args.config}\n")

    # Инициализация графа
    print(f"{Fore.YELLOW}🔧 Инициализация графа...{Style.RESET_ALL}")
    app = build_graph()

    # Формирование начального стейта
    initial_state = {
        "user_task": args.task,
        "config": config,
        "attempts": [],  # Пустой список для Reducer'а
        "judge_feedback": None,
        "final_html_code": None,
        "winner_candidate_index": None
    }

    # Запуск графа
    print(f"{Fore.YELLOW}▶️  Запуск параллельной генерации...{Style.RESET_ALL}\n")

    try:
        # Используем ainvoke для асинхронного выполнения (Playwright требует async)
        final_state = await app.ainvoke(initial_state)

        # Сохранение результатов
        save_experiment_results(final_state, config["system"]["experiments_dir"])

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Остановлено пользователем{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Критическая ошибка: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Запуск асинхронной функции
    asyncio.run(main())
