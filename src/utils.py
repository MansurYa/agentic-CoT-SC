"""
Утилиты для проекта: логирование, сохранение результатов, загрузка конфигов.

Этот модуль содержит вспомогательные функции для настройки окружения,
работы с конфигурационными файлами и сохранения результатов экспериментов
в структурированном виде.
"""

# Стандартные библиотеки
import os
import json
import logging
import base64
from datetime import datetime
from typing import Dict, Any

# Сторонние библиотеки
import yaml
from colorama import init, Fore, Style

# Инициализация цветного вывода для консоли
init(autoreset=True)


def setup_logging(level: str = "INFO") -> None:
    """
    Настраивает систему логирования для всего проекта.

    Устанавливает единый формат логов для всех модулей и подавляет
    избыточный вывод от сторонних библиотек (httpx, openai).

    :param level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    # Подавляем избыточный вывод от сторонних библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_config(path: str = "config/agents_config.yaml") -> dict:
    """
    Загружает конфигурацию проекта из YAML файла.

    :param path: Путь к конфигурационному файлу
    :return: Словарь с параметрами конфигурации
    :raises FileNotFoundError: Если файл конфигурации не найден
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_experiment_results(state: Dict[str, Any], base_dir: str = "experiments") -> str:
    """
    Сохраняет артефакты эксперимента в папку с таймстампом.

    Создаёт структурированную папку с результатами: финальный HTML файл,
    JSON отчёт со всеми метриками, отдельные подпапки для каждого кандидата
    с их кодом и скриншотами. Выводит красивое резюме в консоль.

    :param state: Финальное состояние графа AgenticState
    :param base_dir: Базовая директория для сохранения
    :return: Путь к созданной папке эксперимента
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Сохраняем финальный HTML
    final_html = state.get("final_html_code", "<!-- No code generated -->")
    html_path = os.path.join(exp_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    # 2. Сохраняем подробный отчет (JSON)
    report = {
        "task": state["user_task"],
        "winner": {
            "model": state.get("judge_feedback", {}).get("best_model_name", "N/A"),
            "index": state.get("winner_candidate_index", -1),
            "reasoning": state.get("judge_feedback", {}).get("reasoning", "N/A"),
            "synthesis_advice": state.get("judge_feedback", {}).get("synthesis_advice", "N/A")
        },
        "candidates": []
    }

    # Сериализуем кандидатов (удаляем тяжелые поля для компактности JSON)
    for idx, att in enumerate(state.get("attempts", [])):
        verif = att.get("verification", {})
        cand_data = {
            "index": idx,
            "model": att["model_name_human"],
            "model_id": att["model_config_id"],
            "status": att["status"],
            "error": att.get("error_message"),
            "verification": {
                "score_logic": verif.get("score_logic") if verif else None,
                "score_visual": verif.get("score_visual") if verif else None,
                "critique": verif.get("critique_text") if verif else None,
                "bugs": verif.get("found_bugs") if verif else []
            },
            "execution_logs": att.get("execution_logs", [])[:10],  # Ограничиваем логи
            "usage": att.get("usage", {}),
            "has_screenshot": bool(att.get("screenshot_base64"))
        }
        report["candidates"].append(cand_data)

        # Сохраняем код каждого кандидата в отдельный файл
        if att.get("html_content"):
            cand_dir = os.path.join(exp_dir, f"candidate_{idx}_{att['model_name_human'].replace(' ', '_')}")
            os.makedirs(cand_dir, exist_ok=True)
            with open(os.path.join(cand_dir, "code.html"), "w", encoding="utf-8") as f:
                f.write(att["html_content"])

            # Сохраняем скриншот если есть
            if att.get("screenshot_base64"):
                screenshot_path = os.path.join(cand_dir, "screenshot.jpg")
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(att["screenshot_base64"]))

    report_path = os.path.join(exp_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 3. Красивый вывод в консоль
    print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✅ Эксперимент завершён успешно!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}📁 Результаты сохранены в:{Style.RESET_ALL} {exp_dir}")
    print(f"{Fore.CYAN}📄 Финальное решение:{Style.RESET_ALL} {html_path}")
    print(f"{Fore.CYAN}📊 Подробный отчёт:{Style.RESET_ALL} {report_path}")

    winner_name = state.get("judge_feedback", {}).get("best_model_name", "N/A")
    print(f"\n{Fore.YELLOW}🏆 Победитель:{Style.RESET_ALL} {winner_name}")

    print(f"\n{Fore.MAGENTA}💡 Откройте {html_path} в браузере, чтобы увидеть результат.{Style.RESET_ALL}\n")

    return exp_dir
