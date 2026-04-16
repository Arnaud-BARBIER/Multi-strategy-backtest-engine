from __future__ import annotations

from pathlib import Path


REQUIRED_HOOKS = (
    "run_exit_strategy_instant_user",
    "run_exit_strategy_window_user",
    "run_exit_strategy_stateful_user",
)


def use_user_exit_strategies(path: str | None = None) -> Path:
    if path is None:
        user_file = (Path.cwd() / "user_exit_strategies.py").resolve()
    else:
        user_file = Path(path).expanduser().resolve()

    if not user_file.exists():
        raise FileNotFoundError(f"User exit strategy file not found: {user_file}")

    namespace = {}
    code = user_file.read_text(encoding="utf-8")
    exec(compile(code, str(user_file), "exec"), namespace)

    missing = [name for name in REQUIRED_HOOKS if name not in namespace]
    if missing:
        raise AttributeError(
            f"Module '{user_file}' is missing required hooks: {missing}"
        )

    bridge_path = Path(__file__).with_name("active_user_exit_strategies.py")

    bridge_code = f'''from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_user_file = Path(r"{str(user_file)}")
_spec = importlib.util.spec_from_file_location("_btf_user_exit_strategies", str(_user_file))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load user exit strategies from {{_user_file}}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules["_btf_user_exit_strategies"] = _mod
_spec.loader.exec_module(_mod)

run_exit_strategy_instant_user = _mod.run_exit_strategy_instant_user
run_exit_strategy_window_user = _mod.run_exit_strategy_window_user
run_exit_strategy_stateful_user = _mod.run_exit_strategy_stateful_user
'''
    bridge_path.write_text(bridge_code, encoding="utf-8")
    return user_file


def reset_user_exit_strategies() -> None:
    bridge_path = Path(__file__).with_name("active_user_exit_strategies.py")
    bridge_code = '''from .user_exit_strategies import (
    run_exit_strategy_instant_user,
    run_exit_strategy_window_user,
    run_exit_strategy_stateful_user,
)
'''
    bridge_path.write_text(bridge_code, encoding="utf-8")