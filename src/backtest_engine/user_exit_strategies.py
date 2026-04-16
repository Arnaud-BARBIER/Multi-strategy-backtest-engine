import numpy as np
from numba import njit

from .exit_strategy_system import (
    N_EXIT_ACT_COLS,
    ACT_TYPE,
    EXIT_ACT_NONE,
)


@njit(cache=True)
def run_exit_strategy_instant_user(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,    
    state_global,     
):
    action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)
    action[ACT_TYPE] = EXIT_ACT_NONE
    return action


@njit(cache=True)
def run_exit_strategy_window_user(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,   
    state_global,     
):
    action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)
    action[ACT_TYPE] = EXIT_ACT_NONE
    return action


@njit(cache=True)
def run_exit_strategy_stateful_user(
    strategy_id,
    strategy_rt_matrix,
    i, k,
    opens, highs, lows, closes,
    features,
    pos, exit_rt,
    state_per_pos,    
    state_global,     
):
    action = np.zeros(N_EXIT_ACT_COLS, dtype=np.float64)
    action[ACT_TYPE] = EXIT_ACT_NONE
    return action