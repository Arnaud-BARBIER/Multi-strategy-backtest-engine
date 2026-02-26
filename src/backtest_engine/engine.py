class BacktestEngine:
    from collections import namedtuple
    Bar = namedtuple("Bar", ["Open", "High", "Low", "Close", "name"])

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        self.df = df
        self.cfg = cfg

        # State
        self.positions = []
        self.trades = []
        self.recent_entries = []
        self.last_session_id = None
        self.last_day = None

        # Observation
        self.use_observation = cfg.observation_hours is not None
        self.pending_observations = []
        self.observation_bars = int(cfg.observation_hours * 60 / cfg.timeframe_minutes) if self.use_observation else 0

        # Precompute arrays (exactement comme toi)
        self.opens = df["Open"].to_numpy()
        self.highs = df["High"].to_numpy()
        self.lows = df["Low"].to_numpy()
        self.closes = df["Close"].to_numpy()
        self.signals = df["Signal"].to_numpy()
        self.ema1s = df["EMA1"].to_numpy() if "EMA1" in df.columns else None
        self.ema2s = df["EMA2"].to_numpy() if "EMA2" in df.columns else None
        self.atrs = df["ATR"].to_numpy() if "ATR" in df.columns else None
        self.index = df.index

        self.parsed_windows = [
            self.parse_window(cfg.time_window_1),
            self.parse_window(cfg.time_window_2),
            self.parse_window(cfg.time_window_3),
        ]

        self.exit_mode = "ema" if sum([cfg.EMA1_TP, cfg.EMA2_TP, cfg.EMA_CROSS_TP]) == 1 else "fixed"

    # ---------  Ancienne fonctions du V2, mais en methods) ---------

    @classmethod
    def from_ticker(cls, pipeline: DataPipeline, ticker: str, start: str, end: str, cfg: BacktestConfig):
        df = pipeline.build(ticker, start, end, cfg)
        return cls(df, cfg)

    @staticmethod
    def parse_window(w):
        if w is None:
            return None
        s, e = w.split("-")
        return pd.to_datetime(s).time(), pd.to_datetime(e).time()

    @staticmethod
    def time_in_window_fast(ts, parsed_window):
        if parsed_window is None:
            return False
        start, end = parsed_window
        current = ts.time()
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    @staticmethod
    def _update_be_logic(pos, i, bar, be_trigger_pct, be_offset_pct, be_delay_bars):
        position    = pos["side"]
        entry_price = pos["entry_price"]
        base_sl     = pos["sl"]
        entry_index = pos["entry_index"]
        be_armed      = pos["be_armed"]
        pending_be_sl = pos["pending_be_sl"]
        be_arm_index  = pos["be_arm_index"]
        be_active     = pos["be_active"]

        if be_trigger_pct is None or position == 0:
            return base_sl, be_armed, pending_be_sl, be_arm_index, be_active

        delay_ok = (i - entry_index) >= be_delay_bars

        # ===== LONG =====
        if position == 1 and delay_ok and not be_armed:
            trigger_price = entry_price * (1 + be_trigger_pct)
            if bar.High >= trigger_price:
                be_armed      = True
                be_arm_index  = i
                pending_be_sl = entry_price * (1 + be_offset_pct)

        # ===== SHORT =====
        if position == -1 and delay_ok and not be_armed:
            trigger_price = entry_price * (1 - be_trigger_pct)
            if bar.Low <= trigger_price:
                be_armed      = True
                be_arm_index  = i
                pending_be_sl = entry_price * (1 - be_offset_pct)

        # ===== Activation bougie suivante =====
        if be_armed and i > be_arm_index:
            if position == 1:
                base_sl = max(base_sl, pending_be_sl)
            elif position == -1:
                base_sl = min(base_sl, pending_be_sl)
            be_armed      = False
            be_active     = True
            pending_be_sl = None
            be_arm_index  = None

        return base_sl, be_armed, pending_be_sl, be_arm_index, be_active

    @staticmethod
    def trade_analysis(position, exit_price, entry_price, entry_time, ts, exit_reason,
                    mae=None, mfe=None, mae_h=None, mfe_h=None):

        ret = position * (exit_price - entry_price) / entry_price

        trade = {
            "entry_time": entry_time,
            "exit_time": ts,
            "side": position,
            "entry": entry_price,
            "exit": exit_price,
            "return": ret,
            "reason": exit_reason,
        }

        # --- Ajouter MAE / MFE uniquement si présents ---
        if (mae is not None) and (mfe is not None):
            trade["mae"] = mae
            trade["mfe"] = mfe
            trade["capture_ratio"] = ret / mfe if mfe > 0 else None

        # --- Ajouter Hold metrics uniquement si présents ---
        if (mae_h is not None) and (mfe_h is not None):
            trade["hold_mfe"] = mfe_h
            trade["hold_mae"] = mae_h
            trade["capture_ratio_hold"] = (
                round(ret / mfe_h, 3) if mfe_h > 0 else None
            )

        return trade

    @staticmethod
    def Tp_Sl_prices(side, entry_price, tp_pct=None, sl_pct=None,
                 use_atr=False, atr_value=None, tp_atr_mult=None, sl_atr_mult=None):

        if use_atr is True:
            # Les deux en ATR
            if atr_value is None:
                return None, None
            tp = entry_price + side * atr_value * tp_atr_mult
            sl = entry_price - side * atr_value * sl_atr_mult

        elif use_atr == 1:
            # TP en ATR, SL fixe
            if atr_value is None:
                return None, None
            tp = entry_price + side * atr_value * tp_atr_mult
            sl = entry_price * (1 - side * sl_pct)

        elif use_atr == -1:
            # SL en ATR, TP fixe
            if atr_value is None:
                return None, None
            tp = entry_price * (1 + side * tp_pct)
            sl = entry_price - side * atr_value * sl_atr_mult

        else:
            # Les deux fixe
            tp = entry_price * (1 + side * tp_pct)
            sl = entry_price * (1 - side * sl_pct)

        return tp, sl

    @staticmethod
    def _update_trailing_runner(pos, bar, atr_value, trailing_trigger_pct, trailing_mult):
        side = pos["side"]
        ep   = pos["entry_price"]

        # 1. Armement
        if not pos["runner_active"] and not pos.get("runner_armed", False):
            threshold = ep * (1 + side * trailing_trigger_pct)
            if (side == 1 and bar.High >= threshold) or \
            (side == -1 and bar.Low <= threshold):
                pos["runner_armed"]     = True
                pos["runner_threshold"] = threshold

        # Activation bougie suivante
        if pos.get("runner_armed") and not pos["runner_active"]:
            pos["runner_active"] = True
            pos["runner_armed"]  = False
            initial_sl = bar.Close - side * atr_value * trailing_mult
            if side == 1:
                pos["runner_sl"] = max(initial_sl, ep)
            else:
                pos["runner_sl"] = min(initial_sl, ep)
            return pos

        # Trailing ATR dynamique
        if pos["runner_active"]:
            new_sl = bar.Close - side * atr_value * trailing_mult
            if side == 1:
                pos["runner_sl"] = max(pos["runner_sl"], new_sl, ep)
            else:
                pos["runner_sl"] = min(pos["runner_sl"], new_sl, ep)

        return pos

    def _apply_reverse(self, sig_prev, bar_open, ts):
        cfg = self.cfg
        if not cfg.reverse_mode or sig_prev == 0:
            return

        closed = []
        surviving = []
        for pos in self.positions:
            if pos["side"] == -sig_prev:
                if cfg.fast:
                    closed.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=bar_open,
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=ts,
                        exit_reason="REVERSE",
                    ))
                else:
                    closed.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=bar_open,
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=ts,
                        exit_reason="REVERSE",
                        mae=pos.get("mae", 0.0),
                        mfe=pos.get("mfe", 0.0)
                    ))
            else:
                surviving.append(pos)

        self.positions = surviving
        self.trades.extend(closed)

    def get_active_session(self, ts):
        for idx, w in enumerate(self.parsed_windows):
            if w is not None and self.time_in_window_fast(ts, w):
                return idx
        return None

    def _entry_logic(self, i, ts, bar, sig_prev):

        # Time filter
        cfg = self.cfg
        atr_value = self.atrs[i] if (cfg.use_atr_sl_tp and self.atrs is not None) else None
        # windows est déjà garanti dans __init__
        if any(w is not None for w in self.parsed_windows):
            in_any_window = any(
                self.time_in_window_fast(ts, w) 
                for w in self.parsed_windows 
                if w is not None
            )
            if not in_any_window:
                return None

        if sig_prev == 0:
            return None

        body_pct = abs(self.opens[i-1] - self.closes[i-1]) / self.closes[i-1]
        size_ok = cfg.min_size_pct < body_pct < cfg.max_size_pct

        direction_ok = (
            (sig_prev == 1 and self.closes[i-1] > self.opens[i-1]) or
            (sig_prev == -1 and self.closes[i-1] < self.opens[i-1])
        )

        # Filtre 
        if cfg.Candle_Size_filter:
            if not (size_ok and (not cfg.Previous_Candle_same_direction or direction_ok)):
                return None

        # entrée exécutée si filtre OK ou filtre désactivé
        entry_price = bar.Open

        if cfg.use_atr_sl_tp and atr_value is not None and np.isnan(atr_value):
            return None

        tp, sl = self.Tp_Sl_prices(
            side=sig_prev,
            entry_price=entry_price,
            tp_pct=cfg.tp_pct,
            sl_pct=cfg.sl_pct,
            use_atr=cfg.use_atr_sl_tp,
            tp_atr_mult=cfg.tp_atr_mult,
            sl_atr_mult=cfg.sl_atr_mult,
            atr_value=atr_value
        )

        new_position_dict={
        "side": sig_prev,
        "entry_price": bar.Open,
        "entry_time": ts,
        "entry_index": i,
        "tp": tp,
        "sl": sl,
        "be_armed": False,
        "pending_be_sl": None,
        "be_arm_index": None,
        "be_active": False,
        "runner_active": False,
        "runner_armed":  False,   
        "runner_sl":     None,
        "mae":0.0,
        "mfe":0.0       
        }
        return new_position_dict

    def _max_entries_reset(self,ts):
        cfg = self.cfg
        current_day = ts.date()

        # --- Reset journalier ---
        if cfg.ME_reset_mode == "day":
            if self.last_day is None:
                self.last_day = current_day
            elif current_day != self.last_day:
                self.recent_entries = []
                self.last_day = current_day

        # --- Reset par session ---
        elif cfg.ME_reset_mode == "session":
            current_session = self.get_active_session(ts)
            if self.last_session_id is None:
                self.last_session_id = current_session
            elif current_session != self.last_session_id:
                self.recent_entries = []
                self.last_session_id = current_session

    def _max_entries_filter(self, i, entry):
        cfg = self.cfg
        if not cfg.MaxEntries4Periods:
            return entry

        # Nettoyage fenêtre glissante
        cutoff = i - cfg.ME_Period_Y
        j = 0
        while j < len(self.recent_entries) and self.recent_entries[j] < cutoff:
            j += 1
        self.recent_entries = self.recent_entries[j:]

        # Vérification quota
        if len(self.recent_entries) >= cfg.ME_X:
            return None

        return entry

    def _exit_logic(self, i, bar, pos):
        cfg = self.cfg
        ema1 = self.ema1s[i] if self.ema1s is not None else None
        ema2 = self.ema2s[i] if self.ema2s is not None else None
        
        if not cfg.allow_exit_on_entry_bar and i == pos['entry_index']:
            return None

        side  = pos['side']
        tp    = pos['tp']
        sl    = pos['sl']
        o     = bar.Open
        h     = bar.High
        l     = bar.Low
        close = bar.Close

        # ── RUNNER ACTIF ─────────────────────────────────────────────
        if pos.get("runner_active") and pos["runner_sl"] is not None:
            runner_sl       = pos["runner_sl"]
            threshold_price = pos.get("runner_threshold")
            be_reason       = "BE" if pos.get("be_active", False) else "SL"

            under_threshold = threshold_price is not None and (
                (side == 1 and runner_sl <= threshold_price) or
                (side == -1 and runner_sl >= threshold_price)
            )

            if under_threshold:
                # SL fixe prend le relais
                if side == 1:
                    if o <= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                    elif l <= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                else:
                    if o >= sl:   return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                    elif h >= sl: return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                return None  # TP bloqué

            # runner_sl valide → logique runner
            if side == 1:
                if o <= runner_sl:   return {"exit_price": o,         "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif l <= runner_sl: return {"exit_price": runner_sl, "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif o <= sl:        return {"exit_price": o,         "exit_time": bar.name, "reason": be_reason}
                elif l <= sl:        return {"exit_price": sl,        "exit_time": bar.name, "reason": be_reason}
            else:
                if o >= runner_sl:   return {"exit_price": o,         "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif h >= runner_sl: return {"exit_price": runner_sl, "exit_time": bar.name, "reason": "RUNNER_SL"}
                elif o >= sl:        return {"exit_price": o,         "exit_time": bar.name, "reason": be_reason}
                elif h >= sl:        return {"exit_price": sl,        "exit_time": bar.name, "reason": be_reason}

            return None  # TP bloqué
                
        # =========================
        # FIXED MODE
        # =========================
        if self.exit_mode == "fixed":
            be_reason = "BE" if pos.get("be_active", False) else "SL"

            if side == 1:
                if o <= sl:
                    return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif l <= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                elif h >= tp:                                                      # ← TP enfin accessible
                    return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

            elif side == -1:
                if o >= sl:
                    return {"exit_price": o,  "exit_time": bar.name, "reason": be_reason}
                elif h >= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}
                elif l <= tp:                                                      # ← TP enfin accessible
                    return {"exit_price": tp, "exit_time": bar.name, "reason": "TP"}

        # =========================
        # EMA MODE
        # =========================
        elif self.exit_mode == "ema":

            entry_price = pos["entry_price"]

            # =========================
            # 1️⃣ SL FIXE PRIORITAIRE (intrabar)
            # =========================

            be_reason = "BE" if pos.get("be_active", False) else "SL"

            if side == 1:
                if o <= sl:
                    return {"exit_price": o, "exit_time": bar.name, "reason": be_reason}
                if l <= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

            elif side == -1:
                if o >= sl:
                    return {"exit_price": o, "exit_time": bar.name, "reason": be_reason}
                if h >= sl:
                    return {"exit_price": sl, "exit_time": bar.name, "reason": be_reason}

            # =========================
            # 2️⃣ EMA TP UNIQUEMENT SI PROFIT
            # =========================

            trade_positive = (
                (side == 1 and close > entry_price) or
                (side == -1 and close < entry_price)
            )

            if not trade_positive:
                return None   # on bloque toute sortie EMA en perte

            # ----- EMA1 TP -----
            if cfg.EMA1_TP:
                if side == 1 and close < ema1:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}
                if side == -1 and close > ema1:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA1_TP"}

            # ----- EMA2 TP -----
            if cfg.EMA2_TP:
                if side == 1 and close < ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}
                if side == -1 and close > ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA2_TP"}

            # ----- EMA CROSS TP -----
            if cfg.EMA_CROSS_TP:
                if side == 1 and ema1 < ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}
                if side == -1 and ema1 > ema2:
                    return {"exit_price": close, "exit_time": bar.name, "reason": "EMA_CROSS_TP"}


        return None

    def _update_observations(self, i, bar):
        still_pending = []
        for obs in self.pending_observations:
            favorable = (bar.High - obs["ep"]) / obs["ep"] * obs["side"]
            adverse   = (bar.Low  - obs["ep"]) / obs["ep"] * obs["side"]
            
            t = self.trades[obs["idx"]]
            t["hold_mfe"] = max(t["hold_mfe"], favorable)
            t["hold_mae"] = min(t["hold_mae"], adverse)
            
            if i < obs["end_i"]:
                still_pending.append(obs)
        
        self.pending_observations = still_pending

    def _update_positions(self, i, bar):
        cfg = self.cfg
        surviving = []
        
        for pos in self.positions:
            
            # MAE/MFE
            if not cfg.fast:
                ep   = pos["entry_price"]
                side = pos["side"]
                pos["mfe"] = max(pos["mfe"], (bar.High - ep) / ep * side)
                pos["mae"] = min(pos["mae"], (bar.Low  - ep) / ep * side)

            # BE
            if cfg.be_trigger_pct is not None:
                pos["sl"], \
                pos["be_armed"], \
                pos["pending_be_sl"], \
                pos["be_arm_index"], \
                pos["be_active"] = self._update_be_logic(
                    pos=pos,
                    i=i,
                    bar=bar,
                    be_trigger_pct=cfg.be_trigger_pct,
                    be_offset_pct=cfg.be_offset_pct,
                    be_delay_bars=cfg.be_delay_bars
                )

            # Runner
            if cfg.trailing_trigger_pct is not None and self.atrs is not None:
                pos = self._update_trailing_runner(
                    pos=pos, bar=bar,
                    atr_value = self.atrs[i-1],# on regarde le close. Anti look ahead
                    trailing_trigger_pct=cfg.trailing_trigger_pct,
                    trailing_mult=cfg.runner_trailing_mult
                )

            # Exit
            exit_event = self._exit_logic(i, bar, pos)

            if exit_event is not None:
                if cfg.fast:
                    self.trades.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"],
                        exit_reason=exit_event["reason"],
                    ))
                else:
                    self.trades.append(self.trade_analysis(
                        position=pos["side"],
                        exit_price=exit_event["exit_price"],
                        entry_price=pos["entry_price"],
                        entry_time=pos["entry_time"],
                        ts=exit_event["exit_time"],
                        exit_reason=exit_event["reason"],
                        mae=pos["mae"],
                        mfe=pos["mfe"],
                        mae_h=pos["mae"] if self.use_observation else None,
                        mfe_h=pos["mfe"] if self.use_observation else None
                    ))

                    if self.use_observation:
                        self.pending_observations.append({
                            "idx":   len(self.trades) - 1,
                            "ep":    pos["entry_price"],
                            "side":  pos["side"],
                            "end_i": i + self.observation_bars
                        })
            else:
                surviving.append(pos)

        self.positions = surviving

    # --------- CORE ---------
    def run(self) -> pd.DataFrame:
        cfg = self.cfg

        for i in range(1, len(self.df)):
            ts = self.index[i]
            bar = self.Bar(self.opens[i], self.highs[i], self.lows[i], self.closes[i], ts)
            sig_prev = self.signals[i - 1]

            # 1) reverse mode
            self._apply_reverse(sig_prev, bar.Open, ts)

            # 2) entry logic
            entry = self._entry_logic(i, ts, bar, sig_prev)
            self._max_entries_reset(ts)
            entry = self._max_entries_filter(i, entry)

            if entry is not None:
                if cfg.multi_entry or len(self.positions) == 0:
                    self.positions.append(entry)
                if cfg.MaxEntries4Periods:
                    self.recent_entries.append(i)

            # 3) exit logic loop
            self._update_positions(i, bar)

            # 4) post-trade observation
            if (not cfg.fast) and self.use_observation and self.pending_observations:
                self._update_observations(i, bar)

        return pd.DataFrame(self.trades)
