from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import inspect

import numpy as np
import pandas as pd

from .feature_compiler import CompiledFeatures, to_compiled_features


# =============================================================================
# Helpers
# =============================================================================

def _as_name(fn: Callable, name: str | None = None) -> str:
    if name is not None:
        return str(name)
    if hasattr(fn, "__name__"):
        return fn.__name__
    raise ValueError("Unable to infer a name for the function.")


def _to_numpy_1d(x: Any) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = x.to_numpy(copy=False)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("DataFrame output must have exactly one column if converted to 1D array.")
        arr = x.iloc[:, 0].to_numpy(copy=False)
    elif hasattr(x, "values") and not isinstance(x, np.ndarray):
        arr = np.asarray(x.values)
    else:
        arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D output, got ndim={arr.ndim}.")
    return arr


def _extract_index_from_output(value: Any) -> pd.Index | None:
    if isinstance(value, pd.Series):
        return value.index
    if isinstance(value, pd.DataFrame):
        return value.index
    if hasattr(value, "index") and value.index is not None:
        return pd.DatetimeIndex(value.index)
    return None


def _value_to_name_part(x: Any) -> str:
    if isinstance(x, (np.bool_, bool)):
        return "True" if bool(x) else "False"
    if x is None:
        return "None"
    if isinstance(x, float):
        # nom plus stable
        s = f"{x:.12g}"
        return s.replace(".", "p")
    if isinstance(x, (list, tuple, np.ndarray)):
        return "-".join(_value_to_name_part(v) for v in list(x))
    return str(x).replace(" ", "")

# =============================================================================
# Output containers
# =============================================================================

@dataclass
class FeatureOutput:
    name: str
    values: np.ndarray
    index: pd.Index

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        if self.values.ndim != 1:
            raise ValueError(f"FeatureOutput '{self.name}' must be 1D.")
        self.index = pd.DatetimeIndex(self.index)
        if len(self.values) != len(self.index):
            raise ValueError(
                f"Length mismatch for output '{self.name}': "
                f"{len(self.values)} values vs {len(self.index)} index rows."
            )

    def array(self) -> np.ndarray:
        return self.values

    def series(self) -> pd.Series:
        return pd.Series(self.values, index=self.index, name=self.name)


@dataclass
class FeatureResult:
    feature_name: str
    run_name: str
    outputs: dict[str, FeatureOutput]
    params: dict[str, Any] = field(default_factory=dict)
    source_name: str | None = None
    asset: str | None = None
    tf: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def index(self) -> pd.Index:
        if not self.outputs:
            raise ValueError("Empty FeatureResult has no index.")
        first_key = next(iter(self.outputs))
        return self.outputs[first_key].index

    def output(self, output_name: str = "main") -> FeatureOutput:
        if output_name not in self.outputs:
            raise KeyError(
                f"Unknown output '{output_name}' for run '{self.run_name}'. "
                f"Available: {list(self.outputs.keys())}"
            )
        return self.outputs[output_name]

    def array(self, output_name: str = "main") -> np.ndarray:
        return self.output(output_name).array()

    def series(self, output_name: str = "main") -> pd.Series:
        return self.output(output_name).series()

    def dataframe(
        self,
        output_name: str | None = None,
        align_index: pd.Index | None = None,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        if output_name is not None:
            out = self.output(output_name).series()
            col_name = self.run_name if output_name == "main" else f"{self.run_name}_{output_name}"
            df = out.rename(col_name).to_frame()
        else:
            use_prefix = self.run_name if prefix is None else prefix
            data: dict[str, pd.Series] = {}
            single_main = len(self.outputs) == 1 and "main" in self.outputs
            for out_name, out in self.outputs.items():
                if single_main and out_name == "main":
                    col_name = use_prefix
                else:
                    col_name = f"{use_prefix}_{out_name}"
                data[col_name] = out.series()
            df = pd.DataFrame(data, index=self.index)

        if align_index is not None:
            df = df.reindex(pd.DatetimeIndex(align_index))

        return df

    def compiled(
        self,
        output_name: str | None = None,
        align_index: pd.Index | None = None,
        prefix: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CompiledFeatures:
        merged_meta = {
            "feature_name": self.feature_name,
            "run_name": self.run_name,
            "asset": self.asset,
            "tf": self.tf,
        }
        if meta:
            merged_meta.update(meta)

        return to_compiled_features(
            self.dataframe(output_name=output_name, align_index=align_index, prefix=prefix),
            meta=merged_meta,
        )


# =============================================================================
# Runtime object passed into features
# =============================================================================

class FeatureRuntime:
    def __init__(
        self,
        owner: "Feature",
        source: Any = None,
        source_name: str | None = None,
    ) -> None:
        self._owner = owner
        self._source = source
        self._source_name = source_name

    @property
    def data(self) -> Any:
        return self._owner.data

    @property
    def source(self) -> Any:
        return self._source

    @property
    def source_name(self) -> str | None:
        return self._source_name

    @property
    def source_index(self) -> pd.Index | None:
        if self._source is not None and hasattr(self._source, "index"):
            return pd.DatetimeIndex(self._source.index)
        return None

    def process(self, name: str, **kwargs: Any) -> Any:
        return self._owner.run_process(name, **kwargs)

    def array(self, name: str) -> np.ndarray:
        return self._owner.array(name)

    def series(self, name: str) -> pd.Series:
        return self._owner.series(name)

    def result(self, name: str) -> FeatureResult | FeatureOutput:
        return self._owner.result(name)

    def index(self, name: str | None = None) -> pd.Index:
        if name is None:
            if self.source_index is None:
                raise ValueError("No current source index available.")
            return self.source_index
        return self._owner.index(name)

    def ext(self, name: str, **kwargs: Any) -> Any:
        if not hasattr(self._owner.data, "ext"):
            raise AttributeError("The attached data object has no ext(...) method.")
        return self._owner.data.ext(name, **kwargs)

    def assets(self) -> list[str]:
        if self._source is None:
            return []
        return list(self._source.attrs.get("assets", []))

    def asset_col(self, field: str, asset: str) -> Any:
        if self._source is None or not isinstance(self._source, pd.DataFrame):
            raise ValueError("Current source is not a DataFrame.")

        col = f"{field}_{asset}"
        if col not in self._source.columns:
            raise KeyError(f"Column '{col}' not found in current source.")

        return self._source[col]

    def col(self, name: str) -> Any:
        if self._source is None:
            raise ValueError("No current source available. Pass on=... when calling the feature.")
        if isinstance(self._source, pd.DataFrame):
            if name not in self._source.columns:
                raise KeyError(f"Column '{name}' not found in current source.")
            return self._source[name]
        if isinstance(self._source, pd.Series):
            if name != self._source.name:
                raise KeyError(
                    f"Current source is a Series named '{self._source.name}', not column '{name}'."
                )
            return self._source
        raise TypeError("Current source does not support column access via f.col(...).")


# =============================================================================
# Main Feature object
# =============================================================================

class Feature:
    def __init__(self, data: Any = None) -> None:
        self.data = data

        self._processes: dict[str, Callable[..., Any]] = {}
        self._features: dict[str, Callable[..., Any]] = {}

        # last saved run per feature logical name
        self._last_runs: dict[str, FeatureResult] = {}

        # named saved/exported runs
        self._named_runs: dict[str, FeatureResult] = {}

    # -------------------------------------------------------------------------
    # register
    # -------------------------------------------------------------------------

    def add_process(self, fn: Callable[..., Any], name: str | None = None) -> None:
        name = _as_name(fn, name)
        self._processes[name] = fn

    def add(self, fn: Callable[..., Any], name: str | None = None) -> None:
        name = _as_name(fn, name)
        self._features[name] = fn

    # -------------------------------------------------------------------------
    # list / has / delete
    # -------------------------------------------------------------------------

    def list_processes(self) -> list[str]:
        return sorted(self._processes.keys())

    def list_features(self) -> list[str]:
        return sorted(self._features.keys())

    def list_results(self) -> list[str]:
        names = list(self._last_runs.keys()) + list(self._named_runs.keys())
        return sorted(set(names))

    def has_process(self, name: str) -> bool:
        return name in self._processes

    def has_feature(self, name: str) -> bool:
        return name in self._features

    def delete_process(self, name: str) -> None:
        self._processes.pop(name, None)

    def delete_feature(self, name: str) -> None:
        self._features.pop(name, None)

    def delete_result(self, name: str) -> None:
        if ":" in name:
            base, _ = self._split_result_ref(name)
            self._last_runs.pop(base, None)
            self._named_runs.pop(base, None)
        else:
            self._last_runs.pop(name, None)
            self._named_runs.pop(name, None)

    def clear_results(self) -> None:
        self._last_runs.clear()
        self._named_runs.clear()

    # -------------------------------------------------------------------------
    # process execution
    # -------------------------------------------------------------------------

    def run_process(self, name: str, **kwargs: Any) -> Any:
        if name not in self._processes:
            raise KeyError(f"Unknown process '{name}'. Available: {self.list_processes()}")
        return self._processes[name](**kwargs)

    # -------------------------------------------------------------------------
    # feature execution
    # -------------------------------------------------------------------------

    def __call__(
        self,
        name: str,
        on: Any = None,
        on_ohlcv_matrix: str | None = None,
        matrix_asset: str | None = None,
        matrix_assets: list[str] | None = None,
        save: bool = False,
        export: bool = False,
        save_as: str | None = None,
        **kwargs: Any,
    ) -> FeatureResult:
        if name not in self._features:
            raise KeyError(f"Unknown feature '{name}'. Available: {self.list_features()}")

        on = self._resolve_input_source(
            on=on,
            on_ohlcv_matrix=on_ohlcv_matrix,
            matrix_asset=matrix_asset,
            matrix_assets=matrix_assets,
        )

        feature_fn = self._features[name]
        runtime = FeatureRuntime(owner=self, source=on, source_name=getattr(on, "name", None))

        raw = feature_fn(runtime, **kwargs)

        asset, tf = self._resolve_asset_tf(source=on, raw=raw)
        auto_name = self._build_auto_run_name(
            feature_name=name,
            feature_fn=feature_fn,
            asset=asset,
            tf=tf,
            params=kwargs,
        )

        run_name = save_as or auto_name

        result = self._build_result(
            feature_name=name,
            run_name=run_name,
            raw=raw,
            params=kwargs,
            source=on,
            asset=asset,
            tf=tf,
        )

        if save or export:
            self._last_runs[name] = result
            self._named_runs[run_name] = result

        return result

    # -------------------------------------------------------------------------
    # resolve input
    # -------------------------------------------------------------------------

    def _resolve_input_source(
        self,
        on: Any = None,
        on_ohlcv_matrix: str | None = None,
        matrix_asset: str | None = None,
        matrix_assets: list[str] | None = None,
    ) -> Any:
        n_specified = sum(
            x is not None
            for x in [on, on_ohlcv_matrix]
        )
        if n_specified > 1:
            raise ValueError("Choose either on=... or on_ohlcv_matrix=..., not multiple sources.")

        if on is not None:
            return on

        if on_ohlcv_matrix is None:
            if self.data is not None and hasattr(self.data, "has_main_df") and self.data.has_main_df():
                return self.data.get_main_df()
            return None

        if self.data is None:
            raise ValueError("Feature has no attached data object.")

        if not hasattr(self.data, "ohlcv_matrix"):
            raise AttributeError("Attached data object has no ohlcv_matrix(...) method.")

        if matrix_asset is not None and matrix_assets is not None:
            raise ValueError("Choose either matrix_asset or matrix_assets, not both.")

        if matrix_asset is not None:
            if not hasattr(self.data, "get_ohlcv_matrix_asset"):
                raise AttributeError("Attached data object has no get_ohlcv_matrix_asset(...) method.")
            return self.data.get_ohlcv_matrix_asset(on_ohlcv_matrix, matrix_asset)

        if matrix_assets is not None:
            if not hasattr(self.data, "get_ohlcv_matrix_assets"):
                raise AttributeError("Attached data object has no get_ohlcv_matrix_assets(...) method.")
            return self.data.get_ohlcv_matrix_assets(on_ohlcv_matrix, matrix_assets)

        return self.data.ohlcv_matrix(on_ohlcv_matrix)

    def to_compiled(
        self,
        result: str | FeatureResult,
        output_name: str | None = None,
        align_index: pd.Index | None = None,
        prefix: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CompiledFeatures:
        run = result if isinstance(result, FeatureResult) else self.result(result)
        if isinstance(run, FeatureOutput):
            single = FeatureResult(
                feature_name=str(run.name),
                run_name=str(run.name),
                outputs={"main": run},
            )
            return single.compiled(
                output_name=output_name,
                align_index=align_index,
                prefix=prefix,
                meta=meta,
            )
        return run.compiled(
            output_name=output_name,
            align_index=align_index,
            prefix=prefix,
            meta=meta,
        )
    
    # -------------------------------------------------------------------------
    # result retrieval
    # -------------------------------------------------------------------------

    def result(self, name: str) -> FeatureResult | FeatureOutput:
        base, output_name = self._split_result_ref(name)
        run = self._get_saved_run(base)

        if output_name is None:
            return run
        return run.output(output_name)

    def array(self, name: str) -> np.ndarray:
        base, output_name = self._split_result_ref(name)
        run = self._get_saved_run(base)
        return run.array(output_name or "main")

    def series(self, name: str) -> pd.Series:
        base, output_name = self._split_result_ref(name)
        run = self._get_saved_run(base)
        return run.series(output_name or "main")

    def index(self, name: str) -> pd.Index:
        base, output_name = self._split_result_ref(name)
        run = self._get_saved_run(base)
        if output_name is None:
            return run.index
        return run.output(output_name).index

    # -------------------------------------------------------------------------
    # internal result building
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        feature_name: str,
        run_name: str,
        raw: Any,
        params: dict[str, Any],
        source: Any = None,
        asset: str | None = None,
        tf: str | None = None,
    ) -> FeatureResult:
        normalized = self._normalize_feature_return(raw)
        index = self._resolve_index(normalized, source=source)

        outputs: dict[str, FeatureOutput] = {}
        for out_name, out_value in normalized.items():
            out_arr = _to_numpy_1d(out_value)
            outputs[out_name] = FeatureOutput(
                name=out_name,
                values=out_arr,
                index=index,
            )

        return FeatureResult(
            feature_name=feature_name,
            run_name=run_name,
            outputs=outputs,
            params=dict(params),
            source_name=getattr(source, "name", None),
            asset=asset,
            tf=tf,
        )

    def _normalize_feature_return(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            if not raw:
                raise ValueError("A feature cannot return an empty dict.")
            return raw

        return {"main": raw}

    def _resolve_index(self, outputs: dict[str, Any], source: Any = None) -> pd.Index:
        # 1. source index if provided
        if source is not None and hasattr(source, "index"):
            return pd.DatetimeIndex(source.index)

        # 2. indexed outputs
        found: pd.Index | None = None
        for v in outputs.values():
            idx = _extract_index_from_output(v)
            if idx is not None:
                if found is None:
                    found = pd.DatetimeIndex(idx)
                elif not found.equals(pd.DatetimeIndex(idx)):
                    raise ValueError("Returned outputs do not share the same index.")
        if found is not None:
            return found

        raise ValueError(
            "Unable to resolve feature index. "
            "Pass on=... with an indexed source, or return an indexed object."
        )

    # -------------------------------------------------------------------------
    # internal naming
    # -------------------------------------------------------------------------

    def _build_auto_run_name(
        self,
        feature_name: str,
        feature_fn: Callable[..., Any],
        asset: str | None,
        tf: str | None,
        params: dict[str, Any],
    ) -> str:
        parts = [feature_name]

        if asset is not None:
            parts.append(str(asset))
        if tf is not None:
            parts.append(str(tf))

        ordered_values = self._ordered_param_values(feature_fn, params)
        parts.extend(_value_to_name_part(v) for v in ordered_values)

        return "_".join(parts)

    def _ordered_param_values(
        self,
        feature_fn: Callable[..., Any],
        params: dict[str, Any],
    ) -> list[Any]:
        sig = inspect.signature(feature_fn)
        values: list[Any] = []

        for i, (pname, p) in enumerate(sig.parameters.items()):
            # premier argument = runtime/f
            if i == 0:
                continue

            if pname in params:
                values.append(params[pname])
            elif p.default is not inspect._empty:
                values.append(p.default)

        # kwargs non présents dans la signature, à la fin dans l'ordre d'appel
        declared = {name for i, name in enumerate(sig.parameters.keys()) if i > 0}
        for k, v in params.items():
            if k not in declared:
                values.append(v)

        return values

    def _resolve_asset_tf(
        self,
        source: Any = None,
        raw: Any = None,
    ) -> tuple[str | None, str | None]:
        asset = None
        tf = None

        if source is not None:
            if hasattr(source, "attrs") and isinstance(source.attrs, dict):
                asset = source.attrs.get("asset", asset)
                tf = source.attrs.get("tf", tf)

            if asset is None and hasattr(source, "asset"):
                asset = getattr(source, "asset")
            if tf is None and hasattr(source, "tf"):
                tf = getattr(source, "tf")

        if isinstance(raw, dict):
            values = list(raw.values())
        else:
            values = [raw]

        for v in values:
            if asset is None and hasattr(v, "asset"):
                asset = getattr(v, "asset")
            if tf is None and hasattr(v, "tf"):
                tf = getattr(v, "tf")

        return asset, tf

    # -------------------------------------------------------------------------
    # internal saved-run resolution
    # -------------------------------------------------------------------------

    def _split_result_ref(self, name: str) -> tuple[str, str | None]:
        if ":" in name:
            base, output = name.split(":", 1)
            return base, output
        return name, None

    def _get_saved_run(self, base_name: str) -> FeatureResult:
        if base_name in self._named_runs:
            return self._named_runs[base_name]
        if base_name in self._last_runs:
            return self._last_runs[base_name]
        raise KeyError(
            f"No saved result named '{base_name}'. "
            f"Available saved runs: {self.list_results()}"
        )
