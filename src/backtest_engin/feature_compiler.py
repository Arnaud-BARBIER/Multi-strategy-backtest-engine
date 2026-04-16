import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass(slots=True)
class FeatureSpec:
    name: str
    fn: Callable
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("FeatureSpec.name cannot be empty")
        if not callable(self.fn):
            raise TypeError("FeatureSpec.fn must be callable")


@dataclass(slots=True)
class CompiledFeatures:
    matrix: np.ndarray
    col_map: dict[str, int]
    names: tuple[str, ...]
    index: pd.Index | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.ndim != 2:
            raise ValueError("CompiledFeatures.matrix must be a 2D array")

        self.names = tuple(self.names)
        self.col_map = dict(self.col_map)
        self.meta = dict(self.meta)

        if len(self.names) != self.matrix.shape[1]:
            raise ValueError(
                f"CompiledFeatures.names length mismatch: got {len(self.names)}, "
                f"expected {self.matrix.shape[1]}"
            )

        if set(self.names) != set(self.col_map.keys()):
            raise ValueError("CompiledFeatures.col_map keys must match feature names exactly")

        if self.index is not None:
            self.index = pd.DatetimeIndex(self.index)
            if len(self.index) != self.matrix.shape[0]:
                raise ValueError(
                    f"CompiledFeatures.index length mismatch: got {len(self.index)}, "
                    f"expected {self.matrix.shape[0]}"
                )

    def col(self, name: str) -> int:
        if name not in self.col_map:
            raise KeyError(f"Unknown feature name: {name}")
        return self.col_map[name]

    def cols(self, *names: str) -> tuple[int, ...]:
        return tuple(self.col(name) for name in names)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.index, columns=self.names)


def _default_feature_names(n_cols: int) -> tuple[str, ...]:
    return tuple(f"f{j}" for j in range(n_cols))


def _coerce_feature_frame(
    features: Any,
    names: list[str] | tuple[str, ...] | None = None,
    index: pd.Index | None = None,
    align_index: pd.Index | None = None,
) -> pd.DataFrame:
    has_explicit_index = False

    if isinstance(features, CompiledFeatures):
        df = features.dataframe()
        has_explicit_index = features.index is not None
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
        has_explicit_index = not isinstance(df.index, pd.RangeIndex)
    elif isinstance(features, pd.Series):
        series_name = str(features.name) if features.name is not None else "f0"
        df = features.to_frame(name=series_name)
        has_explicit_index = not isinstance(df.index, pd.RangeIndex)
    elif isinstance(features, dict):
        df = pd.DataFrame(features)
        has_explicit_index = not isinstance(df.index, pd.RangeIndex)
    else:
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("features must be convertible to a 2D feature matrix")
        if names is None:
            names = _default_feature_names(arr.shape[1])
        elif len(names) != arr.shape[1]:
            raise ValueError(
                f"Provided feature names length mismatch: got {len(names)}, expected {arr.shape[1]}"
            )
        df = pd.DataFrame(arr, columns=list(names), index=index)
        has_explicit_index = index is not None

    if names is not None and list(df.columns) != list(names):
        if len(names) != df.shape[1]:
            raise ValueError(
                f"Provided feature names length mismatch: got {len(names)}, expected {df.shape[1]}"
            )
        df.columns = list(names)

    if align_index is not None:
        align_index = pd.DatetimeIndex(align_index)
        if has_explicit_index:
            df = df.reindex(align_index)
        else:
            if len(df) != len(align_index):
                raise ValueError(
                    f"Feature row count mismatch: got {len(df)}, expected {len(align_index)}"
                )
            df.index = align_index
    elif index is not None and not df.index.equals(pd.DatetimeIndex(index)):
        if len(df) != len(index):
            raise ValueError(
                f"Feature index length mismatch: got {len(df)}, expected {len(index)}"
            )
        df.index = pd.DatetimeIndex(index)

    return df


def to_compiled_features(
    features: Any,
    names: list[str] | tuple[str, ...] | None = None,
    index: pd.Index | None = None,
    align_index: pd.Index | None = None,
    meta: dict[str, Any] | None = None,
) -> CompiledFeatures:
    if isinstance(features, CompiledFeatures) and names is None and index is None and align_index is None:
        if meta:
            merged_meta = dict(features.meta)
            merged_meta.update(meta)
            return CompiledFeatures(
                matrix=features.matrix,
                col_map=features.col_map,
                names=features.names,
                index=features.index,
                meta=merged_meta,
            )
        return features

    df = _coerce_feature_frame(
        features=features,
        names=names,
        index=index,
        align_index=align_index,
    )

    clean_names = tuple(str(col) for col in df.columns)
    if len(set(clean_names)) != len(clean_names):
        raise ValueError("Feature column names must be unique")

    return CompiledFeatures(
        matrix=df.to_numpy(dtype=np.float64, copy=False),
        col_map={name: j for j, name in enumerate(clean_names)},
        names=clean_names,
        index=pd.DatetimeIndex(df.index) if isinstance(df.index, pd.DatetimeIndex) else None,
        meta={} if meta is None else dict(meta),
    )


def compile_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray | None = None,
    feature_specs: list[FeatureSpec] | None = None,
) -> CompiledFeatures:
    n = len(closes)

    if len(opens) != n or len(highs) != n or len(lows) != n:
        raise ValueError("OHLC arrays must all have the same length")

    if volumes is not None and len(volumes) != n:
        raise ValueError("volumes must have the same length as closes")

    if not feature_specs:
        return CompiledFeatures(
            matrix=np.zeros((n, 0), dtype=np.float64),
            col_map={},
            names=(),
        )

    names_seen = set()
    cols = []
    names = []

    for spec in feature_specs:
        if spec.name in names_seen:
            raise ValueError(f"Duplicate feature name: {spec.name}")
        names_seen.add(spec.name)

        arr = spec.fn(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            **spec.params,
        )

        arr = np.asarray(arr, dtype=np.float64)

        if arr.ndim != 1:
            raise ValueError(f"Feature '{spec.name}' must return a 1D array, got ndim={arr.ndim}")
        if arr.shape[0] != n:
            raise ValueError(
                f"Feature '{spec.name}' length mismatch: got {arr.shape[0]}, expected {n}"
            )

        cols.append(arr)
        names.append(spec.name)

    matrix = np.column_stack(cols)

    return CompiledFeatures(
        matrix=matrix,
        col_map={name: j for j, name in enumerate(names)},
        names=tuple(names),
    )

# Exemple d'utilisation #
#feature_specs = [
#    FeatureSpec(name="bb_upper", fn=bb_upper_feature, params={"length": 20, "n_std": 2.0}),
#    FeatureSpec(name="ema_fast", fn=ema_feature, params={"length": 20}),
#]
