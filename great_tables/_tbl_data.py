from __future__ import annotations

import re
import warnings
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import narwhals.stable.v1 as nw
from narwhals.typing import IntoDataFrame, IntoDataFrameT, IntoSeries
from typing_extensions import TypeAlias

from ._databackend import AbstractBackend

# Define databackend types ----
# These are resolved lazily (e.g. on isinstance checks) when run dynamically,
# or imported directly during type checking.

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    # the class behind selectors
    from polars.selectors import _selector_proxy_

    PdDataFrame = pd.DataFrame
    PlDataFrame = pl.DataFrame
    PyArrowTable = pa.Table

    PlSelectExpr = _selector_proxy_
    PlExpr = pl.Expr

    PdSeries = pd.Series
    PlSeries = pl.Series
    PyArrowArray = pa.Array
    PyArrowChunkedArray = pa.ChunkedArray

    PdNA = pd.NA
    PlNull = pl.Null

    NpNan = np.nan

    DataFrameLike = Union[PdDataFrame, PlDataFrame, PyArrowTable]
    SeriesLike = Union[PdSeries, PlSeries, PyArrowArray, PyArrowChunkedArray]
    TblData = DataFrameLike

else:
    from abc import ABC

    # we just need this as a static type hint, but singledispatch tries to resolve
    # any hints at runtime. So we need some value for it.
    from typing import Any as _selector_proxy_

    class PdDataFrame(AbstractBackend):
        _backends = [("pandas", "DataFrame")]

    class PlDataFrame(AbstractBackend):
        _backends = [("polars", "DataFrame")]

    class PyArrowTable(AbstractBackend):
        _backends = [("pyarrow", "Table")]

    class PlSelectExpr(AbstractBackend):
        _backends = [("polars.selectors", "_selector_proxy_")]

    class PlExpr(AbstractBackend):
        _backends = [("polars", "Expr")]

    class PdSeries(AbstractBackend):
        _backends = [("pandas", "Series")]

    class PlSeries(AbstractBackend):
        _backends = [("polars", "Series")]

    class PyArrowArray(AbstractBackend):
        _backends = [("pyarrow", "Array")]

    class PyArrowChunkedArray(AbstractBackend):
        _backends = [("pyarrow", "ChunkedArray")]

    class PdNA(AbstractBackend):
        _backends = [("pandas", "NA")]

    class PlNull(AbstractBackend):
        _backends = [("polars", "Null")]

    class NpNan(AbstractBackend):
        _backends = [("numpy", "nan")]

    # TODO: these types are imported throughout gt, so we need to either put
    # those imports under TYPE_CHECKING, or continue to make available dynamically here.
    class DataFrameLike(ABC):
        """Represent some DataFrame"""

    class SeriesLike(ABC):
        """Represent some Series"""

    DataFrameLike.register(PdDataFrame)
    DataFrameLike.register(PlDataFrame)
    DataFrameLike.register(PyArrowTable)
    SeriesLike.register(PdSeries)
    SeriesLike.register(PlSeries)
    SeriesLike.register(PyArrowArray)
    SeriesLike.register(PyArrowChunkedArray)

    TblData = DataFrameLike


# utils ----


def _raise_not_implemented(data: Any):
    raise NotImplementedError(f"Unsupported data type: {type(data)}")


def _raise_pandas_required(msg: Any):
    raise ImportError(msg)


def _re_version(raw_version: str) -> tuple[int, int, int]:
    """Return a semver-like version string as a 3-tuple of integers.

    Note two important caveats: (1) separators like dev are dropped (e.g. "3.2.1dev3" -> (3, 2, 1)),
    and (2) it simply integer converts parts (e.g. "3.2.0001" -> (3,2,1)).
    """

    # Note two major caveats
    regex = r"(?P<major>\d+)\.(?P<minor>\d+).(?P<patch>\d+)"
    return tuple(map(int, re.match(regex, raw_version).groups()))


class Agnostic:
    """This class dispatches a generic in a DataFrame agnostic way.

    It is available for generics like is_na.
    """


# generic functions ----


# copy_frame ----


def copy_frame(df: IntoDataFrameT) -> IntoDataFrameT:
    """Return a copy of the input DataFrame"""
    return nw.from_native(df, eager_only=True).clone().to_native()


# _get_cell ----


def _get_cell(data: IntoDataFrame, row: int, column: str) -> Any:
    """Get the content from a single cell in the input data table"""
    return nw.from_native(data, eager_only=True).item(row=row, column=column)


# _get_column_dtype ----


@singledispatch
def _get_column_dtype(data: DataFrameLike, column: str) -> Any:
    """Get the data type for a single column in the input data table"""
    return data[column].dtype


@_get_column_dtype.register(PyArrowTable)
def _(data: PyArrowTable, column: str) -> Any:
    return data.column(column).type


# group_splits ----
def group_splits(data: IntoDataFrame, group_key: str) -> dict[str | int, list[int]]:
    frame = nw.from_native(data, eager_only=True)
    token = nw.generate_temporary_column_name(8, columns=frame.columns)
    grouped = frame.with_row_index(name=token).select([group_key, token]).group_by(group_key)
    return {k[0]: grp.get_column(token).to_list() for k, grp in grouped}


# eval_select ----

SelectExpr: TypeAlias = Union[
    str,
    list[str],
    int,
    list[int],
    list["str | int"],
    PlSelectExpr,
    list[PlSelectExpr],
    Callable[[str], bool],
    None,
]
_NamePos: TypeAlias = list[tuple[str, int]]


@singledispatch
def eval_select(data: DataFrameLike, expr: SelectExpr, strict: bool = True) -> _NamePos:
    """Return a list of column names selected by expr."""

    raise NotImplementedError(f"Unsupported type: {type(expr)}")


@eval_select.register
def _(
    data: PdDataFrame,
    expr: Union[list[Union[str, int]], Callable[[str], bool]],
    strict: bool = True,
) -> _NamePos:
    if isinstance(expr, (str, int)):
        expr = [expr]

    if isinstance(expr, list):
        return _eval_select_from_list(list(data.columns), expr)
    elif callable(expr):
        # TODO: currently, we call on each string, but we could be calling on
        # pd.DataFrame.columns instead (which would let us use pandas .str methods)
        col_pos = {k: ii for ii, k in enumerate(list(data.columns))}
        return [(col, col_pos[col]) for col in data.columns if expr(col)]

    raise NotImplementedError(f"Unsupported selection expr: {expr}")


@eval_select.register
def _(data: PlDataFrame, expr: Union[list[str], _selector_proxy_], strict: bool = True) -> _NamePos:
    # TODO: how to annotate type of a polars selector?
    # Seems to be polars.selectors._selector_proxy_.
    import polars as pl
    import polars.selectors as cs
    from polars import Expr

    from ._utils import OrderedSet

    pl_version = _re_version(pl.__version__)
    expand_opts = {"strict": False} if pl_version >= (0, 20, 30) else {}

    # just in case _selector_proxy_ gets renamed or something
    # it inherits from Expr, so we can just use that in a pinch
    cls_selector = getattr(cs, "_selector_proxy_", Expr)

    if isinstance(expr, (str, int)):
        expr = [expr]

    if isinstance(expr, list):
        # convert str and int entries to selectors ----
        all_selectors = [
            cs.by_name(x) if isinstance(x, str) else cs.by_index(x) if isinstance(x, int) else x
            for x in expr
        ]

        # validate all entries ----
        _validate_selector_list(all_selectors, **expand_opts)

        # this should be equivalent to reducing selectors using an "or" operator,
        # which isn't possible when there are selectors mixed with expressions
        # like pl.col("some_col")
        final_columns = OrderedSet(
            col_name
            for sel in all_selectors
            for col_name in cs.expand_selector(data, sel, **expand_opts)
        ).as_list()
    else:
        if not isinstance(expr, (cls_selector, Expr)):
            raise TypeError(f"Unsupported selection expr type: {type(expr)}")

        final_columns = cs.expand_selector(data, expr, **expand_opts)

    col_pos = {k: ii for ii, k in enumerate(data.columns)}

    # I don't think there's a way to get the columns w/o running the selection
    return [(col, col_pos[col]) for col in final_columns]


@eval_select.register
def _(
    data: PyArrowTable, expr: Union[list[str], _selector_proxy_], strict: bool = True
) -> _NamePos:
    if isinstance(expr, (str, int)):
        expr = [expr]

    if isinstance(expr, list):
        return _eval_select_from_list(data.column_names, expr)
    elif callable(expr):
        col_pos = {k: ii for ii, k in enumerate(data.column_names)}
        return [(col, col_pos[col]) for col in data.column_names if expr(col)]

    raise NotImplementedError(f"Unsupported selection expr: {expr}")


def _validate_selector_list(selectors: list, strict=True):
    from polars import Expr
    from polars.selectors import is_selector

    for ii, sel in enumerate(selectors):
        if isinstance(sel, Expr):
            if strict:
                raise TypeError(
                    f"Expected a list of selectors, but entry {ii} is a polars Expr, which is only "
                    "supported for polars versions >= 0.20.30."
                )
        elif not is_selector(sel):
            raise TypeError(f"Expected a list of selectors, but entry {ii} is type: {type(sel)}.")


def _eval_select_from_list(
    columns: list[str], expr: list[Union[str, int]]
) -> list[tuple[str, int]]:
    col_pos = {k: ii for ii, k in enumerate(columns)}

    # TODO: should prohibit duplicate names in expr?
    res: list[tuple[str, int]] = []
    for col in expr:
        if isinstance(col, str):
            if col in col_pos:
                res.append((col, col_pos[col]))
        elif isinstance(col, int):
            _pos = col if col >= 0 else len(columns) + col
            res.append((columns[col], _pos))
        else:
            raise TypeError(
                f"eval_select received a list with object of type {type(col)}."
                " Only int and str are supported."
            )
    return res


# create_empty ----


@singledispatch
def create_empty_frame(df: DataFrameLike) -> DataFrameLike:
    """Return a DataFrame with the same shape, but all nan string columns"""
    raise NotImplementedError(f"Unsupported type: {type(df)}")


@create_empty_frame.register
def _(df: PdDataFrame):
    import pandas as pd

    return pd.DataFrame(pd.NA, index=df.index, columns=df.columns, dtype="string")


@create_empty_frame.register
def _(df: PlDataFrame):
    import polars as pl

    return df.clear().cast(pl.Utf8).clear(len(df))


@create_empty_frame.register
def _(df: PyArrowTable):
    import pyarrow as pa

    return pa.table({col: pa.nulls(df.num_rows, type=pa.string()) for col in df.column_names})


# cast_frame_to_string ----


@singledispatch
def cast_frame_to_string(df: DataFrameLike) -> DataFrameLike:
    """Return a copy of the input DataFrame with all columns cast to string"""
    raise NotImplementedError(f"Unsupported type: {type(df)}")


@cast_frame_to_string.register
def _(df: PdDataFrame):
    return df.astype("string")


@cast_frame_to_string.register
def _(df: PlDataFrame):
    import polars as pl
    import polars.selectors as cs

    list_cols = [
        name for name, dtype in zip(df.columns, df.dtypes) if issubclass(dtype.base_type(), pl.List)
    ]

    return df.with_columns(
        cs.by_name(list_cols).map_elements(lambda x: str(x.to_list()), return_dtype=pl.String),
        cs.all().exclude(list_cols).cast(pl.Utf8),
    )


@cast_frame_to_string.register
def _(df: PyArrowTable):
    import pyarrow as pa

    return pa.table(
        {col: df.column(col).cast(pa.string()).combine_chunks() for col in df.column_names}
    )


# replace_null_frame ----


def replace_null_frame(df: IntoDataFrameT, replacement: IntoDataFrameT) -> IntoDataFrameT:
    """Return a copy of the input DataFrame with all null values replaced with replacement"""
    df_nw = nw.from_native(df, eager_only=True)
    replacement_nw = nw.from_native(replacement, eager_only=True)
    exprs = (df_nw[name].fill_null(replacement_nw[name]).alias(name) for name in df_nw.columns)
    return df_nw.select(*exprs).to_native()


# mutate ----


@singledispatch
def eval_transform(df: DataFrameLike, expr: Any) -> list[Any]:
    raise NotImplementedError(f"Unsupported type: {type(df)}")


@eval_transform.register
def _(df: PdDataFrame, expr: Callable[[PdDataFrame], PdSeries]) -> list[Any]:
    res = expr(df)

    if not isinstance(res, PdSeries):
        raise ValueError(f"Result must be a pandas Series. Received {type(res)}")
    elif not len(res) == len(df):
        raise ValueError(
            f"Result must be same length as input data. Observed different lengths."
            f"\n\nInput data: {len(df)}.\nResult: {len(res)}."
        )

    return res.to_list()


@eval_transform.register
def _(df: PlDataFrame, expr: PlExpr) -> list[Any]:
    df_res = df.select(expr)

    if len(df_res.columns) > 1:
        raise ValueError(f"Result must be a single column. Received {len(df_res.columns)} columns.")
    else:
        res = df_res[df_res.columns[0]]

    if not isinstance(res, PlSeries):
        raise ValueError(f"Result must be a polars Series. Received {type(res)}")
    elif not len(res) == len(df):
        raise ValueError(
            f"Result must be same length as input data. Observed different lengths."
            f"\n\nInput data: {len(df)}.\nResult: {len(res)}."
        )

    return res.to_list()


@eval_transform.register
def _(df: PyArrowTable, expr: Callable[[PyArrowTable], PyArrowArray]) -> list[Any]:
    res = expr(df)

    if not isinstance(res, PyArrowArray):
        raise ValueError(f"Result must be an Arrow Array. Received {type(res)}")
    elif not len(res) == len(df):
        raise ValueError(
            f"Result must be same length as input data. Observed different lengths."
            f"\n\nInput data: {df.num_rows}.\nResult: {len(res)}."
        )

    return res.to_pylist()


@singledispatch
def is_na(df: DataFrameLike, x: Any) -> bool:
    raise NotImplementedError(f"Unsupported type: {type(df)}")


@is_na.register
def _(df: PdDataFrame, x: Any) -> bool:
    import pandas as pd

    return pd.isna(x)


@is_na.register(Agnostic)
@is_na.register
def _(df: PlDataFrame, x: Any) -> bool:
    from math import isnan

    import polars as pl

    return x is None or isinstance(x, pl.Null) or (isinstance(x, float) and isnan(x))


@is_na.register
def _(df: PyArrowTable, x: Any) -> bool:
    import pyarrow as pa

    arr = pa.array([x])
    return arr.is_null(nan_is_null=True)[0].as_py()


@singledispatch
def validate_frame(df: DataFrameLike) -> DataFrameLike:
    """Raises an error if a DataFrame is not supported by Great Tables.

    Note that this is only relevant for pandas, which allows duplicate names
    on DataFrames, and multi-index columns (and probably other things).
    """
    raise NotImplementedError(f"Unsupported type: {type(df)}")


@validate_frame.register
def _(df: PdDataFrame) -> PdDataFrame:
    import pandas as pd

    # case 1: multi-index columns ----
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            "pandas DataFrames with MultiIndex columns are not supported."
            " Please use .columns.droplevel() to remove extra column levels,"
            " or combine the levels into a single name per column."
        )

    # case 2: duplicate column names ----
    dupes = df.columns[df.columns.duplicated()]
    if len(dupes):
        raise ValueError(
            f"Column names must be unique. Detected duplicate columns:\n\n {list(dupes)}"
        )

    non_str_cols = [(ii, el) for ii, el in enumerate(df.columns) if not isinstance(el, str)]

    if non_str_cols:
        _col_msg = "\n".join(f"  * Position {ii}: {col}" for ii, col in non_str_cols[:3])
        warnings.warn(
            "pandas DataFrame contains non-string column names. Coercing to strings. "
            "Here are the first few non-string columns:\n\n"
            f"{_col_msg}",
            category=UserWarning,
        )
        new_df = df.copy()
        new_df.columns = [str(el) for el in df.columns]
        return new_df

    return df


@validate_frame.register
def _(df: PlDataFrame) -> PlDataFrame:
    return df


@validate_frame.register
def _(df: PyArrowTable) -> PyArrowTable:
    warnings.warn("PyArrow Table support is currently experimental.")

    if len(set(df.column_names)) != len(df.column_names):
        raise ValueError("Column names must be unique.")

    return df


# to_frame ----


def to_frame(ser: "list[Any] | IntoSeries", name: Optional[str] = None) -> IntoDataFrame:
    # TODO: remove pandas. currently, we support converting a list to a pd.DataFrame
    # in order to support backwards compatibility in the vals.fmt_* functions.

    if nw.dependencies.is_into_series(ser):
        ser_nw = nw.from_native(ser, series_only=True)
        if name:
            ser_nw = ser_nw.rename(name)
        return ser_nw.to_frame().to_native()

    try:
        import pandas as pd
    except ImportError:
        _raise_pandas_required(
            "Passing a plain list of values currently requires the library pandas. "
            "You can avoid this error by passing a polars Series."
        )

    if not isinstance(ser, list):
        raise NotImplementedError(f"Unsupported type: {type(ser)}")

    if not name:
        raise ValueError("name must be specified, when converting a list to a DataFrame.")

    return pd.DataFrame({name: ser})
