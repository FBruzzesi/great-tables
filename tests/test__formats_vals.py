import pandas as pd
import polars as pl
import pytest
from great_tables._formats_vals import _make_one_col_table
import narwhals.stable.v1 as nw


@pytest.mark.parametrize("src", [1, [1], (1,), pd.Series([1]), pl.Series([1])])
def test_roundtrip(src):
    gt = _make_one_col_table(src)

    assert nw.from_native(gt._tbl_data["x"], series_only=True).to_list() == [1]
