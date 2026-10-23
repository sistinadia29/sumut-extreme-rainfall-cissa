"""
Uji sederhana untuk sumut_rain.indices dengan data buatan yang jawabannya diketahui.
Jalankan dari root proyek:  python tests/test_indices.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain.indices import annual_indices, max_consecutive, wet_day_percentile  # noqa: E402


def test_max_consecutive():
    f = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert max_consecutive(f[:, None])[0] == 3
    assert max_consecutive(np.zeros((5, 2), bool)).tolist() == [0, 0]
    assert max_consecutive(np.ones((5, 2), bool)).tolist() == [5, 5]


def test_annual_indices_two_years():
    # Tahun A: 365 hari, Tahun B: 366 hari; satu sel darat + satu sel laut (NaN)
    years = np.array([2001] * 365 + [2004] * 366)
    pr = np.zeros((len(years), 2), dtype=np.float32)
    pr[:, 1] = np.nan                       # sel laut

    # Tahun 2001: hujan 10 mm di hari 10-14 (5 hari), 30 mm di hari 100, 0.5 mm di hari 200
    pr[10:15, 0] = 10
    pr[100, 0] = 30
    pr[200, 0] = 0.5
    # Tahun 2004: hujan 2 mm tiap hari kecuali deret kering 40 hari (hari 50-89)
    b = 365
    pr[b:b + 366, 0] = 2
    pr[b + 50:b + 90, 0] = 0
    # Jendela lintas tahun: 25 mm di hari terakhir 2001 + 25 mm di hari pertama 2004
    pr[364, 0] = 25
    pr[b, 0] = 25

    thr = np.array([20.0, np.nan], dtype=np.float32)
    uy, ind = annual_indices(pr, years, thr95=thr)

    assert uy.tolist() == [2001, 2004]
    # 2001
    assert ind["prcptot"][0, 0] == 50 + 30 + 25       # 0.5 mm bukan hari basah
    assert ind["rx1day"][0, 0] == 30
    assert ind["rx5day"][0, 0] == 50
    assert ind["r20mm"][0, 0] == 2                    # 30 mm dan 25 mm
    assert ind["r95p"][0, 0] == 30 + 25               # > 20 mm
    # 2004
    assert ind["rx1day"][1, 0] == 25
    assert ind["cdd"][1, 0] == 40
    assert ind["rx5day"][1, 0] == 25 + 25 + 2 * 3      # jendela 5 hari terbesar memuat 25+25+2+2+2
    # Sel laut tetap NaN
    assert all(np.isnan(ind[k][:, 1]).all() for k in ind)


def test_wet_day_percentile():
    pr = np.concatenate([np.zeros(50), np.arange(1, 101)]).astype(np.float32)[:, None]
    thr = wet_day_percentile(pr, q=95)
    assert abs(thr[0] - np.percentile(np.arange(1, 101), 95)) < 1e-4


if __name__ == "__main__":
    test_max_consecutive()
    test_annual_indices_two_years()
    test_wet_day_percentile()
    print("Semua uji lolos.")
