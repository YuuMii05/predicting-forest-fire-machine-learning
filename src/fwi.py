"""Canadian Forest Fire Weather Index (FWI) system.

Standard equations of Van Wagner (1987) / Van Wagner & Pickett (1985).
Inputs are daily noon weather: temperature (°C), relative humidity (%),
wind speed (km/h) and 24-hour rainfall (mm). The three moisture codes
(FFMC, DMC, DC) carry over from one day to the next, so the system must be
run day after day, in order.
"""
from __future__ import annotations

import math

import pandas as pd

# Day-length factors for the northern hemisphere (Van Wagner 1987)
DMC_DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DC_DAY_LENGTH = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

# Standard start-up values
FFMC0, DMC0, DC0 = 85.0, 6.0, 15.0


def ffmc_step(ffmc0, T, H, W, rain):
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        if mo > 150.0:
            mo = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf))
        else:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        mo = min(mo, 250.0)
    ed = 0.942 * H ** 0.679 + 11.0 * math.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
    if mo > ed:
        ko = 0.424 * (1.0 - (H / 100.0) ** 1.7) + 0.0694 * math.sqrt(W) * (1.0 - (H / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * T)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = 0.618 * H ** 0.753 + 10.0 * math.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7) + 0.0694 * math.sqrt(W) * (1.0 - ((100.0 - H) / 100.0) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * T)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return min(max(59.5 * (250.0 - m) / (147.2 + m), 0.0), 101.0)


def dmc_step(dmc0, T, H, rain, month):
    T = max(T, -1.1)
    rk = 1.894 * (T + 1.1) * (100.0 - H) * DMC_DAY_LENGTH[month - 1] * 1e-6
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc0 / 43.43)
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)
        elif dmc0 <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc0)
        else:
            b = 6.2 * math.log(dmc0) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = max(244.72 - 43.43 * math.log(mr - 20.0), 0.0)
    else:
        pr = dmc0
    return max(pr + 100.0 * rk, 0.0)


def dc_step(dc0, T, rain, month):
    T = max(T, -2.8)
    pe = max((0.36 * (T + 2.8) + DC_DAY_LENGTH[month - 1]) / 2.0, 0.0)
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-dc0 / 400.0)
        qr = qo + 3.937 * rd
        dr = max(400.0 * math.log(800.0 / qr), 0.0)
    else:
        dr = dc0
    return max(dr + pe, 0.0)


def isi_calc(ffmc, W):
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 19.115 * math.exp(-0.1386 * mo) * (1.0 + mo ** 5.31 / 4.93e7)
    return ff * math.exp(0.05039 * W)


def bui_calc(dmc, dc):
    if dmc == 0 and dc == 0:
        return 0.0
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(bui, 0.0)


def fwi_calc(isi, bui):
    fd = 0.626 * bui ** 0.809 + 2.0 if bui <= 80.0 else 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
    b = 0.1 * isi * fd
    return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647) if b > 1.0 else b


def compute_fwi(df: pd.DataFrame) -> pd.DataFrame:
    """Run the FWI system day by day on ONE location.

    df must be sorted by date and have columns: date, Temperature, RH, Ws, Rain.
    The codes restart from the standard values every 1 March (the usual
    spring start-up), so each fire season is computed independently.
    """
    out = {k: [] for k in ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]}
    ffmc, dmc, dc = FFMC0, DMC0, DC0
    for r in df.itertuples():
        if r.date.month == 3 and r.date.day == 1:
            ffmc, dmc, dc = FFMC0, DMC0, DC0
        T, H, W, P = float(r.Temperature), min(max(float(r.RH), 0.0), 100.0), float(r.Ws), float(r.Rain)
        ffmc = ffmc_step(ffmc, T, H, W, P)
        dmc = dmc_step(dmc, T, H, P, r.date.month)
        dc = dc_step(dc, T, P, r.date.month)
        isi = isi_calc(ffmc, W)
        bui = bui_calc(dmc, dc)
        for k, v in zip(out, [ffmc, dmc, dc, isi, bui, fwi_calc(isi, bui)]):
            out[k].append(round(v, 2))
    return df.assign(**out)
