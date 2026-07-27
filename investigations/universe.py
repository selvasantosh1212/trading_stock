"""Stock universe + sector map for this investigation (35 liquid NSE names
spanning 8 sectors, including two 'ugly' cases - YESBANK (2018-2020 collapse)
and ONGC (long flat/declining stretch) - specifically so the universe isn't
just cherry-picked bull-market winners).
"""

SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "NTPC.NS": "Energy/Utility",
    "POWERGRID.NS": "Energy/Utility",
    "COALINDIA.NS": "Energy/Utility",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",
    "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking/Financials",
    "ICICIBANK.NS": "Banking/Financials",
    "KOTAKBANK.NS": "Banking/Financials",
    "AXISBANK.NS": "Banking/Financials",
    "SBIN.NS": "Banking/Financials",
    "INDUSINDBK.NS": "Banking/Financials",
    "YESBANK.NS": "Banking/Financials",
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma",
    "MARUTI.NS": "Auto",
    "TVSMOTOR.NS": "Auto",
    "M&M.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "HINDUNILVR.NS": "FMCG/Consumer",
    "ITC.NS": "FMCG/Consumer",
    "NESTLEIND.NS": "FMCG/Consumer",
    "BRITANNIA.NS": "FMCG/Consumer",
    "ASIANPAINT.NS": "FMCG/Consumer",
    "TATASTEEL.NS": "Metals/Infra",
    "ULTRACEMCO.NS": "Metals/Infra",
    "LT.NS": "Metals/Infra",
    "BHARTIARTL.NS": "Telecom",
}

SYMBOLS = list(SECTOR_MAP.keys())
