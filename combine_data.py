"""
Outputs the county_year_data.csv by combining datasets
"""

import os, io, requests, zipfile, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

STATE_ABBR_TO_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56',
}
STATE_FIPS_TO_ABBR = {v: k for k, v in STATE_ABBR_TO_FIPS.items()}

# Tax exemptions
# https://www.ncsl.org/technology-and-communication/data-center-tax-incentives
TAX_EXEMPT_YEAR: dict[str, int] = {
    'AZ': 2013, 'CO': 2009, 'DC': 2013, 'DE': 2012,
    'GA': 2009, 'ID': 2012, 'IL': 2012, 'IA': 2009,
    'KY': 2009, 'MI': 2012, 'MN': 2011, 'MS': 2012,
    'NE': 2012, 'NV': 2009, 'NY': 2010, 'NC': 2006,
    'OH': 2010, 'OR': 2007, 'SC': 2012, 'TX': 2001,
    'UT': 2014, 'VA': 2010, 'WA': 2009, 'WY': 2015,
}

# Climate normals
# Source: NOAA NCEI, https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals
STATE_AVG_TEMP_F: dict[str, float] = {
    'AL': 63.2, 'AK': 27.8, 'AZ': 65.4, 'AR': 60.4, 'CA': 59.4,
    'CO': 45.1, 'CT': 50.1, 'DE': 55.3, 'DC': 57.4, 'FL': 70.7,
    'GA': 63.5, 'HI': 70.0, 'ID': 44.4, 'IL': 51.8, 'IN': 51.7,
    'IA': 48.4, 'KS': 55.3, 'KY': 55.6, 'LA': 66.4, 'ME': 44.9,
    'MD': 54.2, 'MA': 48.8, 'MI': 44.8, 'MN': 41.2, 'MS': 64.5,
    'MO': 54.5, 'MT': 42.7, 'NE': 49.7, 'NV': 53.7, 'NH': 45.1,
    'NJ': 53.4, 'NM': 56.3, 'NY': 47.9, 'NC': 59.0, 'ND': 40.4,
    'OH': 51.5, 'OK': 60.4, 'OR': 48.3, 'PA': 50.8, 'RI': 50.6,
    'SC': 63.5, 'SD': 45.2, 'TN': 57.9, 'TX': 64.9, 'UT': 48.6,
    'VT': 43.9, 'VA': 55.9, 'WA': 48.3, 'WV': 51.8, 'WI': 43.1,
    'WY': 42.0,
}

# Water use
# https://pubs.er.usgs.gov/publication/cir1441
STATE_WATER_AVAIL: dict[str, float] = {
    'AL': 420, 'AK': 510, 'AZ': 380, 'AR': 560, 'CA': 490,
    'CO': 450, 'CT': 210, 'DE': 320, 'DC': 140, 'FL': 510,
    'GA': 390, 'HI': 350, 'ID': 680, 'IL': 370, 'IN': 330,
    'IA': 380, 'KS': 420, 'KY': 360, 'LA': 550, 'ME': 310,
    'MD': 290, 'MA': 240, 'MI': 420, 'MN': 380, 'MS': 430,
    'MO': 360, 'MT': 510, 'NE': 470, 'NV': 410, 'NH': 280,
    'NJ': 270, 'NM': 390, 'NY': 310, 'NC': 380, 'ND': 450,
    'OH': 340, 'OK': 390, 'OR': 490, 'PA': 320, 'RI': 200,
    'SC': 400, 'SD': 450, 'TN': 380, 'TX': 460, 'UT': 420,
    'VT': 280, 'VA': 350, 'WA': 490, 'WV': 350, 'WI': 360,
    'WY': 510,
}

# State median household income
# Source: US Census Bureau, Table S1901, 2019 American Community Survey
STATE_MEDIAN_INCOME_2019: dict[str, int] = {
    'AL': 51734, 'AK': 75463, 'AZ': 62055, 'AR': 48952, 'CA': 80440,
    'CO': 77127, 'CT': 78444, 'DE': 70176, 'DC': 92266, 'FL': 59227,
    'GA': 61980, 'HI': 83102, 'ID': 60999, 'IL': 69187, 'IN': 57617,
    'IA': 61836, 'KS': 62087, 'KY': 52295, 'LA': 51073, 'ME': 58924,
    'MD': 86738, 'MA': 85843, 'MI': 59584, 'MN': 74593, 'MS': 45792,
    'MO': 57290, 'MT': 56539, 'NE': 63229, 'NV': 62043, 'NH': 77933,
    'NJ': 85751, 'NM': 51945, 'NY': 72108, 'NC': 57341, 'ND': 64894,
    'OH': 58642, 'OK': 52919, 'OR': 67058, 'PA': 63463, 'RI': 70305,
    'SC': 56227, 'SD': 59533, 'TN': 54833, 'TX': 64034, 'UT': 74197,
    'VT': 63477, 'VA': 80963, 'WA': 78687, 'WV': 48037, 'WI': 64168,
    'WY': 65003,
}

# Inflation adjustment multiplier
INCOME_GROWTH: dict[int, float] = {
    2000: 0.720, 2001: 0.730, 2002: 0.728, 2003: 0.735, 2004: 0.745,
    2005: 0.760, 2006: 0.778, 2007: 0.800, 2008: 0.810, 2009: 0.795,
    2010: 0.812, 2011: 0.818, 2012: 0.828, 2013: 0.838, 2014: 0.858,
    2015: 0.885, 2016: 0.912, 2017: 0.938, 2018: 0.965, 2019: 1.000,
    2020: 0.990, 2021: 1.040, 2022: 1.055, 2023: 1.080,
}

# AI helped us label these
KNOWN_DC_MARKETS: dict[str, float] = {
    # ── Northern Virginia (single-largest US data center market) ──────────────
    '51107': 18.0,  # Loudoun County – Data Center Alley
    '51153': 12.0,  # Prince William County
    '51059': 9.0,   # Fairfax County
    '51013': 6.0,   # Arlington County
    # ── Dallas–Fort Worth ─────────────────────────────────────────────────────
    '48085': 8.0,   # Collin County
    '48113': 7.0,   # Dallas County
    '48121': 6.0,   # Denton County
    '48439': 4.0,   # Tarrant County
    # ── Phoenix ───────────────────────────────────────────────────────────────
    '04013': 10.0,  # Maricopa County
    # ── Chicago ───────────────────────────────────────────────────────────────
    '17031': 7.0,   # Cook County
    '17043': 5.0,   # DuPage County
    '17197': 4.0,   # Will County
    # ── Atlanta ───────────────────────────────────────────────────────────────
    '13121': 6.0,   # Fulton County
    '13135': 5.0,   # Gwinnett County
    '13097': 5.0,   # Douglas County
    '13151': 4.0,   # Henry County
    '13089': 4.0,   # DeKalb County
    # ── Silicon Valley / San Francisco Bay Area ───────────────────────────────
    '06085': 6.0,   # Santa Clara County
    '06013': 4.0,   # Contra Costa County
    '06081': 4.0,   # San Mateo County
    '06037': 4.0,   # Los Angeles County
    # ── New York / New Jersey ─────────────────────────────────────────────────
    '34017': 6.0,   # Hudson County, NJ
    '34003': 5.0,   # Bergen County, NJ
    '36119': 4.0,   # Westchester County, NY
    '36061': 4.0,   # New York County (Manhattan)
    '36059': 3.0,   # Nassau County, NY
    '36103': 3.0,   # Suffolk County, NY
    # ── Seattle / Pacific Northwest ───────────────────────────────────────────
    '53033': 6.0,   # King County, WA
    '53061': 4.0,   # Snohomish County, WA
    # ── Las Vegas / Nevada ────────────────────────────────────────────────────
    '32003': 7.0,   # Clark County (Las Vegas)
    '32031': 3.0,   # Washoe County (Reno)
    # ── Oregon  (Prineville / The Dalles – Google/Facebook/Meta) ─────────────
    '41051': 5.0,   # Multnomah County (Portland)
    '41031': 6.0,   # Jefferson County (Prineville)
    '41055': 6.0,   # Sherman County (The Dalles)
    # ── Iowa  (cloud giants – Google, Microsoft, Meta) ───────────────────────
    '19169': 6.0,   # Story County (Ames)
    '19049': 5.0,   # Dallas County
    '19163': 4.0,   # Scott County (Davenport)
    '19013': 3.0,   # Black Hawk County (Waterloo)
    # ── Idaho ─────────────────────────────────────────────────────────────────
    '16001': 5.0,   # Ada County (Boise)
    # ── Denver / Colorado ─────────────────────────────────────────────────────
    '08031': 5.0,   # Denver County
    '08005': 4.0,   # Arapahoe County
    '08001': 4.0,   # Adams County
    '08059': 3.0,   # Jefferson County
    # ── Nebraska ──────────────────────────────────────────────────────────────
    '31055': 5.0,   # Douglas County (Omaha)
    # ── Ohio ──────────────────────────────────────────────────────────────────
    '39049': 5.0,   # Franklin County (Columbus)
    '39035': 4.0,   # Cuyahoga County (Cleveland)
    '39113': 3.0,   # Montgomery County
    # ── Michigan ──────────────────────────────────────────────────────────────
    '26163': 4.0,   # Wayne County (Detroit)
    '26099': 3.0,   # Macomb County
    # ── Minnesota ─────────────────────────────────────────────────────────────
    '27053': 4.0,   # Hennepin County (Minneapolis)
    '27037': 3.0,   # Dakota County
    # ── North Carolina ────────────────────────────────────────────────────────
    '37119': 5.0,   # Mecklenburg County (Charlotte)
    '37183': 4.0,   # Wake County (Raleigh)
    # ── South Carolina ────────────────────────────────────────────────────────
    '45083': 5.0,   # Richland County
    '45091': 5.0,   # York County (Charlotte metro)
    # ── Florida ───────────────────────────────────────────────────────────────
    '12086': 5.0,   # Miami-Dade County
    '12011': 4.0,   # Broward County
    '12057': 4.0,   # Hillsborough County (Tampa)
    '12095': 4.0,   # Orange County (Orlando)
    # ── Texas (Austin / San Antonio / Houston) ────────────────────────────────
    '48453': 5.0,   # Travis County (Austin)
    '48029': 5.0,   # Bexar County (San Antonio)
    '48201': 5.0,   # Harris County (Houston)
    '48157': 4.0,   # Fort Bend County
    # ── Utah ──────────────────────────────────────────────────────────────────
    '49035': 5.0,   # Salt Lake County
    '49011': 4.0,   # Davis County
    # ── Missouri / Kansas City ───────────────────────────────────────────────
    '29189': 4.0,   # St. Louis County
    '29095': 3.0,   # Jackson County (Kansas City)
    # ── Mid-Atlantic ──────────────────────────────────────────────────────────
    '24031': 4.0,   # Montgomery County, MD
    '24033': 4.0,   # Prince George's County, MD
    '42101': 4.0,   # Philadelphia County, PA
    '42091': 3.0,   # Montgomery County, PA
    # ── Misc emerging markets ─────────────────────────────────────────────────
    '56021': 3.0,   # Laramie County, WY (Cheyenne)
    '38017': 3.0,   # Cass County, ND (Fargo)
    '08123': 3.0,   # Weld County, CO
    '36071': 3.0,   # Orange County, NY (NY metro)
    '06067': 3.0,   # Sacramento County, CA
    '06073': 3.0,   # San Diego County, CA
    '53077': 3.0,   # Yakima County, WA
    '53005': 3.0,   # Benton County, WA (Richland/Kennewick)
}

# Research annual market share data for each state
# Data from CBRE/JLL/Synergy
STATE_DC_MULT: dict[str, float] = {
    "51": 4.5,  # Virginia (Northern Virginia – #1 US market by capacity)
    "48": 3.2,  # Texas (DFW, Austin, Houston)
    "39": 2.8,  # Ohio (Columbus, Cleveland)
    "04": 2.7,  # Arizona (Phoenix)
    "13": 2.3,  # Georgia (Atlanta)
    "32": 2.1,  # Nevada (Las Vegas, Reno)
    "06": 1.9,  # California (Silicon Valley, LA)
    "17": 1.8,  # Illinois (Chicago)
    "53": 1.6,  # Washington (Seattle)
    "37": 1.5,  # North Carolina (Charlotte, Research Triangle)
    "08": 1.4,  # Colorado (Denver)
    "16": 1.4,  # Idaho (Boise)
    "19": 1.3,  # Iowa (cloud-giant campuses)
    "31": 1.3,  # Nebraska (Omaha)
    "49": 1.3,  # Utah (Salt Lake City)
    "41": 1.3,  # Oregon (Portland, Prineville, The Dalles)
    "34": 1.2,  # New Jersey (NYC metro edge)
    "36": 1.2,  # New York (Manhattan, Westchester)
    "45": 1.2,  # South Carolina (Charlotte metro edge)
}


def load_census_population() -> pd.DataFrame:
    """
    Sources: co-est00int-tot.csv, co-est2020-alldata.csv, Census web (2020-2023)
    """
    df00 = pd.read_csv(os.path.join(DATA_DIR, "co-est00int-tot.csv"), encoding="latin-1")
    df00 = df00[df00["SUMLEV"] == 50].copy()
    df00["FIPS"] = df00["STATE"].astype(str).str.zfill(2) + df00["COUNTY"].astype(str).str.zfill(3)
    pop_cols_00 = [f"POPESTIMATE{y}" for y in range(2000, 2010)]
    df00 = df00[["FIPS"] + pop_cols_00]

    df10 = pd.read_csv(os.path.join(DATA_DIR, "co-est2020-alldata.csv"), encoding="latin-1")
    df10 = df10[df10["SUMLEV"] == 50].copy()
    df10["FIPS"] = df10["STATE"].astype(str).str.zfill(2) + df10["COUNTY"].astype(str).str.zfill(3)
    pop_cols_10 = [f"POPESTIMATE{y}" for y in range(2010, 2021)]
    df10 = df10[["FIPS"] + pop_cols_10]

    url = ("https://www2.census.gov/programs-surveys/popest/datasets/"
           "2020-2023/counties/totals/co-est2023-alldata.csv")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df23 = pd.read_csv(io.StringIO(r.text), encoding="latin-1")
        df23 = df23[df23["SUMLEV"] == 50].copy()
        df23["FIPS"] = df23["STATE"].astype(str).str.zfill(2) + df23["COUNTY"].astype(str).str.zfill(3)
        pop_cols_23 = [f"POPESTIMATE{y}" for y in range(2021, 2024)]
        df23 = df23[["FIPS"] + pop_cols_23]
        print(f"Downloaded 2021-2023 Census data ({len(df23)} counties)")
    except Exception as e:
        print("Can't load data")
        df23 = None

    wide = df00.merge(df10, on="FIPS", how="inner")
    if df23 is not None:
        wide = wide.merge(df23, on="FIPS", how="left")
        for y in [2021, 2022, 2023]:
            col = f"POPESTIMATE{y}"
            if col not in wide.columns:
                wide[col] = np.nan
            rate = 1 + 0.004 * (y - 2020) 
            mask = wide[col].isna()
            wide.loc[mask, col] = wide.loc[mask, "POPESTIMATE2020"] * rate
    else:
        for y in [2021, 2022, 2023]:
            rate = 1 + 0.004 * (y - 2020)
            wide[f"POPESTIMATE{y}"] = (wide["POPESTIMATE2020"] * rate).round(0)

    all_pop_cols = [c for c in wide.columns if c.startswith("POPESTIMATE")]
    long = wide.melt(id_vars="FIPS", value_vars=all_pop_cols,
                     var_name="Year", value_name="Population")
    long["Year"] = long["Year"].str.replace("POPESTIMATE", "").astype(int)
    long["Population"] = long["Population"].round(0).astype(int)
    long = long[long["Year"].between(2000, 2023)].copy()
    print(f"    Population rows: {len(long):,}  |  counties: {long['FIPS'].nunique():,}")
    return long

def load_county_areas() -> pd.DataFrame:
    """Uses county_areas.csv."""
    path = os.path.join(DATA_DIR, "county_areas.csv")
    df = pd.read_csv(path, dtype={"GEOID": str})
    df = df[["GEOID", "ALAND_SQMI"]].rename(columns={"GEOID": "FIPS"})
    df["FIPS"] = df["FIPS"].str.zfill(5)
    df = df[df["ALAND_SQMI"] > 0].copy()
    print(f"    Area rows: {len(df):,}")
    return df

def load_eia_rates() -> pd.DataFrame:
    """
    From avgprice_annual.xlsx.
    """
    path = os.path.join(DATA_DIR, "avgprice_annual.xlsx")
    df = pd.read_excel(path, header=1)
    df = df[
        (df["Industry Sector Category"] == "Total Electric Industry") &
        (df["State"] != "US")
    ][["Year", "State", "Industrial"]].copy()
    df.columns = ["Year", "state_abbr", "elec_rate"]
    df = df.dropna(subset=["elec_rate"])
    df = df[df["Year"].between(2000, 2020)]

    ext_rows = []
    base = df[df["Year"] == 2020].set_index("state_abbr")["elec_rate"].to_dict()
    for state_abbr, rate_2020 in base.items():
        for yr, mult in [(2021, 1.04), (2022, 1.13), (2023, 1.03)]:
            prev = ext_rows[-1]["elec_rate"] if ext_rows and ext_rows[-1]["state_abbr"] == state_abbr else rate_2020
            ext_rows.append({"Year": yr, "state_abbr": state_abbr,
                             "elec_rate": round(prev * mult, 3)})
    df = pd.concat([df, pd.DataFrame(ext_rows)], ignore_index=True)

    df["state_fips"] = df["state_abbr"].map(STATE_ABBR_TO_FIPS)
    df = df.dropna(subset=["state_fips"])
    df = df[["state_fips", "Year", "elec_rate"]]
    print(f"    EIA rows: {len(df):,}  |  years: {df['Year'].min()}-{df['Year'].max()}")
    return df

def build_county_base(pop_df: pd.DataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    """
    Starting to build table
    """
    # One row per county (use year 2000 population to get FIPS universe)
    base = pop_df[pop_df["Year"] == 2000][["FIPS"]].copy()
    base["state_fips"] = base["FIPS"].str[:2]
    base["state_abbr"] = base["state_fips"].map(STATE_FIPS_TO_ABBR)

    base = base.merge(area_df, on="FIPS", how="left")

    state_avg_area = base.groupby("state_fips")["ALAND_SQMI"].transform("median")
    base["ALAND_SQMI"] = base["ALAND_SQMI"].fillna(state_avg_area).fillna(500)

    rng = np.random.default_rng(seed=7)
    base["water_avail"] = base["state_abbr"].map(STATE_WATER_AVAIL).fillna(350)
    base["water_avail"] *= rng.uniform(0.7, 1.3, size=len(base))  # ±30% county noise

    base["avg_temp"] = base["state_abbr"].map(STATE_AVG_TEMP_F).fillna(55.0)
    base["avg_temp"] += rng.normal(0, 2.5, size=len(base))

    base["dc_market_score"] = base["FIPS"].map(KNOWN_DC_MARKETS).fillna(1.0)

    base["state_dc_mult"] = base["state_fips"].map(STATE_DC_MULT).fillna(1.0)

    county_boost = np.maximum(1.0, base["dc_market_score"] / 2.0)
    base["dc_mult"] = base["state_dc_mult"] * county_boost

    print(f"  County base: {len(base):,} counties")
    return base


def load_cbp_data() -> pd.DataFrame:
    """
    Downloads Census Bureau County Business Patterns data
    """
    year_to_naics_var: dict[int, str] = {}
    for y in range(2003, 2008): year_to_naics_var[y] = "NAICS2002"
    for y in range(2008, 2012): year_to_naics_var[y] = "NAICS2007"
    for y in range(2012, 2017): year_to_naics_var[y] = "NAICS2012"
    for y in range(2017, 2023): year_to_naics_var[y] = "NAICS2017"

    all_rows = []
    for year, naics_var in sorted(year_to_naics_var.items()):
        url = (f"https://api.census.gov/data/{year}/cbp"
               f"?get=ESTAB&for=county:*&{naics_var}=518210")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"    Warning: CBP {year} returned HTTP {r.status_code}, skipping")
                continue
            data = r.json()
            header = data[0]
            estab_idx  = header.index("ESTAB")
            state_idx  = header.index("state")
            county_idx = header.index("county")
            for row in data[1:]:
                try:
                    estab = max(0, int(row[estab_idx]))
                except (ValueError, TypeError):
                    estab = 0
                fips = str(row[state_idx]).zfill(2) + str(row[county_idx]).zfill(3)
                all_rows.append({"FIPS": fips, "Year": year, "estab": estab})
        except Exception as e:
            print(f"    Warning: CBP {year} failed ({e}), skipping")

    df = pd.DataFrame(all_rows)
    print(f"    CBP rows: {len(df):,}  |  years: {df['Year'].min()}-{df['Year'].max()}")
    print(f"    Unique counties with any NAICS-518210 establishment: "
          f"{df[df['estab']>0]['FIPS'].nunique():,}")
    return df

def build_panel(pop_df: pd.DataFrame,
                base_df: pd.DataFrame,
                elec_df: pd.DataFrame,
                cbp_df:  pd.DataFrame,
                years=range(2000, 2023)) -> pd.DataFrame:
    """
    Returns the full county-year panel with all features and binary target
    """
    YEARS = list(years)
    counties = base_df["FIPS"].tolist()

    panel = pd.DataFrame(
        [(f, y) for f in counties for y in YEARS],
        columns=["FIPS", "Year"]
    )

    panel = panel.merge(pop_df[["FIPS", "Year", "Population"]], on=["FIPS", "Year"], how="left")
    panel["Population"] = panel["Population"].fillna(25000).astype(int)

    static_cols = ["FIPS", "state_fips", "state_abbr", "ALAND_SQMI",
                   "water_avail", "avg_temp", "dc_market_score"]
    panel = panel.merge(base_df[static_cols], on="FIPS", how="left")

    panel["pop_density"] = (panel["Population"] / panel["ALAND_SQMI"].clip(lower=1.0)).round(4)

    panel["_state_income"] = panel["state_abbr"].map(STATE_MEDIAN_INCOME_2019).fillna(60000)
    panel["_pop_adj"] = 1.0 + 0.15 * (panel["Population"].clip(lower=1).apply(np.log1p) / 14.0).clip(upper=1.0)
    panel["_yr_mult"] = panel["Year"].map(INCOME_GROWTH).fillna(1.0)
    panel["median_income"] = (panel["_state_income"] * panel["_pop_adj"] * panel["_yr_mult"]).round(2)
    panel.drop(columns=["_state_income", "_pop_adj", "_yr_mult"], inplace=True)

    panel = panel.merge(elec_df, on=["state_fips", "Year"], how="left")
    # Fill missing (some territories/years) with state median then national default
    state_med = panel.groupby("state_fips")["elec_rate"].transform(
        lambda x: x.fillna(x.median()))
    panel["elec_rate"] = panel["elec_rate"].fillna(state_med).fillna(8.0).round(4)

    def _tax_exempt(row):
        yr = TAX_EXEMPT_YEAR.get(row["state_abbr"])
        return int(yr is not None and row["Year"] >= yr)
    panel["tax_exempt"] = panel.apply(_tax_exempt, axis=1)

    cbp_prev = cbp_df.copy()
    cbp_prev["Year"] = cbp_prev["Year"] + 1  
    cbp_prev = cbp_prev.rename(columns={"estab": "estab_prev"})

    panel = panel.merge(cbp_df.rename(columns={"estab": "estab_curr"}),
                        on=["FIPS", "Year"], how="left")
    panel = panel.merge(cbp_prev, on=["FIPS", "Year"], how="left")
    panel["estab_curr"] = panel["estab_curr"].fillna(0).astype(int)
    panel["estab_prev"] = panel["estab_prev"].fillna(0).astype(int)
    panel["has_new_dc"] = (
        (panel["Year"] >= 2004) & (panel["Year"] <= 2022) &
        (panel["estab_curr"] > panel["estab_prev"])
    ).astype(int)

    panel.drop(columns=["estab_curr", "estab_prev"], inplace=True)

    panel = panel.sort_values(["FIPS", "Year"]).reset_index(drop=True)
    grp = panel.groupby("FIPS")["has_new_dc"]

    panel["dc_lag_1"] = grp.shift(1).fillna(0).astype(int)
    panel["dc_lag_3"] = (
        grp.shift(1).fillna(0) +
        grp.shift(2).fillna(0) +
        grp.shift(3).fillna(0)
    ).fillna(0).astype(int)
    panel["cumulative_dc"] = (grp.cumsum() - panel["has_new_dc"]).astype(int)

    out_cols = [
        "FIPS", "state_fips", "Year",
        "Population", "pop_density", "median_income",
        "elec_rate", "water_avail", "tax_exempt", "avg_temp",
        "dc_lag_1", "dc_lag_3", "cumulative_dc",
        "has_new_dc",
    ]
    df = panel[out_cols].copy()
    print(f"    Panel rows: {len(df):,}  |  DC events: {df['has_new_dc'].sum():,}")
    return df

def main():
    # Load and build data
    pop_df  = load_census_population()
    area_df = load_county_areas()
    elec_df = load_eia_rates()
    cbp_df  = load_cbp_data()

    base_df = build_county_base(pop_df, area_df)
    panel   = build_panel(pop_df, base_df, elec_df, cbp_df)

    # Summary statistics
    train = panel[panel["Year"] < 2020]
    test  = panel[panel["Year"] >= 2020]
    print("\nDataset summary:")
    print(f"  Total rows:           {len(panel):,}")
    print(f"  Unique counties:      {panel['FIPS'].nunique():,}")
    print(f"  Positive rate (all):  {panel['has_new_dc'].mean()*100:.2f}%")
    print(f"  DC events total:      {panel['has_new_dc'].sum():,}")
    print(f"  Counties w/ any DC:   {panel[panel['has_new_dc']==1]['FIPS'].nunique():,}")
    print(f"  Train pos rate:       {train['has_new_dc'].mean()*100:.2f}%")
    print(f"  Test pos rate:        {test['has_new_dc'].mean()*100:.2f}%")

    top_dc = (panel.groupby("FIPS")["has_new_dc"]
              .sum().sort_values(ascending=False).head(15))
    print("\n  Top 15 counties by DC event count:")
    for fips, cnt in top_dc.items():
        mkt = KNOWN_DC_MARKETS.get(fips, 1.0)
        sfips = fips[:2]
        abbr  = STATE_FIPS_TO_ABBR.get(sfips, "??")
        print(f"    {fips} ({abbr})  events={cnt:3d}  market_score={mkt:.1f}")

    # Save
    out_path = os.path.join(DATA_DIR, "county_year_dataset.csv")
    panel.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return panel


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
