#!/usr/bin/env python3
"""
Irrigation Planner (enhanced)
- Added features:
    * Crop stage–specific Kc curves (uses --days_since_planting to compute stage Kc; if not provided, assumes mid-season)
    * Multiple water-source blending: will combine nearby sources to meet event volume
    * NASA POWER Rs fetch to replace crude net radiation approximation (falls back to heuristic if unavailable)
    * 30-day irrigation calendar marking recommended irrigation dates and flagging forecast rain days
- Usage: same as before, with new optional args:
    --days_since_planting INT
    --season_length INT (days, default 120)
    --blend_radius_km FLOAT (radius to search for water sources, default 5 km)

Note: NASA POWER parameter used: ALLSKY_SFC_SW_DWN (daily incoming shortwave, MJ/m2/day).
If unavailable or API fails, code uses Hargreaves heuristic as fallback.
"""
import os
import argparse
import requests
from geopy.distance import geodesic
from math import sqrt, sin, cos, tan, asin, acos, pi, exp, log
from datetime import datetime, timedelta, timezone,date
import random
import calendar
import time
from typing import List, Dict, Any

# -------------------------
# Defaults & Config
# -------------------------
OPENWEATHER_API_KEY = "bd5e378503939ddaee76f12ad7a97608"
NASA_POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/climatology/point"
NASA_POWER_DAILY_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
ISRIC_SOILGRIDS_ENDPOINT = "https://rest.isric.org/soilgrids/v2.0/properties/query"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OSM_TYPES = [
    # flowing/conveyance
    ('way',     'waterway', 'river'),
    ('way',     'waterway', 'canal'),
    ('way',     'waterway', 'stream'),
    ('way',     'waterway', 'drain'),
    ('relation','waterway', 'river'),
    ('relation','waterway', 'canal'),
    # stored water
    ('way',     'natural',  'water'),
    ('relation','natural',  'water'),
    ('way',     'landuse',  'reservoir'),
    ('relation','landuse',  'reservoir'),
    # points
    ('node',    'man_made', 'water_well'),
    ('node',    'amenity',  'drinking_water'),
    ('node',    'natural',  'spring'),
    ('node',    'man_made', 'water_tap'),
]

def build_overpass_query(lat, lon, radius_m):
    parts = []
    for elem, k, v in OSM_TYPES:
        if elem == 'node':
            parts.append(f'node(around:{radius_m},{lat},{lon})["{k}"="{v}"];')
        elif elem == 'way':
            parts.append(f'way(around:{radius_m},{lat},{lon})["{k}"="{v}"];')
        else:
            parts.append(f'relation(around:{radius_m},{lat},{lon})["{k}"="{v}"];')
    body = "\n  ".join(parts)
    # out center returns a centroid for ways/relations so we can compute distance
    return f"""
[out:json][timeout:25];
(
  {body}
);
out tags center;
"""

def km_distance(a, b):
    # simple haversine fallback to avoid adding deps
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(x)))

def fetch_water_sources_osm(lat, lon, radius_km=5.0):
    q = build_overpass_query(lat, lon, int(radius_km*1000))
    r = requests.post(OVERPASS_URL, data={'data': q}, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    for el in data.get('elements', []):
        tags = el.get('tags', {}) or {}
        # center for ways/relations; lat/lon for nodes
        if 'lat' in el and 'lon' in el:
            el_lat, el_lon = el['lat'], el['lon']
        else:
            c = el.get('center')
            if not c:
                continue
            el_lat, el_lon = c['lat'], c['lon']
        # infer type
        wway = tags.get('waterway')
        natural = tags.get('natural')
        landuse = tags.get('landuse')
        man_made = tags.get('man_made')
        amenity = tags.get('amenity')
        water = tags.get('water')

        if man_made == 'water_well':
            src_type = 'Well'
        elif amenity == 'drinking_water':
            src_type = 'DrinkingWater'
        elif man_made == 'water_tap':
            src_type = 'WaterTap'
        elif natural == 'spring':
            src_type = 'Spring'
        elif wway in ('canal',):
            src_type = 'Canal'
        elif wway in ('river', 'stream', 'drain'):
            src_type = 'River'
        elif landuse == 'reservoir' or (natural == 'water' and water in ('reservoir',)):
            src_type = 'Reservoir'
        elif natural == 'water' and water in ('lake', 'pond'):
            src_type = 'LakePond'
        else:
            src_type = 'WaterBody'

        d_km = km_distance((lat, lon), (el_lat, el_lon))
        results.append({
            'name': tags.get('name'),
            'type': src_type,
            'tags': tags,
            'lat': el_lat,
            'lon': el_lon,
            'distance_km': round(d_km, 3),
            'osm_id': el.get('id'),
            'osm_element': el.get('type'),
        })
    # sort by distance
    return sorted(results, key=lambda x: x['distance_km'])

def infer_capacity_m3d(src):
    # Heuristic; refine with WRIS cross-reference if available
    t = src['type']
    if t == 'Canal': return 5000.0
    if t == 'River': return 4000.0
    if t == 'Reservoir': return 2000.0
    if t == 'LakePond': return 1000.0
    if t == 'Well': return 50.0
    if t in ('Spring', 'DrinkingWater', 'WaterTap'): return 10.0
    return 200.0

def find_closest_water_source_live(plot_coords, radius_km=10.0, max_distance_km=50.0):
    lat, lon = plot_coords
    sources = fetch_water_sources_osm(lat, lon, radius_km=radius_km)
    if not sources:
        return {"source_name": None, "distance_km": float('inf'), "capacity_m3_per_day": 0, "type": None}
    s0 = sources[0]
    if s0['distance_km'] > max_distance_km:
        return {"source_name": None, "distance_km": s0['distance_km'], "capacity_m3_per_day": 0, "type": None}
    cap = infer_capacity_m3d(s0)
    return {
        "source_name": s0.get('name') or s0['type'],
        "distance_km": s0['distance_km'],
        "capacity_m3_per_day": cap,
        "type": s0['type'],
        "lat": s0['lat'],
        "lon": s0['lon'],
        "tags": s0['tags'],
    }

def blend_water_sources_live(plot_coords, required_m3, radius_km=5.0):
    lat, lon = plot_coords
    sources = fetch_water_sources_osm(lat, lon, radius_km=radius_km)
    # sort by distance
    allocation = {}
    remaining = float(required_m3)
    for s in sources:
        cap = infer_capacity_m3d(s)
        take = min(cap, remaining)
        allocation[s.get('name') or s['type']] = {
            "type": s['type'],
            "distance_km": s['distance_km'],
            "allocated_m3": round(take, 2),
            "capacity": cap,
            "lat": s['lat'],
            "lon": s['lon'],
        }
        remaining -= take
        if remaining <= 1e-6:
            break
    return allocation, round(max(0.0, remaining), 2)


CROP_KC = {
    "wheat": 1.15,
    "rice": 1.2,
    "maize": 1.15,
    "cotton": 0.9,
    "sugarcane": 1.25,
    "vegetables": 0.95,
    "potato": 0.9,
    "soybean": 1.05,
    "default": 1.0
}

# stage lengths as fractions of season (initial, development, mid, late)
DEFAULT_STAGE_FRAC = (0.2, 0.3, 0.4, 0.1)

SOIL_AWC = {
    "Sandy": 50,
    "Sandy Loam": 120,
    "Loam": 150,
    "Silt Loam": 175,
    "Silty Clay": 200,
    "Clay": 220,
}

IRRIGATION_EFFICIENCY = {
    "drip": 0.9,
    "sprinkler": 0.75,
    "surface": 0.5
}

# Readily available water fraction (FAO-style) by crop - defaults to 0.5 if not present
RAW_FRACTION = {
    "wheat": 0.5,
    "rice": 0.5,
    "maize": 0.5,
    "default": 0.5
}

MAX_IRRIGATION_INTERVAL_DAYS = 10  # cap to avoid unrealistic long intervals

# -------------------------
# Utilities
# -------------------------
def km_distance(a, b):
    return geodesic(a, b).kilometers

def deg2rad(deg):
    return deg * pi / 180.0

def day_of_year(dt):
    return dt.timetuple().tm_yday

# -------------------------
# 1) Weather + Forecast (OpenWeather OneCall)
# -------------------------
def fetch_onecall(lat, lon, api_key):
    """
    Returns dict with keys "current", "hourly", "daily" - simulated if api_key is falsy or request fails.
    """
    if not api_key:
        t = random.uniform(20, 32)
        tmin = t - random.uniform(2, 5)
        tmax = t + random.uniform(2, 5)
        hourly = []
        now = datetime.now(timezone.utc)
        for h in range(48):
            temp = t + random.uniform(-2, 2)
            pop = 0.0 if random.random() > 0.2 else random.uniform(0.1, 0.8)
            rain_1h = round(random.uniform(0, 6) if pop > 0.3 else 0.0, 2)
            hourly.append({
                "dt": int((now + timedelta(hours=h)).timestamp()),
                "temp": temp,
                "wind_speed": random.uniform(0.5, 3.0),
                "humidity": random.randint(40, 80),
                "clouds": int(pop*100),
                "rain_1h": rain_1h,
                "pop": pop
            })
        return {
            "current": {"temp": round(t,1), "temp_min": round(tmin,1), "temp_max": round(tmax,1),
                        "humidity": int(random.uniform(40,80)), "wind_speed": random.uniform(0.5,3.0), "clouds": 10},
            "hourly": hourly,
            "daily": []
        }

    url3 = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric&exclude=minutely,alerts"
    url2 = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric&exclude=minutely,alerts"
    for url in (url3, url2):
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
            hourly = []
            for h in data.get("hourly", [])[:48]:
                rain_1h = 0.0
                if "rain" in h:
                    rain_1h = h.get("rain", {}).get("1h", 0.0)
                hourly.append({
                    "dt": h.get("dt"),
                    "temp": h.get("temp"),
                    "wind_speed": h.get("wind_speed"),
                    "humidity": h.get("humidity"),
                    "clouds": h.get("clouds", 0),
                    "rain_1h": rain_1h,
                    "pop": h.get("pop", 0.0)
                })
            current = data.get("current", {})
            curr_norm = {
                "temp": current.get("temp"),
                "temp_min": None,
                "temp_max": None,
                "humidity": current.get("humidity"),
                "wind_speed": current.get("wind_speed"),
                "clouds": current.get("clouds", 0)
            }
            if "daily" in data and len(data["daily"])>0:
                d0 = data["daily"][0]
                # daily temp structure may vary
                temps = d0.get("temp", {})
                curr_norm["temp_min"] = temps.get("min") if isinstance(temps, dict) else None
                curr_norm["temp_max"] = temps.get("max") if isinstance(temps, dict) else None
            return {"current": curr_norm, "hourly": hourly, "daily": data.get("daily", [])}
        except Exception:
            continue
    # fallback simulated
    return fetch_onecall(lat, lon, "")

def sum_rain_next_hours(onecall_resp, hours=48):
    total = 0.0
    for h in onecall_resp.get("hourly", [])[:hours]:
        total += h.get("rain_1h", 0.0)
    return round(total, 2)

# -------------------------
# 2) Long-term rainfall using NASA POWER Climatology (3-month smoothed)
# -------------------------
def get_long_term_monthly_rainfall(lat, lon, start_year=1981, end_year=2010):
    """
    Returns dict month (1..12) -> monthly mean precipitation (mm/month).
    """
    params = {
        "start": start_year,
        "end": end_year,
        "latitude": lat,
        "longitude": lon,
        "format": "JSON",
        "community": "AG",
        "parameters": "PRECTOT"
    }
    try:
        r = requests.get(NASA_POWER_ENDPOINT, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        p = data.get("properties", {}).get("parameter", {}).get("PRECTOT", {})
        monthly = {}
        order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        for i, mname in enumerate(order, start=1):
            val = p.get(mname)
            if val is None:
                val = 0.0
            monthly[i] = float(val)
        return monthly
    except Exception:
        # fallback realistic-ish seasonal baseline
        base = [40, 45, 60, 90, 120, 200, 250, 220, 150, 60, 30, 25]
        return {i+1: base[i] for i in range(12)}

def compute_effective_rainfall_offset_3month(monthly_mm, month, effectiveness_fraction=0.75):
    """
    Smooth monthly climatology over previous, current, next months and return effective mm/day.
    """
    # ensure month in 1..12
    months = [((month - 2) % 12) + 1, month, (month % 12) + 1]  # prev, current, next
    vals = [monthly_mm.get(m, 0.0) for m in months]
    avg_mm = sum(vals) / 3.0
    days = calendar.monthrange(datetime.now(timezone.utc).year, month)[1]
    mm_per_day = avg_mm / days if days > 0 else 0.0
    effective_mm_day = mm_per_day * effectiveness_fraction
    return round(effective_mm_day, 3)

# -------------------------
# 3) ISRIC SoilGrids lookup
# -------------------------
def get_soil_from_isric(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "property": "sand,clay,silt",
        "depth": "0-5cm"
    }
    try:
        r = requests.get(ISRIC_SOILGRIDS_ENDPOINT, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        properties = data.get("properties", {})
        def extract_mean(prop):
            p = properties.get(prop, {})
            vals = p.get("values", [])
            if len(vals) > 0 and "mean" in vals[0]:
                return float(vals[0]["mean"])
            return None
        clay = extract_mean("clay") or 30.0
        sand = extract_mean("sand") or 40.0
        silt = extract_mean("silt") or 30.0
        total = sand + silt + clay
        if total <= 0:
            sand, silt, clay = 40.0, 30.0, 30.0
            total = 100.0
        sand = sand * 100.0 / total
        silt = silt * 100.0 / total
        clay = clay * 100.0 / total
        if sand > 70:
            text = "Sandy"
        elif clay > 35:
            text = "Clay"
        elif silt > 40:
            text = "Silt Loam"
        elif sand > 40 and silt > 30:
            text = "Sandy Loam"
        else:
            text = "Loam"
        return {"soil_type": text, "sand_pct": round(sand,1), "silt_pct": round(silt,1), "clay_pct": round(clay,1)}
    except Exception:
        return {"soil_type": "Loam", "sand_pct": 40.0, "silt_pct": 30.0, "clay_pct": 30.0}

# -------------------------
# 4) NASA POWER Rs fetch (with neighbor retries) and improved Rn handling
# -------------------------
def _nasa_power_rs_single(lat, lon, day_str):
    params = {
        "start": day_str,
        "end": day_str,
        "latitude": lat,
        "longitude": lon,
        "format": "JSON",
        "community": "AG",
        "parameters": "ALLSKY_SFC_SW_DWN",
        "time-standard": "UTC"
    }
    r = requests.get(NASA_POWER_DAILY_ENDPOINT, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    p = (data.get("properties", {})
             .get("parameter", {})
             .get("ALLSKY_SFC_SW_DWN", {}))
    if isinstance(p, dict) and len(p) > 0:
        # Single-day request -> one key
        val = list(p.values())[0]
        try:
            rs = float(val)
            # Some POWER responses use missing sentinels; treat very negative as missing
            if rs <= -900:
                return None
            return rs  # Typically MJ/m^2/day for this parameter
        except Exception:
            return None
    return None

def _openmeteo_shortwave_mj(lat, lon, day_iso):
    # Open-Meteo daily shortwave_radiation_sum in Wh/m^2 for the day
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "shortwave_radiation_sum",
        "start_date": day_iso,
        "end_date": day_iso,
        "timezone": "auto"
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    daily = data.get("daily", {}) or {}
    vals = daily.get("shortwave_radiation_sum", []) or []
    if len(vals) >= 1 and vals[0] is not None:
        try:
            wh_per_m2 = float(vals[0])
            # Convert Wh/m^2 to MJ/m^2: 1 Wh = 0.0036 MJ
            return wh_per_m2 * 0.0036
        except Exception:
            return None
    return None

def fetch_nasa_power_rs_for_date(lat, lon, date, tries=4, step_deg=0.125,
                                 backfill_days=7, openmeteo_fallback=True):
    """
    Returns daily ALLSKY_SFC_SW_DWN (MJ/m^2/day) for 'date'.
    Strategy:
      1) NASA POWER at the exact coordinate for 'date'.
      2) Backfill: NASA POWER at the exact coordinate for previous 'backfill_days'.
      3) Fallback: Open-Meteo shortwave_radiation_sum converted to MJ/m^2.
    Never returns None if Open-Meteo succeeds; otherwise may return None.
    """
    s = date.strftime("%Y%m%d")

    # 1) NASA POWER for the exact date
    val = _nasa_power_rs_single(lat, lon, s)
    if val is not None:
        return val

    # 2) Backfill to earlier days (more effective than x/y offsets when data is delayed)
    for dback in range(1, max(1, int(backfill_days)) + 1):
        ds = (date - timedelta(days=dback)).strftime("%Y%m%d")
        val = _nasa_power_rs_single(lat, lon, ds)
        if val is not None:
            return val

    # 3) Fallback to Open-Meteo for the requested day (keeps pipeline numeric)
    if openmeteo_fallback:
        day_iso = date.strftime("%Y-%m-%d")
        val = _openmeteo_shortwave_mj(lat, lon, day_iso)
        if val is not None:
            return val

    return None

def estimate_net_radiation(lat, doy, cloudiness_pct, tmin_c, tmax_c, rs_mj_m2_day=None):
    """
    If rs_mj_m2_day provided (from NASA POWER), use it for shortwave. Otherwise fall back to Hargreaves heuristic.
    Returns Rn in MJ/m2/day.
    """
    albedo = 0.23
    if rs_mj_m2_day is None:
        # fallback heuristic using extraterrestrial and Hargreaves
        Ra = extraterrestrial_radiation(lat, doy)
        dt = max(0.1, tmax_c - tmin_c)
        Rs_clear = 0.16 * sqrt(dt) * Ra
        cloud_frac = cloudiness_pct / 100.0
        Rs = Rs_clear * (1.0 - 0.75 * cloud_frac)
    else:
        Rs = rs_mj_m2_day
    Rns = (1 - albedo) * Rs
    # outgoing longwave (approx), using temps
    avgK = ((tmax_c + 273.16)**4 + (tmin_c + 273.16)**4) / 2.0
    sigma = 4.903e-9
    cloud_frac = cloudiness_pct / 100.0
    Rnl = sigma * avgK * (0.5 + 0.4 * cloud_frac)
    Rn = Rns - Rnl
    return max(0.0, Rn)

# -------------------------
# FAO-56 Penman-Monteith (modified to accept Rs override)
# -------------------------
def saturation_vapour_pressure(t_c):
    return 0.6108 * exp((17.27 * t_c) / (t_c + 237.3))

def actual_vapour_pressure_from_rh(tmin, tmax, rh_mean):
    t_mean = (tmax + tmin) / 2.0
    es_t = saturation_vapour_pressure(t_mean)
    ea = es_t * (rh_mean / 100.0)
    return ea

def psychrometric_constant(atm_pressure_kpa=101.3):
    return 0.065

def delta_saturation_vapour_pressure_curve(t_c):
    es = saturation_vapour_pressure(t_c)
    return (4098 * es) / ((t_c + 237.3)**2)

def extraterrestrial_radiation(lat_deg, doy):
    lat = deg2rad(lat_deg)
    dr = 1 + 0.033 * cos(2 * pi / 365.0 * doy)
    delta = 0.409 * sin(2 * pi / 365.0 * doy - 1.39)
    ws = acos(-tan(lat) * tan(delta))
    Gsc = 0.0820
    Ra = (24 * 60 / pi) * Gsc * dr * (ws * sin(lat) * sin(delta) + cos(lat) * cos(delta) * sin(ws))
    return Ra

from math import tan  # ensure tan available

def estimate_eto_penman_monteith(tmean, tmax, tmin, rh_mean, wind_speed_10m, lat_deg, doy, cloudiness_pct, rs_override=None):
    z = 10.0
    if wind_speed_10m is None:
        u2 = 2.0
    else:
        try:
            u2 = wind_speed_10m * 4.87 / log(67.8 * z - 5.42)
        except Exception:
            u2 = wind_speed_10m
    es_tmax = saturation_vapour_pressure(tmax)
    es_tmin = saturation_vapour_pressure(tmin)
    es = (es_tmax + es_tmin) / 2.0
    ea = actual_vapour_pressure_from_rh(tmin, tmax, rh_mean)
    delta = delta_saturation_vapour_pressure_curve(tmean)
    gamma = psychrometric_constant()
    Rn = estimate_net_radiation(lat_deg, doy, cloudiness_pct, tmin, tmax, rs_override)
    G = 0.0
    numerator = 0.408 * delta * (Rn - G) + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    eto = numerator / denominator
    if eto < 0:
        eto = 0.0
    return round(eto, 3)

# -------------------------
# 5) Water source & blending
# -------------------------

def recommend_irrigation_method(distance_km, capacity_m3, soil_type, area_ha, crop_type=None):
    """
    Recommend irrigation method (drip vs sprinkler) using a scoring system.

    Factors considered:
    - Water capacity
    - Distance from source
    - Soil type
    - Area
    - Crop type (optional)
    """

    # Initialize scores
    drip_score = 0
    sprinkler_score = 0

    # 1. Water capacity
    if capacity_m3 < 200:
        drip_score += 2
    else:
        sprinkler_score += 1

    # 2. Distance from source
    if distance_km > 2.0:
        drip_score += 2
    else:
        sprinkler_score += 1

    # 3. Soil type
    if soil_type in ("Sandy", "Sandy Loam"):
        drip_score += 2
    elif soil_type in ("Loam", "Silt Loam"):
        sprinkler_score += 1
    else:  # Clay-heavy soils
        drip_score += 1

    # 4. Area
    if area_ha > 5:
        sprinkler_score += 2
    else:
        drip_score += 1

    # 5. Optional: Crop type
    if crop_type:
        high_value_crops = ["Tomato", "Potato", "Strawberry"]
        if crop_type in high_value_crops:
            drip_score += 2
        else:
            sprinkler_score += 1

    # Final decision
    if drip_score >= sprinkler_score:
        return "drip"
    else:
        return "sprinkler"
 # Output: "drip"



# -------------------------
# 6) Crop-stage Kc curves and irrigation scheduling (with RAW)
# -------------------------
def dynamic_crop_kc(crop, days_since_planting=None, season_length=120):
    """Return current Kc based on a simple 4-stage linear model."""
    kc_mid = CROP_KC.get(crop, CROP_KC["default"] )
    kc_ini = max(0.2, 0.4 * kc_mid)
    kc_end = max(0.5, 0.7 * kc_mid)
    f_ini, f_dev, f_mid, f_late = DEFAULT_STAGE_FRAC
    L_ini = int(round(season_length * f_ini))
    L_dev = int(round(season_length * f_dev))
    L_mid = int(round(season_length * f_mid))
    L_late = season_length - (L_ini + L_dev + L_mid)
    if days_since_planting is None:
        return kc_mid
    d = max(0, min(days_since_planting, season_length))
    if d <= L_ini:
        return round(kc_ini,3)
    d -= L_ini
    if d <= L_dev:
        frac = d / max(1, L_dev)
        return round(kc_ini + frac * (kc_mid - kc_ini), 3)
    d -= L_dev
    if d <= L_mid:
        return round(kc_mid,3)
    d -= L_mid
    frac = d / max(1, L_late)
    return round(kc_mid + frac * (kc_end - kc_mid), 3)

def compute_irrigation_plan(area_ha, crop_kc, eto_mm_day, soil_info,system_type, effective_rain_mm_day=0.0,  raw_fraction=None):
    """
    Uses RAW (readily available water) = RAW_fraction * TAW to compute realistic interval.
    Caps interval to MAX_IRRIGATION_INTERVAL_DAYS.
    """
    area_m2 = area_ha * 10000.0
    etc_mm_day = eto_mm_day * crop_kc
    net_demand_mm_day = max(0.0, etc_mm_day - effective_rain_mm_day)

    root_depth_m = soil_info.get("rooting_depth_cm", 100) / 100.0
    awc_mm_per_m = soil_info.get("awc_mm_per_m", SOIL_AWC.get(soil_info.get("soil_type","Loam"), 150))
    TAW_mm = awc_mm_per_m * root_depth_m  # total available water in mm

    # select RAW fraction
    if raw_fraction is None:
        raw_fraction = RAW_FRACTION.get(crop_kc_to_crop_name(crop_kc), RAW_FRACTION.get("default", 0.5))
    # But crop_kc_to_crop_name is awkward — prefer explicit raw_fraction param from caller.
    # So if raw_fraction None, default to 0.5
    if raw_fraction is None:
        raw_fraction = 0.5

    usable_aw_mm = TAW_mm * raw_fraction  # RAW in mm

    # fallback if net_demand is zero (no irrigation needed)
    if net_demand_mm_day <= 0:
        interval_days = MAX_IRRIGATION_INTERVAL_DAYS
    else:
        interval_days = usable_aw_mm / net_demand_mm_day
        # cap interval to a maximum so we don't produce unrealistic intervals
        interval_days = min(interval_days, MAX_IRRIGATION_INTERVAL_DAYS)

    interval_days = max(1.0, round(interval_days, 1))
    irrigation_depth_mm = usable_aw_mm
    efficiency = IRRIGATION_EFFICIENCY.get(system_type, 0.75)
    volume_per_event_m3 = (irrigation_depth_mm / 1000.0) * area_m2 / efficiency
    daily_demand_m3 = (net_demand_mm_day / 1000.0) * area_m2
    storage_m3 = max(volume_per_event_m3, daily_demand_m3 * interval_days)

    return {
        "etc_mm_day": round(etc_mm_day,3),
        "net_demand_mm_day": round(net_demand_mm_day,3),
        "daily_demand_m3": round(daily_demand_m3,3),
        "TAW_mm": round(TAW_mm,1),
        "usable_aw_mm": round(usable_aw_mm,1),
        "interval_days": interval_days,
        "irrigation_depth_mm": round(irrigation_depth_mm,1),
        "volume_per_event_m3": round(volume_per_event_m3,2),
        "storage_m3": round(storage_m3,2),
        "efficiency": efficiency
    }

def crop_kc_to_crop_name(kc_value):
    # best-effort reverse lookup; not perfect — prefer passing crop name to compute_irrigation_plan
    for k, v in CROP_KC.items():
        if abs(v - kc_value) < 1e-6:
            return k
    return "default"

# -------------------------
# 7) Pumping economics
# -------------------------
def compute_pumping_energy_and_cost(volume_m3, pump_head_m=20.0, pump_eff=0.6, energy_cost_per_kwh=6.0, water_price_per_m3=2.0):
    rho = 1000.0
    g = 9.81
    if pump_eff <= 0:
        pump_eff = 0.6
    energy_kwh = (rho * g * volume_m3 * pump_head_m) / (pump_eff * 3.6e6)
    cost_energy = energy_kwh * energy_cost_per_kwh
    cost_water = water_price_per_m3 * volume_m3
    total_cost = cost_energy + cost_water
    return round(energy_kwh,3), round(cost_energy,2), round(cost_water,2), round(total_cost,2)

# -------------------------
# 8) Calendar helper (fixed timezone handling)
# -------------------------


def build_30_day_calendar(
    lat: float,
    lon: float,
    start_date: date,
    interval_days: float,
    forecast_threshold_mm: float = 2.0,
    forecast_days: int = 16  # Open-Meteo supports up to 16
) -> List[Dict[str, Any]]:
    """
    Build a 30-day irrigation calendar using Open-Meteo daily precipitation forecasts.
    - Fills real forecast mm for available days (up to 'forecast_days'), then 0.0 beyond horizon.
    - No None values are returned.
    - Adds 'has_forecast' to indicate whether the value came from an actual forecast source.
    """
    # Clamp forecast_days to Open-Meteo limits
    if forecast_days is None or forecast_days <= 0:
        forecast_days = 7
    forecast_days = min(16, max(1, int(round(forecast_days))))

    # Query Open-Meteo daily precipitation_sum in local time
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "timezone": "auto",
        "forecast_days": forecast_days
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Build mapping of local-date -> precipitation_sum (mm)
    daily = data.get("daily", {}) or {}
    times = daily.get("time", []) or []
    precs = daily.get("precipitation_sum", []) or []

    daily_precip = {}
    for t, p in zip(times, precs):
        try:
            d = datetime.fromisoformat(t).date()
        except Exception:
            # If timezone=auto ever returns a 'Z', handle fallback
            d = datetime.fromisoformat(t.replace("Z", "+00:00")).date()
        try:
            mm = float(p if p is not None else 0.0)
        except (TypeError, ValueError):
            mm = 0.0
        daily_precip[d] = mm

    # Compute irrigation dates within 30 days starting at start_date
    irrigation_dates = set()
    if interval_days != float('inf') and interval_days and interval_days > 0:
        step = max(1, int(round(interval_days)))
        d = start_date
        while (d - start_date).days < 30:
            irrigation_dates.add(d)
            d = d + timedelta(days=step)

    # Build 30-day calendar
    cal = []
    for i in range(30):
        day = start_date + timedelta(days=i)
        has_fc = day in daily_precip
        rain_mm = round(float(daily_precip.get(day, 0.0)), 2)  # always numeric
        recommended = day in irrigation_dates
        flagged = rain_mm >= forecast_threshold_mm
        skip = recommended and flagged
        cal.append({
            "date": day.isoformat(),
            "recommended_irrigation": recommended,
            "forecast_rain_mm": rain_mm,
            "rain_flag": flagged,
            "skip_due_to_rain": skip,
            "has_forecast": has_fc
        })
    return cal

def _normalize_allocation(allocation):
    """
    Returns a list of rows with keys:
      name, type, distance_km, allocated_m3, capacity
    Accepts either:
      - dict[str -> info dict]  (your current static version)
      - list[info dict with .name/.type] (common in live/OSM lookups)
    """
    rows = []
    if isinstance(allocation, dict):
        items = allocation.items()
    elif isinstance(allocation, list):
        # Expect list of dicts; synthesize a name and pair like (name, info)
        items = [((a.get("name") or a.get("type") or "Source"), a) for a in allocation]
    else:
        items = []

    for name, info in items:
        rows.append({
            "name": str(name),
            "type": info.get("type"),
            "distance_km": float(info.get("distance_km", float("nan"))),
            "allocated_m3": float(info.get("allocated_m3", 0.0)),
            "capacity": float(info.get("capacity", 0.0)),
        })

    # Sort by distance if available
    rows.sort(key=lambda r: (math.isnan(r["distance_km"]), r["distance_km"]))
    return rows

def print_allocation(allocation, unmet, radius_km):
    rows = _normalize_allocation(allocation)
    print(f"\nWater source blending (within radius {radius_km} km):")
    if rows:
        for r in rows:
            t = f" [{r['type']}]" if r.get("type") else ""
            # keep the exact message style used previously
            print(
                f"  {r['name']}{t}: allocated {round(r['allocated_m3'],2)} m3 "
                f"(capacity {int(r['capacity'])} m3/day) distance {round(r['distance_km'],3)} km"
            )
    if (unmet or 0) > 0:
        print(f"  UNMET VOLUME: {round(unmet,2)} m3 — consider storage or splitting the event across days.")
    if not rows and (unmet or 0) > 0:
        print("  No nearby sources found within blend radius.")
import json
import math

def print_closest_source(w, title="Closest water source"):
    """
    Pretty-prints the closest water source info from either:
      - static find_closest_water_source: {source_name, distance_km, capacity_m3_per_day}
      - live OSM version: {source_name, type, distance_km, capacity_m3_per_day, lat, lon, tags}
    """
    print(f"\n{title}:")
    if not isinstance(w, dict) or (not w.get("source_name") and (w.get("capacity_m3_per_day", 0) <= 0)):
        print("  None found within the configured maximum distance.")
        return

    name = w.get("source_name") or "Unnamed source"
    stype = w.get("type")  # may be None for the static version
    dist = w.get("distance_km")
    cap = w.get("capacity_m3_per_day", w.get("capacity"))  # support either key
    lat = w.get("lat")
    lon = w.get("lon")
    tags = w.get("tags") or {}

    # Line 1: name/type
    type_str = f" [{stype}]" if stype else ""
    dist_str = f"{dist:.3f} km" if isinstance(dist, (int, float)) and not math.isnan(dist) else "n/a"
    cap_str = f"{cap:.0f} m3/day" if isinstance(cap, (int, float)) else "n/a"
    print(f"  {name}{type_str}: distance {dist_str}, capacity {cap_str}")

    # Line 2: coords (if available)
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        print(f"  Coords: ({lat:.6f}, {lon:.6f})")

    # Line 3: a few useful tags (if available)
    interesting = ["operator", "intermittent", "water", "waterway", "man_made", "amenity", "name:en"]
    tag_items = [(k, tags[k]) for k in interesting if k in tags]
    if tag_items:
        tag_str = ", ".join([f"{k}={v}" for k, v in tag_items])
        print(f"  Tags: {tag_str}")

def print_closest_source_debug_json(w):
    """Optional: print raw JSON for debugging."""
    try:
        print("\n[debug] raw source object:")
        print(json.dumps(w, indent=2, ensure_ascii=False))
    except Exception:
        pass

# -------------------------
# 9) CLI + Main logic (extended) with timezone fixes and Rs fallback
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Farm Irrigation Planner (enhanced)")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--area_ha", type=float, default=1.0)
    p.add_argument("--crop", type=str, default="wheat")
    p.add_argument("--api_key", type=str, default="")
    p.add_argument("--system", type=str, choices=["drip","sprinkler","surface"], default="sprinkler")
    p.add_argument("--pump_head_m", type=float, default=20.0)
    p.add_argument("--pump_eff", type=float, default=0.6)
    p.add_argument("--energy_cost", type=float, default=6.0)
    p.add_argument("--water_price", type=float, default=2.0)
    p.add_argument("--days_since_planting", type=int, default=None, help="Optional: days since planting to compute stage-specific Kc")
    p.add_argument("--season_length", type=int, default=120, help="Crop season length in days (used for Kc staging)")
    p.add_argument("--blend_radius_km", type=float, default=5.0, help="Search radius to consider other water sources for blending")
    p.add_argument("--max_source_distance_km", type=float, default=50.0, help="Max distance to consider a water source realistic")
    p.add_argument("--raw_fraction", type=float, default=None, help="Optional override for RAW fraction (0-1)")
    p.add_argument("--use_osm_sources", action="store_true", help="Use OSM Overpass to find nearby water sources")
    return p.parse_args()

def main():
    args = parse_args()
    lat, lon = args.lat, args.lon
    area_ha = max(0.01, args.area_ha)
    crop = args.crop.lower()
    api_key = args.api_key or OPENWEATHER_API_KEY

    now = datetime.now(timezone.utc)

    print(f"\nFarm Irrigation Planner — coords=({lat},{lon}) area={area_ha}ha crop={crop.title()}\n")

    # 1) Weather / forecast
    onecall = fetch_onecall(lat, lon, api_key)
    current = onecall.get("current", {})
    tmean = current.get("temp", 25.0)
    tmin = current.get("temp_min") or (tmean - 3)
    tmax = current.get("temp_max") or (tmean + 3)
    wind_10m = current.get("wind_speed", 2.0)
    humidity = current.get("humidity", 60)
    cloudiness = current.get("clouds", 10)

    print("Current weather snapshot (from OneCall or simulated):")
    print(f"  Temp: {tmean}°C  (min {tmin} / max {tmax})")
    print(f"  Humidity: {humidity}%  Wind(10m): {wind_10m} m/s  Clouds: {cloudiness}%")
    forecast_rain_48h = sum_rain_next_hours(onecall, 48)
    print(f"  Forecast rain next 48h (sum): {forecast_rain_48h} mm")

    # 2) Soil via ISRIC
    soil = get_soil_from_isric(lat, lon)
    soil.setdefault("rooting_depth_cm", 100)
    soil.setdefault("awc_mm_per_m", SOIL_AWC.get(soil.get("soil_type","Loam"), 150))
    print(f"\nSoil (ISRIC or fallback): {soil.get('soil_type')} (sand={soil.get('sand_pct')}%, silt={soil.get('silt_pct')}%, clay={soil.get('clay_pct')}%)")
    print(f"  Rooting depth default: {soil['rooting_depth_cm']} cm  AWC per m: {soil['awc_mm_per_m']} mm")

    # 3) Long term rainfall (NASA POWER climatology, 3-month smoothed)
    monthly = get_long_term_monthly_rainfall(lat, lon)
    current_month = now.month
    monthly_current = monthly.get(current_month, 0.0)
    effective_rain_mm_day = compute_effective_rainfall_offset_3month(monthly, current_month, effectiveness_fraction=0.75)
    print(f"\nLong-term average rainfall (3-month smoothed around month {current_month}): {monthly_current} mm/month -> effective ~{effective_rain_mm_day} mm/day usable")

    # 4) Use NASA POWER Rs for ETo if available for today (with neighbor retry)
    today = now.date()
    rs_today = fetch_nasa_power_rs_for_date(lat, lon, today, tries=5, step_deg=0.125)
    if rs_today is None:
        # fallback to heuristic Rs (Hargreaves-like) using tmin/tmax and extraterrestrial Ra
        # compute Ra for today
        doy = day_of_year(now)
        Ra = extraterrestrial_radiation(lat, doy)
        dt = max(0.1, tmax - tmin)
        rs_hargreaves = 0.16 * sqrt(dt) * Ra
        rs_today = rs_hargreaves
        print("\nNASA POWER Rs not available; falling back to Hargreaves Rs approximation.")
    else:
        print(f"\nNASA POWER Rs for today obtained: {rs_today} MJ/m2/day (used for net radiation).")

    # 5) ETo using Penman-Monteith (with Rs override if available)
    if args.use_osm_sources:
        w = find_closest_water_source_live((lat, lon), radius_km=args.blend_radius_km, max_distance_km=args.max_source_distance_km)
    else:
        w = find_closest_water_source_live((lat, lon), max_distance_km=args.max_source_distance_km)
    print_closest_source(w, title="Closest water source (auto)")


    # 11) Recommend irrigation method (heuristic)
    method = recommend_irrigation_method(w['distance_km'], w['capacity_m3_per_day'], soil['soil_type'], area_ha,crop_type=crop)
    print(f"\nRecommended irrigation method (heuristic): {method.upper()}")
    doy = day_of_year(now)
    eto = estimate_eto_penman_monteith(tmean, tmax, tmin, humidity, wind_10m, lat, doy, cloudiness, rs_override=rs_today)
    print(f"\nEstimated reference evapotranspiration (ETo) [FAO-56 PM approx]: {eto} mm/day")

    # 6) Crop Kc and ETc - dynamic
    kc_dynamic = dynamic_crop_kc(crop, args.days_since_planting, season_length=args.season_length)
    print(f"Crop coefficient (Kc) used (stage-specific): {kc_dynamic}")

    # 7) Water source (closest) and method recommendation
    w = find_closest_water_source_live((lat, lon), max_distance_km=args.max_source_distance_km)
    src_name = w['source_name'] or "None within realistic distance"
    print(f"\nClosest water source: {src_name} ({w['distance_km']} km) capacity~{w['capacity_m3_per_day']} m3/day")

    # 8) Irrigation plan (offset rainfall) - pass in raw_fraction if user provided
    plan = compute_irrigation_plan(area_ha, kc_dynamic, eto, {"soil_type": soil['soil_type'], "rooting_depth_cm": soil['rooting_depth_cm'], "awc_mm_per_m": soil['awc_mm_per_m']}, effective_rain_mm_day=effective_rain_mm_day, system_type=method.lower, raw_fraction=args.raw_fraction)
    print("\n--- Irrigation Plan ---")
    print(f"ETc (crop evapotranspiration): {plan['etc_mm_day']} mm/day")
    print(f"Net demand after long-term rainfall offset: {plan['net_demand_mm_day']} mm/day")
    print(f"Daily water demand (net): {plan['daily_demand_m3']} m3/day")
    print(f"Total Available Water (TAW): {plan['TAW_mm']} mm")
    print(f"Readily Available Water used (RAW): {plan['usable_aw_mm']} mm")
    print(f"Recommended interval: Every {plan['interval_days']} day(s) (capped to {MAX_IRRIGATION_INTERVAL_DAYS} days max)")
    print(f"Irrigation depth per event: {plan['irrigation_depth_mm']} mm")
    print(f"Volume per event: {plan['volume_per_event_m3']} m3 (efficiency {int(plan['efficiency']*100)}%)")
    print(f"Suggested on-farm storage (min): {plan['storage_m3']} m3")

    # 9) If forecast predicts substantial rain within 48h, recommend delaying irrigation
    if forecast_rain_48h >= 5.0:
        print("\n⚠️ Rain expected soon: Consider delaying irrigation.")
        print(f"  Forecasted rain next 48h = {forecast_rain_48h} mm (this may cover portions of ET demand).")
    else:
        print("\nNo significant rain forecast in next 48h (per OneCall).")

    # 10) Multiple source blending recommendation
    # allocation, unmet = blend_water_sources((lat, lon), plan['volume_per_event_m3'], WATER_SOURCES, max_radius_km=args.blend_radius_km)
    # print("\nWater source blending (within radius {0} km):".format(args.blend_radius_km))
    # if allocation:
    #     for s, info in allocation.items():
    #         print(f"  {s}: allocated {info['allocated_m3']} m3 (capacity {info['capacity']} m3/day) distance {info['distance_km']} km")
    # if unmet > 0:
    #     print(f"  UNMET VOLUME: {unmet} m3 — consider storage or splitting the event across days.")
    # if not allocation and unmet > 0:
    #     print("  No nearby sources found within blend radius.")


    # 12) Economics: cost per event, energy, check capacity
    volume_event = plan['volume_per_event_m3']
    energy_kwh, cost_energy, cost_water, total_cost = compute_pumping_energy_and_cost(volume_event, pump_head_m=args.pump_head_m, pump_eff=args.pump_eff, energy_cost_per_kwh=args.energy_cost, water_price_per_m3=args.water_price)
    print("\n--- Economics & Pumping ---")
    print(f"Per irrigation event (volume {volume_event} m3):")
    print(f"  Energy required: {energy_kwh} kWh -> cost {cost_energy} ₹")
    print(f"  Water cost: {cost_water} ₹")
    print(f"  Total cost/event: {total_cost} ₹ (pump head {args.pump_head_m} m, eff {args.pump_eff})")

    # 13) Water source capacity check
    if w['capacity_m3_per_day'] <= 0:
        print("\n⚠️ No realistic water source found within the maximum allowed distance.")
        print("  Consider storage (pond/tank) or increasing max_source_distance_km parameter.")
    elif volume_event > w['capacity_m3_per_day']:
        print("\n⚠️ Water source capacity may be insufficient for delivering full event volume in a single day.")
        print("  Consider splitting the event across multiple days or building storage (tank/pond).")
    else:
        print("\nWater source capacity appears sufficient for event (based on simulated capacity).")

    # 14) 30-day calendar with forecast flags
    start_date = datetime.now(timezone.utc).date()
    cal = build_30_day_calendar(lat=lat,lon=lon,start_date=start_date,interval_days=1)
    print("\n--- 30-day irrigation calendar (recommended dates flagged; days with forecast rain flagged) ---")
    for day in cal:
        mark = "[IRR]" if day['recommended_irrigation'] else "     "
        rain_mark = "⚠️" if day['rain_flag'] else "  "
        skip_mark = "[SKIP-Rain]" if day['skip_due_to_rain'] else ""
        print(f"{day['date']} {mark} {skip_mark} forecast_rain={day['forecast_rain_mm']} mm {rain_mark}")

    # 15) Practical tips
    print("\nPractical recommendations:")
    print(" - Install soil moisture sensors to refine irrigation scheduling (avoid overwatering).")
    print(" - Use measured Rs (solar radiation) or NASA POWER daily Rs to improve ETo accuracy.")
    print(" - For best ETo accuracy, use nearby meteorological station data (wind, solar, humidity).")
    print(" - Consider on-farm storage if water source capacity or pumping is limiting.")
    print("\nReport complete.\n")

if __name__ == "__main__":
    main()