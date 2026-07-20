"""
Agentic tools for AquaMind AI.

- calculate_drip_irrigation: existing hydraulic calculator (unchanged logic).
- get_reference_evapotranspiration: NEW. Pulls real FAO-56 Penman-Monteith
  ET0 forecast data from Open-Meteo (free, no API key) for a given
  latitude/longitude, so the agent can reason about actual near-term
  irrigation demand instead of only static document knowledge.
- lookup_crop_coefficient: NEW. Structured lookup over standard FAO-56
  Table 12 crop coefficients (Kc ini / mid / end). This offloads a
  frequently-asked, purely tabular question away from the RAG pipeline
  (where formatting/retrieval noise can distort a single number) and onto
  a deterministic, always-correct source.
"""
from typing import Optional

import requests
from langchain_core.tools import tool

# --- FAO-56 Table 12 (selected common crops), Kc ini / mid / end ---
CROP_KC_TABLE = {
    "tomato":     {"kc_ini": 0.6,  "kc_mid": 1.15, "kc_end": 0.80},
    "maize":      {"kc_ini": 0.3,  "kc_mid": 1.20, "kc_end": 0.60},
    "corn":       {"kc_ini": 0.3,  "kc_mid": 1.20, "kc_end": 0.60},
    "wheat":      {"kc_ini": 0.7,  "kc_mid": 1.15, "kc_end": 0.40},
    "potato":     {"kc_ini": 0.5,  "kc_mid": 1.15, "kc_end": 0.75},
    "cotton":     {"kc_ini": 0.35, "kc_mid": 1.18, "kc_end": 0.70},
    "onion":      {"kc_ini": 0.7,  "kc_mid": 1.05, "kc_end": 0.75},
    "alfalfa":    {"kc_ini": 0.40, "kc_mid": 0.95, "kc_end": 0.90},
    "grape":      {"kc_ini": 0.30, "kc_mid": 0.85, "kc_end": 0.45},
    "citrus":     {"kc_ini": 0.70, "kc_mid": 0.65, "kc_end": 0.70},
    "lettuce":    {"kc_ini": 0.7,  "kc_mid": 1.00, "kc_end": 0.95},
    "cucumber":   {"kc_ini": 0.6,  "kc_mid": 1.00, "kc_end": 0.75},
    "sunflower":  {"kc_ini": 0.35, "kc_mid": 1.15, "kc_end": 0.35},
    "sugarbeet":  {"kc_ini": 0.35, "kc_mid": 1.20, "kc_end": 0.70},
}


@tool
def calculate_drip_irrigation(number_of_emitters: int, flow_rate_per_emitter_Lph: float, operation_hours: float):
    """Calculates the total water volume and flow rate for a drip irrigation system.
    Args:
        number_of_emitters: Total number of drip emitters in the zone.
        flow_rate_per_emitter_Lph: Flow rate of a single emitter in Liters per hour.
        operation_hours: How many hours the system will run.
    """
    total_flow_rate = number_of_emitters * flow_rate_per_emitter_Lph
    total_volume_liters = total_flow_rate * operation_hours
    total_volume_m3 = total_volume_liters / 1000

    return (
        f"\u2699\ufe0f **Calculation Result:**\n"
        f"- Total System Flow Rate = **{total_flow_rate} L/h**\n"
        f"- Total Water Volume = **{total_volume_liters} Liters** ({total_volume_m3} m\u00b3)"
    )


@tool
def get_reference_evapotranspiration(latitude: float, longitude: float, days: int = 5):
    """Fetches real FAO-56 Penman-Monteith reference evapotranspiration (ET0, mm/day)
    forecast for a location, to help estimate near-term crop water demand.
    Use this whenever the user asks about current/upcoming irrigation needs for
    a specific place (city, farm, coordinates), as opposed to generic theory.
    Args:
        latitude: Latitude of the field/farm location.
        longitude: Longitude of the field/farm location.
        days: Number of forecast days to return (1-7, default 5).
    """
    days = max(1, min(days, 7))
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "et0_fao_evapotranspiration,precipitation_sum,temperature_2m_max,temperature_2m_min",
                "forecast_days": days,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        et0 = daily.get("et0_fao_evapotranspiration", [])
        rain = daily.get("precipitation_sum", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])

        if not dates:
            return "\u26a0\ufe0f No forecast data returned for that location."

        lines = ["\U0001f4a7 **Reference Evapotranspiration (ET0) Forecast (FAO-56 Penman-Monteith):**"]
        for i, d in enumerate(dates):
            lines.append(
                f"- {d}: ET0 = **{et0[i]} mm/day**, Rain = {rain[i]} mm, "
                f"Temp = {tmin[i]}\u2013{tmax[i]}\u00b0C"
            )
        avg_et0 = round(sum(et0) / len(et0), 2) if et0 else None
        if avg_et0 is not None:
            lines.append(f"\n**Average ET0 over period: {avg_et0} mm/day**")
            lines.append(
                "Tip: crop water requirement (ETc) = ET0 \u00d7 Kc for the crop's current growth stage."
            )
        return "\n".join(lines)
    except requests.RequestException as e:
        return f"\u26a0\ufe0f Could not fetch weather/ET0 data: {e}"


@tool
def lookup_crop_coefficient(crop_name: str):
    """Looks up the standard FAO-56 crop coefficient (Kc) values for a crop
    at its initial, mid-season, and end-season growth stages. Use this for
    direct 'what is the Kc for X' questions instead of searching documents,
    since this is a precise structured value.
    Args:
        crop_name: Common name of the crop, e.g. 'tomato', 'maize', 'wheat'.
    """
    key = crop_name.strip().lower()
    entry = CROP_KC_TABLE.get(key)
    if not entry:
        available = ", ".join(sorted(CROP_KC_TABLE.keys()))
        return (
            f"No entry for '{crop_name}' in the local Kc table. "
            f"Available crops: {available}. Try the knowledge base for less common crops."
        )
    return (
        f"\U0001f331 **FAO-56 Crop Coefficients for {crop_name.title()}:**\n"
        f"- Initial stage (Kc ini): **{entry['kc_ini']}**\n"
        f"- Mid-season (Kc mid): **{entry['kc_mid']}**\n"
        f"- Late/end-season (Kc end): **{entry['kc_end']}**"
    )
