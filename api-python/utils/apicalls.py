import requests
import pandas as pd
from datetime import date, timedelta
import logging
import time

logger = logging.getLogger(__name__)

def get_weather_summary(lat, lon, max_retries=3):
    """Get weather data with retry logic"""
    for attempt in range(max_retries):
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m",
                "timezone": "auto"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            resp = response.json()
            
            # Validate response structure
            if "hourly" not in resp:
                raise ValueError("Invalid weather API response structure")
            
            hourly_data = resp["hourly"]
            if "temperature_2m" not in hourly_data or "relative_humidity_2m" not in hourly_data:
                raise ValueError("Missing temperature or humidity data in weather response")
            
            df = pd.DataFrame({
                "temperature": hourly_data["temperature_2m"],
                "humidity": hourly_data["relative_humidity_2m"]
            })
            
            # Filter out None values
            df = df.dropna()
            
            if df.empty:
                raise ValueError("No valid weather data received")
            
            avg_temp = df["temperature"].mean()
            avg_hum = df["humidity"].mean()
            
            # Determine season
            month = date.today().month
            if month in [12, 1, 2]:
                season = "winter"
            elif month in [3, 4, 5]:
                season = "summer"
            elif month in [6, 7, 8, 9]:
                season = "rainy"
            else:
                season = "spring"
            
            return avg_temp, avg_hum, season
            
        except Exception as e:
            logger.warning(f"Weather API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, return default values
                logger.error("All weather API attempts failed, using default values")
                return 25.0, 60.0, "summer"  # Default values
            time.sleep(1)  # Wait before retry

def get_soil_ph_first(lat, lon, stat="mean", max_retries=3):
    """Get soil pH with retry logic"""
    for attempt in range(max_retries):
        try:
            url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
            params = {"lat": lat, "lon": lon, "property": "phh2o"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            resp = response.json()
            
            # Validate response structure
            if "properties" not in resp:
                raise ValueError("Invalid soil API response structure")
            
            layers = resp["properties"]["layers"]
            ph_layer = next((layer for layer in layers if layer["name"] == "phh2o"), None)
            
            if ph_layer is None:
                raise ValueError("pH layer not found in soil data")
            
            if not ph_layer.get("depths"):
                raise ValueError("No depth data found for pH")
            
            first_depth = ph_layer["depths"][0]
            val = first_depth["values"].get(stat)
            
            if val is not None:
                ph_value = val / 10.0
                # Validate pH range (should be between 0-14)
                if 0 <= ph_value <= 14:
                    return ph_value
                else:
                    logger.warning(f"Invalid pH value: {ph_value}, using default")
                    
        except Exception as e:
            logger.warning(f"Soil pH API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All soil pH API attempts failed, using default value")
            else:
                time.sleep(1)  # Wait before retry
    
    return 6.5  # Default pH value

def get_total_rainfall(lat, lon, max_retries=3):
    """Get rainfall data with retry logic"""
    for attempt in range(max_retries):
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&daily=precipitation_sum&timezone=auto"
            )
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            resp = response.json()
            
            # Validate response structure
            if "daily" not in resp:
                raise ValueError("Invalid rainfall API response structure")
            
            daily_data = resp["daily"]
            precipitation_data = daily_data.get("precipitation_sum", [])
            
            if not precipitation_data:
                raise ValueError("No precipitation data received")
            
            # Filter out None values and calculate total
            rainfall_values = [v for v in precipitation_data if v is not None]
            total_rainfall = sum(rainfall_values) if rainfall_values else 0
            
            return max(0, total_rainfall)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Rainfall API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All rainfall API attempts failed, using default value")
                return 50.0  # Default rainfall value
            time.sleep(1)  # Wait before retry

def build_single_row(lat, lon):
    """Build a single row of data with comprehensive error handling"""
    try:
        logger.info(f"Fetching data for coordinates: {lat}, {lon}")
        
        # Validate coordinates
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")
        
        # Fetch all data
        temp, hum, season = get_weather_summary(lat, lon)
        ph = get_soil_ph_first(lat, lon)
        rainfall = get_total_rainfall(lat, lon)
        
        # Validate all values are numeric and finite
        if not all(isinstance(x, (int, float)) and not pd.isna(x) for x in [temp, hum, ph, rainfall]):
            raise ValueError("One or more fetched values are invalid")
        
        # Create dataframe
        df = pd.DataFrame([{
            "temperature": float(temp),
            "humidity": float(hum),
            "ph": float(ph),
            "water availability": float(rainfall),
            "season": str(season)
        }])
        
        logger.info(f"Successfully built data row: {df.to_dict('records')[0]}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to build data row: {e}")
        # Return a dataframe with default values as fallback
        default_df = pd.DataFrame([{
            "temperature": 25.0,
            "humidity": 60.0,
            "ph": 6.5,
            "water availability": 50.0,
            "season": "summer"
        }])
        logger.info("Using default values due to API failures")
        return default_df