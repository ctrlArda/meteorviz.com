import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from dotenv import load_dotenv
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point, box     
import warnings
from google import genai 

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# --- 1. PROJECT SETUP AND CONFIGURATION ---
load_dotenv()

# ------ ADD 4 LINES STARTING FROM HERE ------
print("--- VARIABLE CHECK STARTED ---")
api_key_degeri = os.getenv("NASA_API_KEY")
print(f"Read NASA API Key: {api_key_degeri}")
print(f"Working Directory: {os.getcwd()}")
print("--- VARIABLE CHECK FINISHED ---")
# ----------------------------------------------

app = Flask(__name__)
CORS(app)

# Reading and defining NASA and GEMINI API Keys from ENV (CORRECTION MADE HERE)
NASA_API_KEY = os.getenv("NASA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

NEO_LOOKUP_URL = "https://api.nasa.gov/neo/rest/v1/neo/{}"
JPL_LOOKUP_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
WORLDPOP_FILE = "ppp_2020_1km_Aggregated.tif"

# --- PERFORMANCE IMPROVEMENT: Load data into memory at startup ---
WORLDPOP_DATA_SRC = None
if not os.path.exists(WORLDPOP_FILE):
    print("="*60)
    print(f"WARNING: World population data file ('{WORLDPOP_FILE}') not found!")
    print("The program will continue without population data, but population calculation will not work.")
    print("="*60)
else:
    print(f"Loading '{WORLDPOP_FILE}' data into memory...")
    WORLDPOP_DATA_SRC = rasterio.open(WORLDPOP_FILE)
    print("Population data loaded successfully.")


# --- 2. PHYSICAL MODELS AND CONSTANTS ---
TARGET_DENSITY = 2700
GRAVITY = 9.81

def calculate_crater(energy_joules, density, angle_rad):
    if energy_joules <= 0: return 0
    crater_diameter_m = 1.161 * (density / TARGET_DENSITY)**(1/3) * \
                        (energy_joules / (GRAVITY * TARGET_DENSITY))**0.25 * \
                        (np.sin(angle_rad))**(1/3)
    return crater_diameter_m

def calculate_thermal_radius(energy_joules, burn_flux_joules_m2=2.5e5):
    thermal_energy_fraction = 0.4
    return ((thermal_energy_fraction * energy_joules) / (4 * np.pi * burn_flux_joules_m2))**0.5

# NEWLY ADDED FUNCTION
def calculate_air_blast_radius(energy_megatons_tnt):
    """
    Estimates the pressure wave (5 psi) radius. 
    5 psi is the pressure level causing serious structural damage to buildings.
    Formula: R (km) ≈ 1.4 * (Energy in Megatons)^(1/3)
    """
    if energy_megatons_tnt <= 0: return 0
    radius_km = 1.4 * (energy_megatons_tnt**(1/3))
    return radius_km


# --- 3. HUMAN IMPACT CALCULATION ---
def get_population_in_radius(lat, lon, radius_km):
    FAST_ESTIMATION_THRESHOLD_KM = 100 # Fast estimation will be used for radii larger than this limit

    if WORLDPOP_DATA_SRC is None:
        return {"error": f"{WORLDPOP_FILE} not found."}
    
    try:
        src = WORLDPOP_DATA_SRC
        point = Point(lon, lat)
        
        # Convert geographical coordinates to a metric system
        gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        gdf_proj = gdf.to_crs("EPSG:3395") 
        
        # Create the impact circle
        circle_proj = gdf_proj.buffer(radius_km * 1000)
        
        # Decide whether to use fast estimation or detailed calculation
        if radius_km > FAST_ESTIMATION_THRESHOLD_KM:
            # FAST ESTIMATION METHOD
            # Get the bounding box surrounding the circle
            bounds = circle_proj.bounds.values[0]
            bounding_box_proj = box(*bounds)
            
            # Convert the box back to geographical coordinates
            box_geo = gpd.GeoSeries([bounding_box_proj], crs="EPSG:3395").to_crs(src.crs)

            # Read only the box area (much faster)
            out_image, out_transform = mask(src, box_geo, crop=True)
            
            # Calculate the average population density (per pixel)
            valid_pixels = out_image[out_image > 0]
            if len(valid_pixels) == 0:
                return 0 # If it's an unpopulated area (like the ocean)
            
            # Since 1 pixel is approximately 1km², this gives the average population density per pixel
            mean_density_per_pixel = valid_pixels.mean() 
            
            # Calculate the circle's area (in km²)
            circle_area_km2 = np.pi * (radius_km ** 2)
            
            # Calculate the estimated population
            estimated_population = circle_area_km2 * mean_density_per_pixel
            
            return int(estimated_population)
        
        else:
            # DETAILED CALCULATION METHOD (For small radii)
            circle_geo = circle_proj.to_crs(src.crs)
            out_image, out_transform = mask(src, circle_geo, crop=True)
            population = out_image[out_image > 0].sum()
            return int(population)

    except Exception as e:
        return {"error": f"Error processing population data: {e}"}
        
# --- 4. API ENDPOINTS ---
@app.route('/')
def index():
    return "MeteorViz API is Running! (Population Analysis Active)"

@app.route('/lookup_asteroid/<string:spk_id>')
def lookup_asteroid(spk_id):
    if not NASA_API_KEY:
        return jsonify({"error": "NASA API key not configured on the server."}), 500
    try:
        neo_response = requests.get(NEO_LOOKUP_URL.format(spk_id), params={"api_key": NASA_API_KEY})
        neo_response.raise_for_status()
        neo_data = neo_response.json()

        jpl_response = requests.get(JPL_LOOKUP_URL, params={"sstr": spk_id})
        jpl_response.raise_for_status()
        jpl_data = jpl_response.json()
        
        velocity_kms = 20 
        if neo_data.get("close_approach_data") and len(neo_data["close_approach_data"]) > 0:
            velocity_str = neo_data["close_approach_data"][0]["relative_velocity"]["kilometers_per_second"]
            if velocity_str:
                velocity_kms = float(velocity_str)
        result = {
            "source": "NASA NeoWs & JPL Small-Body Database", "name": neo_data.get("name"), "spk_id": spk_id,
            "is_potentially_hazardous": neo_data.get("is_potentially_hazardous_asteroid"),
            "estimated_diameter_km": neo_data.get("estimated_diameter", {}).get("kilometers"),
            "absolute_magnitude": neo_data.get("absolute_magnitude_h"),
            "orbital_period_days": jpl_data.get("orbit", {}).get("period", {}).get("value"),
            "first_observation_date": jpl_data.get("orbit", {}).get("first_obs"),
            "last_observation_date": jpl_data.get("orbit", {}).get("last_obs"), "velocity_kms": velocity_kms
        }
        return jsonify(result)
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Failed to retrieve data from API. ID: {spk_id}. Error: {e.response.text}"}), 404
    except Exception as e:
        return jsonify({"error": f"An unknown error occurred: {e}"}), 500


@app.route('/calculate_human_impact', methods=['POST'])
def calculate_human_impact():
    try:
        data = request.json
        lat, lon = float(data['latitude']), float(data['longitude'])
        mass_kg, velocity_kms = float(data['mass_kg']), float(data['velocity_kms'])
        angle_deg, density = float(data['angle_deg']), float(data['density'])

        velocity_ms = velocity_kms * 1000
        angle_rad = np.deg2rad(angle_deg)
        impact_energy_joules = 0.5 * mass_kg * velocity_ms**2
        
        # Convert energy to Megatons for other calculations
        impact_energy_megatons_tnt = impact_energy_joules / 4.184e15

        # Physical model calculations
        crater_diameter_m = calculate_crater(impact_energy_joules, density, angle_rad)
        thermal_radius_m = calculate_thermal_radius(impact_energy_joules)
        
        # NEW CALCULATIONS
        air_blast_5psi_radius_km = calculate_air_blast_radius(impact_energy_megatons_tnt)
        ejecta_blanket_radius_km = (crater_diameter_m / 1000) * 2.5 # Ejecta blanket area ~2.5 x crater radius
        
        thermal_radius_km = thermal_radius_m / 1000
        
        affected_population = get_population_in_radius(lat, lon, thermal_radius_km)
        
        result = {
            "input_parameters": data,
            "physical_impact": {
                "impact_energy_megatons_tnt": impact_energy_megatons_tnt,
                "crater_diameter_km": crater_diameter_m / 1000,
                "thermal_burn_radius_km": {"second_degree": thermal_radius_km},
                # ADD NEW DATA TO RESPONSE
                "air_blast_radius_km": {"psi_5": air_blast_5psi_radius_km},
                "ejecta_blanket_radius_km": ejecta_blanket_radius_km
            },
            "human_impact_assessment": {
                "analysis_location": {"latitude": lat, "longitude": lon},
                "estimated_population_in_burn_radius": affected_population
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred during calculation: {e}"}), 400
    
@app.route('/ai_analyze', methods=['POST'])
def ai_analyze():
    if not GEMINI_API_KEY:
        # If the key is missing, return an informative JSON response instead of a 500 error
        return jsonify({"error": "Gemini API key not configured on the server."}), 500
    
    try:
        # 1. Get simulation data
        data = request.json
        if not data:
            return jsonify({"error": "No data found for analysis."}), 400

        # 2. Convert incoming data into readable text (Prompt)
        prompt = (
            "Below are data from an asteroid impact simulation. "
            "Write a clear, scientific and public-focused analysis in plain text (no HTML, no bold, single paragraph) "
            "under 150 words. State the impact energy in Megatons and crater diameter in km. "
            "Summarize probable effects on atmosphere, thermal/pressure damage, and seismic impact (estimate Richter magnitude if possible). "
            "Give a brief, practical risk assessment: affected radius, likely casualties/infrastructure impact (qualitative), and 2 short recommended public actions (what authorities and civilians should do). "
            f"\n\nSimulation Data: {data}"
        )


        # 3. Gemini API call
        client = genai.Client(api_key=GEMINI_API_KEY) # Use the key
        
        # Using the model with higher free limits (gemini-2.5-flash-lite)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7, # Increases creativity
                max_output_tokens=300 # Limits response length
            )
        )

        # 4. Send the response back
        return jsonify({"ai_analysis": response.text})

    except Exception as e:
        # This section handles quota, connection, or key errors during the API call.
        print(f"Gemini Analysis Error: {e}")
        return jsonify({"error": f"An error occurred during AI analysis: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
