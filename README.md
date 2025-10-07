## MeteorViz: AI-Driven Near-Earth Object Impact Simulation and Risk Assessment Platform

-----

### Abstract

**MeteorViz** is a comprehensive, open-source tool developed for the NASA Space Apps Challenge to enhance global **Planetary Defense** capabilities. Leveraging **NASA's open NEO data** and combining it with advanced Machine Learning (ML) and Large Language Model (LLM) technologies, the platform provides rapid, geo-localized simulations and actionable risk assessments for potential Near-Earth Object (NEO) impact scenarios. The core innovation lies in training an ML model on physically simulated data to achieve near-instantaneous impact prediction (crater diameter), which is then processed by the **Google Gemini API** to generate plain-language, emergency-response-ready reports for authorities and the public.

-----

### I. Project Objective (Aim)

The primary objective of MeteorViz is to bridge the gap between complex scientific data (NEO characteristics, orbital mechanics, impact physics) and practical, public-facing risk communication.

1.  **Democratize Planetary Defense Data:** Transform raw NASA NEO data into an accessible, interactive web visualization.
2.  **Enable Rapid Simulation:** Develop an accurate Machine Learning model that can predict the key impact outcome—crater diameter—in real-time, overcoming the computational limits of traditional hydrocode simulations.
3.  **Provide Actionable Intelligence:** Utilize advanced AI (Gemini) to convert technical simulation results (e.g., energy in Megatons, crater size) into understandable public safety guidance and comprehensive risk reports suitable for emergency management organizations.
4.  **Promote Open Science:** Create a fully open-source system, including the synthetic dataset and trained ML model, adhering to the NASA Space Apps theme of **"Explore Open Science Together."**

-----

### II. Methodology and Solution Architecture

The MeteorViz system is built on a robust, three-stage architecture: **Data Synthesis**, **Machine Learning Modeling**, and **AI-Enhanced Application (Web Platform)**.

#### 1\. Data Synthesis and Modeling Pipeline (`create_dataset_from_nasa.py`, `train_model.py`)

  * **Data Acquisition:** A Python script (`create_dataset_from_nasa.py`) queries the NASA NeoWs (Near-Earth Object Web Service) API to retrieve real orbital and physical data (diameter, velocity) for hundreds of known NEOs.
  * **Physics-Based Augmentation:** Since critical impact parameters (mass, density, angle) are often missing or assumed, the script **synthetically generates** this information based on accepted physical ranges and statistical distributions (e.g., assigning rock, ice, or iron composition).
  * **Crater Calculation:** The augmented data is passed through established physical cratering models to calculate the ground truth **Crater Diameter** (`crater_diameter_m`) for the resulting `nasa_impact_dataset.csv`.
  * **Machine Learning Training:** The synthetic dataset is used to train a **Gradient Boosting Regressor (GBR)** model (`train_model.py`). GBR was chosen for its high accuracy in regression tasks and ability to handle non-linear relationships between impact features (Mass, Velocity, Angle, Density) and the target crater diameter. The trained model is saved as `impact_model.pkl`.

#### 2\. AI-Enhanced Web Application (`app.py`, `index.html`)

  * **Core Logic (Flask Backend):** The `app.py` Flask application serves as the API. It handles user requests, retrieves real-time NEO data from NASA, loads the pre-trained `impact_model.pkl` for rapid simulation, and integrates external services.
  * **Real-time Simulation:** When a user selects a potential impactor, the application feeds the NEO's characteristics into the local, pre-trained ML model, instantly generating a highly accurate prediction for crater diameter and impact energy.
  * **LLM Integration (Gemini API):** The simulation results (numerical data: coordinates, energy, crater diameter) are passed to the **Gemini API** via a detailed prompt. The prompt instructs the model to act as a **risk analyst** and generate a structured, non-technical risk report that includes:
      * Summary of immediate impact effects (thermal, pressure, seismic).
      * Estimated affected radius and qualitative casualty/infrastructure impact.
      * Two clear, prioritized public action recommendations.
  * **Visualization and User Experience:** The web interface (`index.html`, `style.css`) uses **Leaflet.js** for an interactive map showing the predicted impact location and the hazard zone radius, and **Three.js** for a modern, engaging space-themed background.

-----

### III. Key Technologies and Stack

| Component | Technology | Role in Project |
| :--- | :--- | :--- |
| **Backend/API** | **Python**, **Flask** | Core application logic, routing, and serving the predictive model. |
| **Machine Learning** | **Scikit-learn** (Gradient Boosting Regressor), **Pandas** | Training the crater prediction model and managing the synthetic dataset. |
| **Artificial Intelligence** | **Google Gemini API** (gemini-2.5-flash-lite) | Advanced translation of numerical simulation data into actionable, plain-language risk reports and public guidance. |
| **Data Source** | **NASA Open APIs** (NeoWs) | Accessing real-time, official Near-Earth Object data. |
| **Frontend/Visualization** | **HTML5**, **Tailwind CSS**, **JavaScript** | User interface, responsiveness, and modern design. |
| **Geo-Visualization** | **Leaflet.js** | Interactive mapping to display impact locations and hazard radii on Earth. |
| **Atmosphere/Aesthetics** | **Three.js** | Creating the dynamic, immersive space nebula background. |

-----

### IV. Implementation Guide for Setup (Setup Instructions)

This guide assumes a Unix-like environment (Linux/macOS) but is adaptable to Windows.

#### Prerequisites

1.  **Python:** Ensure Python 3.8+ is installed.
2.  **API Keys:**
      * **NASA API Key:** Obtain one from NASA's API website.
      * **Gemini API Key:** Obtain one from Google AI Studio.
3.  **Environment Variables:** Create a file named `.env` in the root directory to store your keys:
    ```
    NASA_API_KEY="YOUR_NASA_KEY_HERE"
    GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
    ```

#### Step 1: Clone Repository and Setup Environment

```bash
# Assuming the project files are in the current directory
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # (Dependencies: flask, numpy, pandas, scikit-learn, joblib, requests, python-dotenv, google-genai, rasterio, geopandas, shapely)
```

#### Step 2: Generate the Synthetic Dataset

The training data must be created by fetching real NASA data and applying the physical modeling assumptions.

```bash
python create_dataset_from_nasa.py
# This script will take a few minutes to complete and will output:
# nasa_impact_dataset.csv
```

#### Step 3: Train the Machine Learning Model

Use the generated CSV file to train the Gradient Boosting Regressor model.

```bash
python train_model.py
# This script will print the model's performance (e.g., MAE, R2 Score) and save:
# impact_model.pkl
```

#### Step 4: Run the Web Application

The Flask server will host the API and the web interface.

```bash
python app.py
# The application will start, typically accessible at:
# http://127.0.0.1:5000/
```

#### Step 5: Access the Platform

Open a web browser and navigate to the local host address. The application is now fully operational, utilizing the NASA API for object lookup, the local ML model for simulation, and the Gemini API for intelligent risk analysis.

-----

### Conclusion and Future Work

MeteorViz successfully demonstrates the potent synergy between open science data (NASA NEOs), traditional physical modeling, and cutting-edge generative AI. By creating a fast, AI-powered predictive layer, we deliver a tool that enhances global situational awareness and supports decision-making in high-pressure planetary defense scenarios.

**Future Enhancements:**

1.  **High-Resolution Terrain Data:** Integrate NASA Shuttle Radar Topography Mission (SRTM) data to more accurately model terrain effects (e.g., airburst vs. ground impact).
2.  **Advanced ML Models:** Experiment with Neural Networks (TensorFlow/PyTorch) for potentially higher prediction accuracy on the synthetic dataset.
3.  **Real-Time Dashboard:** Develop a public dashboard to track the top 10 highest-risk NEOs as identified by the NASA Sentry table, visualized via MeteorViz.
