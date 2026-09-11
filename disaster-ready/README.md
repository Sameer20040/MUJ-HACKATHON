# DisasterReady

A production-ready disaster management platform as a Streamlit app with live data feeds.

## Features

- Live hazard monitoring from keyless APIs (USGS earthquakes, Open-Meteo weather/flood, GDACS global alerts)
- Multi-region support (India, United States, Global)
- Live tracking: user location + proximity alerts, simulated asset movement
- Advanced features: polygon centroids on India map, heatmaps, state boundaries, export capabilities
- Scenario generation from live data
- Stress testing of response plans
- Resource optimization recommendations
- GeoJSON and PDF export

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Data Sources

- **Earthquakes**: USGS FDSN (Global)
- **Weather**: Open-Meteo forecast (Global)
- **River/Flood**: Open-Meteo Flood API (Global)
- **Official Alerts**: GDACS (Global)

## Implementation Details

- Built with Streamlit, SQLAlchemy 2.0 ORM, and SQLite
- Async data ingestion with retry logic
- Background daemon threads for live data updates and asset tracking simulation
- Region derivation via coordinate bounding boxes
- Browser geolocation via streamlit-geolocation
- Simulated asset movement using shapely LineString interpolation
- Proximity alerting with optional SMTP email
- India state boundaries as GeoJSON overlay with tooltips and centroid markers
- Sensor reading heatmap via folium.plugins.HeatMap

## Project Structure

- `app.py`: Main Streamlit application
- `src/`: Source code (models, services, connectors, tracking)
- `seed_data/`: Initial data (sensor sites, India states GeoJSON)
- `requirements.txt`: Python dependencies

## License

MIT