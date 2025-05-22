# coding: utf-8

import streamlit as st
import geemap.foliumap as geemap # THIS IS THE CRITICAL CHANGE: Explicitly import foliumap
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit_folium
ee.Initialize()
# --- Set Streamlit Page Configuration FIRST ---
st.set_page_config(layout="wide")

# --- Initialize session state for coordinates and map interactions (after set_page_config) ---
if 'lat' not in st.session_state:
    st.session_state.lat = 39.7  # Default initial latitude
if 'lon' not in st.session_state:
    st.session_state.lon = 78.5  # Default initial longitude
if 'marker_geojson' not in st.session_state: # For displaying marker on map
    st.session_state.marker_geojson = None
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# --- Placeholder for Rosetta if the actual library is not found ---
# --- Remove or comment out this section if you have the rosetta library ---
try:
    from rosetta import rosetta, SoilData
    st.sidebar.success("Rosetta library loaded successfully.")
except ImportError:
    st.sidebar.warning("Rosetta library not found. Using a placeholder. Hydraulic parameters will be random.")
    def rosetta_placeholder(model_version, soil_data_obj):
        num_samples = soil_data_obj.data.shape[0]
        # theta_r, theta_s, log10_alpha, log10_n, log10_Ksat
        mean = np.random.rand(num_samples, 5)
        mean[:, 0] = np.random.uniform(0.01, 0.1, num_samples) # theta_r
        mean[:, 1] = np.random.uniform(0.3, 0.6, num_samples) # theta_s
        mean[:, 2] = np.log10(np.random.uniform(0.001, 0.1, num_samples)) # log10_alpha
        mean[:, 3] = np.log10(np.random.uniform(1.1, 3.0, num_samples))   # log10_n
        mean[:, 4] = np.log10(np.random.uniform(1, 100, num_samples))   # log10_Ksat
        # Ensure theta_s > theta_r
        for i in range(num_samples):
            if mean[i,1] <= mean[i,0]:
                mean[i,1] = mean[i,0] + np.random.uniform(0.1, 0.3)
        stdev = np.random.rand(num_samples, 5) * 0.1
        codes = np.random.randint(0, 2, num_samples) # Simulate success codes
        return mean, stdev, codes

    class SoilDataPlaceholder:
        def __init__(self, data_array):
            self.data = data_array
        @staticmethod
        def from_array(data_array):
            return SoilDataPlaceholder(data_array)

    rosetta = rosetta_placeholder
    SoilData = SoilDataPlaceholder
# --- End of Placeholder ---


# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    st.error(f"Failed to initialize Earth Engine: {e}. Please ensure EE is authenticated.")
    st.stop()

# --- Earth Engine Data Asset Paths ---
# These are standard SoilGrids asset IDs. User might need to change these.
# Clay content (g/kg)
clay_0 = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
clay_10 = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b10')
clay_30 = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b30')
clay_60 = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b60')
clay_100 = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b100')
clay_200 = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b200')

sand_0 = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
sand_10 = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b10')
sand_30 = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b30')
sand_60 = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b60')
sand_100 = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b100')
sand_200 = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b200')

silt_0 = ee.Image(100).subtract(clay_0.add(sand_0))
silt_10 = ee.Image(100).subtract(clay_10.add(sand_10))
silt_30 = ee.Image(100).subtract(clay_30.add(sand_30))
silt_60 = ee.Image(100).subtract(clay_60.add(sand_60))
silt_100 = ee.Image(100).subtract(clay_100.add(sand_100))
silt_200 = ee.Image(100).subtract(clay_200.add(sand_200))

sbd_0 = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0').divide(100)
sbd_10 = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b10').divide(100)
sbd_30 = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b30').divide(100)
sbd_60 = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b60').divide(100)
sbd_100 = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b100').divide(100)
sbd_200 = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b200').divide(100)

fc_0 = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b0')
fc_10 = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b10')
fc_30 = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b30')
fc_60 = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b60')
fc_100 = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b100')
fc_200 = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b200')

# Helper function to safely get EE value
def get_ee_value(image, point, band_name_original, default_value=None):
    try:
        reduction = image.reduceRegion(
            reducer=ee.Reducer.firstNonNull(),
            geometry=point,
            scale=250  # SoilGrids native resolution
        )
        # GEE server-side objects can't be directly checked for emptiness with 'if not val:'
        # .getInfo() brings it client-side.
        val = reduction.getInfo()

        if val and band_name_original in val and val[band_name_original] is not None:
            return val[band_name_original]
        # Fallback for single band images or if band name is 'constant'
        elif val:
            # Check if the primary band name of the image (could be 'mean' before rename) is in the result
            img_band_names = image.bandNames().getInfo()
            if img_band_names and img_band_names[0] in val and val[img_band_names[0]] is not None:
                 return val[img_band_names[0]]
            if 'constant' in val and val['constant'] is not None: # For silt as in user code
                 return val['constant']

        # If still not found, issue a warning and return default
        asset_id_str = image.id().getInfo() if image.id() else 'Unknown Image'
        st.warning(f"Data not found for band '{band_name_original}' in asset '{asset_id_str}' at point. Using default: {default_value}.")
        return default_value
    except Exception as e:
        asset_id_str = image.id().getInfo() if image.id() else 'Unknown Image'
        st.warning(f"Error extracting data for asset '{asset_id_str}': {str(e)}. Using default: {default_value}.")
        return default_value

# Extract soil parameters function
def extract_soil_parameters(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    default_val_texture = 33.3 # Default for texture if null
    default_val_bd = 1.35       # Default for BD if null
    default_val_fc = 0.25       # Default for FC if null

    # Clay, Sand, Silt from SoilGrids are g/kg -> % (divide by 10 or * 0.1)
    clay_values = [
        get_ee_value(clay_0, point, 'b0', default_val_texture*10) * 0.1, get_ee_value(clay_10, point, 'b10', default_val_texture*10) * 0.1,
        get_ee_value(clay_30, point, 'b30', default_val_texture*10) * 0.1, get_ee_value(clay_60, point, 'b60', default_val_texture*10) * 0.1,
        get_ee_value(clay_100, point, 'b100', default_val_texture*10) * 0.1, get_ee_value(clay_200, point, 'b200', default_val_texture*10) * 0.1
    ]
    sand_values = [
        get_ee_value(sand_0, point, 'b0', default_val_texture*10) * 0.1, get_ee_value(sand_10, point, 'b10', default_val_texture*10) * 0.1,
        get_ee_value(sand_30, point, 'b30', default_val_texture*10) * 0.1, get_ee_value(sand_60, point, 'b60', default_val_texture*10) * 0.1,
        get_ee_value(sand_100, point, 'b100', default_val_texture*10) * 0.1, get_ee_value(sand_200, point, 'b200', default_val_texture*10) * 0.1
    ]
    silt_values = [
        get_ee_value(silt_0, point, 'constant', default_val_texture*10) * 0.1, get_ee_value(silt_10, point, 'constant', default_val_texture*10) * 0.1,
        get_ee_value(silt_30, point, 'constant', default_val_texture*10) * 0.1, get_ee_value(silt_60, point, 'constant', default_val_texture*10) * 0.1,
        get_ee_value(silt_100, point, 'constant', default_val_texture*10) * 0.1, get_ee_value(silt_200, point, 'constant', default_val_texture*10) * 0.1
    ]
    # Bulk Density (sbd/bdod) from SoilGrids is cg/cm³ -> g/cm³ (divide by 100 or * 0.01)
    sbd_values = [
        val if val is not None else default_val_bd for val in [
            get_ee_value(sbd_0, point, 'b0', default_val_bd*100) * 0.01,
            get_ee_value(sbd_10, point, 'b10', default_val_bd*100) * 0.01,
            get_ee_value(sbd_30, point, 'b30', default_val_bd*100) * 0.01,
            get_ee_value(sbd_60, point, 'b60', default_val_bd*100) * 0.01,
            get_ee_value(sbd_100, point, 'b100', default_val_bd*100) * 0.01,
            get_ee_value(sbd_200, point, 'b200', default_val_bd*100) * 0.01
        ]
    ]
    # Field Capacity (FC) / Water content at 33kPa from SoilGrids (projects/...) is cm³/dm³ (volumetric per mille) -> cm³/cm³ (divide by 1000 or * 0.001)
    fc_values = [
        val if val is not None else default_val_fc for val in [
            get_ee_value(fc_0, point, 'b0', default_val_fc*1000) * 0.001,
            get_ee_value(fc_10, point, 'b10', default_val_fc*1000) * 0.001,
            get_ee_value(fc_30, point, 'b30', default_val_fc*1000) * 0.001,
            get_ee_value(fc_60, point, 'b60', default_val_fc*1000) * 0.001,
            get_ee_value(fc_100, point, 'b100', default_val_fc*1000) * 0.001,
            get_ee_value(fc_200, point, 'b200', default_val_fc*1000) * 0.001
        ]
    ]


    # Normalize texture fractions to sum to 100%
    for i in range(len(sand_values)):
        # Handle cases where any texture component might be None from get_ee_value due to EE error
        s_i, si_i, c_i = sand_values[i], silt_values[i], clay_values[i]
        if any(v is None for v in [s_i, si_i, c_i]):
            st.warning(f"Texture data missing for layer {i}. Using fallback values (33.3% each).")
            sand_values[i], silt_values[i], clay_values[i] = 33.3, 33.3, 33.3
            continue

        current_sum = s_i + si_i + c_i
        if current_sum <= 0: # Avoid division by zero if all are zero or negative (unlikely)
            st.warning(f"Texture sum for layer {i} is {current_sum:.2f}%. Setting to default 33.3/33.3/33.3.")
            sand_values[i], silt_values[i], clay_values[i] = 33.3, 33.3, 33.3
        elif abs(current_sum - 100.0) > 1.0: # Allow minor deviation (e.g. 99.9 to 100.1)
            # st.info(f"Normalizing texture sum for layer {i} from {current_sum:.2f}% to 100%.")
            sand_values[i] = (s_i / current_sum) * 100
            silt_values[i] = (si_i / current_sum) * 100
            clay_values[i] = (c_i / current_sum) * 100


    data = {
        'Layer (cm)': [0, 10, 30, 60, 100, 200], # These are top depths of layers
        'Sand (%)': sand_values,
        'Silt (%)': silt_values,
        'Clay (%)': clay_values,
        'Bulk Density (g/cm³)': sbd_values,
        'Field Capacity (cm³/cm³)': fc_values
    }
    df = pd.DataFrame(data)
    return df

# Use Rosetta model, analyze, and plot
def get_soil_parameters_and_plot_streamlit(lat, lon):
    with st.spinner(f"Extracting GEE soil data for Lat: {lat:.4f}, Lon: {lon:.4f}..."):
        soil_data_df = extract_soil_parameters(lat, lon)

    st.subheader("Soil Parameters (Input to Rosetta)")
    st.dataframe(soil_data_df.style.format("{:.2f}"))

    if soil_data_df.isnull().values.any():
        st.error("Soil data contains missing (NaN) values after GEE extraction. Cannot proceed with Rosetta accurately.")
        st.dataframe(soil_data_df[soil_data_df.isnull().any(axis=1)])
        st.warning("Attempting to fill NaN with 0 for Rosetta. Results will be affected.")
        soil_data_df = soil_data_df.fillna(0) # Simplistic fill, better to use means or stop

    # Placeholder for Wilting Point (cm³/cm³). User might want to fetch this from GEE too.
    # e.g. from ee.Image("projects/soilgrids-isric/wwp_mean_1500kpa_0-5cm") etc. and scale by 0.001
    wilting_point_placeholder = np.repeat(0.1, 6)

    # Prepare data array for Rosetta
    # Model 1 in original code took 6 columns: Sand, Silt, Clay, BD, FC, WP
    try:
        rosetta_input_data = np.array([
            soil_data_df['Sand (%)'].astype(float),
            soil_data_df['Silt (%)'].astype(float),
            soil_data_df['Clay (%)'].astype(float),
            soil_data_df['Bulk Density (g/cm³)'].astype(float),
            soil_data_df['Field Capacity (cm³/cm³)'].astype(float), # FIXED: Removed extra characters
            wilting_point_placeholder.astype(float) # Assumed Wilting Point
        ]).T
    except Exception as e:
        st.error(f"Error preparing data for Rosetta: {e}")
        st.write("Soil Dataframe:")
        st.dataframe(soil_data_df)
        return

    if np.isnan(rosetta_input_data).any():
        st.error("NaN values detected in data prepared for Rosetta, even after attempted fill. Aborting Rosetta.")
        st.dataframe(pd.DataFrame(rosetta_input_data, columns=['Sand', 'Silt', 'Clay', 'BD', 'FC', 'WP_est']))
        return

    try:
        with st.spinner("Running Rosetta model..."):
            soil_data_obj = SoilData.from_array(rosetta_input_data)
            # Using model 1 as per user's original code, assuming it takes 6 inputs
            mean, stdev, codes = rosetta(1, soil_data_obj)
    except Exception as e:
        st.error(f"Error during Rosetta model execution: {e}")
        st.write("Data passed to Rosetta:")
        st.dataframe(pd.DataFrame(rosetta_input_data, columns=['Sand', 'Silt', 'Clay', 'BD', 'FC', 'WP_est']))
        return

    theta_r = mean[:, 0]      # Residual volumetric water content (cm³/cm³)
    theta_s = mean[:, 1]      # Saturated volumetric water content (cm³/cm³)
    log10_alpha = mean[:, 2]  # log10(alpha) (alpha in cm⁻¹)
    log10_n = mean[:, 3]      # log10(n) (n dimensionless)
    log10_Ksat = mean[:, 4]   # log10(Ksat) (Ksat in cm/day or user's Rosetta unit)

    inferred_data = {
        'Layer (cm)': soil_data_df['Layer (cm)'],
        'θr (cm³/cm³)': theta_r,
        'θs (cm³/cm³)': theta_s,
        'α (cm⁻¹)': 10**log10_alpha,
        'n (dim)': 10**log10_n,
        'log10(Ksat)': log10_Ksat
    }
    inferred_df = pd.DataFrame(inferred_data)

    st.subheader("Hydraulic Parameters (Output from Rosetta)")
    st.dataframe(inferred_df.style.format({
        "θr (cm³/cm³)" : "{:.3f}", "θs (cm³/cm³)" : "{:.3f}",
        "α (cm⁻¹)" : "{:.4f}", "n (dim)" : "{:.2f}", "log10(Ksat)": "{:.2f}"
    }))

    # Plotting Soil Water Retention Curves
    st.subheader("Soil Water Retention Curves (SWRC)")
    fig, ax = plt.subplots(figsize=(10, 7))
    # Pressure head (psi) from pF -1 (0.1 cm, near saturation) to pF 7 (10,000,000 cm, oven dry)
    psi_cm = np.logspace(-1, 7, 200)

    for i in range(len(soil_data_df)):
        tr, ts = theta_r[i], theta_s[i]
        alpha_val, n_val = 10**log10_alpha[i], 10**log10_n[i]

        # Basic checks for physical plausibility before plotting
        if not (np.isfinite(tr) and np.isfinite(ts) and np.isfinite(alpha_val) and np.isfinite(n_val)):
            st.caption(f"Layer {soil_data_df['Layer (cm)'][i]} cm: Skipping plot due to non-finite parameters.")
            continue
        if ts <= tr :
            st.caption(f"Layer {soil_data_df['Layer (cm)'][i]} cm: Skipping plot (θs ≤ θr). θs={ts:.3f}, θr={tr:.3f}")
            continue
        if n_val <= 1:
            st.caption(f"Layer {soil_data_df['Layer (cm)'][i]} cm: Skipping plot (n ≤ 1). n={n_val:.2f}")
            continue
        if alpha_val <= 0:
            st.caption(f"Layer {soil_data_df['Layer (cm)'][i]} cm: Skipping plot (α ≤ 0). α={alpha_val:.4f}")
            continue

        m_vg = 1 - (1 / n_val)
        theta_psi = tr + (ts - tr) / (1 + (alpha_val * psi_cm)**n_val)**m_vg

        # Define layer depth strings for labels
        current_depth = soil_data_df['Layer (cm)'][i]
        if i + 1 < len(soil_data_df['Layer (cm)']):
            next_depth = soil_data_df['Layer (cm)'][i+1]
            layer_label = f"{current_depth}-{next_depth} cm"
        else: # Last layer, e.g. 200 cm refers to 100-200 cm, or 200+
            # Assuming 'Layer (cm)' are top depths and standard SoilGrids depths used
            if current_depth == 0: layer_label = "0-10 cm"
            elif current_depth == 10: layer_label = "10-30 cm"
            elif current_depth == 30: layer_label = "30-60 cm"
            elif current_depth == 60: layer_label = "60-100 cm"
            elif current_depth == 100: layer_label = "100-200 cm"
            elif current_depth == 200: layer_label = "200+ cm" # From original code's implication
            else: layer_label = f"{current_depth}+ cm" # Fallback

        ax.plot(theta_psi, np.log10(psi_cm), label=layer_label)

    ax.set_ylabel('log₁₀(Pressure Head / cm)  |  pF')
    ax.set_xlabel('Volumetric Water Content (θ / cm³cm⁻³)')
    # Dynamic x-axis limit based on max theta_s, with a minimum sensible range
    max_thetas = inferred_df['θs (cm³/cm³)'].max() if not inferred_df.empty and 'θs (cm³/cm³)' in inferred_df else 0.5
    ax.set_xlim(0, max(0.5, max_thetas * 1.1) if pd.notna(max_thetas) else 0.5)
    ax.set_ylim(-1, 7) # pF range
    ax.set_title('Soil Water Retention Curves')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Soil Layer")
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust for external legend
    st.pyplot(fig)

# --- Streamlit App UI ---
st.title("🛰️ Soil Hydraulic Properties Analyzer")
st.markdown("""
    Extract soil properties from Google Earth Engine (SoilGrids), estimate hydraulic parameters
    using the Rosetta model, and visualize Soil Water Retention Curves (SWRC).
""")

# Sidebar for inputs
with st.sidebar:
    st.header("📍 Input Coordinates")

    # Manual Lat/Lon input
    # Use st.session_state values to ensure consistency after map click
    manual_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                 value=st.session_state.lat, step=0.0001, format="%.4f", key="manual_lat_input")
    manual_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                 value=st.session_state.lon, step=0.0001, format="%.4f", key="manual_lon_input")

    # Update session state if manual input changes
    if manual_lat != st.session_state.lat:
        st.session_state.lat = manual_lat
    if manual_lon != st.session_state.lon:
        st.session_state.lon = manual_lon

    st.markdown("---")
    st.markdown("**OR** Click on the map (updates fields above):")

    # Create a geemap Map object centered on current coordinates
    # geemap.Map() creates an ipyleaflet map by default, but also works fine with streamlit_folium
    # as geemap's map objects are designed to be compatible.
    m = geemap.Map(center=[st.session_state.lat, st.session_state.lon], zoom=5, draw_export=False, search_control=False, layer_control=True)
    m.add_basemap('HYBRID')

    # Add a marker for the current session state coordinates
    st.session_state.marker_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [st.session_state.lon, st.session_state.lat]
                },
                "properties": {"name": "Selected Point"}
            }
        ]
    }
    m.add_geojson(st.session_state.marker_geojson, layer_name="Selected Location", style={'color': 'yellow', 'fillColor': 'yellow', 'opacity': 0.8, 'weight': 2})


    # Display map and capture click events using streamlit_folium.st_folium
    map_data = streamlit_folium.st_folium(m, height=350, key="main_map")

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        # Check if click is different from current session state to avoid unnecessary reruns/updates
        if abs(clicked_lat - st.session_state.lat) > 1e-5 or abs(clicked_lon - st.session_state.lon) > 1e-5:
            st.session_state.lat = clicked_lat
            st.session_state.lon = clicked_lon
            # Rerun to update number_input widgets and map marker
            st.experimental_rerun()

    st.markdown("---")
    # Button to trigger analysis
    if st.button("🚀 Analyze Soil Properties", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

    st.markdown("---")
    st.markdown("Developed with [geemap](https://geemap.org) & [Streamlit](https://streamlit.io). Soil data from [SoilGrids](https://soilgrids.org).")


# Main content area for results
if st.session_state.run_analysis:
    if st.session_state.lat is not None and st.session_state.lon is not None:
        st.success(f"Analyzing for Latitude: **{st.session_state.lat:.4f}**, Longitude: **{st.session_state.lon:.4f}**")
        get_soil_parameters_and_plot_streamlit(st.session_state.lat, st.session_state.lon)
    else:
        st.error("Please provide valid latitude and longitude.")
    st.session_state.run_analysis = False # Reset flag after analysis
elif not st.session_state.run_analysis and 'lat' in st.session_state : # Initial state or after input change but no analysis yet
     st.info(f"Current coordinates: Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f}. Click 'Analyze Soil Properties' in the sidebar.")