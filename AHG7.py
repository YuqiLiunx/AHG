# coding: utf-8

import streamlit as st
import geemap.foliumap as geemap
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit_folium
import os # Import os for path handling
from datetime import datetime, timedelta # For generating dummy dates

# --- Initialize Earth Engine (should be done once globally) ---
try:
    ee.Initialize()
except Exception as e:
    st.error(f"Failed to initialize Earth Engine: {e}. 请检查您的Earth Engine认证。")
    st.stop()

# --- Set Streamlit Page Configuration FIRST ---
st.set_page_config(layout="wide", page_title="农田水盐运移与作物产量估算云平台")

# --- Initialize session state for coordinates and map interactions (after set_page_config) ---
if 'lat' not in st.session_state:
    st.session_state.lat = 34.0  # Default initial latitude (e.g., somewhere in China)
if 'lon' not in st.session_state:
    st.session_state.lon = 108.0  # Default initial longitude
if 'marker_geojson' not in st.session_state: # For displaying marker on map
    st.session_state.marker_geojson = None
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'run_phydrus_simulation' not in st.session_state:
    st.session_state.run_phydrus_simulation = False
if 'run_aquacrop_simulation' not in st.session_state: # New state for AquaCrop
    st.session_state.run_aquacrop_simulation = False
if 'inferred_df' not in st.session_state: # To store Rosetta results across reruns
    st.session_state.inferred_df = None


# --- Placeholder for Rosetta if the actual library is not found ---
try:
    from rosetta import rosetta, SoilData
    st.sidebar.success("Rosetta 库加载成功。")
except ImportError:
    st.sidebar.warning("Rosetta 库未找到。将使用占位符，水力参数将为随机值。")
    def rosetta_placeholder(model_version, soil_data_obj):
        num_samples = soil_data_obj.data.shape[0]
        mean = np.random.rand(num_samples, 5)
        mean[:, 0] = np.random.uniform(0.01, 0.1, num_samples) # theta_r
        mean[:, 1] = np.random.uniform(0.3, 0.6, num_samples) # theta_s
        mean[:, 2] = np.log10(np.random.uniform(0.001, 0.1, num_samples)) # log10_alpha
        mean[:, 3] = np.log10(np.random.uniform(1.1, 3.0, num_samples))   # log10_n
        mean[:, 4] = np.log10(np.random.uniform(1, 100, num_samples))   # log10_Ksat
        for i in range(num_samples):
            if mean[i,1] <= mean[i,0]:
                mean[i,1] = mean[i,0] + np.random.uniform(0.1, 0.3)
        stdev = np.random.rand(num_samples, 5) * 0.1
        codes = np.random.randint(0, 2, num_samples)
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

# --- Integration for phydrus ---
phydrus_exe_path = os.path.join(os.getcwd(), "hydrus") # Expect hydrus executable in the app root
phydrus_exe_available = os.path.exists(phydrus_exe_path) and os.access(phydrus_exe_path, os.X_OK)

try:
    import phydrus as ps # Assuming 'phydrus' folder is in the root directory
    st.sidebar.success("phydrus Python 库加载成功。")
    phydrus_python_lib_available = True
    if not phydrus_exe_available:
        st.sidebar.warning("警告: HYDRUS-1D 可执行文件 'hydrus' 未找到或不可执行。phydrus 模拟将被跳过或使用模拟数据。")
except ImportError as e:
    st.sidebar.warning(f"phydrus Python 库未找到: {e}。土壤水分运动模拟功能已禁用。")
    phydrus_python_lib_available = False
    # Define a dummy ps object to avoid NameError if not imported
    class DummyPs:
        def Model(self, *args, **kwargs):
            return None
        def get_empty_material_df(self, n):
            return pd.DataFrame()
        def create_profile(self, h):
            return pd.DataFrame()
    ps = DummyPs()
# --- End of Integration for phydrus ---

# --- Integration for AQUACROP-OSPy ---
try:
    import aquacrop as ac
    st.sidebar.success("AQUACROP-OSPy 库加载成功。")
    aquacrop_available = True
except ImportError as e:
    st.sidebar.warning(f"AQUACROP-OSPy 库未找到: {e}。作物产量模拟功能已禁用。")
    aquacrop_available = False
    # Define a dummy ac object to avoid NameError if not imported
    class DummyAc:
        def Soil(self, *args, **kwargs): return None
        def Crop(self, *args, **kwargs): return None
        def Weather(self, *args, **kwargs): return None
        def InitialWaterContent(self, *args, **kwargs): return None
        def IrrigationManagement(self, *args, **kwargs): return None
        def FieldManagement(self, *args, **kwargs): return None
        def GroundWater(self, *args, **kwargs): return None
        def Simulation(self, *args, **kwargs): return None
        def AquaCropModel(self, *args, **kwargs): return None
        def create_soil_profile(self, *args, **kwargs): return None
        def create_crop_model(self, *args, **kwargs): return None
        def create_weather_file(self, *args, **kwargs): return None

    ac = DummyAc()
# --- End of Integration for AQUACROP-OSPy ---


# --- Earth Engine Data Asset Paths ---
# 使用try-except包装，以防EE初始化失败后仍尝试访问
try:
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
except Exception as e:
    st.error(f"加载 Earth Engine 数据资产失败: {e}. 请确认您的EE环境已正确设置。")
    # 如果EE初始化失败，这里就不用st.stop()了，因为上面已经停止了


# Helper function to safely get EE value
def get_ee_value(image, point, band_name_original, default_value=None):
    try:
        reduction = image.reduceRegion(
            reducer=ee.Reducer.firstNonNull(),
            geometry=point,
            scale=250
        )
        val = reduction.getInfo()

        if val and band_name_original in val and val[band_name_original] is not None:
            return val[band_name_original]
        elif val:
            img_band_names = image.bandNames().getInfo()
            if img_band_names and img_band_names[0] in val and val[img_band_names[0]] is not None:
                 return val[img_band_names[0]]
            if 'constant' in val and val['constant'] is not None:
                 return val['constant']

        asset_id_str = image.id().getInfo() if image.id() else '未知图像'
        st.warning(f"在资产 '{asset_id_str}' 中未找到波段 '{band_name_original}' 的数据。使用默认值: {default_value}。")
        return default_value
    except Exception as e:
        asset_id_str = image.id().getInfo() if image.id() else '未知图像'
        st.warning(f"提取资产 '{asset_id_str}' 数据时出错: {str(e)}。使用默认值: {default_value}。")
        return default_value

# Extract soil parameters function
def extract_soil_parameters(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    default_val_texture = 33.3
    default_val_bd = 1.35
    default_val_fc = 0.25

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
        s_i, si_i, c_i = sand_values[i], silt_values[i], clay_values[i]
        if any(v is None for v in [s_i, si_i, c_i]):
            st.warning(f"第 {i} 层纹理数据缺失。使用默认值 (沙、粉、黏土各 33.3%)。")
            sand_values[i], silt_values[i], clay_values[i] = 33.3, 33.3, 33.3
            continue

        current_sum = s_i + si_i + c_i
        if current_sum <= 0:
            st.warning(f"第 {i} 层纹理总和为 {current_sum:.2f}%。设置为默认值 33.3/33.3/33.3。")
            sand_values[i], silt_values[i], clay_values[i] = 33.3, 33.3, 33.3
        elif abs(current_sum - 100.0) > 1.0:
            sand_values[i] = (s_i / current_sum) * 100
            silt_values[i] = (si_i / current_sum) * 100
            clay_values[i] = (c_i / current_sum) * 100

    data = {
        '深度 (cm)': [0, 10, 30, 60, 100, 200],
        '沙粒含量 (%)': sand_values,
        '粉粒含量 (%)': silt_values,
        '黏粒含量 (%)': clay_values,
        '容重 (g/cm³)': sbd_values,
        '田间持水量 (cm³/cm³)': fc_values
    }
    df = pd.DataFrame(data)
    return df

# Use Rosetta model, analyze, and plot
def get_soil_parameters_and_plot_streamlit(lat, lon):
    with st.spinner(f"正在从 Google Earth Engine 提取土壤数据 (纬度: {lat:.4f}, 经度: {lon:.4f})..."):
        soil_data_df = extract_soil_parameters(lat, lon)

    st.subheader("📊 土壤原始属性数据")
    st.caption("来自 SoilGrids 的土壤纹理、容重和田间持水量数据。")
    st.dataframe(soil_data_df.style.format("{:.2f}"), use_container_width=True)

    if soil_data_df.isnull().values.any():
        st.error("GEE 提取后土壤数据包含缺失 (NaN) 值。无法准确进行 Rosetta 估计。")
        st.dataframe(soil_data_df[soil_data_df.isnull().any(axis=1)], use_container_width=True)
        st.warning("尝试用 0 填充 NaN 以供 Rosetta 使用。请注意，这将影响结果的准确性。")
        soil_data_df = soil_data_df.fillna(0)

    wilting_point_placeholder = np.repeat(0.1, 6) # Using placeholder for now

    try:
        rosetta_input_data = np.array([
            soil_data_df['沙粒含量 (%)'].astype(float),
            soil_data_df['粉粒含量 (%)'].astype(float),
            soil_data_df['黏粒含量 (%)'].astype(float),
            soil_data_df['容重 (g/cm³)'].astype(float),
            soil_data_df['田间持水量 (cm³/cm³)'].astype(float),
            wilting_point_placeholder.astype(float)
        ]).T
    except Exception as e:
        st.error(f"为 Rosetta 模型准备数据时出错: {e}")
        st.write("当前土壤数据框:")
        st.dataframe(soil_data_df, use_container_width=True)
        return None # Return None if data prep fails

    if np.isnan(rosetta_input_data).any():
        st.error("为 Rosetta 准备的数据中检测到 NaN 值，即使尝试填充后也如此。中止 Rosetta 运行。")
        st.dataframe(pd.DataFrame(rosetta_input_data, columns=['沙粒', '粉粒', '黏粒', '容重', '田间持水量', '凋萎点_估算']), use_container_width=True)
        return None # Return None if NaN values persist

    try:
        with st.spinner("正在运行 Rosetta 模型估算水力参数..."):
            soil_data_obj = SoilData.from_array(rosetta_input_data)
            mean, stdev, codes = rosetta(1, soil_data_obj)
    except Exception as e:
        st.error(f"Rosetta 模型执行期间出错: {e}")
        st.write("传递给 Rosetta 的数据示例:")
        st.dataframe(pd.DataFrame(rosetta_input_data, columns=['沙粒', '粉粒', '黏粒', '容重', '田间持水量', '凋萎点_估算']), use_container_width=True)
        return None # Return None if Rosetta fails

    theta_r = mean[:, 0]
    theta_s = mean[:, 1]
    log10_alpha = mean[:, 2]
    log10_n = mean[:, 3]
    log10_Ksat = mean[:, 4]

    inferred_data = {
        '深度 (cm)': soil_data_df['深度 (cm)'],
        '残余含水量 (θr, cm³/cm³)': theta_r,
        '饱和含水量 (θs, cm³/cm³)': theta_s,
        '进气值 (α, cm⁻¹)': 10**log10_alpha,
        '孔隙分布指数 (n, 无量纲)': 10**log10_n,
        '饱和导水率 (log10(Ksat), log10(cm/天))': log10_Ksat,
        '田间持水量 (FC, cm³/cm³)' : soil_data_df['田间持水量 (cm³/cm³)'] # Re-add FC for AQUACROP-OSPy
    }
    inferred_df = pd.DataFrame(inferred_data)

    st.subheader("💧 土壤水力参数 (Rosetta 输出)")
    st.caption("基于土壤质地、容重和田间持水量估算的范·格尼腾 (Van Genuchten) 参数。")
    st.dataframe(inferred_df.style.format({
        "残余含水量 (θr, cm³/cm³)" : "{:.3f}", "饱和含水量 (θs, cm³/cm³)" : "{:.3f}",
        "进气值 (α, cm⁻¹)" : "{:.4f}", "孔隙分布指数 (n, 无量纲)" : "{:.2f}", "饱和导水率 (log10(Ksat), log10(cm/天))": "{:.2f}",
        "田间持水量 (FC, cm³/cm³)" : "{:.3f}"
    }), use_container_width=True)

    # Plotting Soil Water Retention Curves
    st.subheader("📈 土壤水分特征曲线 (SWRC)")
    st.caption("展示不同土壤层的水分含量与水压头（pF值）的关系。")
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted figure size for more space

    for i in range(len(soil_data_df)):
        tr, ts = theta_r[i], theta_s[i]
        alpha_val, n_val = 10**log10_alpha[i], 10**log10_n[i]

        if not (np.isfinite(tr) and np.isfinite(ts) and np.isfinite(alpha_val) and np.isfinite(n_val)):
            st.caption(f"层 {soil_data_df['深度 (cm)'][i]} cm: 由于参数非有限，跳过绘图。")
            continue
        if ts <= tr :
            st.caption(f"层 {soil_data_df['深度 (cm)'][i]} cm: 跳过绘图 (θs ≤ θr)。 θs={ts:.3f}, θr={tr:.3f}")
            continue
        if n_val <= 1:
            st.caption(f"层 {soil_data_df['深度 (cm)'][i]} cm: 跳过绘图 (n ≤ 1)。 n={n_val:.2f}")
            continue
        if alpha_val <= 0:
            st.caption(f"层 {soil_data_df['深度 (cm)'][i]} cm: 跳过绘图 (α ≤ 0)。 α={alpha_val:.4f}")
            continue

        psi_cm = np.logspace(-1, 7, 200) # Pressure head from 0.1 cm to 10^7 cm
        m_vg = 1 - (1 / n_val)
        theta_psi = tr + (ts - tr) / (1 + (alpha_val * psi_cm)**n_val)**m_vg

        current_depth = soil_data_df['深度 (cm)'][i]
        next_depth_index = i + 1
        if next_depth_index < len(soil_data_df['深度 (cm)']):
            next_depth = soil_data_df['深度 (cm)'][next_depth_index]
            layer_label = f"{current_depth}-{next_depth} cm"
        else:
            layer_label = f"{current_depth}+ cm" # For the last layer

        ax.plot(theta_psi, np.log10(psi_cm), label=layer_label)

    ax.set_ylabel('log₁₀(水压头 / cm)  |  pF')
    ax.set_xlabel('体积含水量 (θ / cm³cm⁻³)')
    max_thetas = inferred_df['饱和含水量 (θs, cm³/cm³)'].max() if not inferred_df.empty and '饱和含水量 (θs, cm³/cm³)' in inferred_df else 0.5
    ax.set_xlim(0, max(0.5, max_thetas * 1.1) if pd.notna(max_thetas) else 0.5)
    ax.set_ylim(-1, 7)
    ax.set_title('土壤水分特征曲线')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="土壤层") # 调整图例位置
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # 留出右侧空间给图例
    st.pyplot(fig)

    return inferred_df # Return inferred data for phydrus simulation

# --- Streamlit App UI ---
st.title("🛰️农田水盐运移与作物产量估算云平台")
st.markdown("""
    本平台旨在利用地球科学数据、水文模型及产量模型，为农业决策提供智能分析。
    功能包括：**遥感数据提取**、**土壤水力特性估算**、**水盐运移模拟**及**作物产量估算**。
""")

# Sidebar for general controls
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.markdown("---")
    st.markdown("**关于本平台**")
    st.info("""
        本应用集成了以下技术：
        - **Google Earth Engine**: 用于提取全球尺度的气候、土壤等环境数据。
        - **Rosetta**: 估算土壤水力参数。
        - **Hydrus**: 用于土壤水分运动和盐分运移模拟。
        - **AquaCrop**: 进行作物水分生产力模拟。
        \n开发基于 [geemap](https://geemap.org), [Streamlit](https://streamlit.io), [phydrus](https://github.com/phydrus/phydrus.git) & [AQUACROP-OSPy](https://aquacrop.github.io/aquacrop/)。土壤数据来源于 [SoilGrids](https://soilgrids.org)。
    """)
    st.markdown("---")
    # Add a reset button for convenience
    if st.button("🔄 重置应用状态", help="点击此按钮将清除所有模拟结果并重置输入。", use_container_width=True):
        for key in st.session_state.keys():
            if key.startswith('run_'): # Reset run flags
                st.session_state[key] = False
            elif key in ['inferred_df', 'marker_geojson']: # Reset data
                st.session_state[key] = None
        st.session_state.lat = 34.0
        st.session_state.lon = 108.0
        st.experimental_rerun()

# --- Main content area ---

# Section for Location Selection
with st.expander("📍 **选择分析地点 (点击地图或手动输入)**", expanded=True):
    st.markdown("通过**点击地图**选择地点，或在下方**手动输入**经纬度。")

    m = geemap.Map(center=[st.session_state.lat, st.session_state.lon], zoom=5, draw_export=False, search_control=True, layer_control=True)
    m.add_basemap('HYBRID')

    # Add marker for the selected location
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
    m.add_geojson(st.session_state.marker_geojson, layer_name="当前选择地点", style={'color': 'yellow', 'fillColor': 'yellow', 'opacity': 0.8, 'weight': 2})

    # Increased height for the map and added use_container_width=True
    map_data = streamlit_folium.st_folium(m, height=500, key="main_map", use_container_width=True)

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        if abs(clicked_lat - st.session_state.lat) > 1e-5 or abs(clicked_lon - st.session_state.lon) > 1e-5:
            st.session_state.lat = clicked_lat
            st.session_state.lon = clicked_lon
            st.toast(f"地点已更新至: 纬度 {clicked_lat:.4f}, 经度 {clicked_lon:.4f}！", icon="✅") # 及时反馈
            st.experimental_rerun() # Rerun to update the map marker and number inputs

    st.markdown("---")
    st.markdown("或者手动输入经纬度：")
    col1, col2 = st.columns(2)
    with col1:
        manual_lat = st.number_input("纬度", min_value=-90.0, max_value=90.0,
                                     value=st.session_state.lat, step=0.0001, format="%.4f",
                                     help="输入地理位置的纬度，范围在 -90 到 90 之间。", key="manual_lat_input")
    with col2:
        manual_lon = st.number_input("经度", min_value=-180.0, max_value=180.0,
                                     value=st.session_state.lon, step=0.0001, format="%.4f",
                                     help="输入地理位置的经度，范围在 -180 到 180 之间。", key="manual_lon_input")

    # Update session state if manual input changes
    if manual_lat != st.session_state.lat:
        st.session_state.lat = manual_lat
    if manual_lon != st.session_state.lon:
        st.session_state.lon = manual_lon
    
    # Check if manual input was directly changed without clicking map
    if st.button("更新地点并开始分析", type="primary", use_container_width=True, help="点击此按钮将使用当前输入的经纬度并开始土壤属性分析。"):
        if abs(manual_lat - st.session_state.lat) > 1e-5 or abs(manual_lon - st.session_state.lon) > 1e-5:
            st.session_state.lat = manual_lat
            st.session_state.lon = manual_lon
            st.toast(f"地点已手动更新至: 纬度 {st.session_state.lat:.4f}, 经度 {st.session_state.lon:.4f}！", icon="✅")
        st.session_state.run_analysis = True
        st.experimental_rerun() # Ensure analysis starts immediately

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📊 土壤属性分析 (Rosetta)", "💧 土壤水分模拟 (Hydrus-1D)", "🧂 盐分运移模拟 (待开发)", "🌾 作物产量模拟 (AQUACROP-OSPy)"])

with tab1:
    st.header("📊 土壤属性分析 (Rosetta)")
    if st.session_state.run_analysis:
        if st.session_state.lat is not None and st.session_state.lon is not None:
            st.info(f"正在为经纬度 **{st.session_state.lat:.4f}**, **{st.session_state.lon:.4f}** 进行土壤属性分析...")
            inferred_df = get_soil_parameters_and_plot_streamlit(st.session_state.lat, st.session_state.lon)
            if inferred_df is not None:
                st.session_state.inferred_df = inferred_df
                st.success("土壤属性分析和水力参数估算成功完成！请查看下方结果和图表。")
            else:
                st.session_state.inferred_df = None # Clear if analysis failed
                st.error("土壤属性分析失败。请检查输入或尝试其他地点。")
        else:
            st.error("请提供有效的经纬度以开始分析。")
        st.session_state.run_analysis = False # Reset flag after analysis
    elif st.session_state.inferred_df is not None:
        st.info("已完成土壤属性分析。您可以在下方查看上次分析的结果。")
        # Re-display results if already computed
        # Note: Calling this function again will re-extract data, which might be slow if not cached.
        # For a truly intelligent UI, if inferred_df exists, we should just display it directly
        # and skip the extraction/rosetta steps to improve responsiveness for re-tabbing.
        # For simplicity in this example, we re-run the plotting part.
        get_soil_parameters_and_plot_streamlit(st.session_state.lat, st.session_state.lon)
    else:
        st.info("请在 '选择分析地点' 区域选择地点或手动输入经纬度，然后点击 '更新地点并开始分析' 按钮。")


with tab2:
    st.header("💧 土壤水分模拟 (Hydrus-1D)")
    if not phydrus_python_lib_available:
        st.warning("phydrus Python 库未找到。土壤水分运动模拟功能已禁用。请确保已正确安装 `phydrus` 库。")
    elif st.session_state.inferred_df is None:
        st.warning("在运行土壤水分模拟之前，请先在 '土壤属性分析' 标签页中成功运行分析，以获取 Rosetta 参数。")
    else:
        st.info(f"""
            **重要提示：** 此功能依赖于外部的 `HYDRUS-1D` 可执行程序。
             `hydrus` 可执行文件需要在应用程序的根目录 (`{phydrus_exe_path}`) 并且**具有执行权限**。
        """)

        st.subheader("模拟参数设置")
        col_phydrus_1, col_phydrus_2 = st.columns(2)
        with col_phydrus_1:
            sim_duration_days = st.number_input("模拟持续时间 (天)", min_value=1, max_value=365*5, value=7, step=1,
                                                 help="模拟土壤水分运动的总天数。")
            initial_theta = st.slider("初始体积含水量 (θ)", min_value=0.05, max_value=0.55, value=0.3, step=0.01,
                                      help="模拟开始时土壤的初始平均体积含水量。")
        with col_phydrus_2:
            st.markdown("**上部边界条件**")
            upper_bc_type = st.radio("类型", ["恒定通量 (cm/天)", "大气边界"], index=0, key="phydrus_upper_bc_type",
                                     help="选择土壤顶部边界条件类型。恒定通量模拟持续降雨或灌溉；大气边界更复杂，考虑蒸发和降雨。")
            upper_bc_value = st.number_input("通量/降雨量 (cm/天)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                              help="恒定通量的大小，或大气边界条件下的降雨量（简化）。")

        st.markdown("**下部边界条件**")
        lower_bc_type = st.radio("类型 ", ["自由排水", "恒定水压头 (cm)"], index=0, key="phydrus_lower_bc_type",
                                 help="选择土壤底部边界条件类型。自由排水允许水分从底部流出；恒定水压头模拟地下水位。")
        lower_bc_value_pressure = -100.0 # Default for constant pressure head
        if lower_bc_type == "恒定水压头 (cm)":
             lower_bc_value_pressure = st.number_input("水压头 (cm)", min_value=-5000.0, max_value=0.0, value=-100.0, step=10.0,
                                                       help="恒定水压头边界条件下的水压头值，通常为负值（表示吸力），0 表示饱和。")

        if st.button("▶️ 运行土壤水分模拟", use_container_width=True, key="run_phydrus_btn"):
            st.session_state.run_phydrus_simulation = True

        if st.session_state.run_phydrus_simulation:
            st.subheader("模拟结果")
            inferred_df = st.session_state.inferred_df
            
            # Take parameters for the top layer (0-10cm) for simplicity, or you can average/layer
            theta_r_sim = inferred_df.iloc[0]['残余含水量 (θr, cm³/cm³)']
            theta_s_sim = inferred_df['饱和含水量 (θs, cm³/cm³)'].iloc[0]
            alpha_sim = inferred_df.iloc[0]['进气值 (α, cm⁻¹)'].clip(min=1e-5) # Ensure alpha is positive
            n_sim = inferred_df.iloc[0]['孔隙分布指数 (n, 无量纲)'].clip(min=1.1) # Ensure n > 1
            Ksat_sim = 10**inferred_df.iloc[0]['饱和导水率 (log10(Ksat), log10(cm/天))'] # Convert from log10(Ksat)

            st.markdown(f"**用于顶部土层 (0-{inferred_df.iloc[1]['深度 (cm)']}cm) 的水力参数:**")
            st.markdown(f"- 残余含水量 (θr): `{theta_r_sim:.3f}` cm³/cm³")
            st.markdown(f"- 饱和含水量 (θs): `{theta_s_sim:.3f}` cm³/cm³")
            st.markdown(f"- 进气值 (α): `{alpha_sim:.4f}` cm⁻¹")
            st.markdown(f"- 孔隙分布指数 (n): `{n_sim:.2f}` (无量纲)")
            st.markdown(f"- 饱和导水率 (Ksat): `{Ksat_sim:.2f}` cm/天")

            if not phydrus_exe_available:
                st.warning("由于 `HYDRUS-1D` 可执行文件未找到或不可执行，以下显示的是**模拟结果的占位符**，并非实际的 `phydrus` 模拟输出。")
                # Generate dummy results for visualization if HYDRUS-1D executable is not found
                sim_results_theta = np.zeros((sim_duration_days + 1, 20))
                sim_results_h = np.zeros((sim_duration_days + 1, 20))
                nodes = np.linspace(0, -50, 20) # Dummy depth for visualization

                for t_idx in range(sim_duration_days + 1):
                    # 模拟含水量随时间简单变化
                    if upper_bc_type == "恒定通量 (cm/天)":
                        # 简单模拟降雨/通量导致含水量增加
                        delta_theta_time = (upper_bc_value / 50) * (t_idx / sim_duration_days) # 随时间线性增加
                        sim_results_theta[t_idx, :] = initial_theta + delta_theta_time * np.exp(nodes/50) # 随深度指数衰减
                        sim_results_theta[t_idx, :] = np.clip(sim_results_theta[t_idx, :], theta_r_sim, theta_s_sim)
                    else: # For "大气边界", still using constant flux for demo
                        sim_results_theta[t_idx, :] = initial_theta

                    # 简单模拟从含水量到水压头的转换 (反向 Van Genuchten)
                    relative_theta = (sim_results_theta[t_idx, :] - theta_r_sim) / (theta_s_sim - theta_r_sim)
                    # 避免对数运算的零或负数
                    relative_theta = np.clip(relative_theta, 1e-6, 1 - 1e-6)
                    try:
                        m_vg_inv = 1 - (1 / n_sim)
                        pressure_head_mock = - ( ( ( (theta_s_sim - theta_r_sim) / (sim_results_theta[t_idx, :] - theta_r_sim) )**(1/m_vg_inv) - 1 )**(1/n_sim) ) / alpha_sim
                        pressure_head_mock[np.isinf(pressure_head_mock)] = -10000 # Cap extreme values
                        pressure_head_mock[np.isnan(pressure_head_mock)] = -10000 # Cap extreme values
                        sim_results_h[t_idx, :] = pressure_head_mock
                    except RuntimeWarning:
                        sim_results_h[t_idx, :] = -10000 # Set to dry if calculation fails

                fig_phydrus, ax_phydrus = plt.subplots(figsize=(10, 6))
                ax_phydrus.plot(np.mean(sim_results_theta, axis=1), np.arange(sim_duration_days + 1))
                ax_phydrus.set_xlabel('平均体积含水量 (θ)')
                ax_phydrus.set_ylabel('模拟天数')
                ax_phydrus.set_title('模拟土壤含水量随时间变化 (占位符)')
                ax_phydrus.grid(True)
                st.pyplot(fig_phydrus)
                st.caption("此图为示意图，仅在 HYDRUS-1D 可执行文件不可用时显示。")


            else:
                # --- Actual phydrus simulation logic (requires HYDRUS-1D executable) ---
                st.success("HYDRUS-1D 可执行文件已找到，尝试运行模拟...")
                try:
                    # Create a temporary workspace for HYDRUS-1D files
                    ws = "phydrus_workspace"
                    if not os.path.exists(ws):
                        os.makedirs(ws)
                    
                    ml = ps.Model(exe_name=phydrus_exe_path, ws_name=ws, name="model",
                                  description="Soil Water Movement Simulation from Streamlit",
                                  mass_units="mmol", time_unit="day", length_unit="cm")

                    ml.add_time_info(tinit=0, tmax=sim_duration_days, print_times=True, dt=0.01,
                                     dtmax=0.1, printinit=1)

                    m = ml.get_empty_material_df(n=1) # Single homogeneous layer for simplicity
                    # L - pore connectivity (default is 0.5)
                    m.loc[1] = [theta_r_sim, theta_s_sim, alpha_sim, n_sim, Ksat_sim, 0.5]
                    ml.add_material(m)

                    depth_sim = 100 # cm, example depth, can be adjusted dynamically based on layers
                    profile_nodes = np.linspace(0, -depth_sim, 50) # 50 nodes for 100 cm depth
                    
                    # Convert initial_theta to pressure head for profile initialization
                    try:
                        m_vg_init = 1 - (1 / n_sim)
                        if initial_theta <= theta_r_sim + 1e-6:
                            initial_pressure_head = -100000.0 # Very dry
                        elif initial_theta >= theta_s_sim - 1e-6:
                            initial_pressure_head = 0.0 # Saturated
                        else:
                            ratio_theta = (theta_s_sim - theta_r_sim) / (initial_theta - theta_r_sim)
                            # Ensure the term inside power is positive and not too close to zero
                            term_inside_power = (ratio_theta**(1/m_vg_init)) - 1
                            if term_inside_power <= 0:
                                initial_pressure_head = -100000.0 # Handle edge case
                            else:
                                initial_pressure_head = - ( (term_inside_power)**(1/n_sim) ) / alpha_sim
                        
                        initial_pressure_head = np.clip(initial_pressure_head, -10000.0, 0.0)
                    except Exception as e:
                        st.warning(f"无法计算初始水压头: {e}。默认为 -100 cm。")
                        initial_pressure_head = -100.0
                    
                    profile_df = ps.create_profile(h=initial_pressure_head, num_nodes=50, top=0, bot=depth_sim)
                    profile_df['Mat'] = 1 # All nodes assigned to material 1
                    ml.add_profile(profile_df)

                    # Water flow boundary conditions
                    if upper_bc_type == "恒定通量 (cm/天)":
                        # top_bc: 1=constant flux, 2=atmospheric bc, 3=variable flux, 4=free drainage
                        # bot_bc: 0=free drainage, 1=constant pressure head, 2=constant flux
                        ml.add_waterflow(top_bc=1, rtop=upper_bc_value)
                    elif upper_bc_type == "大气边界":
                        # Atmospheric boundary conditions are more complex and require more params (Precipitation, Evaporation, Root Water Uptake)
                        # For simplicity, we'll still use constant flux as a placeholder for rain for now.
                        st.warning("大气边界条件在 HYDRUS 中更复杂，需要更多参数。此处仍使用恒定通量作为示例。")
                        ml.add_waterflow(top_bc=1, rtop=upper_bc_value)
                    
                    if lower_bc_type == "自由排水":
                        ml.add_waterflow(top_bc=ml.waterflow.top_bc, bot_bc=0) # 0: free drainage
                    elif lower_bc_type == "恒定水压头 (cm)":
                        ml.add_waterflow(top_bc=ml.waterflow.top_bc, bot_bc=1, hbot=lower_bc_value_pressure) # 1: constant pressure head

                    ml.write_input()
                    
                    with st.spinner("正在运行 HYDRUS-1D 模拟...这可能需要一些时间，请耐心等待。"):
                        rs = ml.simulate() # This is the call that requires the external HYDRUS executable
                    
                    if rs.success:
                        st.success("HYDRUS-1D 模拟成功完成！")
                        df_tlevel = ml.read_tlevel()
                        
                        # Plotting Cumulative Bottom Flux (or other relevant output)
                        fig_sim_tlevel, ax_tlevel = plt.subplots(figsize=(10, 6))
                        if 'vBot[L/T]' in df_tlevel.columns:
                            ax_tlevel.plot(df_tlevel['Time'], df_tlevel['vBot[L/T]'], label='底部通量 (cm/天)', color='blue')
                            ax_tlevel.set_ylabel('底部通量 (cm/天)')
                            ax_tlevel.set_title('底部通量随时间变化')
                        else:
                            st.warning("在 HYDRUS 输出中未找到 'vBot[L/T]' 列进行绘图。")
                            # Fallback to something else, e.g., WaterContent or PressureHead if available
                        
                        ax_tlevel.set_xlabel('时间 (天)')
                        ax_tlevel.grid(True)
                        ax_tlevel.legend()
                        st.pyplot(fig_sim_tlevel)

                        st.markdown("---")
                        st.subheader("详细模拟数据")
                        st.info("以下是模拟期间土壤水分和水压头随深度和时间的变化情况（部分示例）。")
                        # You would typically parse .OUT_T_VAR.DAT or .OUT_M_OBS.DAT for full profiles
                        # For now, just showing a message and the workspace path
                        st.text(f"HYDRUS 输出文件路径: {ml.ws_path}")
                        st.markdown("请前往指定路径查看完整的模拟输出文件 (`T_LEVEL.OUT`, `LOOK.OUT`, etc.)。")
                        
                    else:
                        st.error(f"HYDRUS-1D 模拟失败: {rs.error_message}。请检查 HYDRUS 输入文件和日志以获取更多详细信息。")
                        st.text(f"HYDRUS log: {rs.log}")

                except Exception as e:
                    st.error(f"phydrus 模拟设置或执行期间出现意外错误: {e}。请检查您的 phydrus 库和 HYDRUS-1D 可执行文件。")
                    st.info("请记住 phydrus 需要有效的 HYDRUS-1D 可执行文件，并且其 API 设置模拟可能很复杂。")

            st.session_state.run_phydrus_simulation = False # Reset flag

with tab3:
    st.header("🧂 盐分运移模拟 (待开发)")
    st.info("此部分将用于模拟土壤中的盐分运移过程。")
    st.write("未来版本将提供以下功能:")
    st.markdown("- 允许用户**输入初始土壤盐度**和**灌溉水盐度**。")
    st.markdown("- 模拟不同水管理情景下（如不同灌溉量、淋洗事件）的**盐分累积或淋洗动态**。")
    st.markdown("- **可视化土壤盐度剖面**随时间的变化，帮助评估盐渍化风险。")
    st.markdown("- 与作物产量模拟联动，评估**盐度对作物生长的影响**。")

with tab4:
    st.header("🌾 作物产量模拟 (AQUACROP-OSPy)")
    if not aquacrop_available:
        st.warning("AQUACROP-OSPy 库未找到。作物产量模拟功能已禁用。请确保已正确安装 `aquacrop` 库。")
    elif st.session_state.inferred_df is None:
        st.warning("在运行作物产量模拟之前，请先在 '土壤属性分析' 标签页中成功运行分析，以获取土壤参数。")
    else:
        st.info("""
            此功能将使用 Rosetta 模型输出的土壤水力参数和**合成气候数据**来模拟作物生长和产量。
        """)

        st.subheader("模拟参数设置")
        col_aquacrop_1, col_aquacrop_2 = st.columns(2)
        with col_aquacrop_1:
            crop_type = st.selectbox("选择作物类型", ["Maize", "Wheat", "Rice", "Soybean"], index=0,
                                     help="选择要模拟的作物种类。不同作物有不同的生长参数和水分需求。")
            planting_date_str = st.date_input("种植日期", value=pd.to_datetime("2000-05-01"),
                                            help="作物开始生长的日期。", key="aquacrop_planting_date")
        with col_aquacrop_2:
            sim_end_date_str = st.date_input("模拟结束日期", value=pd.to_datetime("2000-09-30"),
                                             help="作物生长模拟结束的日期。", key="aquacrop_end_date")
            
        # Convert date to datetime object for AQUACROP-OSPy
        planting_date = pd.to_datetime(planting_date_str)
        sim_end_date = pd.to_datetime(sim_end_date_str)

        if st.button("🌿 运行作物产量模拟", use_container_width=True, key="run_aquacrop_btn"):
            st.session_state.run_aquacrop_simulation = True
        
        if st.session_state.run_aquacrop_simulation:
            st.subheader("模拟结果")

            inferred_df = st.session_state.inferred_df
            
            # 1. Prepare Soil Profile for AQUACROP-OSPy
            theta_r_ac = inferred_df.iloc[0]['残余含水量 (θr, cm³/cm³)']
            theta_s_ac = inferred_df['饱和含水量 (θs, cm³/cm³)'].iloc[0]
            Ksat_ac = 10**inferred_df.iloc[0]['饱和导水率 (log10(Ksat), log10(cm/天))']
            theta_fc_ac = inferred_df.iloc[0]['田间持水量 (FC, cm³/cm³)']
            
            # Define soil layers for AQUACROP-OSPy. Assume a 100 cm deep profile with 10 cm layers.
            thicknesses = [10.0] * 10 # 10 layers of 10 cm each for 100 cm total depth
            theta_fc_layers = [theta_fc_ac] * 10
            theta_s_layers = [theta_s_ac] * 10
            theta_r_layers = [theta_r_ac] * 10
            k_sat_layers = [Ksat_ac] * 10 # cm/day

            try:
                aquacrop_soil = ac.Soil(thicknesses, theta_fc_layers, theta_s_layers, theta_r_layers, k_sat_layers)
                
                st.markdown("**AQUACROP-OSPy 土壤剖面设置 (基于 Rosetta 结果)**")
                st.dataframe(pd.DataFrame({
                    '层厚度 (cm)': thicknesses,
                    'θ_fc': theta_fc_layers,
                    'θ_s': theta_s_layers,
                    'θ_r': theta_r_layers,
                    'Ksat (cm/天)': k_sat_layers
                }).head(3).style.format("{:.3f}"), use_container_width=True) # Show first few layers
                if len(thicknesses) > 3:
                    st.caption(f"总共 {len(thicknesses)} 层，此处显示前 3 层数据。")

                # 2. Define Crop
                aquacrop_crop = ac.Crop(crop_type)
                st.markdown(f"**选择作物类型: ** `{crop_type}`")

                # 3. Generate Synthetic Weather Data (for demonstration)
                dates = pd.date_range(planting_date, sim_end_date, freq='D')
                n_days = len(dates)
                
                if n_days > 0:
                    # More realistic dummy weather patterns for different months/seasons
                    monthly_temps = {
                        1: {'Tmax': 5, 'Tmin': -5, 'ETo': 1}, 2: {'Tmax': 8, 'Tmin': -2, 'ETo': 1.5},
                        3: {'Tmax': 15, 'Tmin': 3, 'ETo': 2.5}, 4: {'Tmax': 22, 'Tmin': 8, 'ETo': 3.5},
                        5: {'Tmax': 28, 'Tmin': 15, 'ETo': 5}, 6: {'Tmax': 32, 'Tmin': 20, 'ETo': 6.5},
                        7: {'Tmax': 35, 'Tmin': 23, 'ETo': 7}, 8: {'Tmax': 33, 'Tmin': 21, 'ETo': 6},
                        9: {'Tmax': 27, 'Tmin': 16, 'ETo': 4.5}, 10: {'Tmax': 20, 'Tmin': 10, 'ETo': 3},
                        11: {'Tmax': 12, 'Tmin': 4, 'ETo': 2}, 12: {'Tmax': 6, 'Tmin': -3, 'ETo': 1.2}
                    }
                    
                    precip = np.random.uniform(0, 5, n_days) # Base 0-5 mm rainfall
                    # Add some "rainy days" with higher precipitation
                    rainy_days_idx = np.random.choice(n_days, int(n_days * 0.15), replace=False)
                    precip[rainy_days_idx] += np.random.uniform(5, 30, len(rainy_days_idx)) # Some heavy rain events

                    t_max = np.zeros(n_days)
                    t_min = np.zeros(n_days)
                    eto = np.zeros(n_days)

                    for i, date in enumerate(dates):
                        month_data = monthly_temps.get(date.month, {'Tmax': 20, 'Tmin': 10, 'ETo': 4}) # Fallback
                        t_max[i] = np.random.normal(month_data['Tmax'], 2) # Normal distribution around mean
                        t_min[i] = np.random.normal(month_data['Tmin'], 2)
                        eto[i] = np.random.normal(month_data['ETo'], 0.5)
                    
                    # Ensure Tmin < Tmax and all values are positive
                    t_min = np.minimum(t_min, t_max - 5).clip(min=-20) # Tmin at least 5 deg C below Tmax
                    eto = eto.clip(min=0.1) # ETo must be positive

                    weather_df = pd.DataFrame({
                        'Date': dates,
                        'Rain': precip,
                        'Tmax': t_max,
                        'Tmin': t_min,
                        'ETo': eto
                    }).set_index('Date')
                    
                    aquacrop_weather = ac.Weather(weather_df)
                    st.markdown("**合成气候数据 (前 7 天示例)**")
                    st.caption("请注意：这些数据是为演示目的生成的随机数据。")
                    st.dataframe(weather_df.head(7).style.format("{:.2f}"), use_container_width=True)
                else:
                    st.warning("模拟天数不足，无法生成气候数据。请调整种植日期和结束日期，确保至少有一天。")
                    aquacrop_weather = None


                # 4. Initialize Simulation
                if aquacrop_weather is not None:
                    model = ac.AquaCropModel(
                        sim_start_time=f"{planting_date.year}/{planting_date.month}/{planting_date.day}",
                        sim_end_time=f"{sim_end_date.year}/{sim_end_date.month}/{sim_end_date.day}",
                        weather_data=aquacrop_weather,
                        soil_profile=aquacrop_soil,
                        crop_params=aquacrop_crop,
                    )

                    model.initial_water_content(initial_water_content=ac.InitialWaterContent.FC)
                    # model.IrrigationManagement = ac.IrrigationManagement() # No irrigation for now
                    # model.FieldManagement = ac.FieldManagement() # Default field management

                    # 5. Run Simulation
                    with st.spinner("正在运行 AQUACROP-OSPy 模拟...这可能需要一些时间。"):
                        model.run_model(till_termination=True)

                    # 6. Extract Results
                    sim_results = model.get_simulation_results()

                    if sim_results is not None and not sim_results.empty:
                        st.success("AQUACROP-OSPy 模拟成功完成！")
                        st.subheader("🌾 作物产量模拟概览")
                        
                        # Display key summary results with explanations
                        st.markdown(f"**模拟总天数:** `{len(sim_results)}` 天。")
                        st.markdown(f"**最终生物量产量:** `{sim_results['Biomass'].iloc[-1]:.2f}` kg/ha。")
                        st.markdown(f"**最终作物产量 (收成):** `{sim_results['Yield'].iloc[-1]:.2f}` kg/ha。")
                        st.markdown(f"**模拟总蒸散量 (ET):** `{sim_results['Evapotranspiration'].sum():.2f}` mm (作物和土壤水分蒸发总量)。")
                        st.markdown(f"**模拟总降雨量:** `{weather_df['Rain'].sum():.2f}` mm。")

                        # Plotting key results
                        fig_ac, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True) # sharex ensures x-axes are linked

                        # Biomass and Yield
                        axes[0].plot(sim_results['time_calendar'], sim_results['Biomass'], label='生物量', color='green')
                        axes[0].plot(sim_results['time_calendar'], sim_results['Yield'], label='作物产量', linestyle='--', color='darkgreen')
                        axes[0].set_ylabel('质量 (kg/ha)')
                        axes[0].set_title('生物量和作物产量累积')
                        axes[0].legend()
                        axes[0].grid(True)

                        # Green Cover
                        axes[1].plot(sim_results['time_calendar'], sim_results['CC'], label='绿冠覆盖度', color='olivedrab')
                        axes[1].set_ylabel('绿冠覆盖度 (%)')
                        axes[1].set_title('作物绿冠覆盖度变化')
                        axes[1].legend()
                        axes[1].grid(True)

                        # Water fluxes
                        axes[2].plot(sim_results['time_calendar'], sim_results['Evapotranspiration'], label='蒸散量 (ET)', color='blue')
                        axes[2].plot(sim_results['time_calendar'], sim_results['SurfaceRunoff'], label='地表径流', linestyle=':', color='red')
                        axes[2].plot(sim_results['time_calendar'], sim_results['DeepPercolation'], label='深层渗漏', linestyle='-.', color='purple')
                        axes[2].set_ylabel('水分 (mm/天)')
                        axes[2].set_title('每日水分通量')
                        axes[2].legend()
                        axes[2].grid(True)

                        plt.xlabel('日期')
                        plt.tight_layout()
                        st.pyplot(fig_ac)

                        st.markdown("---")
                        st.subheader("原始模拟结果 (前 5 天数据)")
                        st.dataframe(sim_results.head().style.format("{:.2f}"), use_container_width=True)

                    else:
                        st.error("AQUACROP-OSPy 模拟未返回结果或失败。请检查输入参数。")
                else:
                    st.warning("由于没有有效的气候数据，AQUACROP-OSPy 模拟未运行。请调整种植日期和结束日期。")

            except Exception as e:
                st.error(f"AQUACROP-OSPy 模拟期间出错: {e}。请检查您的输入和库安装。")
                st.info("AQUACROP-OSPy 模型的输入参数可能需要更精细的调整，或合成气候数据可能存在问题。")
            
            st.session_state.run_aquacrop_simulation = False # Reset flag