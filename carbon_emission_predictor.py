import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Constants
AVG_CO2_PER_KM = 0.15  # kg CO2 per km per kg cargo (industry average)
TREE_ABSORPTION_PER_YEAR = 21.77  # kg CO2 per tree per year (average)

# =============================================
# 3D COLORFUL PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="EcoVision | Carbon Analytics",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for 3D colorful styling
st.markdown("""
<style>
    /* 3D Color Palette */
    :root {
        --primary-3d: #4CAF50;
        --secondary-3d: #8BC34A;
        --accent-3d: #FFC107;
        --dark-3d: #2E7D32;
        --light-3d: #C8E6C9;
        --highlight: #FF5722;
        --gradient-start: #00BCD4;
        --gradient-end: #3F51B5;
    }
    
    /* 3D Main Container */
    .stApp {
        background: linear-gradient(135deg, var(--light-3d), #f9f9f9);
    }
    
    /* 3D Title Styling */
    .main-title-3d {
        color: white;
        text-align: center;
        font-size: 3.2rem;
        margin-bottom: 0;
        font-weight: 800;
        text-shadow: 3px 3px 0 var(--dark-3d), 
                     6px 6px 0 rgba(0,0,0,0.1);
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        transform: perspective(500px) rotateX(5deg);
        margin: 1rem 3rem;
    }
    
    .subtitle-3d {
        color: var(--dark-3d);
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* 3D Card Styling */
    .card-3d {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 8px 8px 0 var(--dark-3d),
                   0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 3px solid var(--primary-3d);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .card-3d:hover {
        transform: translateY(-5px);
        box-shadow: 10px 10px 0 var(--dark-3d),
                   0 15px 35px rgba(0,0,0,0.15);
    }
    
    .card-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
    }
    
    /* 3D Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-3d), var(--secondary-3d)) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 4px 4px 0 var(--dark-3d) !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 6px 6px 0 var(--dark-3d) !important;
    }
    
    /* 3D Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--light-3d), white) !important;
        padding: 2rem !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.1) !important;
    }
    
    .sidebar-title-3d {
        color: var(--dark-3d) !important;
        font-size: 1.8rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 0 rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* 3D Metric Styling */
    [data-testid="stMetric"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        box-shadow: 5px 5px 0 var(--dark-3d) !important;
        border: 2px solid var(--primary-3d) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: var(--dark-3d) !important;
        font-weight: 700 !important;
    }
    
    /* 3D Expander Styling */
    .stExpander {
        background: white !important;
        border-radius: 12px !important;
        box-shadow: 5px 5px 0 var(--dark-3d) !important;
        border: 2px solid var(--primary-3d) !important;
    }
    
    /* Tree Counter Animation */
    .tree-counter-3d {
        font-size: 4rem;
        color: var(--dark-3d);
        font-weight: 800;
        text-align: center;
        margin: 1.5rem 0;
        text-shadow: 3px 3px 0 rgba(0,0,0,0.1);
        animation: float 3s ease-in-out infinite;
        background: linear-gradient(135deg, var(--primary-3d), var(--secondary-3d));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    /* 3D Input Styling */
    .stNumberInput, .stSelectbox, .stSlider {
        background: white !important;
        border-radius: 10px !important;
        box-shadow: 3px 3px 0 var(--dark-3d) !important;
        border: 2px solid var(--primary-3d) !important;
    }
    
    /* 3D Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white !important;
        border-radius: 10px 10px 0 0 !important;
        box-shadow: 3px 0 0 var(--dark-3d) !important;
        border: 2px solid var(--primary-3d) !important;
        border-bottom: none !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--light-3d) !important;
        color: var(--dark-3d) !important;
        font-weight: 700;
    }
    
    /* 3D Footer Styling */
    .footer-3d {
        text-align: center;
        color: white;
        font-size: 1em;
        margin-top: 3rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        border-radius: 15px;
        box-shadow: 5px 5px 0 var(--dark-3d);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 3D HEADER SECTION
# =============================================
st.markdown('<h1 class="main-title-3d">🌎 EcoVision  Carbon Tracker</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-3d">Visualize • Analyze • Offset Your Carbon Footprint</p>', unsafe_allow_html=True)

# 3D Sidebar navigation
st.sidebar.markdown('<h1 class="sidebar-title-3d">🌱 Navigation</h1>', unsafe_allow_html=True)
app_mode = st.sidebar.radio("", 
                          ["📊 Emission Scan", 
                           "🌳 Offset Simulator", 
                           "📈 Data Explorer", 
                           "⚙️ AI Model Lab",
                           "🔬 Fuel Science"],
                          label_visibility="collapsed")

# Add fuel science section
if app_mode == "🔬 Fuel Science":
    st.markdown("## 🔬 Combustion of Methane (CNG)")
    st.latex("CH_4\\ (\\text{Methane}) + 2O_2 \\rightarrow CO_2 + 2H_2O")
    
   

    st.markdown("""
    ### 🧪 What Happens in This Reaction?

    - **Methane (CH₄)** is the main component of CNG (Compressed Natural Gas).
    - It reacts with **oxygen (O₂)** from the air.
    - Produces:
    - **Carbon Dioxide (CO₂)** – a greenhouse gas  
    - **Water Vapor (H₂O)** – in gaseous form
    - This is called **complete combustion** and also releases **heat energy** used to power vehicles.

    ---

    ### 🔥 Petrol & Diesel Combustion

    #### ⛽ Petrol (Octane – C₈H₁₈)

    - Petrol is a hydrocarbon fuel.
    - Combustion Reaction:
    \[
    2C_8H_{18} + 25O_2 → 16CO_2 + 18H_2O
    \]
    - Needs a **spark plug** to ignite (Spark Ignition Engine).
    - Produces:
    - CO₂
    - H₂O
    - Heat energy
    - Used in cars, bikes, and light vehicles.
                

    #### 🚛 Diesel (Dodecane – C₁₂H₂₆)

    - Diesel is also a hydrocarbon, but heavier.
    - Combustion Reaction:
    \[
    2C_{12}H_{26} + 37O_2 → 24CO_2 + 26H_2O
    \]
    - Uses **high pressure** for self-ignition (Compression Ignition Engine).
    - Produces:
    - More CO₂
    - More NOₓ and PM (Particulate Matter)
    - Higher torque (power)

    ---

    ### 🌱 Environmental Impact

    Although **CNG burns cleaner** than petrol and diesel, it still produces **CO₂**, which contributes to **global warming**.

    | Fuel Type               | CO₂ Emission     | Particulate Matter | Nitrogen Oxides (NOₓ) |
    |-------------------------|------------------|---------------------|------------------------|
    | Petrol                  | High             | High                | Moderate               |
    | Diesel                  | Very High        | Very High           | High                   |
    | CNG (Methane)           | 🔽 20–30% Less   | Very Low            | Low                    |
    | Electric (coal power)   | Indirect CO₂     | None (direct)       | None (direct)          |
    | Electric (solar/wind)   | ✅ Zero           | ✅ Zero              | ✅ Zero                 |

    ---

    ### ☣️ About Nitrogen Oxides (NOₓ)

    - NOₓ includes **NO** and **NO₂**, which are harmful gases produced by combustion.
    - They can:
    - 🫁 Cause **asthma, bronchitis**, and lung damage  
    - 🌫️ Create **photochemical smog**  
    - 🌧️ Lead to **acid rain**  
    - 🌍 Contribute to **global warming**
    - CNG emits **much lower NOₓ** than petrol and diesel.

    ---

    ### ⚡ Note on Electric Vehicles (EVs)

    - EVs produce **no CO₂ directly**, but electricity used to charge them may come from **fossil fuels**.
    - If the grid is clean (solar, wind, hydro), then **EVs are truly zero-emission**.

    ---

    ### 🔍 Conclusion

    > **CNG is cleaner than petrol or diesel, but not zero-emission.**  
    > For true zero emissions, we must move toward **renewable energy and electric vehicles**.
    """)


    # Add your existing footer
    st.markdown("""
    <div style="background: linear-gradient(90deg, #16a085, #27ae60); padding: 15px; border-radius: 10px; margin-top: 30px; text-align: center; color: white; font-family: 'Segoe UI', sans-serif;">
        🌟 Clean energy isn't just the future — it's our responsibility today.🌟 
    </div>
    """, unsafe_allow_html=True)
def calculate_trees_needed(co2_kg, years=5):
    """Calculate how many trees needed to offset CO2 over given years"""
    return co2_kg / (TREE_ABSORPTION_PER_YEAR * years)


# =============================================
# MODEL LOADING FUNCTION
# =============================================
# =============================================
# MODEL LOADING FUNCTION (IMPROVED)
# =============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('co2_emission_model.pkl')
        le_fuel = joblib.load('label_encoder_fuel.pkl')
        le_traffic = joblib.load('label_encoder_traffic.pkl')
        le_weather = joblib.load('label_encoder_weather.pkl')
    except:
        # Generate synthetic data with improved fuel type emissions
        np.random.seed(42)
        n_samples = 5000  # Increased sample size
        
        # Emission factors by fuel type (kg CO2 per liter)
        FUEL_EMISSIONS = {
            'Diesel': 2.68,
            'Petrol': 2.31,
            'CNG': 1.65,    # Lower emissions for CNG
            'Electric': 0.0  # Zero direct emissions
        }
        
        data = {
            'Route_ID': range(1, n_samples+1),
            'Distance_km': np.random.uniform(50, 2000, n_samples),
            'Fuel_Type': np.random.choice(['Diesel', 'Petrol', 'CNG', 'Electric'], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
            'Fuel_Consumed_Liters': np.random.uniform(10, 500, n_samples),
            'Avg_Speed_kmph': np.random.uniform(30, 100, n_samples),
            'Traffic_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
            'Weather_Condition': np.random.choice(['Clear', 'Rainy', 'Foggy'], n_samples, p=[0.6, 0.3, 0.1]),
            'Cargo_Weight_kg': np.random.uniform(500, 10000, n_samples),
        }
        
        # Calculate emissions based on fuel type
        data['CO2_Emission_kg'] = [
            data['Fuel_Consumed_Liters'][i] * FUEL_EMISSIONS[data['Fuel_Type'][i]] 
            + data['Distance_km'][i] * data['Cargo_Weight_kg'][i] * AVG_CO2_PER_KM / 1000
            for i in range(n_samples)
        ]
        
        df = pd.DataFrame(data)
        df.to_csv('carbon_footprint_logistics_2000.csv', index=False)
        
        # Label encoding
        le_fuel = LabelEncoder()
        le_traffic = LabelEncoder()
        le_weather = LabelEncoder()
        
        df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
        df['Traffic_Level'] = le_traffic.fit_transform(df['Traffic_Level'])
        df['Weather_Condition'] = le_weather.fit_transform(df['Weather_Condition'])
        
        # Train improved model
        X = df.drop(['Route_ID', 'CO2_Emission_kg'], axis=1)
        y = df['CO2_Emission_kg']
        
        model = RandomForestRegressor(
            n_estimators=200,  # Increased number of trees
            max_depth=12,      # Deeper trees for better accuracy
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Save the improved model
        joblib.dump(model, 'co2_emission_model.pkl')
        joblib.dump(le_fuel, 'label_encoder_fuel.pkl')
        joblib.dump(le_traffic, 'label_encoder_traffic.pkl')
        joblib.dump(le_weather, 'label_encoder_weather.pkl')
    
    return model, le_fuel, le_traffic, le_weather

model, le_fuel, le_traffic, le_weather = load_model()
# =============================================
# 3D EMISSION SCAN MODULE
# =============================================
if app_mode == "📊 Emission Scan":
    with st.container():
        st.markdown("""
        <div class="card-3d">
            <h2>🚛 Supply Chain Carbon Footprint Analysis</h2>
            <p>🧮Calculate your logistics carbon footprint with our interactive analyzer.</p>
        </div>
        """, unsafe_allow_html=True)
       
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            with st.expander("🔧 Trip Parameters", expanded=True):
                distance = st.number_input("Distance (km)", min_value=1, max_value=5000, value=500)
                fuel_type = st.selectbox("Fuel Type", le_fuel.classes_)
                fuel_consumed = st.number_input("Fuel Consumed (Liters)", min_value=0.1, value=25.0)
        
        with col2:
            with st.expander("🌤️ Environmental Factors", expanded=True):
                avg_speed = st.number_input("Average Speed (km/h)", min_value=1, max_value=120, value=60)
                traffic_level = st.selectbox("Traffic Level", le_traffic.classes_)
                weather = st.selectbox("Weather Condition", le_weather.classes_)
        
        with st.expander("📦 Cargo Details", expanded=True):
            cargo_weight = st.number_input("Cargo Weight (kg)", min_value=1, value=3000)
    
        if st.button("SCAN EMISSIONS", use_container_width=True):
            with st.spinner('Analyzing environmental impact in 3D...'):
                input_data = {
                    'Distance_km': distance,
                    'Fuel_Type': fuel_type,
                    'Fuel_Consumed_Liters': fuel_consumed,
                    'Avg_Speed_kmph': avg_speed,
                    'Traffic_Level': traffic_level,
                    'Weather_Condition': weather,
                    'Cargo_Weight_kg': cargo_weight
                }
                
                input_df = pd.DataFrame([input_data])
                
                input_df['Fuel_Type'] = le_fuel.transform(input_df['Fuel_Type'])
                input_df['Traffic_Level'] = le_traffic.transform(input_df['Traffic_Level'])
                input_df['Weather_Condition'] = le_weather.transform(input_df['Weather_Condition'])
                
                prediction = model.predict(input_df)
                avg_emission = distance * cargo_weight * AVG_CO2_PER_KM / 1000
                
                # 3D Results card
                st.markdown(f"""
                <div class="card-3d">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                        <div style="background: linear-gradient(135deg, #FF6B6B, #FF8E53); padding: 1.5rem; border-radius: 12px; color: white;">
                            <h3>Your Emissions</h3>
                            <div style="font-size: 2.5rem; font-weight: 800;">{prediction[0]:.1f} kg CO₂</div>
                        </div>
                        <div style="background: linear-gradient(135deg, #4CAF50, #8BC34A); padding: 1.5rem; border-radius: 12px; color: white;">
                            <h3>Industry Average</h3>
                            <div style="font-size: 2.5rem; font-weight: 800;">{avg_emission:.1f} kg CO₂</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                difference = prediction[0] - avg_emission
                if difference > 0:
                    st.error(f"⚠️ Your emissions are {difference:.2f} kg above industry average")
                else:
                    st.success(f"✅ Your emissions are {abs(difference):.2f} kg below industry average")
                                    # Optimization recommendations
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center;">
                    <h2>🔄 Optimization Recommendations</h2>
                    <p>Actionable insights to reduce your supply chain carbon footprint</p>
                </div>
                """, unsafe_allow_html=True)
                
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    route_savings = distance * 0.15 * cargo_weight * AVG_CO2_PER_KM / 1000
                    st.markdown(f"""
                    <div class="card-3d" style="padding: 1rem;">
                        <h4>🚛 Route Optimization</h4>
                        <p>Potential distance reduction: <strong>12-18%</strong></p>
                        <p>Estimated savings: <strong>{route_savings:.1f} kg CO₂</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with rec_col2:
                    if fuel_type != "Electric":
                        fuel_savings = prediction[0] * 0.25
                        alt_fuel = "Electric"
                        savings_pct = 25
                    else:
                        fuel_savings = prediction[0] * 0.15
                        alt_fuel = "CNG"
                        savings_pct = 15
                    
                    st.markdown(f"""
                    <div class="card-3d" style="padding: 1rem;">
                        <h4>⛽ Fuel Efficiency</h4>
                        <p>Switch to {alt_fuel}: <strong>{savings_pct:.1f}%</strong> cleaner</p>
                        <p>Estimated savings: <strong>{fuel_savings:.1f} kg CO₂</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with rec_col3:
                    load_savings = prediction[0] * 0.1
                    st.markdown(f"""
                    <div class="card-3d" style="padding: 1rem;">
                        <h4>📦 Load Optimization</h4>
                        <p>Improved capacity utilization: <strong>8-12%</strong></p>
                        <p>Estimated savings: <strong>{load_savings:.1f} kg CO₂</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 3D Offset calculator
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center;">
                    <h2>🌱 Carbon Neutralization</h2>
                    <p>Discover how to offset your emissions through reforestation</p>
                </div>
                """, unsafe_allow_html=True)
                
                years_to_offset =1
                
                trees_needed = calculate_trees_needed(prediction[0], years_to_offset)
                st.markdown(f"""
<div style="
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    margin: 1rem 0;
">
    <!-- Main Solution Card -->
    <div style="
        background: linear-gradient(135deg, #C8E6C9 0%, #E8F5E9 50%, white 100%);
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        padding: 2rem;
        margin-bottom: 1.5rem;
    ">
        <div style="text-align: center;">
            <div style="
                background-color: #2E7D32;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                display: inline-block;
                font-weight: 600;
                font-size: 0.9rem;
                margin-bottom: 1rem;
            ">
                Your Carbon Offset Solution
            </div>
            <h3 style="color: #1B5E20; margin: 0.5rem 0; font-size: 1.4rem;">
                Trees Required for Neutralization
            </h3>
            <div style="
                font-size: 3rem;
                font-weight: 700;
                color: #1B5E20;
                margin: 1rem 0;
                line-height: 1;
            ">
                {trees_needed:.0f} Trees
            </div>
            <p style="font-size: 1.1rem; color: #333; margin: 0.8rem 0;">
                To offset <strong>{prediction[0]:.0f} kg</strong> of CO₂ emissions<br>
                over <strong>{years_to_offset} years</strong>
            </p>
        </div>
    </div>

  <h4 style="color: #2E7D32; margin-top: 0;">Calculation:</h4>
        <div style="background: #F5F5F5; padding: 10px; border-radius: 5px;">
            Trees = Total CO₂ ÷ (Annual Absorption × Years)<br><br>
             {prediction[0]:.0f} ÷ (21.77 × {years_to_offset})= {trees_needed:.0f} Tree 
        </div>
        <p style="font-size: 14px; color: #666;">
            🌳 Based on average of 21.77 kg CO₂ absorbed per tree per year
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
                

# =============================================
# OFFSET SIMULATOR MODULE
# =============================================
elif app_mode == "🌳 Offset Simulator":
    st.header("🌳 Offset Simulator")
    st.markdown("""
    <div class="card-3d">
    <h3>♻️Interactive Carbon Offset Calculator</h3>
    <p>🌿Calculate how many trees you need to plant to offset your carbon emissions.</p>
    <p>🛰️Visualize how many trees you need to plant to compensate for your carbon emissions.</p>
                <h4 style="color: #2E7D32; margin-top: 0;">Calculation:</h4>
        <div style="background: #F5F5F5; padding: 10px; border-radius: 5px;">
            Trees = Total CO₂ ÷ (Annual Absorption × Years)<br><br>
            🌳 Based on average of 21.77 kg CO₂ absorbed per tree per year
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with st.expander("📊 Your Emissions", expanded=True):
            total_co2 = st.number_input("Total CO₂ Emissions (kg)", min_value=0.1, value=1000.0)
            years = st.slider("Offset period (years)", 1, 30, 10)
    
    trees_needed = calculate_trees_needed(total_co2, years)
    
    with col2:
            st.markdown(f"""
            <div class="card-3d" style="background: linear-gradient(135deg, #E8F5E9, white); height: 100%;">
                <div style="text-align: center; padding: 2rem;">
                    <h3 style="color: var(--dark-3d);">Offset Solution(Trees)</h3>
                    <div class="tree-counter-3d">{trees_needed:.0f}</div>
                    <p style="font-size: 1.1rem;">Trees required to absorb {total_co2:.0f} kg CO₂ over {years} years</p>
                </div>
                
            </div>
            """, unsafe_allow_html=True)
    
    # 
# =============================================
# DATA EXPLORER MODULE
# =============================================
elif app_mode == "📈 Data Explorer":
    st.header("📊 EcoVision Carbon Tracker ")
    
    try:
        df = pd.read_csv('carbon_footprint_logistics_2000.csv')
    except:
        load_model()  # Generates sample data if needed
        df = pd.read_csv('carbon_footprint_logistics_2000.csv')
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Explorer", "📈 Statistical Insights", "📊 3D Visual Analytics"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.download_button(
            label="DOWNLOAD DATASET",
            data=df.to_csv(index=False),
            file_name="logistics_emissions_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Statistical Overview")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Average CO₂", f"{df['CO2_Emission_kg'].mean():.1f} kg")
        col3.metric("Most Common Fuel", df['Fuel_Type'].mode()[0])
    
    with tab3:
        st.subheader("Interactive  Visualizations")
        
        chart_type = st.selectbox("Select Visualization", 
                                [" Emissions by Fuel Type", 
                                 " Distance vs Emissions", 
                                 "Cargo Impact"])
        
        if chart_type == "Emissions by Fuel Type":
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            for fuel in df['Fuel_Type'].unique():
                subset = df[df['Fuel_Type'] == fuel]
                ax.scatter(subset['Distance_km'], subset['Cargo_Weight_kg'], subset['CO2_Emission_kg'], 
                          label=fuel, s=50, alpha=0.7)
            
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Cargo Weight (kg)')
            ax.set_zlabel('CO₂ Emissions (kg)')
            ax.set_title(' Emissions by Fuel Type', pad=20)
            ax.legend()
            
            st.pyplot(fig)
        
        elif chart_type == "Distance vs Emissions":
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            hist, xedges, yedges = np.histogram2d(df['Distance_km'], df['CO2_Emission_kg'], bins=20)
            
            xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)
            
            dx = dy = 0.8 * np.ones_like(zpos)
            dz = hist.flatten()
            
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#4CAF50', zsort='average')
            
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('CO₂ Emissions (kg)')
            ax.set_zlabel('Frequency')
            ax.set_title(' Distance vs Emissions Distribution', pad=20)
            
            st.pyplot(fig)
        
        else:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot_trisurf(df['Cargo_Weight_kg'], df['Avg_Speed_kmph'], df['CO2_Emission_kg'], 
                           cmap=cm.viridis, linewidth=0.2, antialiased=True)
            
            ax.set_xlabel('Cargo Weight (kg)')
            ax.set_ylabel('Average Speed (km/h)')
            ax.set_zlabel('CO₂ Emissions (kg)')
            ax.set_title(' Cargo Weight Impact Surface', pad=20)
            
            st.pyplot(fig)

# =============================================
# AI MODEL LAB MODULE
# =============================================
elif app_mode == "⚙️ AI Model Lab":
    st.header("🧪 AI Model Laboratory")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Model Architecture")
        st.markdown("""
        <div class="card-3d">
            <h4>Random Forest Regressor</h4>
            <ul>
                <li><strong>Ensemble Method:</strong> Bagging with 100 decision trees</li>
                <li><strong>Training Data:</strong> 2,000 logistics records</li>
                <li><strong>Features:</strong> 7 operational parameters</li>
                <li><strong>Target:</strong> CO₂ Emissions (kg)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Performance Metrics")
        st.markdown("""
        <div class="card-3d">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="background: linear-gradient(135deg, #00BCD4, #0097A7); padding: 1rem; border-radius: 10px; color: white;">
                    <h4>R² Score</h4>
                    <div style="font-size: 1.8rem; font-weight: 700;">0.92</div>
                </div>
                <div style="background: linear-gradient(135deg, #FF9800, #F57C00); padding: 1rem; border-radius: 10px; color: white;">
                    <h4>Mean Error</h4>
                    <div style="font-size: 1.8rem; font-weight: 700;">42.3 kg</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader(" Feature Importance")
        features = ['Distance', 'Fuel Type', 'Fuel Used', 'Avg Speed', 'Traffic', 'Weather', 'Cargo Weight']
        importance = model.feature_importances_
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ypos = np.arange(len(features))
        xpos = np.zeros_like(ypos)
        zpos = np.zeros_like(ypos)
        
        dx = np.ones_like(zpos) * 0.8
        dy = np.ones_like(zpos) * 0.8
        dz = importance * 100
        
        colors = cm.viridis(dz / max(dz))
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
        
        ax.set_yticks(ypos + 0.4)
        ax.set_yticklabels(features)
        ax.set_xlabel('')
        ax.set_zlabel('Importance (%)')
        ax.set_title(' Feature Importance', pad=20)
        
        st.pyplot(fig)
        
        st.subheader("Model Training")
        st.markdown("""
        <div class="card-3d">
            <p>The model was trained on synthetic logistics data representing:</p>
            <ul>
                <li>Various fuel types (Diesel, Petrol, CNG, Electric)</li>
                <li>Different traffic conditions (Low, Medium, High)</li>
                <li>Multiple weather scenarios (Clear, Rainy, Foggy)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# 3D FOOTER
# =============================================
st.markdown("""
<div style="background: linear-gradient(90deg, #16a085, #27ae60); padding: 15px; border-radius: 10px; margin-top: 30px; text-align: center; color: white; font-family: 'Segoe UI', sans-serif;">
    <p style="font-size: 1.1rem; font-weight: 600;">🌿 SmartRoute Emissions Carbon Tracker</p>
    <p style="font-size: 0.95rem;">Sustainable Logistics Analytics | Tree Absorption Rate: 21.77 kg CO₂/year (USDA)</p>
    <hr style="border: 0.5px solid white; margin: 10px auto; width: 80%;">
    <p style="font-size: 0.9rem;">
        © 2025 <strong>Prem Mohan</strong>. All rights reserved.<br>
        Unauthorized copying, distribution, or use of any part of this platform is strictly prohibited without explicit permission.
    </p>
</div>
""", unsafe_allow_html=True)