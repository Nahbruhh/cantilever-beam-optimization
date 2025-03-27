import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from page_format import HTML_Template,MainCSS

# --- App Title and Description ---
def display_title_and_description():
    st.title("Cantilever Beam Optimization Demo")
    st.write("### Optimize: Weight, Deflection, or Stress")
    with st.expander("Connect with Me"):
        st.markdown("""
                    **Author**: Copyright (c) 2025 **Nguyen Manh Tuan**
    <style>
        .social-buttons img {
            transition: opacity 0.3s;
        }
        .social-buttons img:hover {
            opacity: 0.7;
        }
    </style>
    <div class="social-buttons" style="display: flex; gap: 20px; align-items: center;">
        <a href="https://github.com/Nahbruhh" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="40">
        </a>
        <a href="https://www.linkedin.com/in/manh-tuan-nguyen19/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" width="100">
        </a>
    </div>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
    ### Problem Description
    This application optimizes a cantilever beam's design by minimizing one of the following objectives:
    - **Weight** (kg): Proportional to cross-sectional area.
    - **Deflection** (mm): At the free end of the beam.
    - **Stress** (MPa): Maximum stress at the fixed end.

    #### Constraints
    - Deflection: $\delta \leq \delta_{\\text{max}}$ (user-defined maximum deflection).
    - Stress: $\sigma \leq \sigma_y$ (user-defined yield strength).

    #### Parameters
    - **Variables**: Width $b$ (mm), Height $h$ (mm)
    - **User-Defined**: Length $L$ (mm), Load $P$ (N), Yield Strength $\sigma_y$ (MPa), Max Deflection $\delta_{\\text{max}}$ (mm)
    - **Fixed**: Young's Modulus $E$ (GPa) and Density $\\rho = 7850 \\, \\text{kg/m}^3$ (steel)

    #### Approach
    Uses ML-based methods (Random Forest, Gradient Boosting, Neural Network) to optimize the design.
    """)
      
        
    with col2:
        st.image("assets/beam.webp")
    st.divider()

# inputs
def get_user_inputs():
    with st.sidebar:
        st.header("Input Parameters 🎛️")
        
        # Initialize session state
        if 'b_value' not in st.session_state:
            st.session_state.b_value = 20.00
        if 'h_value' not in st.session_state:
            st.session_state.h_value = 70.00
        
        # Sliders for ranges
        b_min, b_max = st.slider("Width (b) Range (mm)", 10.0, 100.0, (10.0, 20.0))
        h_min, h_max = st.slider("Height (h) Range (mm)", 10.0, 100.0, (80.0, 100.0))
        
        
        
        # st.session_state.b_value = max(b_min, min(b_max, st.session_state.b_value))
        # st.session_state.h_value = max(h_min, min(h_max, st.session_state.h_value))
        
        # Number inputs
        b = st.number_input("Width (b) (mm)", min_value=10.0, value=st.session_state.b_value, step=0.01, key="b_input")
        h = st.number_input("Height (h) (mm)", min_value=10.0, value=st.session_state.h_value, step=0.01, key="h_input")
        st.session_state.b_value = b
        st.session_state.h_value = h
        
        # Other parameters
        L = st.number_input("Length (L) (mm)", min_value=500.0, max_value=2000.0, value=709.42, step=0.01)
        P = st.number_input("Load (P) (N)", min_value=500.0, max_value=5000.0, value=1000.0, step=0.01)
        E = st.number_input("Young's Modulus (E) (GPa)", min_value=100.0, max_value=300.0, value=200.0, step=0.01)
        sigma_y = st.number_input("Yield Strength (σ_y) (MPa)", min_value=100.0, max_value=500.0, value=250.0, step=0.01)
        delta_max = st.number_input("Max Deflection (δ_max) (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.01)
        
        doe_points = st.slider("DOE Points 🔢", 200, 1000, 100)
        objective = st.selectbox("Optimization Objective 🎯", ["Minimize Deflection","Minimize Weight",  "Minimize Stress"], index=0)
        
        methods = {"ML-based": ["Random Forest", "Gradient Boosting", "Neural Network"]}
        opt_type = st.selectbox("Optimization Type 🛠️", list(methods.keys()))
        opt_method = st.selectbox("Method ⚙️", methods[opt_type])
        
        # Buttons
        submit_btn = st.button("Submit ✅", help="Calculate results", use_container_width=True)
        doe_btn = st.button("DOE Plot 📈", help="Visualize DOE sampling", use_container_width=True)
        corr_btn = st.button("Correlation 🔍", help="Show parameter correlation", use_container_width=True)
        optimize_btn = st.button("Optimize 🚀", help="Run optimization", use_container_width=True)
        benchmark_btn = st.button("Benchmark ⏱️", help="Compare methods", use_container_width=True)
        clear_btn = st.button("Clear 🗑️", help="Clear history and cache", use_container_width=True)
    
    return (b, h, L, P, E, sigma_y, delta_max, doe_points, objective, opt_method, 
            submit_btn, doe_btn, corr_btn, optimize_btn, benchmark_btn, clear_btn, 
            b_min, b_max, h_min, h_max)

# --- Computation Functions ---
def compute_properties(b, h, L, P, E, rho=7850):
    b_m = b / 1000
    h_m = h / 1000
    L_m = L / 1000
    E_Pa = E * 1e9
    
    # Weight
    area = b_m * h_m
    volume = area * L_m
    weight = rho * volume
    
    # Deflection
    I = (b_m * h_m**3) / 12
    delta = (P * L_m**3) / (3 * E_Pa * I) * 1000
    
    # Stress
    sigma = (6 * P * L_m) / (b_m * h_m**2) / 1e6
    
    return weight, delta, sigma

def weight_objective(bh, L, rho=7850):
    b, h = bh
    b_m, h_m = b / 1000, h / 1000
    area = b_m * h_m
    volume = area * (L / 1000)
    return rho * volume

def deflection_objective(bh, L, P, E):
    b, h = bh
    b_m, h_m = b / 1000, h / 1000
    I = (b_m * h_m**3) / 12
    return (P * (L / 1000)**3) / (3 * (E * 1e9) * I) * 1000

def stress_objective(bh, L, P):
    b, h = bh
    b_m, h_m = b / 1000, h / 1000
    return (6 * P * (L / 1000)) / (b_m * h_m**2) / 1e6

def deflection_constraint(bh, delta_max, L, P, E):
    return delta_max - deflection_objective(bh, L, P, E)

def stress_constraint(bh, sigma_y, L, P):
    return sigma_y - stress_objective(bh, L, P)

@st.cache_data
def generate_doe_samples(doe_points, b_min, b_max, h_min, h_max, L):
    start_time = time.time()
    b_doe = np.random.uniform(b_min, b_max, doe_points)
    h_doe = np.random.uniform(h_min, h_max, doe_points)
    weights = [weight_objective([b, h], L) for b, h in zip(b_doe, h_doe)]
    doe_time = time.time() - start_time
    return b_doe, h_doe, weights, doe_time

@st.cache_data
def run_optimization(method, objective, bounds, doe_points, L, P, E, sigma_y, delta_max, doe_time=0):
    try:
        start_time = time.time()
        b_min, b_max = bounds[0]
        h_min, h_max = bounds[1]

        obj_func = {
            "Minimize Deflection": lambda bh: deflection_objective(bh, L, P, E),
            "Minimize Weight": lambda bh: weight_objective(bh, L),
            "Minimize Stress": lambda bh: stress_objective(bh, L, P)
        }[objective]

        b_samples, h_samples, y_doe, doe_time = generate_doe_samples(doe_points, b_min, b_max, h_min, h_max, L)
        X = np.column_stack((b_samples, h_samples))
        y = np.array([obj_func([b, h]) for b, h in X])
        constraints_valid = np.zeros(doe_points)

        for i in range(doe_points):
            params = [b_samples[i], h_samples[i]]
            constraints_valid[i] = 1 if (deflection_constraint(params, delta_max, L, P, E) >= 0 and 
                                         stress_constraint(params, sigma_y, L, P) >= 0) else 0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
        }[method]

        feasible_idx = constraints_valid == 1
        if np.sum(feasible_idx) > 0:
            X_train = X_scaled[feasible_idx]
            y_train = y[feasible_idx]
            model.fit(X_train, y_train)
        else:
            X_train = X_scaled
            y_train = y
            model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        r2 = r2_score(y_train, y_pred_train)
        mse = mean_squared_error(y_train, y_pred_train)
        mae = mean_absolute_error(y_train, y_pred_train)

        n_pred = 100
        b_grid = np.linspace(b_min, b_max, n_pred)
        h_grid = np.linspace(h_min, h_max, n_pred)
        B, H = np.meshgrid(b_grid, h_grid)
        X_pred = np.column_stack((B.ravel(), H.ravel()))
        X_pred_scaled = scaler.transform(X_pred)

        y_pred = model.predict(X_pred_scaled)
        constraints_valid_pred = np.zeros(len(X_pred))
        for i, params in enumerate(X_pred):
            constraints_valid_pred[i] = 1 if (deflection_constraint(params, delta_max, L, P, E) >= 0 and 
                                              stress_constraint(params, sigma_y, L, P) >= 0) else 0

        feasible_pred = constraints_valid_pred == 1
        if np.sum(feasible_pred) > 0:
            feasible_y = y_pred[feasible_pred]
            feasible_X = X_pred[feasible_pred]
            opt_idx = np.argmin(feasible_y)
            opt_bh = feasible_X[opt_idx]
            opt_value = feasible_y[opt_idx]
        else:
            opt_bh = np.array([b_min, h_min])
            opt_value = obj_func(opt_bh)

        opt_bh = np.clip(opt_bh, [b_min, h_min], [b_max, h_max])
        end_time = time.time()
        runtime = (end_time - start_time) - doe_time

        return opt_bh, opt_value, runtime, r2, mse, mae
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return [b_min, h_min], float('inf'), 0, 0, float('inf'), float('inf')

# --- app
def main_app():
    if 'history' not in st.session_state:
        st.session_state.history = {}
    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = {}

    display_title_and_description()
    inputs = get_user_inputs()
    (b, h, L, P, E, sigma_y, delta_max, doe_points, objective, opt_method, 
     submit_btn, doe_btn, corr_btn, optimize_btn, benchmark_btn, clear_btn, 
     b_min, b_max, h_min, h_max) = inputs

    if submit_btn:
        if b < b_min or b > b_max:
            st.warning(f"**Width (b) = {b:.2f} mm is outside the design range [{b_min:.2f}, {b_max:.2f}] mm.**")
        if h < h_min or h > h_max:
            st.warning(f"**Height (h) = {h:.2f} mm is outside the design range [{h_min:.2f}, {h_max:.2f}] mm.**")
        weight, delta, sigma = compute_properties(b, h, L, P, E)
        st.write(f"**Weight**: {weight:.4f} kg ")
        st.write(f"**Deflection**: {delta:.4f} mm ")
        st.write(f"**Stress**: {sigma:.2f} MPa ")
        if delta <= delta_max and sigma <= sigma_y:
            st.success("Design satisfies constraints! ✅")
        else:
            st.error("Design violates constraints! ❌")

    if optimize_btn:
        opt_bh, opt_value, runtime, r2, mse, mae = run_optimization(opt_method, objective, [(b_min, b_max), (h_min, h_max)], doe_points, L, P, E, sigma_y, delta_max)
        opt_weight, opt_delta, opt_sigma = compute_properties(opt_bh[0], opt_bh[1], L, P, E)
        
        st.session_state.opt_results[opt_method] = {
            "opt_bh": opt_bh, "opt_value": opt_value, "runtime": runtime,
            "r2": r2, "mse": mse, "mae": mae,
            "weight": opt_weight, "deflection": opt_delta, "stress": opt_sigma
        }
        
        st.markdown(f"""**Optimized Design ({opt_method})**   
                    **Objective:** {objective} 
                    """) 
        st.write(f"- **Width (b):** {opt_bh[0]:.2f} mm")
        st.write(f"- **Height (h):** {opt_bh[1]:.2f} mm") 
                    
        if objective == "Minimize Weight":
            st.write(f"- **Predicted Minimum Weight:** {opt_value:.2f} kg")
            st.write(f"- **Calculated Weight:** {opt_weight:.4f} kg")
        elif objective == "Minimize Deflection":
            st.write(f"- **Predicted Minimum Deflection:** {opt_value:.2f} mm")
            st.write(f"- **Calculated Deflection:** {opt_delta:.4f} mm")
        else:
            st.write(f"- **Predicted Minimum Stress:** {opt_value:.2f} MPa")
            st.write(f"- **Calculated Stress:** {opt_sigma:.2f} MPa")
        
        st.write(f"- **Calculated Deflection:** {opt_delta:.4f} mm (Constraint: ≤ {delta_max:.2f} mm)")
        st.write(f"- **Calculatd Stress:** {opt_sigma:.2f} MPa (Constraint: ≤ {sigma_y:.2f} MPa)")
        
        
        
        if opt_delta <= delta_max and opt_sigma <= sigma_y:
            st.success("Optimized design satisfies constraints! ✅")
        else:
            st.error("Optimized design violates constraints! ❌")
        st.info(f"""
            #### ML Model Performance:
            - **R² Score:** {r2:.4f} (Higher values for better fit of the meta model)
            - **Mean Squared Error (MSE):** {mse:.4f} (Average squared difference between predicted and actual values)
            - **Mean Absolute Error (MAE):** {mae:.4f} (Average absolute difference between predicted and actual values)
            """)
        r2_threshold = 0.98  # You can adjust this threshold as needed
        if r2 < r2_threshold:
            st.warning(f"The {opt_method} model: Low R² {r2:.4f}, which is below the threshold of {r2_threshold:.2f}. Consider increasing DOE points for better prediction.")
        
        st.session_state.history[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = {
            "method": opt_method, "objective": objective,"doe_points": doe_points,
            "b": opt_bh[0], "h": opt_bh[1],
            "weight": opt_weight, "deflection": opt_delta, "stress": opt_sigma,
            "runtime": runtime, "r2": r2, # "mse": mse,  # "mae": mae
        }

    if doe_btn:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("### DOE Parameters")
            st.warning(f"""
                    - **DOE Sampling Points:** {doe_points}
                    - **b range:** {b_min} - {b_max} mm
                    - **h range:** {h_min} - {h_max} mm
                    """)
            st.info("""
            **Design of Experiments (DOE)** is a method to systematically explore how input variables (e.g., width, height) affect outcomes (e.g., weight).  
            - **Purpose**: Identify optimal designs efficiently.  
            - **How it Works**: Randomly samples the design space and evaluates constraints.  
            - **Here**: We use DOE to generate training data for ML optimization.
            """)
        with col2:
            b_doe, h_doe, weights, _ = generate_doe_samples(doe_points, b_min, b_max, h_min, h_max, L)
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(b_doe, h_doe, c=weights, cmap='viridis')
            plt.colorbar(scatter, label='Weight (kg)')
            ax.set_xlabel('Width (b) (mm)')
            ax.set_ylabel('Height (h) (mm)')
            ax.set_title('DOE Sampling')
            st.pyplot(fig)

    if corr_btn:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("### Correlation Parameters")
            st.warning(f"""
                    - **DOE Sampling Points:** {doe_points}
                    - **b range:** {b_min} - {b_max} mm
                    - **h range:** {h_min} - {h_max} mm
                    """)
            st.info("""
            **Correlation** measures how strongly variables (e.g., width, stress) are related.  
            - **Purpose**: Understand dependencies between parameters.  
            - **How it Works**: Values range from -1 (negative relation) to 1 (positive relation), with 0 meaning no relation.  
            - **Here**: We visualize correlations to reveal design insights.
            """)

        with col2:
            b_samples, h_samples, weights, _ = generate_doe_samples(doe_points, b_min, b_max, h_min, h_max, L)
            _, deltas, sigmas = zip(*[compute_properties(b, h, L, P, E) for b, h in zip(b_samples, h_samples)])
            df = pd.DataFrame({'b': b_samples, 'h': h_samples, 'weight': weights, 'deflection': deltas, 'stress': sigmas})
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Parameter Correlation')
            st.pyplot(fig)

    if benchmark_btn:
        st.header("Benchmark Results ⏱️")
        if not st.session_state.opt_results:
            st.warning("Please run 'Optimize' first to generate results for benchmarking.")
        else:
            benchmark_results = []
            for method in ["Random Forest", "Gradient Boosting", "Neural Network"]:
                if method in st.session_state.opt_results:
                    result = st.session_state.opt_results[method]
                    benchmark_results.append({
                        "Method": method, "Objective": objective, "Optimal Value": result["opt_value"],
                        "Runtime (s)": result["runtime"], "R² Score": result["r2"], "MSE": result["mse"], "MAE": result["mae"]
                    })
            if benchmark_results:
                benchmark_df = pd.DataFrame(benchmark_results)
                st.dataframe(benchmark_df.style.format({
                    "Optimal Value": "{:.4f}", "Runtime (s)": "{:.3f}", "R² Score": "{:.4f}", "MSE": "{:.4f}", "MAE": "{:.4f}"
                }))
                fig, ax = plt.subplots(figsize=(12, 4))
                benchmark_df.plot(kind='bar', x='Method', y='Runtime (s)', ax=ax)
                ax.set_title('Method Comparison - Runtime')
                st.pyplot(fig)
            else:
                st.warning("No optimization results available for benchmarking.")

    if clear_btn:
        st.session_state.history.clear()
        st.session_state.opt_results.clear()
        st.session_state.b_value = 20.00
        st.session_state.h_value = 70.00
        st.cache_data.clear()
        st.success("History and cache cleared! ✅")

    st.header("History Log ⏳")
    if st.session_state.history:
        history_df = pd.DataFrame.from_dict(st.session_state.history, orient='index')
        history_df.columns = ["Method", "Objective","DOE Points", "Width b (mm)", "Height h (mm)", "Weight (kg)", "Deflection δ (mm)", "Stress σ (MPa)", "Runtime (s)", "R²"] 
        st.dataframe(history_df.style.format({
            "b (mm)": "{:.2f}",
            "Height h (mm)": "{:.2f}",
            "Weight (kg)": "{:.2f}",
            "Deflection δ (mm)": "{:.4f}", 
            "Stress σ (MPa)": "{:.2f}",     
            "Runtime (s)": "{:.3f}",
            "R²": "{:.4f}"
            
        }),use_container_width=True)
    else:
        st.write("No history available.")

# --- Entry Point ---
if __name__ == "__main__":
    st.set_page_config(page_title="Cantilever Beam Optimization", layout="wide")
    st.html(HTML_Template.base_style.substitute(css=MainCSS.initial_page_styles))
    main_app()