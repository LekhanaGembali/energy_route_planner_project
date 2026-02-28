import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Define the file name where the "brain" will be saved
MODEL_FILENAME = "ev_energy_model.joblib"

def generate_synthetic_ev_data(num_samples=5000):
    """Generates realistic EV telemetry data using physics + random noise."""
    np.random.seed(42) 
    
    lengths = np.random.uniform(10.0, 2000.0, num_samples)    
    speeds = np.random.uniform(10.0, 80.0, num_samples)       
    slopes = np.random.uniform(-15.0, 15.0, num_samples)      
    
    mass_kg = 1600.0
    gravity = 9.81
    air_density = 1.225
    drag_coeff = 0.28
    frontal_area = 2.2
    rolling_res = 0.012
    regen_eff = 0.65

    energy_wh_list = []
    
    for length, speed_kph, slope in zip(lengths, speeds, slopes):
        speed_mps = speed_kph / 3.6
        theta = np.arctan(slope / 100.0)
        
        f_roll = rolling_res * mass_kg * gravity * np.cos(theta)
        f_grad = mass_kg * gravity * np.sin(theta)
        f_aero = 0.5 * air_density * drag_coeff * frontal_area * (speed_mps ** 2)
        
        total_force = f_roll + f_grad + f_aero
        energy_wh = (total_force * length) / 3600.0
        
        if energy_wh < 0:
            energy_wh *= regen_eff
            
        noise_factor = np.random.uniform(0.85, 1.15)
        final_energy = max(energy_wh * noise_factor, -length * 0.05)
        energy_wh_list.append(final_energy)
        
    df = pd.DataFrame({
        'length_m': lengths,
        'speed_kph': speeds,
        'slope_percent': slopes,
        'energy_consumed_wh': energy_wh_list
    })
    
    return df

def train_energy_model():
    """Trains a new model from scratch."""
    df = generate_synthetic_ev_data(num_samples=5000)
    
    X = df[['length_m', 'speed_kph', 'slope_percent']]
    y = df['energy_consumed_wh']
    
    # Train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    return rf_model

def get_trained_model():
    """
    The Production Manager: Checks if the model file exists on the hard drive.
    If yes -> Loads it instantly.
    If no -> Trains a new one and saves it to the hard drive.
    """
    if os.path.exists(MODEL_FILENAME):
        # Read the pre-trained brain from disk
        return joblib.load(MODEL_FILENAME)
    else:
        # Train it from scratch
        model = train_energy_model()
        # Save the brain to disk for next time
        joblib.dump(model, MODEL_FILENAME)
        return model

def predict_energy_dynamic(length_meters: float, speed_kph: float, slope_percent: float, model) -> float:
    """Inference Engine: Uses the loaded Random Forest to predict energy cost."""
    if model is None:
        model = get_trained_model()
        
    features = np.array([[length_meters, speed_kph, slope_percent]])
    predicted_wh = model.predict(features)[0]
    
    return float(predicted_wh)