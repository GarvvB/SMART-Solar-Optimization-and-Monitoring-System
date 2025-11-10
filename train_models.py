from src.model_train import train_and_save_models

# Update the correct paths
gen_path = "data/Plant_1_Generation_Data.csv"
weather_path = "data/solar_weather.csv"

# Train both models and save them in /models
train_and_save_models(gen_path, weather_path)
