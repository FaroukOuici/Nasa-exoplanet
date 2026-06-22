from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from xgboost import XGBRegressor
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import numpy as np


# The model get 89% of accuracy with this parameters
class Model():
    def __init__(self):
        try:
            exoplanet = NasaExoplanetArchive.query_criteria_async(
                table="pscomppars",
                select="top 10000 pl_rade , pl_bmasse , st_teff , st_rad"
            )
            table = exoplanet.to_table()
            self.df = table.to_pandas()
        except Exception as e:
            print(f"Error : {e}")
        print(f"the data was :\n {self.df.info()}")

    
    def cleaning(self): # This Function is used to cleaning the data 
        self.df = self.df.dropna(subset=["pl_bmasse"])
        target_cols = ['pl_rade','st_teff','st_rad']
        geo_means = np.exp(np.log(self.df[target_cols].replace(0, np.nan)).mean())
        self.df[target_cols] = self.df[target_cols].fillna(geo_means)
        # Rename every Column
        self.df.rename(columns = {'st_teff':'star_temp','st_rad':'star_radius','pl_bmasse':'pl_mass','pl_rade':'pl_radius'},inplace=True)
        print(f'The new data is {self.df.head()}')
        print(f'the data become :\n {self.df.info()}')

    def transform(self):
        try:
            self.df["pl_mass_log"] = np.log1p(self.df["pl_mass"]) # We are using the log function to make the data more normal
            self.df["pl_radius_log"] = np.log1p(self.df["pl_radius"])
        except Exception as e:
            print(f"Error : {e}")

    
    def data_split(self):
        try:
            # here we split the data and prepare it for training
            self.transform()
            X = self.df[["star_temp","star_radius","pl_radius_log"]]
            y = self.df["pl_mass_log"]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error : {e}")

        self.train()
        
    def train(self):
        try:
            # Here We introduce the model and train it
            self.model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_estimators=150,
                max_depth=5,
                learning_rate=0.04,
                n_jobs=1,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
        except Exception as e:
            print(f"Error : {e}")

        self.test()

    def test(self):
        try:
            y_real_test = np.expm1(self.y_test)
            y_pred_real = np.expm1(self.y_pred)
            r2 = r2_score(self.y_test, self.y_pred)
            Mean_Absolute_Error = mean_absolute_error(y_real_test , y_pred_real)
            print(f"R2 Score : {r2}")
            print(f"Mean Squared Error : {Mean_Absolute_Error}")
        except Exception as e:
            print(f"Error : {e}")

    def save_model(self):
        # We save the model here(please note that if you are using kaggle this will be in the Kaggle/working directory)
        self.model.save_model("xg_planets.json")

    def exam(self):
# Here you can test the model in new data and get the result just get a planet mass and it's radius and stellar temperature,radius and put the radius of the star,temprature along the radius of the planet and you will get the prediction
        new_planets = pd.DataFrame({
        "star_temp":   [3200.0, 5200.0, 6100.0],   
        "star_radius": [0.28,   0.79,   1.35],     
        "pl_radius":   [0.85,   3.23,   10.66]     
        })
        new_planets["pl_radius_log"] = np.log1p(new_planets["pl_radius"])
        X_new = new_planets[["star_temp","star_radius","pl_radius_log"]]
        y_new = self.model.predict(X_new)
        pred_new = np.expm1(y_new)
        for i,mass in enumerate(pred_new):
            print(f"The planet {i} mass is {mass:.2f} Eu")



# This is the main function that summons all the functions
if __name__ == "__main__":
    obj = Model()
    obj.cleaning()
    obj.data_split()
    obj.save_model()
    obj.exam()
