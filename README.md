# IMDB Movie Ratings Prediction

## Project Overview
This project aims to predict movie ratings using machine learning techniques. It involves data preprocessing, feature selection, and training multiple regression models to evaluate their performance on the IMDB dataset.

## Dataset
The dataset used for this project is `movie_metadata.csv`, which contains information about various movies, including director name, actor names, budget, duration, genre, and IMDB rating.

### Features in the Dataset
- `director_name`: Name of the director
- `actor_1_name`, `actor_2_name`, `actor_3_name`: Names of the lead actors
- `budget`: Budget of the movie
- `duration`: Duration of the movie in minutes
- `genres`: Movie genres
- `imdb_score`: Target variable (IMDB rating)
- And other relevant attributes

## Dependencies
This project requires the following Python libraries:

```bash
numpy
pandas
seaborn
matplotlib
scikit-learn
xgboost
```

## Setup Instructions
1. Clone the repository or download the notebook.
2. Install the required dependencies using:

   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn xgboost
   ```

3. Place the `movie_metadata.csv` file in the working directory.
4. Run the Jupyter Notebook `IMDB Movie Ratings Prediction.ipynb` step by step.

## Model Training
The notebook implements and evaluates the following regression models:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **K-Nearest Neighbors (KNN) Regressor**
- **XGBoost Regressor**

## Evaluation Metrics
The models are evaluated based on the following metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

## Results and Observations
After training and evaluating different models, the results indicate the best-performing model based on accuracy and error rates.

