# Electron Collision Mass Predictor Model

## Description

This project focuses on applying machine learning regression techniques to the "CERN Electron Collision Data" dataset. The primary goal is to build and evaluate models capable of predicting the invariant mass (M) of a dielectron system using the kinematic properties (energy, momentum components, direction) of two detected electrons as input features. The notebook explores data, performs preprocessing, engineers new features based on physics principles, trains various regression algorithms, tunes their hyperparameters, and evaluates their performance to understand how well these features determine the invariant mass.

## Data Source

The dataset, titled "CERN Electron Collision Data", contains information from 100,000 dielectron events and is sourced from the CERN Open Data Portal. The data is loaded from `dielectron.csv`.

## Preprocessing & Exploratory Data Analysis

The notebook performs the following preprocessing and EDA steps:
* Initial inspection of data shape, column names, and unique values.
* Renaming of the `px1 ` column to `px1` to fix a trailing space.
* Checking for and dropping duplicate rows.
* Checking for and dropping rows with missing values.
* Dropping irrelevant columns such as `Run` and `Event`.
* Analyzing column distributions and correlations using Pearson and Spearman heatmaps.
* Visualizing data distributions with box plots and histograms, noting the need for transformations due to skewed distributions.

## Feature Engineering

The project engineers several new features derived from the kinematic properties of the electrons, based on the invariant mass formula and other physical insights. These include:
* `delta_eta`: Difference in pseudorapidity.
* `dphi`: Difference in azimuthal angle (adjusted to be within -π to π).
* `delta_R`: Angular distance, calculated from `delta_eta` and `dphi`.
* `E_sum`: Sum of energies (`E1` + `E2`).
* `system_px`, `system_py`, `system_pz`: Components of the total system momentum.
* `system_pt_mag`: Magnitude of the transverse momentum of the system.
* `Q_product`: Product of the charges (`Q1` \* `Q2`).
* `E_asymmetry`: Energy asymmetry between the two electrons.
* `pt_asymmetry`: Transverse momentum asymmetry between the two electrons.

Several features, including `delta_R`, `M` (target variable), `E_sum`, `system_pt_mag`, `pt1`, and `pt2`, are transformed using `cp.log1p` (log1p transformation) to handle their skewed distributions.

## Modeling

The project evaluates several regression models, both without and with the engineered features, utilizing GPU-accelerated libraries (cuML, CuPy, XGBoost with CUDA). Data is scaled using `RobustScaler` before model training.

### Models Tested (Without Feature Engineering)

* **Linear Regression**: Showed poor performance due to the inherently non-linear relationship of invariant mass.
* **Polynomial Regression**: Achieved high R² (approx. 0.97) by using 2nd-degree polynomial features, effectively capturing non-linearity.
* **Random Forest Regressor**: Demonstrated strong performance with an R² of 0.975 on the test set, showing some overfitting.
* **XGBoost Regressor**: Delivered exceptional results with an R² of 0.989 on the test set, outperforming Random Forest and showing better generalization.

### Models Tested (With Feature Engineering)

* **Linear Regression**: Dramatically improved to an R² of 0.931 on the test set, showcasing the effectiveness of feature engineering in making complex relationships accessible to simpler models.
* **Polynomial Regression**: Achieved near-perfect results with a test set R² of 0.998, precisely learning the underlying physical formula.
* **Random Forest Regressor**: Also delivered outstanding performance with a test set R² of 0.998.
* **XGBoost Regressor**: Achieved top-tier performance on par with Polynomial and Random Forest models on the feature-engineered dataset.

## Results

The inclusion of engineered features, particularly those derived from the physical invariant mass formula, significantly improved the performance of all models, especially the Linear Regression. Among all models tested, the **Polynomial Regression**, **Random Forest Regressor**, and **XGBoost Regressor** (all with feature engineering) achieved exceptional and very similar R² scores (around 0.998) on the test set, demonstrating their superior predictive capability for this task. The notebook highlights XGBoost as the top performer due to its robust generalization without feature engineering, and its continued strong performance with engineered features.

## Deployment

The best-performing XGBoost model is saved using `joblib` as `xgb_invariant_mass_model.joblib` for future deployment.

## Prerequisites

This project heavily relies on NVIDIA RAPIDS libraries for GPU acceleration. Ensure you have a compatible NVIDIA GPU and CUDA Toolkit installed before proceeding.
