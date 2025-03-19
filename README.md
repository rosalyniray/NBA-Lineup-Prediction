# NBA-Lineup-Prediction

## Project Overview
This project aims to predict the optimal fifth player for a home team in an NBA game, given partial lineup data and other game-related features. The goal is to maximize the home team's overall performance by leveraging historical NBA game data from 2007 to 2015. The project involves data preprocessing, feature engineering, model training, and evaluation, with a focus on adhering to feature constraints and ensuring model explainability.

---

## Table of Contents
1. [Project Objective](#project-objective)
2. [Dataset](#dataset)
3. [Feature Restrictions](#feature-restrictions)
4. [Model Architecture](#model-architecture)
5. [Code Structure](#code-structure)
6. [How to Run the Code](#how-to-run-the-code)
7. [Results](#results)
8. [Deliverables](#deliverables)
9. [Evaluation Criteria](#evaluation-criteria)
10. [Contributors](#contributors)

---

## Project Objective
The primary objective of this project is to build a machine learning model that predicts the optimal fifth player for a home team in an NBA game. The model uses historical game data, including player statistics, team compositions, and game outcomes, to recommend a player who maximizes the home team's performance.

---

## Dataset
The dataset contains NBA game data from 2007 to 2015, including:
- Game-related features (e.g., season, starting minute, home/away team).
- Player statistics and team compositions.
- Game outcomes (e.g., win/loss, performance metrics).

### Feature Restrictions
Only features specified in the provided `allowed_features.txt` file may be used for model training and testing. The code strictly adheres to these constraints.

---

## Model Architecture
The project uses a combination of machine learning models and techniques:
1. **Gradient Boosting Regressor (GBR)**:
   - Primary model for predicting lineup effectiveness.
   - Handles non-linear relationships and provides feature importance scores.
2. **K-Means Clustering**:
   - Groups players based on performance metrics to identify similar players.
   - Cluster labels are used as additional features.
3. **Naive Bayes**:
   - Predicts the likelihood of a player being the optimal fifth player based on historical data.
4. **FP-Growth**:
   - Identifies frequent player combinations that historically perform well.
   - Frequent patterns are used as additional features.

---

## Code Structure
The repository is organized as follows:
```
nba-lineup-prediction/
├── data/
│   ├── processed/            # Processed data files
│   └── raw/
│   └── meta/
├── backup/                   # Test data files
├── models/                   # Saved models and encoders
│── data_processor.py         # Data loading and preprocessing
│── feature_engineering.py    # Feature engineering and encoding
│── model_trainer.py          # Model training and evaluation
│── predictor.py              # Prediction and recommendation logic
│── test_script.py            # Script for testing the model                              
└── main.py                   # Main script to run the pipeline
```

---

## How to Run the Code
### Prerequisites
- Python 3.8 or higher.
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Steps to Run the Code
1. **Training the Model**:
   - Run the training pipeline:
     ```bash
     python main.py --train
     ```
   - This will preprocess the data, train the model, and save it to the `models/` directory.

2. **Making Predictions**:
   - Run the interactive prediction script:
     ```bash
     python main.py --predict
     ```
   - Follow the prompts to input lineup data and get the optimal fifth player recommendation.

3. **Testing the Model**:
   - Run the test script to evaluate the model on the provided test data:
     ```bash
     python test_script.py
     ```
     - --row n to test a row
     - --range n-m to test a range
     - --detailed for full output
     - --result to get accuracy of entire program

---

## Results
The model's performance is evaluated based on:
- **R² Score**: Measures how well the model explains the variance in lineup effectiveness.
- **RMSE**: Measures the average prediction error.
- **Accuracy**: Percentage of correct predictions on the test data.

### Sample Output
```
Model R² on training data: 0.85
Model R² on test data: 0.82
Model RMSE on test data: 0.12

Top 5 recommended players:
1. Player A (effectiveness score: 0.95)
2. Player B (effectiveness score: 0.93)
3. Player C (effectiveness score: 0.91)
```
---

## Contributors
- Rosalyn Sayim

---

For any questions or issues, please open an issue in the repository or contact [your email address].
