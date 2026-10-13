# Health App Subscription Churn Prediction (Synthetic)

## Subtitle
Realistic synthetic dataset for practicing binary classification on health app subscription churn

## SYNTHETIC DISCLOSURE
⚠️ **IMPORTANT**: This is a completely synthetic dataset generated for educational and benchmarking purposes. No real user data was used or implied. All names, IDs, dates, and other fields are artificially generated.

## The Story
FitLife is a fictional health and wellness mobile application that offers personalized workout plans, nutrition tracking, and mindfulness exercises through a subscription model. Users can subscribe monthly or annually to access premium features.

The business problem: Predict which subscribers are likely to cancel (churn) their subscription in the next month based on their usage patterns, demographic information, and engagement metrics.

## Hidden Factors Driving Behavior
1. **Health Motivation** (latent): Intrinsic drive to maintain fitness goals
2. **Price Sensitivity** (latent): How much cost affects decision-making
3. **Tech Comfort** (latent): Ease with using mobile applications
4. **Social Influence** (latent): Impact of friends/family using the app
5. **Time Availability** (latent): How much free time user has for health activities

## How Hidden Factors Drive Observed Columns
- High health motivation → More workout sessions, longer streaks, goal completion
- High price sensitivity → More likely to check pricing pages, react to discounts
- High tech comfort → More feature usage, fewer app crashes, longer sessions
- High social influence → More social features used, more friends connected
- High time availability → More frequent logins, longer total time in app

## Target Variable
**churned** (binary): Whether the user cancelled their subscription within 30 days of observation period (1 = churned, 0 = active)

## File List
1. `health_app_churn_50k.csv` - Main dataset (50,000 rows, 26 columns)
2. `generate_dataset.py` - Seeded, reproducible data generator with constraints
3. `data_dictionary.csv` - Complete column descriptions, types, ranges
4. `benchmark.py` - Model benchmarking script
5. `benchmark_results.txt` - Raw output from 5-fold cross-validation
6. `README.md` - This file

## Column Overview
- **ID Columns**: user_id (unique identifier)
- **Target**: churned (0/1)
- **Predictors**:
  - Continuous numeric: age, weekly_workout_minutes, avg_session_duration, monthly_cost, etc.
  - Discrete counts: workouts_per_week, streaks_longest, goals_set, etc.
  - Binary flags: has_friend_on_app, used_nutrition_tracking, etc.
  - Ordinal ratings: satisfaction_score (1-5), ease_of_use_rating (1-5)
  - Categorical: subscription_type, primary_goal, region, device_type, etc.
  - Date/Timestamp: signup_date (features extracted: tenure_months, signup_weekday, signup_month)

## Data Quality Notes
- Missing values: 3-6% in 5 columns with documented mechanisms
- Outliers: <1% documented in 2 numeric columns
- Hard constraints: 0 violations (all rows satisfy logical constraints)
- Target rate: ~25% positive class (mildly imbalanced)
- No target leakage: All predictors known at prediction time

## Benchmark Results (5-fold CV)
- Majority Class Baseline ROC-AUC: 0.5000
- Logistic Regression ROC-AUC: 0.7764
- Gaussian Naive Bayes ROC-AUC: 0.7239
- Decision Tree (max_depth=5) ROC-AUC: 0.8121
- Random Forest ROC-AUC: 0.8481
- Gradient Boosting ROC-AUC: 0.8665

## Suggested Uses
- Practicing classification algorithms
- Feature engineering exercises
- Handling imbalanced datasets
- Pipeline construction (imputation, scaling, encoding)
- Model comparison and evaluation

## Known Limitations
- Synthetic nature may not capture all real-world complexities
- Relationships are based on plausible assumptions, not empirical data
- Limited temporal dynamics (static snapshot)

## License
CC BY 4.0 (Attribution Required)

## Tags
healthcare, fitness, churn-prediction, synthetic-data, binary-classification, tabular-data, machine-learning, benchmarking