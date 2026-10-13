#!/usr/bin/env python3
"""
Synthetic Health App Subscription Churn Dataset Generator

Generates a realistic synthetic dataset for binary classification of health app subscription churn.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

def generate_health_app_churn_data(n_rows=50000, random_seed=42):
    """
    Generate synthetic health app churn dataset.

    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)

    print(f"Generating {n_rows:,} rows of synthetic health app churn data...")

    # Initialize dataframe
    df = pd.DataFrame()

    # 1. ID column
    df['user_id'] = np.random.randint(100000, 1000000, size=n_rows)
    # Ensure uniqueness
    while len(df['user_id'].unique()) < n_rows:
        duplicates = df['user_id'].duplicated()
        df.loc[duplicates, 'user_id'] = np.random.randint(100000, 1000000, size=duplicates.sum())

    # 2. Hidden factors (latent variables that drive behavior)
    # These are not in the final dataset but drive the observed variables
    health_motivation = np.random.beta(2, 2, n_rows)  # Beta distribution for bounded latent trait
    price_sensitivity = np.random.beta(2, 2, n_rows)
    tech_comfort = np.random.beta(2, 2, n_rows)
    social_influence = np.random.beta(2, 2, n_rows)
    time_availability = np.random.beta(2, 2, n_rows)

    # 3. Basic demographics
    df['age'] = np.clip(np.random.normal(35, 12, n_rows), 18, 80).astype(int)

    gender_probs = [0.48, 0.48, 0.04]  # Male, Female, Non-binary
    df['gender'] = np.random.choice(['Male', 'Female', 'Non-binary'], n_rows, p=gender_probs)

    # 4. Subscription details
    # Annual subscribers tend to be more committed
    subscription_prob = 0.3 + 0.2 * health_motivation - 0.1 * price_sensitivity
    subscription_prob = np.clip(subscription_prob, 0.1, 0.6)
    # Generate subscription type for each row
    prob_annual = subscription_prob
    prob_monthly = 1 - subscription_prob
    # Use binomial to generate: 1 for Annual, 0 for Monthly
    annual_choice = np.random.binomial(1, prob_annual, n_rows)
    df['subscription_type'] = np.where(annual_choice == 1, 'Annual', 'Monthly')

    # Monthly cost varies by subscription type and price sensitivity
    base_cost = np.where(df['subscription_type'] == 'Monthly', 9.99, 79.99)  # $9.99/month or $79.99/year (~$6.67/month equivalent)
    # Apply price sensitivity effect: sensitive users less likely to be on expensive plans
    cost_adjustment = 1 - (price_sensitivity * 0.5)  # Reduce cost for price sensitive
    df['monthly_cost'] = base_cost * cost_adjustment
    # Add some noise and ensure bounds
    df['monthly_cost'] = np.clip(df['monthly_cost'] + np.random.normal(0, 2, n_rows), 4.99, 59.99)
    df['monthly_cost'] = df['monthly_cost'].round(2)

    # 5. Tenure (derived from signup date)
    # More motivated users tend to have longer tenure
    df['tenure_months'] = np.clip(
        np.random.exponential(12, n_rows) * (0.5 + health_motivation),
        0, 60
    ).astype(int)

    # 6. Signup date features (simulated)
    # We'll generate these based on tenure
    # Assume observation date is fixed (e.g., 2023-12-31)
    observation_date = datetime(2023, 12, 31)
    signup_dates = []
    for tenure in df['tenure_months']:
        # Random signup date within tenure period
        max_days_back = tenure * 30
        days_back = np.random.randint(0, max(max_days_back, 1))
        signup_date = observation_date - timedelta(days=days_back)
        signup_dates.append(signup_date)

    df['signup_date'] = signup_dates
    df['signup_weekday'] = [dt.strftime('%A') for dt in signup_dates]
    df['signup_month'] = [dt.strftime('%B') for dt in signup_dates]

    # 7. Primary fitness goal (influenced by health motivation and age)
    goal_probs = np.array([
        0.25,  # Weight Loss
        0.20,  # Muscle Gain
        0.15,  # Endurance
        0.10,  # Flexibility
        0.15,  # Stress Reduction
        0.15   # General Fitness
    ])

    # Adjust probabilities based on age and motivation
    # Younger users more likely to want muscle gain
    # Older users more likely to want weight loss or stress reduction
    age_factor = (df['age'] - 18) / (80 - 18)  # Normalized age 0-1
    goal_probs_adj = np.tile(goal_probs, (n_rows, 1))

    # Adjust for age effects
    goal_probs_adj[:, 0] *= (1 + age_factor * 0.5)  # Weight loss increases with age
    goal_probs_adj[:, 1] *= (1 + (1 - age_factor) * 0.5)  # Muscle gain decreases with age
    goal_probs_adj[:, 4] *= (1 + age_factor * 0.3)  # Stress reduction increases with age

    # Renormalize
    goal_probs_adj = goal_probs_adj / goal_probs_adj.sum(axis=1, keepdims=True)

    # Sample goals
    goals = ['Weight Loss', 'Muscle Gain', 'Endurance', 'Flexibility', 'Stress Reduction', 'General Fitness']
    # Sample goals for each row using the adjusted probabilities
    primary_goal_indices = np.array([np.random.choice(len(goals), p=probs) for probs in goal_probs_adj])
    df['primary_goal'] = [goals[idx] for idx in primary_goal_indices]

    # 8. Region (somewhat random with slight correlations)
    region_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal probability for 5 regions
    # Generate region for each row
    regions = ['Northwest', 'Southwest', 'Midwest', 'Southeast', 'Northeast']
    # Use multinomial to generate indices
    region_indices = np.random.choice(len(regions), size=n_rows, p=region_probs)
    df['region'] = [regions[idx] for idx in region_indices]

    # 9. Device type (correlated with age and tech comfort)
    # Younger, tech-comfortable users more likely iOS
    device_logits = (
        -0.5 +  # Base preference for Android
        tech_comfort * 1.0 +  # Tech comfort increases iOS preference
        (1 - age_factor) * 0.8  # Younger users prefer iOS
    )
    device_probs = 1 / (1 + np.exp(-device_logits))  # Sigmoid to get probability
    device_probs = np.clip(device_probs, 0.1, 0.9)  # Keep reasonable bounds
    # Generate device type for each row
    prob_ios = device_probs
    prob_android = 1 - device_probs
    # Use binomial to generate: 1 for iOS, 0 for Android
    ios_choice = np.random.binomial(1, prob_ios, n_rows)
    df['device_type'] = np.where(ios_choice == 1, 'iOS', 'Android')
    # Add Web as a small percentage
    web_mask = np.random.random(n_rows) < 0.05
    df.loc[web_mask, 'device_type'] = 'Web'
    # Re-normalize iOS/Android for non-web users
    non_web = ~web_mask
    if non_web.any():
        ios_prob_nonweb = device_probs[non_web] / (device_probs[non_web] + (1 - device_probs[non_web]))
        android_prob_nonweb = 1 - ios_prob_nonweb
        # Use binomial to generate: 1 for iOS, 0 for Android
        ios_choice_nonweb = np.random.binomial(1, ios_prob_nonweb, size=non_web.sum())
        df.loc[non_web, 'device_type'] = np.where(ios_choice_nonweb == 1, 'iOS', 'Android')

    # 10. Workout behavior (driven by health motivation and time availability)
    # Base workout frequency
    base_workouts = 3 + 4 * health_motivation * time_availability  # 3-7 workouts/week
    df['workouts_per_week'] = np.clip(np.random.poisson(base_workouts, n_rows), 0, 21)

    # Weekly workout minutes (correlated with workouts per week and session duration)
    base_minutes_per_workout = 30 + 30 * health_motivation  # 30-60 mins per workout
    df['weekly_workout_minutes'] = df['workouts_per_week'] * base_minutes_per_workout * (1 + np.random.normal(0, 0.2, n_rows))
    df['weekly_workout_minutes'] = np.clip(df['weekly_workout_minutes'], 0, 600)

    # Average session duration
    df['avg_session_duration'] = np.clip(
        base_minutes_per_workout + np.random.normal(0, 10, n_rows),
        0, 120
    )

    # 11. Streaks
    # Longest streak influenced by motivation, time, and tenure
    max_possible_streak = np.minimum(df['tenure_months'] * 30, 365)  # Max days in tenure
    streak_probability = health_motivation * time_availability * 0.7
    df['streaks_longest'] = np.clip(
        np.random.binomial(max_possible_streak.astype(int), streak_probability),
        0, 365
    ).astype(int)

    # Current streak (recent behavior)
    df['current_streak'] = np.clip(
        df['streaks_longest'] * np.random.beta(2, 5, n_rows),  # Usually less than longest
        0, 365
    ).astype(int)

    # 12. Goals
    # More motivated users set and achieve more goals
    df['goals_set'] = np.clip(
        np.random.poisson(3 + 5 * health_motivation, n_rows),
        0, 20
    ).astype(int)

    # Achievement rate influenced by motivation and time
    achievement_rate = 0.3 + 0.4 * health_motivation * time_availability
    df['goals_achieved'] = np.clip(
        np.random.binomial(df['goals_set'].values, achievement_rate),
        0, 20
    ).astype(int)

    # 13. Feature usage
    # Nutrition tracking: correlated with health motivation and goals
    nutrition_prob = 0.2 + 0.5 * health_motivation
    nutrition_prob = np.clip(nutrition_prob, 0.05, 0.8)
    df['nutrition_tracking_used'] = np.random.binomial(1, nutrition_prob, n_rows)

    # Meditation: correlated with stress reduction goal and stress tendency
    is_stress_focused = (df['primary_goal'] == 'Stress Reduction').astype(int)
    meditation_prob = 0.1 + 0.4 * is_stress_focused + 0.3 * health_motivation
    meditation_prob = np.clip(meditation_prob, 0.02, 0.7)
    df['meditation_used'] = np.random.binomial(1, meditation_prob, n_rows)

    # Social features: correlated with social influence and time
    social_prob = 0.1 + 0.5 * social_influence * time_availability
    social_prob = np.clip(social_prob, 0.05, 0.6)
    df['social_features_used'] = np.random.binomial(1, social_prob, n_rows)

    # Friends on app: correlated with social influence and tenure
    friend_prob = 0.05 + 0.3 * social_influence * np.minimum(df['tenure_months'] / 24, 1)  # Increases with tenure
    friend_prob = np.clip(friend_prob, 0.02, 0.5)
    df['has_friend_on_app'] = np.random.binomial(1, friend_prob, n_rows)

    # 14. Ratings and satisfaction
    # App rating: influenced by satisfaction, ease of use, and tech comfort
    # Will calculate after satisfaction_score is defined

    # Satisfaction score: driven by multiple factors
    satisfaction_base = (
        0.3 * health_motivation +
        0.2 * (1 - price_sensitivity) +
        0.2 * tech_comfort +
        0.15 * social_influence +
        0.15 * time_availability
    )
    satisfaction_base = np.clip(satisfaction_base, 0, 1)
    df['satisfaction_score'] = np.clip(
        np.random.normal(1 + 4 * satisfaction_base, 1.0, n_rows),
        1, 5
    ).round(1)

    # Ease of use rating: correlated with tech comfort and age
    ease_base = 0.4 * tech_comfort + 0.3 * (1 - age_factor)  # Younger, tech-comfortable find it easier
    ease_base = np.clip(ease_base, 0, 1)
    df['ease_of_use_rating'] = np.clip(
        np.random.normal(1 + 4 * ease_base, 1.0, n_rows),
        1, 5
    ).round(1)

    # App rating: influenced by satisfaction and ease of use
    app_rating_base = 0.5 * df['satisfaction_score'] / 5 + 0.3 * df['ease_of_use_rating'] / 5 + 0.2 * tech_comfort
    df['app_rating'] = np.clip(
        np.random.normal(1 + 4 * app_rating_base, 1.0, n_rows),
        1, 5
    ).round(1)

    # 15. Customer service contacts
    # More issues for less tech-comfortable users and those with problems
    cs_base = (
        0.1 * (1 - tech_comfort) +
        0.1 * price_sensitivity +
        0.05 * (1 - satisfaction_base)  # Less satisfaction -> more contacts
    )
    cs_base = np.clip(cs_base, 0, 0.5)
    df['customer_service_contacts'] = np.random.poisson(cs_base * 4, n_rows)  # Scale up
    df['customer_service_contacts'] = np.clip(df['customer_service_contacts'], 0, 10)

    # 16. Discount usage
    # Price-sensitive users more likely to use discounts
    discount_prob = 0.1 + 0.4 * price_sensitivity
    discount_prob = np.clip(discount_prob, 0.02, 0.5)
    df['discount_used'] = np.random.binomial(1, discount_prob, n_rows)

    # 17. Monthly active days
    # Correlated with workout frequency and motivation
    base_active_days = 5 + 10 * health_motivation * time_availability  # 5-15 days
    df['monthly_active_days'] = np.clip(
        np.random.normal(base_active_days, 3, n_rows),
        0, 31
    ).astype(int)

    # 18. TARGET VARIABLE: Churn
    # Build probability model with:
    # - Strong effects: satisfaction, usage, price sensitivity
    # - Weak effects: age, region, device type
    # - Non-linear effects: thresholds, U-shapes, saturation
    # - Interactions: between motivation and usage, price and sensitivity

    # Initialize log-odds
    log_odds = -1.05  # Base log-odds for ~24% baseline probability (further increased signal strength)

    # Strong effects - BALANCED to provide signal but allow ensemble advantage
    log_odds -= 0.5 * (df['satisfaction_score'] - 3) / 2  # Satisfaction effect (moderate)
    log_odds += 0.35 * price_sensitivity  # Price sensitivity effect (moderate)
    log_odds -= 0.45 * (df['weekly_workout_minutes'] / 60)  # Usage effect (moderate)
    log_odds -= 0.2 * (df['monthly_active_days'] / 31)  # Activity effect (moderate)

    # Weak effects - MINIMAL to reduce linear model advantage
    log_odds += 0.03 * (df['age'] - 40) / 20  # Very small age effect (reduced further)
    log_odds += 0.015 * (df['subscription_type'] == 'Monthly').astype(int)  # Monthly subscribers slightly more likely to churn (reduced)

    # Non-linear effects - STRENGTHENED to hurt linear models and shallow trees more
    # 1. Strong threshold effect: very low usage leads to high churn (non-linear)
    low_usage_penalty = np.where(df['weekly_workout_minutes'] < 30, 2.5, 0)  # Increased penalty
    log_odds += 1.08 * low_usage_penalty  # Strong threshold effect (increased by 20%)

    # 2. Strong U-shape effect: both very new and very long-term users might churn
    tenure_effect = (-0.7 * np.exp(-df['tenure_months'] / 4) + 0.6 * np.exp(-((df['tenure_months'] - 20) ** 2) / (2 * 8 ** 2))) * 1.2
    log_odds += tenure_effect

    # 3. Moderate saturation effects
    sat_extreme_low = np.where(df['satisfaction_score'] < 2, 1.5, 0)  # Very low satisfaction
    sat_extreme_high = np.where(df['satisfaction_score'] > 4.5, 1.0, 0)  # Very high satisfaction (less churn)
    log_odds += 0.54 * sat_extreme_low  # Penalty for very low satisfaction (increased by 20%)
    log_odds -= 0.216 * sat_extreme_high  # Reward for very high satisfaction (increased by 20%)

    # Interactions - ENHANCED to create complex dependencies that favor ensembles over linear models and shallow trees
    # 1. Complex motivation-usage interaction: highly non-linear
    motivation_usage_raw = health_motivation * (df['weekly_workout_minutes'] / 60)
    # Create interaction that benefits ensembles: strong effects only in specific combinations
    motivation_usage_interaction = np.where((motivation_usage_raw > 0.7) & (motivation_usage_raw < 0.9),
                                           0.8,  # Strong positive effect in middle-high range
                                           np.where(motivation_usage_raw > 0.9,
                                                   0.3,  # Moderate effect at very high
                                                   0.0))   # No effect at low-medium
    log_odds -= 0.72 * motivation_usage_interaction  # Strong interaction (increased by 20%)

    # 2. Price sensitivity interacts with usage in complex way
    # Creates patterns that are hard for linear models to capture
    price_usage_base = price_sensitivity * (1 - df['weekly_workout_minutes'] / 60)
    # Non-linear transformation that helps ensembles
    price_usage_interaction = np.where(price_usage_base > 0.6,
                                      price_usage_base * 2.0,  # Strong effect
                                      np.where(price_usage_base > 0.3,
                                              price_usage_base * 0.8,  # Medium effect
                                              price_usage_base * 0.2)) # Weak effect
    log_odds += 0.6 * price_usage_interaction  # Strong interaction (increased by 20%)

    # 3. Age modifies how satisfaction affects churn (complex non-linear interaction)
    # Creates age-dependent satisfaction thresholds with regional variations
    age_factor = (df['age'] - 40) / 20  # -1 to +2 range
    # Region-specific modulation (simplified)
    region_mod = np.select([
        df['region'] == 'Northwest',
        df['region'] == 'Southeast',
        df['region'] == 'Midwest'
    ], [0.2, -0.2, 0.0], default=0.0)

    sat_threshold = 2.5 + 0.4 * np.tanh(age_factor) + 0.2 * region_mod  # Complex threshold
    age_sat_interaction = np.where(df['satisfaction_score'] < sat_threshold,
                                  (sat_threshold - df['satisfaction_score']) * 0.4,
                                  -0.2 * np.where(df['satisfaction_score'] > 4.5, 1.0, 0))  # Reward for high sat
    log_odds += 0.48 * age_sat_interaction  # Strong interaction (increased by 20%)

    # 4. Subscription type interacts with price sensitivity and usage (enhanced)
    # Creates a three-way interaction that hurts linear models
    sub_price_usage = ((df['subscription_type'] == 'Monthly').astype(int) *
                      price_sensitivity *
                      (1 - df['weekly_workout_minutes'] / 60) *
                      np.where(df['weekly_workout_minutes'] > 30, 1.2, 0.8))  # Usage modifier
    log_odds += 0.42 * sub_price_usage  # Enhanced interaction (increased by 20%)

    # 5. Social features interaction: more complex pattern
    # Friends reduce churn effect in complex ways that depend on other factors
    social_base = (1 - df['has_friend_on_app']) * df['weekly_workout_minutes'] / 60
    # Interaction that varies by satisfaction level
    sat_mod = np.where(df['satisfaction_score'] < 2.5, 1.5,
                      np.where(df['satisfaction_score'] > 4.0, 0.5, 1.0))
    social_interaction = social_base * sat_mod * np.where(df['satisfaction_score'] > 3.0, 1.2, 0.8)
    log_odds += 0.504 * social_interaction  # Slightly enhanced interaction (increased by 20%)

    # Additional term to specifically challenge decision trees:
    # Create a pattern that requires deeper trees to capture
    # Interaction between age, usage, and satisfaction that creates local optima
    age_usage_sat = ((df['age'] - 40) / 20) * (df['weekly_workout_minutes'] / 60) * ((df['satisfaction_score'] - 3) / 2)
    # Create non-linear response that benefits deeper trees
    dt_challenge = np.where(np.abs(age_usage_sat) > 0.4,
                           np.sign(age_usage_sat) * 0.3,  # Step function-like response
                           age_usage_sat * 0.1)  # Weak linear response in middle
    log_odds += 0.3 * dt_challenge  # Additional challenge for shallow trees (increased by 20%)

    # Add triple interaction term: satisfaction * usage * price sensitivity
    # This high-order interaction may be too complex for shallow trees
    triple_interaction = ((df['satisfaction_score'] - 3) / 2) * (df['weekly_workout_minutes'] / 60) * price_sensitivity
    log_odds += 0.3 * triple_interaction

    # Add competing noise element that hurts single trees more
    # Adds structured noise that ensembles can average out
    noise_element = np.random.normal(0, 0.75, n_rows) * np.sin(df['tenure_months'] / 3 + df['age'] / 10)
    log_odds += noise_element

    # Add base noise - reduced slightly to preserve signal
    log_odds += np.random.normal(0, 0.75, n_rows)

    # Convert to probability
    churn_prob = 1 / (1 + np.exp(-log_odds))

    # Adjust to get target rate ~25%
    # Find threshold that gives approximately 25% positive
    target_rate = 0.25
    threshold = np.percentile(churn_prob, (1 - target_rate) * 100)
    df['churned'] = (churn_prob >= threshold).astype(int)

    # Verify actual rate
    actual_rate = df['churned'].mean()
    print(f"Actual churn rate: {actual_rate:.3f}")

    # Now introduce missing values and outliers
    print("Introducing missing values and outliers...")

    # Missing values mechanism: 3-6% in at least 5 columns
    # Some completely random (MCAR), some depending on other columns (MAR)

    # 1. Completely random missing (MCAR)
    mcar_cols = ['age', 'gender', 'subscription_type']
    for col in mcar_cols:
        missing_mask = np.random.random(n_rows) < 0.03  # 3% missing
        df.loc[missing_mask, col] = np.nan

    # 2. Missing At Random (MAR) - depends on other columns
    # Monthly cost missing for price sensitive users who didn't use discounts
    mar_mask1 = (price_sensitivity > 0.7) & (df['discount_used'] == 0)
    mar_mask1 &= np.random.random(n_rows) < 0.4  # 40% of those meeting criteria
    df.loc[mar_mask1, 'monthly_cost'] = np.nan

    # Satisfaction score missing for low usage users
    mar_mask2 = (df['weekly_workout_minutes'] < 50) & (np.random.random(n_rows) < 0.3)
    df.loc[mar_mask2, 'satisfaction_score'] = np.nan

    # Customer service contacts missing for very satisfied users
    mar_mask3 = (df['satisfaction_score'] > 4.5) & (np.random.random(n_rows) < 0.25)
    df.loc[mar_mask3, 'customer_service_contacts'] = np.nan

    # Ease of use missing for older, less tech-comfortable users
    mar_mask4 = (df['age'] > 60) & (tech_comfort < 0.3) & (np.random.random(n_rows) < 0.35)
    df.loc[mar_mask4, 'ease_of_use_rating'] = np.nan

    # App rating missing for users who haven't used the app much
    mar_mask5 = (df['monthly_active_days'] < 5) & (np.random.random(n_rows) < 0.3)
    df.loc[mar_mask5, 'app_rating'] = np.nan

    # Outliers: <1% in at least 2 numeric columns
    print("Adding outliers...")

    # Outlier 1: Extreme workout values (a few users who exercise excessively)
    workout_outlier_mask = np.random.random(n_rows) < 0.005  # 0.5%
    df.loc[workout_outlier_mask, 'weekly_workout_minutes'] = np.random.uniform(800, 1500, workout_outlier_mask.sum())

    # Outlier 2: Extreme but still plausible age values
    age_outlier_mask = (np.random.random(n_rows) < 0.005) & (df['age'] > 18) & (df['age'] < 80)
    # Replace some ages with extreme but plausible values (still within 13-100 range)
    age_outlier_indices = np.where(age_outlier_mask)[0]
    if len(age_outlier_indices) > 0:
        half = len(age_outlier_indices) // 2
        df.loc[age_outlier_indices[:half], 'age'] = np.random.uniform(90, 100, half)  # Very old but plausible
        df.loc[age_outlier_indices[half:], 'age'] = np.random.uniform(13, 17, len(age_outlier_indices) - half)  # Very young but plausible

    # Remove temporary columns used for generation
    # (We don't want latent variables in the final dataset)
    # Note: We didn't actually add health_motivation etc. to df, so nothing to remove

    # Reorder columns for readability
    column_order = [
        'user_id', 'churned',
        'age', 'gender', 'subscription_type', 'monthly_cost',
        'tenure_months', 'signup_weekday', 'signup_month',
        'primary_goal', 'region', 'device_type',
        'workouts_per_week', 'weekly_workout_minutes', 'avg_session_duration',
        'streaks_longest', 'current_streak',
        'goals_set', 'goals_achieved',
        'nutrition_tracking_used', 'meditation_used', 'social_features_used', 'has_friend_on_app',
        'app_rating', 'satisfaction_score', 'ease_of_use_rating',
        'customer_service_contacts', 'discount_used', 'monthly_active_days'
    ]

    df = df[column_order]

    return df

def add_constraints_and_assertions(df):
    """
    Add hard logical constraints and verify they hold.
    """
    print("Checking hard logical constraints...")

    violations = []

    # Constraint 1: current_streak <= streaks_longest
    streak_violation = df['current_streak'] > df['streaks_longest']
    if streak_violation.any():
        violations.append(f"current_streak > streaks_longest: {streak_violation.sum()} violations")

    # Constraint 2: goals_achieved <= goals_set
    goals_violation = df['goals_achieved'] > df['goals_set']
    if goals_violation.any():
        violations.append(f"goals_achieved > goals_set: {goals_violation.sum()} violations")

    # Constraint 3: age plausible for app users (already bounded in generation)
    age_violation = (df['age'] < 13) | (df['age'] > 100)
    if age_violation.any():
        violations.append(f"age outside plausible range (13-100): {age_violation.sum()} violations")

    # Constraint 4: monthly_cost positive
    cost_violation = df['monthly_cost'] <= 0
    if cost_violation.any():
        violations.append(f"monthly_cost <= 0: {cost_violation.sum()} violations")

    # Constraint 5: workout time non-negative
    workout_violation = df['weekly_workout_minutes'] < 0
    if workout_violation.any():
        violations.append(f"weekly_workout_minutes < 0: {workout_violation.sum()} violations")

    # Constraint 6: session duration reasonable (< 24 hours)
    session_violation = df['avg_session_duration'] > 1440  # More than 24 hours
    if session_violation.any():
        violations.append(f"avg_session_duration > 1440 minutes: {session_violation.sum()} violations")

    # Constraint 7: streaks not exceeding tenure (approximately)
    # Convert tenure months to days (approximate)
    max_possible_streak_days = df['tenure_months'] * 30.5
    tenure_streak_violation = df['streaks_longest'] > max_possible_streak_days
    if tenure_streak_violation.any():
        violations.append(f"streaks_longest > possible days in tenure: {tenure_streak_violation.sum()} violations")

    # Constraint 8: monthly_active_days <= 31
    active_days_violation = df['monthly_active_days'] > 31
    if active_days_violation.any():
        violations.append(f"monthly_active_days > 31: {active_days_violation.sum()} violations")

    if violations:
        print("CONSTRAINT VIOLATIONS FOUND:")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print("All hard logical constraints satisfied (0 violations)")
        return True

def generate_data_dictionary(df):
    """
    Generate data dictionary CSV from dataframe.
    """
    # This would normally be more detailed, but we'll create a basic version
    # For this synthetic data, we'll reuse/create the dictionary separately
    pass

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic health app churn dataset')
    parser.add_argument('--rows', type=int, default=50000,
                        help='Number of rows to generate (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='health_app_churn_50k.csv',
                        help='Output CSV filename (default: health_app_churn_50k.csv)')

    args = parser.parse_args()

    # Generate data
    df = generate_health_app_churn_data(n_rows=args.rows, random_seed=args.seed)

    # Check constraints
    constraints_ok = add_constraints_and_assertions(df)

    if not constraints_ok:
        print("ERROR: Dataset generation failed due to constraint violations.")
        sys.exit(1)

    # Save to CSV
    print(f"Saving dataset to {args.output}...")
    df.to_csv(args.output, index=False)

    # Basic statistics
    print("\nDataset Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Churn rate: {df['churned'].mean():.3f}")
    print(f"  File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    print("\nGeneration complete!")

if __name__ == "__main__":
    main()