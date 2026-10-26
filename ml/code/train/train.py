import os
import pandas as pd
import pickle
import logging
import sys
import glob
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.preprocessing import PowerTransformer
from scipy import stats
import psutil
import time
import sklearn

# Logging setup
root_logger = logging.getLogger()

# Console Handler
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# File Handler
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(f"{log_path}/logfile.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logFormatter)
root_logger.addHandler(file_handler)

# Application Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_directory_info(current_directory=os.getcwd()):
    logger.info("Current working directory: %s", current_directory)
    try:
        directory_contents = os.listdir(current_directory)
        logger.info("Directory contents: %s", directory_contents)
    except (OSError, PermissionError) as e:
        logger.warning("Cannot list directory contents: %s", str(e))

def load_data(file_path):
    log_directory_info(file_path)
    all_files = glob.glob(os.path.join(file_path, "*.csv"))
    
    if not all_files:
        raise ValueError(f"No CSV files found in directory: {file_path}")
    
    logger.info("Reading files with Pandas. Found %d CSV file(s)", len(all_files))
    
    try:
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        if df.empty:
            raise ValueError(f"All CSV files in {file_path} are empty")
        logger.info("Successfully loaded data with shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading CSV files: %s", str(e))
        raise

def prepare_data(df):
    """
    Prepare data for regression model training.
    Filters out rows with NULL target values and prepares feature columns.
    """
    logger.info("Preparing data for regression model...")
    
    # Log available columns
    logger.info("Available columns in the dataset: %s", list(df.columns))
    
    # Verify target column exists
    target_col = 'pct_change_cltv_oct_to_nov'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    # Filter out rows where target variable is NULL
    initial_rows = len(df)
    df_clean = df[df[target_col].notna()].copy()
    filtered_rows = len(df_clean)
    logger.info("Filtered out %d rows with NULL target values. Remaining: %d", 
                initial_rows - filtered_rows, filtered_rows)
    
    if df_clean.empty:
        raise ValueError(f"All rows were filtered out. No valid target values found in column '{target_col}'")
    
    # Check for infinite values in target
    if np.isinf(df_clean[target_col]).any():
        logger.warning("Found infinite values in target variable. Filtering them out.")
        df_clean = df_clean[~np.isinf(df_clean[target_col])].copy()
        logger.info("Remaining rows after filtering infinite values: %d", len(df_clean))
        
        if df_clean.empty:
            raise ValueError("All rows were filtered out due to infinite target values")
    
    # Log target variable statistics
    logger.info("Target variable (%s) statistics:", target_col)
    logger.info("  Mean: %.4f", df_clean[target_col].mean())
    logger.info("  Median: %.4f", df_clean[target_col].median())
    logger.info("  Std: %.4f", df_clean[target_col].std())
    logger.info("  Min: %.4f", df_clean[target_col].min())
    logger.info("  Max: %.4f", df_clean[target_col].max())
    
    return df_clean

def check_normality(series, alpha=0.05, max_samples=5000):
    """
    Check if a numerical series follows a normal distribution.
    Uses D'Agostino-Pearson test (more powerful than Shapiro-Wilk for larger samples).
    
    Parameters:
    -----------
    series : pd.Series
        Numerical series to test
    alpha : float
        Significance level (default 0.05)
    max_samples : int
        Maximum samples to use for test (for performance)
    
    Returns:
    --------
    dict : Dictionary with test results
    """
    # Remove NaN and infinite values
    clean_series = series.dropna()
    clean_series = clean_series[np.isfinite(clean_series)]
    
    if len(clean_series) < 8:  # Minimum samples needed for D'Agostino-Pearson
        return {'is_normal': False, 'p_value': 0.0, 'reason': 'insufficient_samples'}
    
    # Sample if too large (for performance)
    if len(clean_series) > max_samples:
        clean_series = clean_series.sample(n=max_samples, random_state=42)
    
    try:
        # D'Agostino-Pearson test (tests both skewness and kurtosis)
        stat, p_value = stats.normaltest(clean_series)
        is_normal = p_value > alpha
    except Exception as e:
        logger.warning(f"Error in normality test for {series.name}: {str(e)}")
        return {'is_normal': False, 'p_value': 0.0, 'reason': 'test_error'}
    
    return {
        'is_normal': is_normal,
        'p_value': p_value,
        'statistic': stat,
        'sample_size': len(clean_series)
    }

def transform_numerical_features(train_df, test_df, target_col='pct_change_cltv_oct_to_nov', 
                                  exclude_cols=None, normality_alpha=0.05):
    """
    Check normality of numerical features and apply transformations if needed.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    target_col : str
        Name of target column (excluded from transformation)
    exclude_cols : list
        Additional columns to exclude from transformation (e.g., 'user_id')
    normality_alpha : float
        Significance level for normality test
    
    Returns:
    --------
    tuple : (transformed_train_df, transformed_test_df, transformation_info)
    """
    if exclude_cols is None:
        exclude_cols = []
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)
    
    logger.info("Checking normality of numerical features...")
    
    # Identify numerical columns (exclude target and other specified columns)
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    logger.info("Found %d numerical feature columns to check", len(numerical_cols))
    
    if len(numerical_cols) == 0:
        logger.warning("No numerical feature columns found to transform. Only target and excluded columns present.")
    
    # Copy dataframes for transformation
    train_transformed = train_df.copy()
    test_transformed = test_df.copy()
    
    transformation_info = {}
    transformed_count = 0
    
    for col in numerical_cols:
        # Check normality on training data only
        normality_result = check_normality(train_df[col], alpha=normality_alpha)
        
        if not normality_result['is_normal']:
            logger.info("Column '%s' is not normally distributed (p-value: %.4f)", 
                       col, normality_result['p_value'])
            
            # Determine best transformation
            best_transform = None
            best_p = 0.0
            
            # Test different transformations on training data
            valid_train = train_df[col].dropna()
            valid_train_mask = ~(pd.isna(train_df[col]) | np.isinf(train_df[col]))
            
            if valid_train_mask.sum() == 0:
                continue
            
            valid_train_values = valid_train.values
            
            # Try log transformation (if all values > 0)
            if (valid_train_values > 0).all():
                try:
                    log_values = np.log1p(valid_train_values - valid_train_values.min() + 1e-10)
                    _, p = stats.normaltest(log_values)
                    if p > best_p:
                        best_p = p
                        best_transform = ('log', None)
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    logger.debug("Log transformation failed for column '%s': %s", col, str(e))
            
            # Try square root (if all values >= 0)
            if (valid_train_values >= 0).all():
                try:
                    sqrt_values = np.sqrt(valid_train_values + 1e-10)
                    _, p = stats.normaltest(sqrt_values)
                    if p > best_p:
                        best_p = p
                        best_transform = ('sqrt', None)
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    logger.debug("Square root transformation failed for column '%s': %s", col, str(e))
            
            # Try Yeo-Johnson (handles negative values)
            try:
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                yj_transformed = pt.fit_transform(valid_train_values.reshape(-1, 1)).flatten()
                _, p = stats.normaltest(yj_transformed)
                if p > best_p:
                    best_p = p
                    best_transform = ('yeojohnson', pt)  # Store fitted transformer
            except (ValueError, RuntimeError, FloatingPointError, AttributeError) as e:
                logger.debug("Yeo-Johnson transformation failed for column '%s': %s", col, str(e))
            
            # Apply best transformation if found
            if best_transform is not None:
                transform_type, transformer = best_transform
                
                # Transform training data
                if transform_type == 'log':
                    train_transformed.loc[valid_train_mask, col] = np.log1p(
                        valid_train_values - valid_train_values.min() + 1e-10
                    )
                elif transform_type == 'sqrt':
                    train_transformed.loc[valid_train_mask, col] = np.sqrt(
                        valid_train_values + 1e-10
                    )
                elif transform_type == 'yeojohnson':
                    train_transformed.loc[valid_train_mask, col] = transformer.transform(
                        valid_train_values.reshape(-1, 1)
                    ).flatten()
                
                # Apply same transformation to test data
                valid_test_mask = ~(pd.isna(test_df[col]) | np.isinf(test_df[col]))
                if valid_test_mask.sum() > 0:
                    valid_test_values = test_df.loc[valid_test_mask, col].values
                    
                    if transform_type == 'log':
                        # Use same min from training
                        train_min = valid_train_values.min()
                        test_transformed.loc[valid_test_mask, col] = np.log1p(
                            valid_test_values - train_min + 1e-10
                        )
                    elif transform_type == 'sqrt':
                        test_transformed.loc[valid_test_mask, col] = np.sqrt(
                            valid_test_values + 1e-10
                        )
                    elif transform_type == 'yeojohnson':
                        # Use fitted transformer
                        test_transformed.loc[valid_test_mask, col] = transformer.transform(
                            valid_test_values.reshape(-1, 1)
                        ).flatten()
                
                # Check normality after transformation
                post_normality = check_normality(train_transformed[col], alpha=normality_alpha)
                
                # Store transformation info (including transformer for Yeo-Johnson)
                transform_params = {}
                if transform_type == 'log':
                    transform_params['train_min'] = float(valid_train_values.min())
                elif transform_type == 'yeojohnson':
                    # Save the fitted transformer with sklearn version info for compatibility
                    transform_params['transformer'] = transformer
                    transform_params['sklearn_version'] = sklearn.__version__
                
                transformation_info[col] = {
                    'original_p_value': normality_result['p_value'],
                    'transform_type': transform_type,
                    'post_transform_p_value': post_normality['p_value'],
                    'improved': post_normality['p_value'] > normality_result['p_value'],
                    'parameters': transform_params
                }
                
                logger.info("  Applied %s transformation. New p-value: %.4f (improved: %s)", 
                           transform_type, post_normality['p_value'],
                           post_normality['p_value'] > normality_result['p_value'])
                transformed_count += 1
            else:
                logger.warning("  Could not find suitable transformation for '%s'", col)
                transformation_info[col] = {
                    'original_p_value': normality_result['p_value'],
                    'transform_type': 'none',
                    'reason': 'no_suitable_transform'
                }
        else:
            transformation_info[col] = {
                'original_p_value': normality_result['p_value'],
                'transform_type': 'none',
                'reason': 'already_normal'
            }
    
    logger.info("Applied transformations to %d out of %d numerical features", 
               transformed_count, len(numerical_cols))
    
    return train_transformed, test_transformed, transformation_info

def split_data(df):
    """
    Split data into training and testing sets for regression.
    """
    logger.info("Splitting data into train and test sets...")
    
    # Separate features and target
    target_col = 'pct_change_cltv_oct_to_nov'
    
    # Validate that we have enough data to split
    if len(df) < 2:
        raise ValueError(f"Insufficient data for train/test split. Need at least 2 rows, got {len(df)}")
    
    # Drop target and user_id (user_id is identifier, not a feature)
    feature_cols = [col for col in df.columns if col not in [target_col, 'user_id']]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Only target and user_id columns present.")
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Validate splits are not empty
    if train.empty:
        raise ValueError("Training set is empty after split")
    if test.empty:
        raise ValueError("Test set is empty after split")
    
    logger.info("Training data shape: %s", train.shape)
    logger.info("Testing data shape: %s", test.shape)
    logger.info("Number of features: %d", len(feature_cols))
    
    return train, test

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics matching metrics_definition.json format.
    Returns a dictionary with all regression metrics.
    """
    # Filter out NaN and infinite values for metric calculation
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = np.array(y_true[mask])
    y_pred_clean = np.array(y_pred[mask])
    
    if len(y_true_clean) == 0:
        logger.warning("No valid predictions for metric calculation")
        return {}
    
    # Calculate standard regression metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_clean, y_pred_clean)
    median_ae = median_absolute_error(y_true_clean, y_pred_clean)
    mean_error = np.mean(y_pred_clean - y_true_clean)  # Bias
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Handle division by zero
    non_zero_mask = y_true_clean != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
    else:
        mape = np.nan
        logger.warning("Cannot calculate MAPE: all actual values are zero")
    
    metrics = {
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Mean Error': mean_error,
        'Median Absolute Error': median_ae,
        'MSE': mse
    }
    
    return metrics

def train_and_export_model(train, test, model_dir):
    """
    Train regression model to predict pct_change_cltv_oct_to_nov.
    """
    logger.info("Training the CLTV change regression model using AutoGluon...")
    
    target_col = 'pct_change_cltv_oct_to_nov'
    
    predictor = TabularPredictor(
        label=target_col,
        path=model_dir,
        problem_type='regression',
        eval_metric='root_mean_squared_error'
    )
    
    logger.info("Starting model training...")
    # NOTE: Using test set as tuning_data is acceptable for AutoGluon.
    # AutoGluon uses tuning_data for early stopping and model selection,
    # but final evaluation should be done on a separate validation set if available.
    predictor.fit(
        train_data=train,
        tuning_data=test,
        presets='medium_quality',
        use_bag_holdout=True  # Ensures bagging uses holdout validation
    )
    
    logger.info("Evaluating model on test data...")
    evaluation_metrics = predictor.evaluate(test)
    logger.info(f"AutoGluon evaluation metrics: {evaluation_metrics}")
    
    # Get predictions for custom metric calculation
    y_test = test[target_col].values
    y_pred = predictor.predict(test.drop(columns=[target_col]))
    
    # Calculate all regression metrics
    regression_metrics = calculate_regression_metrics(y_test, y_pred)
    
    # Log metrics in format matching metrics_definition.json
    logger.info("\n" + "="*60)
    logger.info("REGRESSION MODEL EVALUATION METRICS")
    logger.info("="*60)
    for metric_name, metric_value in regression_metrics.items():
        if not np.isnan(metric_value):
            logger.info(f"{metric_name}: {metric_value:.4f}")
    logger.info("="*60 + "\n")
    
    # Get leaderboard of top models
    logger.info("Fetching leaderboard of top 10 models...")
    leaderboard = predictor.leaderboard(test, extra_info=True, silent=False)
    top_10_models = leaderboard['model'].head(10).tolist()
    logger.info(f"Top 10 models: {top_10_models}")
    
    # Evaluate top 10 models on test data
    logger.info("\nEvaluating top 10 individual models:")
    logger.info("="*60)
    for model_name in top_10_models:
        if 'ensemble' in model_name.lower():
            logger.warning(f"Skipping ensemble model: {model_name}")
            continue
        try:
            # Get predictions from this specific model
            y_pred_model = predictor.predict(test.drop(columns=[target_col]), model=model_name)
            
            # Calculate regression metrics for this model
            model_metrics = calculate_regression_metrics(y_test, y_pred_model)
            
            logger.info(f"\nModel: {model_name}")
            for metric_name, metric_value in model_metrics.items():
                if not np.isnan(metric_value):
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
    
    logger.info("="*60 + "\n")
    
    logger.info("Saving trained model...")
    predictor.save()
    logger.info(f"Model saved in directory: {model_dir}")
    return predictor

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution of {func.__name__} completed in {execution_time:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def main():
    try:
        logger.info("Executing CLTV Change Regression Training Script")
        dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train/')
        model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')
        os.makedirs(model_dir, exist_ok=True)
        
        # Load data
        df = load_data(dataset_path)
        logger.info("Loaded dataset with shape: %s", df.shape)
        
        # Prepare data for regression
        df = prepare_data(df)
        
        # Split data into train and test sets (before transformation to avoid data leakage)
        train, test = split_data(df)
        
        # Check normality and apply transformations to numerical features
        logger.info("\n" + "="*60)
        logger.info("NORMALITY CHECK AND FEATURE TRANSFORMATION")
        logger.info("="*60)
        train_transformed, test_transformed, transformation_info = transform_numerical_features(
            train, test, 
            target_col='pct_change_cltv_oct_to_nov',
            exclude_cols=['user_id'],
            normality_alpha=0.05
        )
        
        # Save transformation info for inference
        transformation_info_path = os.path.join(model_dir, "feature_transformations.pkl")
        try:
            with open(transformation_info_path, 'wb') as f:
                pickle.dump(transformation_info, f)
            logger.info("Transformation info saved to: %s", transformation_info_path)
        except (pickle.PicklingError, IOError) as e:
            logger.error("Failed to save transformation info: %s", str(e))
            # Log which transformers might be causing issues
            problematic_cols = []
            for col, info in transformation_info.items():
                if info.get('transform_type') == 'yeojohnson':
                    problematic_cols.append(col)
            if problematic_cols:
                logger.warning("Columns with Yeo-Johnson transformers that may have serialization issues: %s", 
                             problematic_cols)
            raise
        logger.info("="*60 + "\n")
        
        # Train and export model using transformed data
        predictor = train_and_export_model(train_transformed, test_transformed, model_dir)
        
        logger.info("CLTV Change Regression Training Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
