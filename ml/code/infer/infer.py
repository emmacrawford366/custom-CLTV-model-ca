import os
import pandas as pd
import pickle
import logging
import sys
import io
import numpy as np
from flask import Flask, request as flask_request, jsonify, Response
from autogluon.tabular import TabularPredictor
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

# Flask app setup
app = Flask(__name__)

# Global variables
model = None
feature_transformations = None

# Load the trained model
def load_model():
    global model
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")
    logger.info("Loading model from directory: %s", model_dir)
    try:
        model = TabularPredictor.load(model_dir)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load model: %s", str(e), exc_info=True)
        raise

# Load the feature transformations
def load_feature_transformations():
    global feature_transformations
    transformations_path = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model/"), "feature_transformations.pkl")
    logger.info("Loading feature transformations from: %s", transformations_path)
    try:
        with open(transformations_path, "rb") as f:
            feature_transformations = pickle.load(f)
        
        # Validate transformation structure
        if not isinstance(feature_transformations, dict):
            logger.warning("Feature transformations file has unexpected structure. Expected dict, got %s", 
                         type(feature_transformations).__name__)
            feature_transformations = {}
        else:
            # Check for sklearn version mismatches
            current_sklearn_version = sklearn.__version__
            for col, info in feature_transformations.items():
                if isinstance(info, dict) and info.get('transform_type') == 'yeojohnson':
                    saved_version = info.get('parameters', {}).get('sklearn_version')
                    if saved_version and saved_version != current_sklearn_version:
                        logger.warning("Sklearn version mismatch for column '%s': saved with %s, current is %s. "
                                     "Transformations may not work correctly.", 
                                     col, saved_version, current_sklearn_version)
        
        transformed_count = len([k for k, v in feature_transformations.items() 
                                if isinstance(v, dict) and v.get('transform_type') != 'none'])
        logger.info("Feature transformations loaded successfully. Found transformations for %d features.", 
                   transformed_count)
    except FileNotFoundError:
        logger.warning("Feature transformations file not found at %s. Continuing without transformations.", 
                      transformations_path)
        feature_transformations = {}
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        logger.error("Failed to unpickle feature transformations: %s", str(e), exc_info=True)
        logger.warning("Continuing without transformations. Model may not work correctly.")
        feature_transformations = {}
    except Exception as e:
        logger.error("Failed to load feature transformations: %s", str(e), exc_info=True)
        raise

# Load the inference dataset
def load_data(request_obj):
    """
    Load inference data from Flask request.
    
    Parameters:
    -----------
    request_obj : Flask request object
        The Flask request containing CSV data
    
    Returns:
    --------
    pd.DataFrame : Loaded inference data
    """
    logger.info("Loading the input data.")
    try:
        csv_data = io.StringIO(request_obj.data.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        if df.empty:
            raise ValueError("Input CSV file is empty")
        
        logger.info("Parsed input data shape: %s", df.shape)
        logger.info("Available columns in the dataset: %s", list(df.columns))
        return df
    except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
        logger.error("Failed to parse CSV data: %s", str(e), exc_info=True)
        raise ValueError(f"Invalid CSV data: {str(e)}") from e
    except Exception as e:
        logger.error("Failed to load inference data: %s", str(e), exc_info=True)
        raise

def apply_feature_transformations(df):
    """
    Apply the same feature transformations that were used during training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features to transform
    
    Returns:
    --------
    pd.DataFrame : Dataframe with transformed features
    """
    if feature_transformations is None or len(feature_transformations) == 0:
        logger.info("No feature transformations to apply.")
        return df
    
    if not isinstance(feature_transformations, dict):
        logger.warning("Feature transformations is not a dictionary. Skipping transformations.")
        return df
    
    df_transformed = df.copy()
    transformed_count = 0
    
    logger.info("Applying feature transformations...")
    
    for col, transform_info in feature_transformations.items():
        # Validate transformation info structure
        if not isinstance(transform_info, dict):
            logger.warning("Invalid transformation info structure for column '%s'. Skipping.", col)
            continue
        
        transform_type = transform_info.get('transform_type', 'none')
        
        if transform_type == 'none' or col not in df.columns:
            continue
        
        try:
            # Get valid values (non-NaN, non-infinite)
            valid_mask = ~(pd.isna(df[col]) | np.isinf(df[col]))
            
            if valid_mask.sum() == 0:
                logger.warning("No valid values to transform for column '%s'", col)
                continue
            
            valid_values = df.loc[valid_mask, col].values
            
            # Apply transformation based on type
            if transform_type == 'log':
                # Use the same train_min from training
                parameters = transform_info.get('parameters', {})
                if not isinstance(parameters, dict):
                    logger.warning("Invalid parameters structure for log transformation of '%s'. Skipping.", col)
                    continue
                
                train_min = parameters.get('train_min')
                if train_min is None:
                    logger.warning("No train_min found for log transformation of '%s'. Skipping.", col)
                    continue
                df_transformed.loc[valid_mask, col] = np.log1p(valid_values - train_min + 1e-10)
                transformed_count += 1
                logger.debug("Applied log transformation to '%s' using train_min=%.4f", col, train_min)
                
            elif transform_type == 'sqrt':
                df_transformed.loc[valid_mask, col] = np.sqrt(valid_values + 1e-10)
                transformed_count += 1
                logger.debug("Applied sqrt transformation to '%s'", col)
                
            elif transform_type == 'yeojohnson':
                # Use the fitted transformer from training
                parameters = transform_info.get('parameters', {})
                if not isinstance(parameters, dict):
                    logger.warning("Invalid parameters structure for Yeo-Johnson transformation of '%s'. Skipping.", col)
                    continue
                
                transformer = parameters.get('transformer')
                if transformer is None:
                    logger.warning("No transformer found for Yeo-Johnson transformation of '%s'. Skipping.", col)
                    continue
                
                # Validate transformer type
                if not hasattr(transformer, 'transform'):
                    logger.warning("Transformer for '%s' does not have transform method. Skipping.", col)
                    continue
                
                df_transformed.loc[valid_mask, col] = transformer.transform(
                    valid_values.reshape(-1, 1)
                ).flatten()
                transformed_count += 1
                logger.debug("Applied Yeo-Johnson transformation to '%s'", col)
            else:
                logger.warning("Unknown transformation type '%s' for column '%s'. Skipping.", transform_type, col)
                
        except (ValueError, AttributeError, TypeError) as e:
            logger.error("Error applying transformation to column '%s': %s", col, str(e))
            continue
        except Exception as e:
            logger.error("Unexpected error applying transformation to column '%s': %s", col, str(e), exc_info=True)
            continue
    
    logger.info("Applied transformations to %d features.", transformed_count)
    return df_transformed

# Health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    health = model is not None  # Check if the model is loaded
    status = 200 if health else 503
    return jsonify(status="healthy" if health else "unhealthy"), status

@app.route('/execution-parameters', methods=['GET'])
def execution_parameters():
    return jsonify({"SupportsExecutionParameters": False}), 200

# Inference endpoint
@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        logger.info("Received an inference request.")

        # Validate model is loaded
        if model is None:
            error_msg = "Model is not loaded. Cannot perform inference."
            logger.error(error_msg)
            return jsonify(error=error_msg), 503

        # Load the input data
        input_data = load_data(flask_request)
        
        # Validate input data is not empty
        if input_data.empty:
            error_msg = "Input data is empty. Cannot perform inference."
            logger.error(error_msg)
            return jsonify(error=error_msg), 400

        # Verify required columns exist (user_id and features)
        if 'user_id' not in input_data.columns:
            error_msg = "'user_id' column is required in the input data"
            logger.error(error_msg)
            return jsonify(error=error_msg), 400
        
        # Store user_id for output
        user_ids = input_data['user_id'].copy()
        
        # Validate user_ids are not all null
        if user_ids.isna().all():
            error_msg = "All user_id values are null"
            logger.error(error_msg)
            return jsonify(error=error_msg), 400
        
        # Apply feature transformations (same as training)
        input_data_transformed = apply_feature_transformations(input_data)
        
        # Prepare data for prediction (drop user_id if model doesn't need it)
        # The model was trained without user_id as a feature
        prediction_data = input_data_transformed.drop(columns=['user_id'], errors='ignore')
        
        # Validate prediction data is not empty
        if prediction_data.empty:
            error_msg = "No feature columns available for prediction after transformations"
            logger.error(error_msg)
            return jsonify(error=error_msg), 400
        
        # Validate feature columns exist
        if len(prediction_data.columns) == 0:
            error_msg = "No feature columns found for prediction"
            logger.error(error_msg)
            return jsonify(error=error_msg), 400
        
        logger.info("Data prepared for prediction. Shape: %s", prediction_data.shape)
        logger.info("Features for prediction: %s", list(prediction_data.columns))
        
        # Generate predictions (regression: predict() not predict_proba())
        logger.info("Generating predictions...")
        try:
            predictions = model.predict(prediction_data)
        except Exception as e:
            error_msg = f"Model prediction failed: {str(e)}. Check that feature columns match training schema."
            logger.error(error_msg, exc_info=True)
            return jsonify(error=error_msg), 500
        
        # Validate predictions
        if predictions is None:
            error_msg = "Model returned None predictions"
            logger.error(error_msg)
            return jsonify(error=error_msg), 500
        
        if len(predictions) == 0:
            error_msg = "Model returned empty predictions"
            logger.error(error_msg)
            return jsonify(error=error_msg), 500
        
        # Check for invalid predictions
        if np.isnan(predictions).any():
            nan_count = np.isnan(predictions).sum()
            logger.warning("Found %d NaN predictions out of %d total", nan_count, len(predictions))
        
        if np.isinf(predictions).any():
            inf_count = np.isinf(predictions).sum()
            logger.warning("Found %d infinite predictions out of %d total", inf_count, len(predictions))
        
        # Validate user_ids and predictions have same length
        if len(user_ids) != len(predictions):
            error_msg = f"Length mismatch: user_ids ({len(user_ids)}) != predictions ({len(predictions)})"
            logger.error(error_msg)
            return jsonify(error=error_msg), 500
        
        # Create output dataframe
        output_data = pd.DataFrame({
            'user_id': user_ids,
            'pct_change_cltv_oct_to_nov': predictions
        })
        
        # Validate output is not empty
        if output_data.empty:
            error_msg = "Output dataframe is empty"
            logger.error(error_msg)
            return jsonify(error=error_msg), 500
        
        # Sort by predicted change (descending - highest positive change first)
        output_data.sort_values(by='pct_change_cltv_oct_to_nov', ascending=False, inplace=True)
        
        logger.info("Predictions generated successfully. Shape: %s", output_data.shape)
        logger.info("Predicted CLTV change statistics:")
        logger.info("  Mean: %.4f", output_data['pct_change_cltv_oct_to_nov'].mean())
        logger.info("  Median: %.4f", output_data['pct_change_cltv_oct_to_nov'].median())
        logger.info("  Min: %.4f", output_data['pct_change_cltv_oct_to_nov'].min())
        logger.info("  Max: %.4f", output_data['pct_change_cltv_oct_to_nov'].max())
        
        # Optional: Filter top N percent if needed (can be configured via environment variable)
        top_percent = os.environ.get("top_pct")
        if top_percent:
            try:
                top_percent = float(top_percent)
                if top_percent <= 0 or top_percent > 100:
                    logger.warning("Invalid top_pct value: %s. Ignoring filter.", top_percent)
                else:
                    top_n = int(len(output_data) * (top_percent / 100))
                    if top_n > 0:
                        output_data = output_data.head(top_n)
                        logger.info("Returning top %.1f%% predictions (%d rows)", top_percent, top_n)
                    else:
                        logger.warning("Top percent filter resulted in 0 rows. Returning all predictions.")
            except (ValueError, TypeError) as e:
                logger.warning("Invalid top_pct environment variable: %s. Ignoring filter.", str(e))

        # Validate output before converting to CSV
        if output_data.empty:
            error_msg = "Output data is empty after filtering"
            logger.error(error_msg)
            return jsonify(error=error_msg), 500

        # Convert DataFrame to CSV
        try:
            output_csv = io.StringIO()
            output_data.to_csv(output_csv, index=False)
            output_csv.seek(0)  # Move to the start of the StringIO buffer
        except Exception as e:
            error_msg = f"Failed to convert predictions to CSV: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify(error=error_msg), 500

        # Create Flask Response with CSV
        response = Response(output_csv.getvalue(), mimetype='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=cltv_change_predictions.csv'
        logger.info("Returning predictions as CSV file.")
        return response, 200

    except ValueError as e:
        logger.error("Validation error during inference: %s", str(e), exc_info=True)
        return jsonify(error=str(e)), 400
    except Exception as e:
        logger.error("Unexpected error during inference: %s", str(e), exc_info=True)
        return jsonify(error=f"Internal server error: {str(e)}"), 500

if __name__ == '__main__':
    # Load the trained model and feature transformations
    load_model()
    load_feature_transformations()
    app.run(host='0.0.0.0', port=8080)
