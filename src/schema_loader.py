import json
import os
from typing import Dict, List

def safe_json_load(file_path: str):
    """Safely load JSON file with regex escape handling."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Fix regex patterns by escaping single backslashes
    import re
    content = re.sub(r'"([^"]*\\[^\\][^"]*?)"', lambda m: '"' + m.group(1).replace('\\', '\\\\') + '"', content)
    return json.loads(content)

def load_schema(schema_name: str, schema_dir: str = "~/SageMaker/ACM_Model_Creation/API/schema") -> Dict:
    """
    Load a schema from a JSON file.
    
    Args:
        schema_name (str): Name of the schema file (without .json extension)
        schema_dir (str): Directory containing schema files
        
    Returns:
        Dict: The loaded schema
        
    Raises:
        FileNotFoundError: If the schema file doesn't exist
    """
    schema_path = os.path.join(os.path.expanduser(schema_dir), f"{schema_name}.json")
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    try:
        return safe_json_load(schema_path)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON in schema file {schema_path}: {str(e)}")

def list_available_schemas(schema_dir: str = "~/SageMaker/ACM_Model_Creation/API/schema") -> List[str]:
    """
    List all available schema files.
    
    Args:
        schema_dir (str): Directory containing schema files
        
    Returns:
        List[str]: List of schema names (without .json extension)
    """
    if not os.path.exists(os.path.expanduser(schema_dir)):
        return []
    
    schemas = []
    for file in os.listdir(os.path.expanduser(schema_dir)):
        if file.endswith('.json'):
            schemas.append(file[:-5])  # Remove .json extension
    
    return sorted(schemas)

def load_metrics(metrics_name: str, metrics_dir: str = "~/SageMaker/ACM_Model_Creation/API/metrics_definitions") -> List[Dict]:
    """
    Load metrics definitions from a JSON file.
    
    Args:
        metrics_name (str): Name of the metrics file (without .json extension)
        metrics_dir (str): Directory containing metrics files
        
    Returns:
        List[Dict]: The loaded metrics definitions
    """
    metrics_path = os.path.join(os.path.expanduser(metrics_dir), f"{metrics_name}.json")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    try:
        return safe_json_load(metrics_path)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON in metrics file {metrics_path}: {str(e)}")

def list_available_metrics(metrics_dir: str = "~/SageMaker/ACM_Model_Creation/API/metrics_definitions") -> List[str]:
    """
    List all available metrics definition files.
    
    Args:
        metrics_dir (str): Directory containing metrics files
        
    Returns:
        List[str]: List of metrics names (without .json extension)
    """
    if not os.path.exists(os.path.expanduser(metrics_dir)):
        return []
    
    metrics = []
    for file in os.listdir(os.path.expanduser(metrics_dir)):
        if file.endswith('.json'):
            metrics.append(file[:-5])  # Remove .json extension
    
    return sorted(metrics)

def load_cmaa_config(config_name: str, config_dir: str = "~/SageMaker/ACM_Model_Creation/API/cmaa") -> Dict:
    """
    Load CMAA configuration from a JSON file.
    
    Args:
        config_name (str): Name of the config file (without .json extension)
        config_dir (str): Directory containing config files
        
    Returns:
        Dict: The loaded CMAA configuration
    """
    config_path = os.path.join(os.path.expanduser(config_dir), f"{config_name}.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"CMAA config file not found: {config_path}")
    
    try:
        return safe_json_load(config_path)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON in CMAA config file {config_path}: {str(e)}")

def list_available_cmaa_configs(config_dir: str = "~/SageMaker/ACM_Model_Creation/API/cmaa") -> List[str]:
    """
    List all available CMAA config files.
    
    Args:
        config_dir (str): Directory containing config files
        
    Returns:
        List[str]: List of config names (without .json extension)
    """
    if not os.path.exists(os.path.expanduser(config_dir)):
        return []
    
    configs = []
    for file in os.listdir(os.path.expanduser(config_dir)):
        if file.endswith('.json'):
            configs.append(file[:-5])  # Remove .json extension
    
    return sorted(configs)