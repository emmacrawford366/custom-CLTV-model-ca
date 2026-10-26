import boto3
import json
import os
import shutil
import subprocess
import base64
from pprint import pprint
from datetime import datetime
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError

from statistics import mean
from src.schema_loader import load_schema, list_available_schemas, load_metrics, list_available_metrics, load_cmaa_config, list_available_cmaa_configs


class ACMPipeline:
    def __init__(self, acm_algorithm_name: str, instance_id: str, membershipIdentifier:str, amc_account_id: str, use_case_type = 'audience', aws_account_id: str = None, region: str = 'us-east-1', base_path:str = '/home/ec2-user/SageMaker/ACM_Model_Creation/ml/code', metrics_name: str = None, cmaa_config_name: str = None, schema_name: str = None, verbose = False, **kwargs):
        """
        Initialize the ACM Pipeline with AWS configurations.
        
        Args:
            acm_algorithm_name (str): Name of the ACM algorithm
            region (str): AWS region (default: 'us-east-1')
            account_id (str): AWS account ID (optional)
        """

        self.acm_algorithm_name = acm_algorithm_name
        self.region = region
        self.instance_id = instance_id
        self.amc_account_id = amc_account_id
        self.membershipIdentifier = membershipIdentifier
        self.sts_client = boto3.client('sts')
        self.aws_account_id = aws_account_id or self._get_account_id()
        self.ecr_client = boto3.client('ecr')
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=self.region)
        self.verbose = verbose
        self.base_path = base_path
        self.metrics_name = metrics_name
        self.cmaa_config_name = cmaa_config_name
        self.schema_name = schema_name
        self.use_case_type = use_case_type.lower()
        
        self.cleanroomsml = boto3.client('cleanroomsml', region_name=self.region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        self.dynamodb = boto3.resource('dynamodb',region_name=self.region)
        
        # Store service names - will raise KeyError if missing
        self.dynamodb_table_configs = kwargs['dynamodb_table_configs']
        self.dynamodb_table_executions = kwargs['dynamodb_table_executions']
        self.step_function_arn = kwargs['step_function_arn']
        self.ecr_access_role = kwargs['ecr_access_role']
        self.cloudwatch_access_role = kwargs['cloudwatch_access_role']
        self.ecr_repo_name = kwargs['ecr_repo_name']
        self.container_cli = kwargs.get('container_cli') or os.getenv('ACM_CONTAINER_CLI') or self._detect_container_cli()

    def _get_account_id(self) -> str:
        """Get AWS account ID using STS client."""
        return self.sts_client.get_caller_identity()['Account']

    def _detect_container_cli(self) -> Optional[str]:
        """Detect available container CLI in preference order."""
        if shutil.which('docker'):
            return 'docker'
        if shutil.which('finch'):
            return 'finch'
        return None

    def _get_container_cli(self) -> str:
        """Return configured container CLI and validate it is available."""
        cli = self.container_cli
        if not cli:
            raise FileNotFoundError(
                "No supported container CLI found. Install Docker or Finch, or set ACM_CONTAINER_CLI."
            )
        if not shutil.which(cli):
            raise FileNotFoundError(
                f"Configured container CLI '{cli}' was not found in PATH."
            )
        return cli

    def load_schema(self, schema_name: str, schema_file_path = None) -> Dict:
        """Load a schema from the schema directory."""
        if schema_file_path is not None:
            return load_schema(schema_name, schema_dir = schema_file_path)
        else:
            return load_schema(schema_name)

    def list_schemas(self) -> List[str]:
        """List all available schemas."""
        return list_available_schemas()

    def load_metrics(self, metrics_name: str, metrics_definition_directory = None) -> List[Dict]:
        """Load metrics definitions from the metrics directory."""
        if metrics_definition_directory is not None:
            return load_metrics(metrics_name, metrics_dir = metrics_definition_directory)
        else:
            return load_metrics(metrics_name)

    def list_metrics(self) -> List[str]:
        """List all available metrics definitions."""
        return list_available_metrics()

    def load_cmaa_config(self, config_name: str, cmaa_config_directory = None) -> Dict:
        """Load CMAA configuration from the config directory."""
        if cmaa_config_directory is not None:
            return load_cmaa_config(config_name, cmaa_config_directory)
        else:
            return load_cmaa_config(config_name)

    def list_cmaa_configs(self) -> List[str]:
        """List all available CMAA configurations."""
        return list_available_cmaa_configs()

    def start_pipeline(self, payload: Dict[str, Any], schema_file_path= None, cmaa_config_directory= None, metrics_definition_directory=None) -> Dict[str, Any]:
        """
        Start and coordinate the entire ACM pipeline process.
        
        Args:
            payload (dict): Configuration payload for the pipeline
            
        Returns:
            dict: Result from the training pipeline execution
        """
        try:
            # Assemble Payload
            payload['acm_algorithm_name'] = self.acm_algorithm_name
            payload['instance_id'] = self.instance_id
            payload['account_id'] = self.amc_account_id
            payload['use_case_type'] = self.use_case_type
            
            # Add schema if specified and not already in payload
            if self.schema_name and 'schema' not in payload:
                if schema_file_path:
                    print('Schema in file path')
                    payload['schema'] = self.load_schema(self.schema_name, schema_file_path)
                else:
                    payload['schema'] = self.load_schema(self.schema_name)

            if self.use_case_type not in ['measurement', 'audience']:
                raise Exception(f"Error: Invalid use case type '{self.use_case_type}'. Must be 'measurement' or 'audience'.")           

        except Exception as e:
            print('Error assembling the payload: {e}')
            raise
            
        try:
            workflow_type = payload.get('workflow_type')
            if workflow_type not in ['train', 'inference']:
                raise Exception(f"Error: Invalid workflow type '{workflow_type}'. Must be 'train', 'inference'.")

            ####Check to see if data_type exists. If it does, re-assign existing value. If it does not create the key with a Null value. 
            if (self.use_case_type == 'measurement') and (payload.get('data_type') is None):
                raise Exception(f"Error: data_type must be specified for measurement use case type.")
            data_type = payload.get('data_type')
            payload['data_type'] = data_type


            print(f"Starting ACM pipeline for algorithm: {self.acm_algorithm_name}")
            print(f"Workflow type: {workflow_type}")

            # If this is the first run on this algorithm - A default image for train and conditionally for infer must be created to create the CMA
            self._build_default_images()
            
            # Step 1 & 2: Build and push container
            print("\nStep 1 & 2: Building and pushing container...")
            ecr_image = self._build_and_push_container(workflow_type)
            
            # Step 3: Create model associations
            print("\nStep 3: Creating model associations...")
            ACM_Algorithm_Configs = self.check_algorithm_exists()
            if not ACM_Algorithm_Configs['exists']: 
                print("ACM algorithm does not exist in DynamoDB, Creating a CMA, CMAA and storing to dynamoDB")
                if metrics_definition_directory:
                    model_algorithm_arn = self._create_model_associations(metrics_definition_directory)
                else:
                    model_algorithm_arn = self._create_model_associations()

                if cmaa_config_directory:
                    cmaa_arn = self._create_configured_model_algorithm_association(model_algorithm_arn, cmaa_config_directory)
                else: 
                    cmaa_arn = self._create_configured_model_algorithm_association(model_algorithm_arn)
                
                self.model_algorithm_arn = model_algorithm_arn
                self.cmaa_arn = cmaa_arn
                self._store_algorithm_config()
            else:
                print(f'Skipping creating model association as one already exists with name: {self.acm_algorithm_name}')
                self.model_algorithm_arn = ACM_Algorithm_Configs['model_algorithm_arn']
                self.cmaa_arn = ACM_Algorithm_Configs['cmaa_arn']


            # Step 4: Update ML Configuration ARN 
            print("\nStep 4: Updating ML Configuration Role ARN...")
            updated_ml_configuration_role = self._update_ml_role_configuration()
            print(f"ML Configuration Role ARN updated successfully to: {updated_ml_configuration_role}")


            # Step 5: Start training pipeline
            print("\nStep 5: Starting the training pipeline...")
            current_time = datetime.now().isoformat()
            payload['configured_model_algorithm_associations'] = self.cmaa_arn
            payload['current_time'] = current_time
            
            result = self._start_training_pipeline(
                state_machine_arn=self.step_function_arn,
                payload=payload
            )

            #Step 6 Save execution details to dynamo
            print("\n Step 6: Saving execution details to dynamoDB.")
            execution_arn = result['execution_arn']
            print("Saving execution details to dynamoDB")
            self._create_execution_record(execution_arn, payload)
            print(f"All execution parameters saved. View them here: https://console.aws.amazon.com/dynamodb/home?region={self.region}#tables:selected={self.dynamodb_table_executions}")
            
            print("\nPipeline started successfully!")
            print(f"Monitor progress at: {result['console_url']}")
            
            return result
            
        except Exception as e:
            print(f"\nError in ACM pipeline: {str(e)}")
            print(f"\n💡 Tip: Use pipeline.display_error_summary() or check Error_Monitor.ipynb for detailed error analysis")
            raise

    def _build_default_images(self):
        """Build default images for train and conditionally for infer workflows."""
        print("\nChecking to see if default images exist")
        
        # Always build training image
        full_image_name_train = self.generate_ecr_name('train', output='full_image_name')
        if not self._image_exists(full_image_name_train):
            self._build_and_push_container('train')
        else:
            print(f"Image {full_image_name_train} already exists, skipping build")
        
        # Only build inference image for audience use-cases
        if self.use_case_type == 'audience':
            full_image_name_infer = self.generate_ecr_name('inference', output='full_image_name')
            if not self._image_exists(full_image_name_infer):
                self._build_and_push_container('inference')
            else:
                print(f"Image {full_image_name_infer} already exists, skipping build")
        else:
            print(f"Measurement model detected (use_case_type: {self.use_case_type}), skipping inference image build")
    
    def _image_exists(self, full_image_name: str) -> bool:
        """Check if an image exists in ECR."""
        try:
            # Extract tag from full_image_name (format: account.dkr.ecr.region.amazonaws.com/repo:tag)
            tag = full_image_name.split(':')[-1]
            repo_name = self.ecr_repo_name
            
            self.ecr_client.describe_images(
                repositoryName=repo_name,
                imageIds=[{'imageTag': tag}]
            )
            return True
        except self.ecr_client.exceptions.ImageNotFoundException:
            return False
        except Exception:
            return False

    def _get_working_directory(self, workflow_type: str) -> str:
        """Get the appropriate working directory based on workflow type."""
        base_path = self.base_path
        if workflow_type == 'train':
            return f"{base_path}/train"
        elif workflow_type == 'inference':
            return f"{base_path}/infer"
        else:
            raise ValueError(f"Invalid workflow type: {workflow_type}")

    def generate_ecr_name(self, workflow_type, output):
        repository_name = self.ecr_repo_name
        if output == 'repository_name':
            return repository_name
        ecr_uri = f"{self.aws_account_id}.dkr.ecr.{self.region}.amazonaws.com"

        if output == 'container_name':
            return f"{self.acm_algorithm_name}_{workflow_type}_container"
            
        if output == 'ecr_uri':
            return ecr_uri
        full_image_name = f"{ecr_uri}/{repository_name}:{self.acm_algorithm_name}_{workflow_type}_image"
        if output == 'full_image_name':
            return full_image_name
        return full_image_name

    def _build_and_push_container(self, workflow_type: str) -> str:
        """
        Build Docker container and push to ECR.
        
        Returns:
            str: ECR image URI
        """
        working_directory = self._get_working_directory(workflow_type)
        self._build_container(workflow_type, working_directory)
        return self._push_to_ecr(workflow_type)

    def _build_container(self, workflow_type: str, working_directory: str):
        """Build Docker container."""
        print("Building Docker container...")
        os.chdir(working_directory)
        container_cli = self._get_container_cli()
        target_platform = os.environ.get('ACM_CONTAINER_PLATFORM', 'linux/amd64')
        
        container_name = self.generate_ecr_name(workflow_type, output='container_name')
        print(f'Container name: {container_name}')
        print(f'Container CLI: {container_cli}')
        print(f'Container platform: {target_platform}')

        try:
            docker_build_response = subprocess.run(
                [
                    container_cli,
                    'build',
                    '--no-cache',
                    '--platform',
                    target_platform,
                    '-t',
                    f'{container_name}:latest',
                    '-f',
                    'Dockerfile',
                    '.'
                ],
                capture_output=True,
                text=True,
                check=True
            )
            if self.verbose:
                print('Docker build message...')
                print(docker_build_response.stderr)
            print(f'{workflow_type} Docker container build completed successfully')
        except subprocess.CalledProcessError as e:
            print(f"Docker build failed: {e.stderr}")
            raise



    def _push_to_ecr(self, workflow_type: str) -> str:
        """
        Push Docker container to ECR.
        
        Returns:
            str: Full ECR image URI
        """
        container_name = self.generate_ecr_name(workflow_type, output='container_name')
        full_image_name = self.generate_ecr_name(workflow_type, output='full_image_name')
        ecr_uri = self.generate_ecr_name(workflow_type, output='ecr_uri')
        container_cli = self._get_container_cli()


        try:
            # Get ECR login token with boto3 to avoid AWS CLI dependency.
            auth_response = self.ecr_client.get_authorization_token()
            auth_data = auth_response.get('authorizationData', [])
            if not auth_data:
                raise RuntimeError('No ECR authorization data returned by AWS.')

            token_b64 = auth_data[0].get('authorizationToken')
            if not token_b64:
                raise RuntimeError('Missing ECR authorization token in AWS response.')

            token_decoded = base64.b64decode(token_b64).decode('utf-8')
            _, ecr_password = token_decoded.split(':', 1)
            
            subprocess.run(
                [container_cli, 'login', '--username', 'AWS', '--password-stdin', ecr_uri],
                input=ecr_password,
                capture_output=True,
                text=True,
                check=True
            )

            # Tag and push image
            subprocess.run(
                [container_cli, 'tag', f'{container_name}:latest', full_image_name],
                check=True
            )
            
            docker_push_response = subprocess.run(
                [container_cli, 'push', full_image_name],
                capture_output=True,
                text=True,
                check=True
            )
            if self.verbose:
                print('Docker push completed successfully')
                print('Docker push message...')
                print(docker_push_response.stdout)

            print(f"Container successfully pushed to: {full_image_name}")
            return full_image_name

        except subprocess.CalledProcessError as e:
            print(f"Error pushing to ECR: {e.stderr}")
            raise
    @staticmethod
    def list_algorithms(algorithm_name: str = None, instance_id: str = None, region: str = 'us-east-1'):
        """
        List algorithm configurations with detailed information from the ACM_Algorithm_Configs table.
        
        Args:
            algorithm_name (str, optional): Filter by specific algorithm name
            instance_id (str, optional): Filter by specific instance ID (requires algorithm_name)
            region (str, optional): AWS region (default: 'us-east-1')

        Returns:
            List[Dict]: A list of dictionaries containing detailed algorithm configurations
        """
        try:
            # Get environment variables
            env_vars = get_cloudformation_env_vars()
            table_name = env_vars.get('dynamodb_table_configs')
            if not table_name:
                raise ValueError(
                    "Unable to resolve DynamoDB configs table. Set DYNAMODB_TABLE_CONFIGS or ensure IAM allows dynamodb:ListTables for auto-discovery."
                )
            
            # Create DynamoDB client
            dynamodb = boto3.resource('dynamodb', region_name=region)
            table = dynamodb.Table(table_name)

            if algorithm_name and instance_id:
                # Get specific configuration
                response = table.get_item(
                    Key={
                        'acm_algorithm_name': algorithm_name,
                        'instance_id': instance_id
                    }
                )
                algorithms = [response['Item']] if 'Item' in response else []
            elif algorithm_name:
                # Query by algorithm name
                response = table.query(
                    KeyConditionExpression='acm_algorithm_name = :algo_name',
                    ExpressionAttributeValues={
                        ':algo_name': algorithm_name
                    }
                )
                algorithms = response['Items']
                
                # Handle pagination for query
                while 'LastEvaluatedKey' in response:
                    response = table.query(
                        KeyConditionExpression='acm_algorithm_name = :algo_name',
                        ExpressionAttributeValues={':algo_name': algorithm_name},
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    algorithms.extend(response['Items'])
            else:
                # Get all algorithm configurations
                response = table.scan()
                algorithms = response['Items']
                
                # Handle pagination for scan
                while 'LastEvaluatedKey' in response:
                    response = table.scan(
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    algorithms.extend(response['Items'])
            
            # Sort by algorithm name and then by last_updated (most recent first)
            algorithms.sort(
                key=lambda x: (x['acm_algorithm_name'], x.get('last_updated', '')),
                reverse=True
            )
            
            # Print the configurations
            print(f"Found {len(algorithms)} algorithm configuration(s):")
            for i, config in enumerate(algorithms):
                print(f"\n--- Configuration {i+1} ---")
                print(f"Algorithm Name: {config.get('acm_algorithm_name')}")
                print(f"Instance ID: {config.get('instance_id')}")
                print(f"AMC Account ID: {config.get('amc_account_id')}")
                print(f"Membership Identifier: {config.get('membershipIdentifier')}")
                print(f"Region: {config.get('region')}")
                print(f"Use Case Type: {config.get('use_case_type')}")
                print(f"Base Path: {config.get('base_path')}")
                print(f"Metrics Used: {config.get('metrics_used')}")
                print(f"CMAA Config Used: {config.get('cmaa_config_used')}")
                print(f"Schema Name: {config.get('schema_name')}")
                print(f"Last Updated: {config.get('last_updated')}")
            
            return algorithms

        except Exception as e:
            print(f"Error listing algorithms: {str(e)}")
            raise

    @classmethod
    def from_existing_config(cls, acm_algorithm_name: str, instance_id: str, **kwargs):
        """
        Create an ACMPipeline instance from an existing configuration stored in DynamoDB.
        
        Args:
            acm_algorithm_name (str): Name of the algorithm configuration to load
            instance_id (str): Instance ID for the configuration to load
            **kwargs: CloudFormation environment variables (required)
            
        Returns:
            ACMPipeline: Initialized pipeline instance with the existing configuration
            
        Note:
            Most configuration parameters are immutable once stored. If you need to change
            parameters like use_case_type, base_path, metrics_name, etc., create a new
            algorithm with a different name.
        """
        try:
            # Get environment variables for table name
            env_vars = get_cloudformation_env_vars()
            table_name = env_vars.get('dynamodb_table_configs')
            if not table_name:
                raise ValueError(
                    "Unable to resolve DynamoDB configs table. Set DYNAMODB_TABLE_CONFIGS or ensure IAM allows dynamodb:ListTables for auto-discovery."
                )
            
            dynamodb = boto3.resource('dynamodb', region_name=kwargs.get('region', 'us-east-1'))
            table = dynamodb.Table(table_name)
            
            # Get the specific configuration
            response = table.get_item(
                Key={
                    'acm_algorithm_name': acm_algorithm_name,
                    'instance_id': instance_id
                }
            )
            
            if 'Item' not in response:
                raise ValueError(f"No configuration found for algorithm '{acm_algorithm_name}' with instance_id '{instance_id}'")
            
            config = response['Item']
            
            # Extract configuration parameters (immutable)
            init_params = {
                'acm_algorithm_name': config['acm_algorithm_name'],
                'instance_id': config['instance_id'],
                'membershipIdentifier': config['membershipIdentifier'],
                'amc_account_id': config['amc_account_id'],
                'region': config.get('region', 'us-east-1'),
                'aws_account_id': config.get('aws_account_id'),
                'metrics_name': config.get('metrics_used'),
                'cmaa_config_name': config.get('cmaa_config_used'),
                'use_case_type': config.get('use_case_type'),
                'base_path': config.get('base_path'),
                'schema_name': config.get('schema_name')
            }
            
            # Add CloudFormation environment variables
            init_params.update(kwargs)
            
            # Create the instance
            instance = cls(**init_params)
            
            # Print success message and configuration details
            print(f"Successfully loaded: {instance.acm_algorithm_name}")
            print(f"\n--- Loaded Configuration ---")
            print(f"Algorithm Name: {instance.acm_algorithm_name}")
            print(f"Instance ID: {instance.instance_id}")
            print(f"AMC Account ID: {instance.amc_account_id}")
            print(f"Membership Identifier: {instance.membershipIdentifier}")
            print(f"Region: {instance.region}")
            print(f"Use Case Type: {getattr(instance, 'use_case_type', 'Not set')}")
            print(f"Base Path: {getattr(instance, 'base_path', 'Not set')}")
            print(f"Metrics Name: {instance.metrics_name}")
            print(f"CMAA Config Name: {instance.cmaa_config_name}")
            print(f"Schema Name: {getattr(instance, 'schema_name', 'Not set')}")
            print(f"Last Updated: {config.get('last_updated')}")
            
            return instance
            
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    def _store_algorithm_config(self) -> None:
        """
        Store or update the algorithm configuration in DynamoDB.
        Contains static information about the algorithm like ARNs.
        """
        try:
            table = self.dynamodb.Table(self.dynamodb_table_configs)
            
            item = {
                'acm_algorithm_name': self.acm_algorithm_name,
                'model_algorithm_arn': self.model_algorithm_arn,
                'cmaa_arn': self.cmaa_arn,
                'region': self.region,
                'aws_account_id': self.aws_account_id,
                'last_updated': datetime.now().isoformat(),
                'instance_id': self.instance_id,
                'membershipIdentifier': self.membershipIdentifier,
                'amc_account_id': self.amc_account_id,
                'metrics_used': self.metrics_name,
                'cmaa_config_used': self.cmaa_config_name,
                'use_case_type': self.use_case_type,
                'base_path': self.base_path,
                'schema_name': self.schema_name
            }
    
            table.put_item(Item=item)
            print(f"Stored/Updated algorithm configuration for {self.acm_algorithm_name}")
    
        except Exception as e:
            print(f"Error storing algorithm configuration: {str(e)}")
            raise
    
    def _create_execution_record(self, execution_arn: str, payload: Dict[str, Any]) -> None:
        """
        Create a record in DynamoDB for the pipeline execution.
        Stores information about individual runs.
        
        Args:
            execution_arn (str): The ARN of the Step Function execution
            payload (dict): The payload used to start the pipeline
        """
        try:
            table = self.dynamodb.Table(self.dynamodb_table_executions)

            def _summarize_channels(channels):
                summarized = []
                for channel in channels or []:
                    sql_query = channel.get('sql_query') or ''
                    summarized.append({
                        'name': channel.get('name'),
                        'time_start': channel.get('time_start'),
                        'time_end': channel.get('time_end'),
                        'sql_query_length': len(sql_query),
                        'sql_query_omitted': bool(sql_query)
                    })
                return summarized
    
            # Get parameters based on workflow type
            workflow_type = payload.get('workflow_type')
            current_time = payload.get('current_time')
            training_params = payload.get('training_parameters', {})
            inference_params = payload.get('inference_parameters', {})
            training_channels = payload.get('training_input_channels', [])
            inference_channels = payload.get('inference_input_channels', [])

            # Keep DynamoDB records compact to avoid item-size overflows during later updates.
            payload_summary = {
                'workflow_type': payload.get('workflow_type'),
                'instance_id': payload.get('instance_id'),
                'acm_algorithm_name': payload.get('acm_algorithm_name'),
                'use_case_type': payload.get('use_case_type'),
                'data_type': payload.get('data_type'),
                'training_parameters': {
                    'name': training_params.get('name'),
                    'instance_type': training_params.get('instance_type'),
                    'instance_count': training_params.get('instance_count'),
                    'vol_size': training_params.get('vol_size'),
                    'timeout': training_params.get('timeout')
                } if training_params else None,
                'inference_parameters': {
                    'name': inference_params.get('name'),
                    'instance_type': inference_params.get('instance_type'),
                    'instance_count': inference_params.get('instance_count'),
                    'vol_size': inference_params.get('vol_size'),
                    'trained_model_id': inference_params.get('trained_model_id')
                } if inference_params else None,
                'training_input_channels': _summarize_channels(training_channels),
                'inference_input_channels': _summarize_channels(inference_channels)
            }

            schema_obj = payload.get('schema') if payload.get('schema') else None
            schema_summary = None
            if isinstance(schema_obj, dict):
                schema_summary = {
                    'keys': list(schema_obj.keys()),
                    'num_keys': len(schema_obj)
                }
    
            # Create execution record
            item = {
                'acm_algorithm_name': self.acm_algorithm_name, 
                'execution_timestamp': current_time,  # Sort key
                'execution_arn': execution_arn,
                'status': 'RUNNING',
                'workflow_type': workflow_type,
                'last_updated': current_time,
                'payload': json.dumps(payload_summary),
                'console_url': f"https://console.aws.amazon.com/states/home?region={self.region}#/executions/details/{execution_arn}",
                'schema_used': json.dumps(schema_summary) if schema_summary else None,
                
                # Training specific fields
                'training_channels_count': len(training_channels) if workflow_type == 'train' else None,
                'training_instance_type': training_params.get('instance_type'),
                'training_instance_count': training_params.get('instance_count'),
                'training_volume_size': training_params.get('vol_size'),
                'training_timeout': training_params.get('timeout'),
                'training_model_name': training_params.get('name'),
                'training_description': training_params.get('description'),
                'training_channels': json.dumps(_summarize_channels(training_channels)) if workflow_type == 'train' else None,
                
                # Inference specific fields
                'inference_channels_count': len(inference_channels) if workflow_type == 'inference' else None,
                'inference_instance_type': inference_params.get('instance_type'),
                'inference_instance_count': inference_params.get('instance_count'),
                'inference_volume_size': inference_params.get('vol_size'),
                'inference_model_name': inference_params.get('name'),
                'trained_model_id': inference_params.get('trained_model_id'),
                'inference_channels': json.dumps(_summarize_channels(inference_channels)) if workflow_type == 'inference' else None
            }
            # Put item in DynamoDB
            table.put_item(Item=item)
            print(f"Created {workflow_type} execution record in DynamoDB for {self.acm_algorithm_name}")
    
        except Exception as e:
            print(f"Error creating execution record in DynamoDB: {str(e)}")
            raise
    
    def check_algorithm_exists(self):
        """
        Check if the algorithm configuration exists in DynamoDB and return its details.
        
        Returns:
            Dict with algorithm configuration if exists, None if it doesn't.
            Example return:
            {
                'exists': True,
                'model_algorithm_arn': 'arn:aws:...',
                'cmaa_arn': 'arn:aws:...',
                'region': 'us-east-1',
                'account_id': '123456789012',
                'last_updated': '2023-08-17T12:34:56.789'
            }
        """
        try:
            table = self.dynamodb.Table(self.dynamodb_table_configs)
            
            response = table.get_item(
                Key={
                    'acm_algorithm_name': self.acm_algorithm_name,
                    'instance_id': self.instance_id
                }
            )
            
            if 'Item' in response:
                config = response['Item']
                return {
                    'exists': True,
                    'model_algorithm_arn': config.get('model_algorithm_arn'),
                    'cmaa_arn': config.get('cmaa_arn'),
                    'region': config.get('region'),
                    'aws_account_id': config.get('aws_account_id'),
                    'last_updated': config.get('last_updated'),
                    'instance_id':config.get('instance_id'),
                    'membershipIdentifier': config.get('membershipIdentifier'),
                    'amc_account_id': config.get('amc_account_id'),
                    'use_case_type': config.get('use_case_type'),
                    'base_path': config.get('base_path'),
                    'schema_name': config.get('schema_name')
                }
            
            return {
                'exists': False,
                'model_algorithm_arn': None,
                'cmaa_arn': None,
                'region': None,
                'aws_account_id': None,
                'last_updated': None,
                'instance_id': None,
                'membershipIdentifier': None,
                'amc_account_id': None,
                'use_case_type': None,
                'base_path': None,
                'schema_name': None
            }
            
        except Exception as e:
            print(f"Error checking algorithm configuration: {str(e)}")
            raise
    
    def _create_model_associations(self, metrics_definition_directory=None) -> None:
        """Create model associations in ACM."""
        full_image_name_train = self.generate_ecr_name(workflow_type='train', output='full_image_name')

        if metrics_definition_directory:
            metrics = self.load_metrics(self.metrics_name, metrics_definition_directory)
        else:
            metrics = load_metrics(self.metrics_name)

        config_params = {
            'name': self.acm_algorithm_name,
            'description': f'{self.acm_algorithm_name} model association',
            'roleArn': self.ecr_access_role,
            'trainingContainerConfig': {
                'imageUri': full_image_name_train,
                'metricDefinitions': metrics
            }
        }
        
        # Only add inference container for audience use-cases
        if self.use_case_type == 'audience':
            full_image_name_infer = self.generate_ecr_name(workflow_type='inference', output='full_image_name')
            config_params['inferenceContainerConfig'] = {
                'imageUri': full_image_name_infer
            }
            if self.verbose:
                print(f'Training image name: {full_image_name_train}')
                print(f'Inference image name: {full_image_name_infer}')
        else:
            if self.verbose:
                print(f'Training image name: {full_image_name_train}')
                print(f'Measurement model detected (use_case_type: {self.use_case_type}), skipping inference container')

        # Add tags to match the tagging strategy in app.py
        config_params['tags'] = {
            'Application': 'ACM-Custom-Models',
            'Component': 'ML-Pipeline'
        }

        try:
            configured_model_algorithm = self.cleanroomsml.create_configured_model_algorithm(**config_params)
            print(f"configuredModelAlgorithmArn is: {configured_model_algorithm['configuredModelAlgorithmArn']}")
            return configured_model_algorithm['configuredModelAlgorithmArn']
        except Exception as e:
            print(f"Error creating configured model algorithm: {str(e)}")
            raise # Re-raise the exception
    
    def _create_configured_model_algorithm_association(self, configured_model_algorithm, cmaa_config_directory=None):
        """
        Create a configured model algorithm association in ACM.
        
        Args:
            configured_model_algorithm (str): The configured model algorithm ARN
            cmaa_config_directory (str, optional): Directory containing CMAA config files
        """
        try:
            if cmaa_config_directory:
                # Load CMAA config from specified directory
                privacy_config = self.load_cmaa_config(self.cmaa_config_name, cmaa_config_directory)
            else:
                privacy_config = self.load_cmaa_config(self.cmaa_config_name)
                
            cmaa = self.cleanroomsml.create_configured_model_algorithm_association(
                membershipIdentifier=self.membershipIdentifier,
                configuredModelAlgorithmArn=configured_model_algorithm,
                name=f'{self.acm_algorithm_name}cmaa',
                description=f'{self.acm_algorithm_name}cmaa',
                privacyConfiguration=privacy_config,
                tags={
                    'Application': 'ACM-Custom-Models',
                    'Component': 'ML-Pipeline'
                }
            )
            print(f"CMAA association complete. CMAA ARN is: {cmaa['configuredModelAlgorithmAssociationArn']}")
            return cmaa['configuredModelAlgorithmAssociationArn']
        except Exception as e:
            print(f"Error creating configured model algorithm association: {str(e)}")
            raise
    
    def _update_ml_role_configuration(self):
        #### Check to see if a role is already configured. If it is, return existing role, if not update to use the cloudwatch_access_role deployed by cloud formation. 
        
        #Check to see if a role is already set
        try: 
            response = self.cleanroomsml.get_ml_configuration(
            membershipIdentifier=self.membershipIdentifier
            )
            #If yes, return existing role 
            print('ml_role_configuration already exists')
            return response['defaultOutputLocation']['roleArn']
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                try:
                    # Create ml_role_configuration
                    response = self.cleanroomsml.put_ml_configuration(
                        membershipIdentifier=self.membershipIdentifier,
                        defaultOutputLocation = {
                            'roleArn': self.cloudwatch_access_role}
                    )
                    print("ml_role_configuration does not exists, associating role")
                    return self.cloudwatch_access_role
                except Exception as e:
                    print(f"Error creating ml_role_configuration: {str(e)}")
                    raise
            else:
                print(f"Error updating the ml_role_configuration: {str(e)}")
                raise 

    def _start_training_pipeline(self, state_machine_arn: str, payload):
        """Start the training pipeline execution."""
        try:
            execution_name = self.acm_algorithm_name + datetime.now().strftime("%m%d%H%M")
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=execution_name,
                input=json.dumps(payload)
            )
            
            execution_arn = response['executionArn']
            result = {
                'execution_arn': execution_arn,
                'execution_name': execution_name,
                'start_date': response['startDate'],
                'status': 'RUNNING',
                'workflow_type': payload.get('workflow_type'),
                'training_channels_count': len(payload.get('training_input_channels', [])),
                'console_url': f"https://console.aws.amazon.com/states/home?region={self.region}#/executions/details/{execution_arn}"
            }
            
            return result
            
        except Exception as e:
            print(f"Error starting training pipeline: {str(e)}")
            raise

    def stop_pipeline(self, execution_arn: str, reason: str = "Manual stop from SageMaker notebook") -> Dict:
        """Stop a running pipeline execution."""
        try:
            response = self.stepfunctions_client.stop_execution(
                executionArn=execution_arn,
                error="ManualStop",
                cause=reason
            )
            print(f"Pipeline stopped: {execution_arn}")
            return response
        except Exception as e:
            print(f"Error stopping pipeline: {str(e)}")
            raise
    
    def get_pipeline_status(self, execution_arn: str, full_payload: bool = False):
        """Get the current status and output of a pipeline execution."""
        try:
            response = self.stepfunctions_client.describe_execution(executionArn=execution_arn)
            status = response['status']

            
            if status == "SUCCEEDED":
                output = json.loads(response.get('output', '{}'))
                if not full_payload:
                    # Initialize dictionary for key features
                    key_features = {}
                    
                    # Get payload data for easier access
                    payload = output.get('Payload', {}).get('body', {}).get('Payload', {})
                    
                    # Extract common fields
                    key_features['status'] = output.get('Payload', {}).get('body', {}).get('updated_status')
                    key_features['instance_id'] = payload.get('instance_id')
                    key_features['use_case_type'] = payload.get('use_case_type')
                    key_features['number_of_processed_channels'] = len(payload.get('processed_channels', []))
                    key_features['workflow_type'] = payload.get('workflow_type')
                    
                    # Extract workflow-specific fields
                    if payload.get('workflow_type') == 'train':
                        key_features['trained_model_id'] = output.get('Payload', {}).get('body', {}).get('trained_model_id')
                        training_monitoring = payload.get('training_job_monitoring', {}).get('training_job_monitoring_result', {})
                        key_features['training_job_status'] = training_monitoring.get('status')
                        model_details = training_monitoring.get('modelDetails', {})
                        key_features['create_time'] = model_details.get('createTime')
                        
                        # Extract additional features for train workflow
                        key_features['data_type'] = payload.get('data_type')
                        
                        # Extract pre-signed URLs from mlDataExportStatus
                        ml_data_export_status = payload.get('mlDataExportStatus', {}).get('mlDataExport', {})
                        key_features['pre_signed_urls'] = ml_data_export_status.get('preSignedS3Uris', [])
                        
                        # Extract storage location from downloadedFiles
                        downloaded_files = payload.get('downloadedFiles', {}).get('downloadResult', {})
                        key_features['storage_location'] = downloaded_files.get('storage_location')
                    
                    elif payload.get('workflow_type') == 'inference':
                        # Extract trainedModelInferenceId from the body level
                        key_features['trainedModelInferenceId'] = output.get('Payload', {}).get('body', {}).get('trained_model_inference_id')
                        key_features['modeledDatasetId'] = payload.get('modeledDataset', {}).get('modeledDatasetId')
                        # Extract from get_modeled_dataset_upload in body
                        upload_details = output.get('Payload', {}).get('body', {}).get('get_modeled_dataset_upload', {}).get('uploadDetails', {})
                        key_features['estimatedRowCount'] = upload_details.get('estimatedRowCount')
                        key_features['modeledDatasetUploadJobId'] = upload_details.get('modeledDatasetUploadJobId')
                        inference_monitoring = payload.get('monitoring', {}).get('inference_monitoring_result', {})
                        key_features['inference_job_status'] = inference_monitoring.get('status')
                        job_details = inference_monitoring.get('jobDetails', {})
                        key_features['create_time'] = job_details.get('createTime')
                    
                    return key_features
                return output
            elif status == "RUNNING":
                return {"status": status}
            else:       
                return {
                    "status": status,
                    "error": response.get('error', 'See logs'),
                    "logs": json.dumps(response, indent=2, default=str)
                }
            
        except Exception as e:
            print(f"Error getting pipeline status: {str(e)}")
            raise

    def list_trained_model_metrics(self, trained_model_id,custom_metrics=True):
        """List the Metrics for Trained Model"""
        response = self.cloudwatch.list_metrics(Namespace='/aws/cleanroomsml/TrainedModels',Dimensions=[{'Name': 'TrainedModelName'}] if custom_metrics else [])['Metrics']
        return [metric for metric in response if metric['Dimensions'][0]['Value'].startswith(trained_model_id)]

    def get_trained_model_metrics(self, trained_model_id, start_time, metric_query=None):
        """Get the Metrics Data for Trained Model"""
        if metric_query == None:
            metric_query = []
        metrics = self.list_trained_model_metrics(trained_model_id)
        start_time = start_time

        for i,j in enumerate(metrics):
            metric_query.append({
                                  "Id": f"m{i}",
                                  "MetricStat": {
                                        "Metric": j,
                                        "Period": 60,
                                        "Stat": "Average"
                                  }
                                })
        if not metric_query:
            raise ValueError("No active CloudWatch metrics found for this trained model")


        return self.cloudwatch.get_metric_data(MetricDataQueries=metric_query,
                StartTime=datetime.strptime(start_time.split('.')[0],"%Y-%m-%dT%H:%M:%S"),
                EndTime=datetime.now())

    def get_trained_model_metrics_agg(self, trained_model_id, start_time):
        """Get the Metrics Data for Trained Model at aggregate level"""
        metrics_result = self.get_trained_model_metrics(trained_model_id, start_time)['MetricDataResults']
        aggregated_metrics = {}
        for metric in metrics_result:
            values = metric.get('Values', [])
            label = metric.get('Label', metric.get('Id', 'unknown_metric'))

            # CloudWatch can return empty value arrays when points are not yet emitted.
            if values:
                aggregated_metrics[label] = mean(values)
            else:
                aggregated_metrics[label] = None

        return aggregated_metrics

    ###TODO - Add in a sort on training type - infer or Train
    def get_algorithm_executions(self, limit=None, status_filter=None):
        """
        Get all executions for the current algorithm from DynamoDB.
        
        Args:
            limit (int, optional): Maximum number of records to return
            status_filter (str, optional): Filter results by status (e.g., 'RUNNING', 'SUCCEEDED', 'FAILED')
        
        Returns:
            List[Dict]: List of execution records with key information
        """
        try:
            table = self.dynamodb.Table(self.dynamodb_table_executions)
            
            # Base query parameters
            query_params = {
                'KeyConditionExpression': '#algo = :algo_name',
                'ExpressionAttributeNames': {
                    '#algo': 'acm_algorithm_name'
                },
                'ExpressionAttributeValues': {
                    ':algo_name': self.acm_algorithm_name
                }
            }
            
            # Add status filter if provided
            if status_filter:
                query_params['FilterExpression'] = '#status = :status'
                query_params['ExpressionAttributeNames']['#status'] = 'status'
                query_params['ExpressionAttributeValues'][':status'] = status_filter
            
            # Add limit if provided
            if limit:
                query_params['Limit'] = limit
    
            response = table.query(**query_params)
            executions = response['Items']
            
            # Handle pagination
            while 'LastEvaluatedKey' in response and (not limit or len(executions) < limit):
                query_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = table.query(**query_params)
                executions.extend(response['Items'])
                
                if limit:
                    executions = executions[:limit]
    
            # Process and format the results
            formatted_executions = []
            for execution in executions:
                execution_info = {
                    'execution_timestamp': execution.get('execution_timestamp'),
                    'status': execution.get('status'),
                    'workflow_type': execution.get('workflow_type'),
                    'execution_arn': execution.get('execution_arn'),
                    'console_url': execution.get('console_url')
                }
                
                # Add workflow-specific information
                if execution.get('workflow_type') == 'train':
                    execution_info.update({
                        'training_model_name': execution.get('training_model_name'),
                        'training_instance_type': execution.get('training_instance_type'),
                        'training_instance_count': execution.get('training_instance_count'),
                        'training_channels_count': execution.get('training_channels_count'),
                        'training_description': execution.get('training_description'),
                        'trained_model_id': execution.get('trained_model_id')

                    })
                elif execution.get('workflow_type') == 'inference':
                    execution_info.update({
                        'inference_model_name': execution.get('inference_model_name'),
                        'inference_instance_type': execution.get('inference_instance_type'),
                        'inference_instance_count': execution.get('inference_instance_count'),
                        'inference_channels_count': execution.get('inference_channels_count'),
                        'trained_model_id': execution.get('trained_model_id'),
                        'trainedModelInferenceId': execution.get('trainedModelInferenceId')
                    })
    
                formatted_executions.append(execution_info)
    
            # Sort by execution timestamp in descending order (most recent first)
            formatted_executions.sort(
                key=lambda x: x['execution_timestamp'],
                reverse=True
            )
    
            return formatted_executions
    
        except Exception as e:
            print(f"Error retrieving algorithm executions: {str(e)}")
            raise
    
    def get_latest_execution(self, workflow_type=None):
        """
        Get the most recent execution for this algorithm.
        
        Args:
            workflow_type (str, optional): Filter by workflow type ('train' or 'inference')
            
        Returns:
            Dict: Latest execution information
        """
        try:
            executions = self.get_algorithm_executions(limit=1)
            
            if workflow_type:
                executions = self.get_algorithm_executions()
                executions = [e for e in executions if e['workflow_type'] == workflow_type]
                if executions:
                    return executions[0]
                return None
                
            return executions[0] if executions else None
            
        except Exception as e:
            print(f"Error retrieving latest execution: {str(e)}")
            raise
    
    def get_algorithm_errors(self, limit=10):
        """
        Get all executions with errors for this algorithm.
        
        Args:
            limit (int): Maximum number of error records to return
            
        Returns:
            List[Dict]: List of executions with error information
        """
        try:
            table = self.dynamodb.Table(self.dynamodb_table_executions)
            
            response = table.query(
                KeyConditionExpression='acm_algorithm_name = :algo',
                FilterExpression='#has_errors = :true',
                ExpressionAttributeNames={
                    '#has_errors': 'has_errors'
                },
                ExpressionAttributeValues={
                    ':algo': self.acm_algorithm_name,
                    ':true': True
                },
                Limit=limit,
                ScanIndexForward=False  # Most recent first
            )
            
            return response['Items']
            
        except Exception as e:
            print(f"Error retrieving algorithm errors: {str(e)}")
            raise
    
    def check_for_errors(self, execution_arn=None):
        """
        Check if there are any errors in recent executions or a specific execution.
        
        Args:
            execution_arn (str, optional): Specific execution to check
            
        Returns:
            Dict: Error summary with details
        """
        try:
            if execution_arn:
                # Check specific execution
                table = self.dynamodb.Table(self.dynamodb_table_executions)
                response = table.scan(
                    FilterExpression='execution_arn = :arn AND #has_errors = :true',
                    ExpressionAttributeNames={
                        '#has_errors': 'has_errors'
                    },
                    ExpressionAttributeValues={
                        ':arn': execution_arn,
                        ':true': True
                    }
                )
                
                if response['Items']:
                    return {
                        'has_errors': True,
                        'error_count': len(response['Items'][0].get('error_list', [])),
                        'errors': response['Items'][0].get('error_list', [])
                    }
                else:
                    return {'has_errors': False, 'error_count': 0, 'errors': []}
            else:
                # Check recent executions for this algorithm
                errors = self.get_algorithm_errors(limit=5)
                if errors:
                    total_errors = sum(len(exec.get('error_list', [])) for exec in errors)
                    return {
                        'has_errors': True,
                        'executions_with_errors': len(errors),
                        'total_error_count': total_errors,
                        'recent_errors': errors
                    }
                else:
                    return {'has_errors': False, 'executions_with_errors': 0, 'total_error_count': 0}
                    
        except Exception as e:
            print(f"Error checking for errors: {str(e)}")
            raise
    
    def display_error_summary(self):
        """
        Display a user-friendly summary of recent errors for this algorithm.
        """
        try:
            error_check = self.check_for_errors()
            
            if not error_check['has_errors']:
                print(f"✅ No recent errors found for algorithm: {self.acm_algorithm_name}")
                return
            
            print(f"⚠️  Error Summary for Algorithm: {self.acm_algorithm_name}")
            print("=" * 60)
            print(f"Executions with errors: {error_check['executions_with_errors']}")
            print(f"Total error count: {error_check['total_error_count']}")
            
            for i, execution in enumerate(error_check['recent_errors'], 1):
                timestamp = execution.get('execution_timestamp', 'Unknown')
                workflow_type = execution.get('workflow_type', 'Unknown')
                status = execution.get('status', 'Unknown')
                
                print(f"\n🔴 Execution {i}:")
                print(f"   Time: {timestamp}")
                print(f"   Type: {workflow_type.upper()}")
                print(f"   Status: {status}")
                
                for j, error in enumerate(execution.get('error_list', []), 1):
                    category = error.get('error_category', 'UNKNOWN')
                    message = error.get('error_message', 'No message')
                    lambda_name = error.get('lambda_name', 'Unknown')
                    suggestion = error.get('suggestion', 'No suggestion available')
                    
                    print(f"\n   Error {j}: {category}")
                    print(f"   Lambda: {lambda_name}")
                    print(f"   Message: {message}")
                    print(f"   💡 Suggestion: {suggestion}")
                
                console_url = execution.get('console_url')
                if console_url:
                    print(f"\n   🔗 Step Functions: {console_url}")
                
                print("-" * 60)
                
        except Exception as e:
            print(f"Error displaying error summary: {str(e)}")
            raise 


def get_cloudformation_env_vars():
    """
    Pull all CloudFormation environment variables and return them as a dictionary.
    
    Returns:
        dict: Dictionary containing all CloudFormation service names and ARNs
    """
    def _get_first_set_env(*keys):
        for key in keys:
            value = os.getenv(key)
            if value:
                return value
        return None

    def _autodiscover_dynamodb_table_name(name_hints):
        # In notebook contexts env vars are often not exported. Use table name hints as a safe fallback.
        try:
            dynamodb_client = boto3.client('dynamodb')
            paginator = dynamodb_client.get_paginator('list_tables')
            candidate_tables = []

            for page in paginator.paginate():
                for table_name in page.get('TableNames', []):
                    normalized_name = table_name.lower()
                    if any(hint in normalized_name for hint in name_hints):
                        candidate_tables.append(table_name)

            return sorted(candidate_tables)[0] if candidate_tables else None
        except Exception:
            return None

    def _autodiscover_ecr_repo_name(table_hint: Optional[str] = None):
        try:
            ecr_client = boto3.client('ecr')
            paginator = ecr_client.get_paginator('describe_repositories')
            repo_names = []

            for page in paginator.paginate():
                for repo in page.get('repositories', []):
                    repo_name = repo.get('repositoryName')
                    if repo_name:
                        repo_names.append(repo_name)

            if not repo_names:
                return None

            hint = (table_hint or '').lower()
            filtered = repo_names

            # Prefer country-specific repos when DynamoDB stack names expose that hint.
            if '-ca-' in hint:
                ca_repos = [name for name in filtered if '-ca-' in name.lower()]
                if ca_repos:
                    filtered = ca_repos
            elif '-uk-' in hint:
                uk_repos = [name for name in filtered if '-uk-' in name.lower()]
                if uk_repos:
                    filtered = uk_repos

            acm_repos = [name for name in filtered if 'acm-repo' in name.lower()]
            if acm_repos:
                return sorted(acm_repos)[0]

            return sorted(filtered)[0]
        except Exception:
            return None

    def _autodiscover_step_function_arn(table_hint: Optional[str] = None):
        try:
            sfn_client = boto3.client('stepfunctions')
            paginator = sfn_client.get_paginator('list_state_machines')
            candidates = []

            for page in paginator.paginate():
                for sm in page.get('stateMachines', []):
                    name = sm.get('name', '')
                    arn = sm.get('stateMachineArn')
                    if arn:
                        candidates.append((name.lower(), arn))

            if not candidates:
                return None

            hint = (table_hint or '').lower()
            filtered = candidates

            # Prefer country-specific state machines when table names expose that hint.
            if '-ca-' in hint:
                ca = [item for item in filtered if '-ca-' in item[0] or 'canada' in item[0]]
                if ca:
                    filtered = ca
            elif '-uk-' in hint:
                uk = [item for item in filtered if '-uk-' in item[0] or 'uk' in item[0]]
                if uk:
                    filtered = uk

            # Prefer pipeline/workflow state machines.
            prioritized = [
                item for item in filtered
                if ('pipeline' in item[0]) or ('workflow' in item[0]) or ('acm' in item[0])
            ]
            if prioritized:
                filtered = prioritized

            filtered = sorted(filtered, key=lambda x: x[0])
            return filtered[0][1]
        except Exception:
            return None

    dynamodb_table_configs = _get_first_set_env(
        'DYNAMODB_TABLE_CONFIGS',
        'dynamodb_table_configs',
        'ACM_DYNAMODB_TABLE_CONFIGS'
    )
    if not dynamodb_table_configs:
        dynamodb_table_configs = _autodiscover_dynamodb_table_name(
            ['algorithm_configs', 'acm_algorithm_configs']
        )

    dynamodb_table_executions = _get_first_set_env(
        'DYNAMODB_TABLE_EXECUTIONS',
        'dynamodb_table_executions',
        'ACM_DYNAMODB_TABLE_EXECUTIONS'
    )
    if not dynamodb_table_executions:
        dynamodb_table_executions = _autodiscover_dynamodb_table_name(
            ['algorithm_executions', 'acm_algorithm_executions']
        )

    ecr_repo_name = _get_first_set_env('ECR_REPO_NAME', 'ecr_repo_name', 'ACM_ECR_REPO_NAME')
    if not ecr_repo_name:
        ecr_repo_name = _autodiscover_ecr_repo_name(dynamodb_table_configs)

    step_function_arn = _get_first_set_env('STEP_FUNCTION_ARN', 'step_function_arn', 'ACM_STEP_FUNCTION_ARN')
    if not step_function_arn:
        step_function_arn = _autodiscover_step_function_arn(dynamodb_table_configs)

    return {
        'dynamodb_table_configs': dynamodb_table_configs,
        'dynamodb_table_executions': dynamodb_table_executions,
        'secrets_name': _get_first_set_env('SECRETS_NAME', 'secrets_name', 'ACM_SECRETS_NAME'),
        'step_function_arn': step_function_arn,
        'ecr_access_role': _get_first_set_env('ECR_ACCESS_ROLE', 'ecr_access_role', 'ACM_ECR_ACCESS_ROLE'),
        'cloudwatch_access_role': _get_first_set_env('CLOUDWATCH_ACCESS_ROLE', 'cloudwatch_access_role', 'ACM_CLOUDWATCH_ACCESS_ROLE'),
        'ecr_repo_name': ecr_repo_name
    }

def check_pipeline_errors(algorithm_name=None):
    """
    Quick function to check for pipeline errors without initializing full pipeline.
    
    Args:
        algorithm_name (str, optional): Specific algorithm to check
        
    Returns:
        Dict: Error summary
    """
    try:
        env_vars = get_cloudformation_env_vars()
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(env_vars['dynamodb_table_executions'])
        
        if algorithm_name:
            # Check specific algorithm
            response = table.query(
                KeyConditionExpression='acm_algorithm_name = :algo',
                FilterExpression='#has_errors = :true',
                ExpressionAttributeNames={'#has_errors': 'has_errors'},
                ExpressionAttributeValues={':algo': algorithm_name, ':true': True},
                Limit=5,
                ScanIndexForward=False
            )
        else:
            # Check all algorithms
            from datetime import datetime, timedelta
            cutoff_time = (datetime.now() - timedelta(hours=24)).isoformat()
            
            response = table.scan(
                FilterExpression='#has_errors = :true AND #timestamp > :cutoff',
                ExpressionAttributeNames={
                    '#has_errors': 'has_errors',
                    '#timestamp': 'execution_timestamp'
                },
                ExpressionAttributeValues={':true': True, ':cutoff': cutoff_time}
            )
        
        return {
            'has_errors': len(response['Items']) > 0,
            'error_executions': response['Items']
        }
        
    except Exception as e:
        print(f"Error checking pipeline errors: {str(e)}")
        return {'has_errors': False, 'error_executions': []}