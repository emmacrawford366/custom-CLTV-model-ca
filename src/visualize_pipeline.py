import json
import boto3
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.ACMPipelineClass import get_cloudformation_env_vars


def create_timeline(algorithm_name=None, days=7):

    # Initialize
    cloud_formation_env_vars = get_cloudformation_env_vars()
    dynamodb = boto3.resource('dynamodb')
    executions_table = dynamodb.Table(cloud_formation_env_vars['dynamodb_table_executions'])
    """Create enhanced timeline with algorithm and error overlays"""
    try:
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        if algorithm_name:
            response = executions_table.query(
                KeyConditionExpression='acm_algorithm_name = :algo',
                FilterExpression='#timestamp > :cutoff',
                ExpressionAttributeNames={'#timestamp': 'execution_timestamp'},
                ExpressionAttributeValues={':algo': algorithm_name, ':cutoff': cutoff_time},
                ScanIndexForward=False
            )
        else:
            response = executions_table.scan(
                FilterExpression='#timestamp > :cutoff',
                ExpressionAttributeNames={'#timestamp': 'execution_timestamp'},
                ExpressionAttributeValues={':cutoff': cutoff_time}
            )
        
        executions = response['Items']
        
        if not executions:
            print(f"📊 No executions found in the last {days} days")
            return
        
        # Prepare enhanced data
        data = []
        for execution in executions:
            timestamp = datetime.fromisoformat(execution['execution_timestamp'].replace('Z', '+00:00'))
            status = execution.get('status', 'UNKNOWN')
            workflow_type = execution.get('workflow_type', 'unknown')
            has_errors = execution.get('has_errors', False)
            algorithm = execution.get('acm_algorithm_name', 'unknown')
            
            # Extract error type
            error_type = 'None'
            if has_errors and 'error_list' in execution:
                error_list = execution['error_list']
                if error_list:
                    error_type = error_list[0].get('error_category', 'Unknown')
            
            if status == 'COMPLETED' and not has_errors:
                result = 'Success'
            elif status == 'FAILED' or has_errors:
                result = 'Failed'
            else:
                result = 'Running'
            
            data.append({
                'timestamp': timestamp,
                'result': result,
                'workflow_type': workflow_type,
                'algorithm': algorithm,
                'error_type': error_type
            })
        
        df = pd.DataFrame(data)
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Enhanced timeline scatter with overlays
        colors = {'Success': 'green', 'Failed': 'red', 'Running': 'orange'}
        markers = {'train': 'o', 'inference': 's'}
        
        for result in ['Success', 'Failed', 'Running']:
            for wf_type in ['train', 'inference']:
                subset = df[(df['result'] == result) & (df['workflow_type'] == wf_type)]
                if not subset.empty:
                    ax.scatter(subset['timestamp'], [result] * len(subset), 
                              c=colors[result], marker=markers.get(wf_type, '^'), 
                              s=100, alpha=0.7, 
                              label=f'{result} ({wf_type.title()})')
                    
                    # Add overlays for failed executions
                    if result == 'Failed':
                        for idx, row in subset.iterrows():
                            # Show algorithm name and error type
                            label_text = f"{row['algorithm']}\n{row['error_type']}"
                            ax.annotate(label_text, 
                                       (row['timestamp'], result),
                                       xytext=(10, 10), textcoords='offset points',
                                       fontsize=8, alpha=0.9,
                                       bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='lightcoral', alpha=0.8),
                                       arrowprops=dict(arrowstyle='->', 
                                                     connectionstyle='arc3,rad=0.1'))
                    
                    # Add algorithm names for successful executions (if multiple algorithms)
                    elif result == 'Success' and not algorithm_name:
                        for idx, row in subset.iterrows():
                            ax.annotate(row['algorithm'], 
                                       (row['timestamp'], result),
                                       xytext=(5, -15), textcoords='offset points',
                                       fontsize=7, alpha=0.7,
                                       bbox=dict(boxstyle='round,pad=0.2', 
                                               facecolor='lightgreen', alpha=0.6))
        
        ax.set_ylabel('Execution Result', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title(f'Enhanced Pipeline Timeline - Last {days} Days\n(Shows algorithm names and error types)', 
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Enhanced summary stats
        total = len(df)
        success = len(df[df['result'] == 'Success'])
        failed = len(df[df['result'] == 'Failed'])
        
        print(f"\n📈 Enhanced Summary (Last {days} days):")
        print(f"   Total: {total} | ✅ Success: {success} ({success/total*100:.1f}%) | ❌ Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Error type breakdown
        failed_df = df[df['result'] == 'Failed']
        if not failed_df.empty:
            print(f"\n🔍 Error Types:")
            error_counts = failed_df['error_type'].value_counts()
            for error_type, count in error_counts.items():
                print(f"   • {error_type}: {count} failures")
        
        # Algorithm breakdown (if viewing all algorithms)
        if not algorithm_name:
            print(f"\n📊 Algorithm Performance:")
            algo_stats = df.groupby('algorithm').agg({
                'result': lambda x: f"{(x == 'Success').sum()}/{len(x)} ({(x == 'Success').sum()/len(x)*100:.1f}%)"
            })
            for algo, stats in algo_stats.iterrows():
                print(f"   • {algo}: {stats['result']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def get_recent_errors(hours=24):
    cloud_formation_env_vars = get_cloudformation_env_vars()
    dynamodb = boto3.resource('dynamodb')
    executions_table = dynamodb.Table(cloud_formation_env_vars['dynamodb_table_executions'])
    """Get all recent errors from DynamoDB"""
    cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    try:
        # Scan for executions with errors
        response = executions_table.scan(
            FilterExpression='#has_errors = :true AND #timestamp > :cutoff',
            ExpressionAttributeNames={
                '#has_errors': 'has_errors',
                '#timestamp': 'execution_timestamp'
            },
            ExpressionAttributeValues={
                ':true': True,
                ':cutoff': cutoff_time
            }
        )
        
        # Sort by timestamp descending (newest first)
        items = response['Items']
        return sorted(items, key=lambda x: x.get('execution_timestamp', ''), reverse=True)
    except Exception as e:
        print(f"❌ Error fetching recent errors: {str(e)}")
        return []

def display_error_summary(errors):
    cloud_formation_env_vars = get_cloudformation_env_vars()
    dynamodb = boto3.resource('dynamodb')
    executions_table = dynamodb.Table(cloud_formation_env_vars['dynamodb_table_executions'])
    """Display a clean summary of errors"""
    if not errors:
        print("✅ No errors found in the last 24 hours!")
        return
    
    print(f"⚠️  Found {len(errors)} execution(s) with errors:")
    print("=" * 80)
    
    for execution in errors:
        algorithm_name = execution.get('acm_algorithm_name', 'Unknown')
        timestamp = execution.get('execution_timestamp', 'Unknown')
        workflow_type = execution.get('workflow_type', 'Unknown')
        status = execution.get('status', 'Unknown')
        
        print(f"\n🔴 Algorithm: {algorithm_name}")
        print(f"   Time: {timestamp}")
        print(f"   Type: {workflow_type.upper()}")
        print(f"   Status: {status}")
        
        # Display individual errors
        error_list = execution.get('error_list', [])
        error_log_count = 0
        
        for i, error in enumerate(error_list, 1):
            category = error.get('error_category', 'UNKNOWN')
            message = error.get('error_message', 'No message')
            lambda_name = error.get('lambda_name', 'Unknown')
            suggestion = error.get('suggestion', 'No suggestion available')
            
            print(f"\n   Error {i}: {category}")
            print(f"   Lambda: {lambda_name}")
            print(f"   Message: {message}")
            print(f"   💡 Suggestion: {suggestion}")
            
            # Check if this error has optional error log fields
            if 'error_log_content' in error or 'error_log_s3_location' in error or 'error_log_original_url' in error:
                error_log_count += 1
                print(f"\n   📄 Error Log Available:")
                
                if 'error_log_s3_location' in error:
                    print(f"      S3 Location: {error['error_log_s3_location']}")
                
                # if 'error_log_original_url' in error:
                #     original_url = error['error_log_original_url']
                #     print(f"      Original URL: {original_url[:80]}...")
                
                # Display error content (formatted for readability)
                if 'error_log_content' in error:
                    content = error['error_log_content']
                    print(f"\n   📋 Error Log Content:")
                    print("   " + "─" * 60)
                    
                    # Split content into lines for better formatting
                    lines = content.split('\n')
                    displayed_lines = 0
                    max_lines = 20  # Limit to first 20 lines for readability
                    
                    for line in lines:
                        if displayed_lines >= max_lines:
                            remaining_lines = len(lines) - displayed_lines
                            print(f"   ... [{remaining_lines} more lines - view full content in S3]")
                            break
                        
                        # Add proper indentation and handle long lines
                        if len(line) > 80:
                            # Break long lines at reasonable points
                            words = line.split(' ')
                            current_line = "   "
                            for word in words:
                                if len(current_line + word) > 80:
                                    print(current_line)
                                    current_line = "   " + word + " "
                                else:
                                    current_line += word + " "
                            if current_line.strip():
                                print(current_line.rstrip())
                        else:
                            print(f"   {line}")
                        
                        displayed_lines += 1
                    
                    print("   " + "─" * 60)
        
        # Display legacy error log URLs and S3 locations if available (for backward compatibility)
        error_log_urls = execution.get('error_log_presigned_urls', [])
        error_log_s3_locations = execution.get('error_log_s3_locations', [])
        
        if error_log_urls or error_log_s3_locations:
            print(f"\n   🔗 Error Log Resources:")
            if error_log_urls:
                print(f"      Original URLs: {len(error_log_urls)} found")
                for i, url in enumerate(error_log_urls[:3], 1):  # Show first 3
                    print(f"        {i}. {url[:80]}...")
                if len(error_log_urls) > 3:
                    print(f"        ... and {len(error_log_urls) - 3} more")
            
            if error_log_s3_locations:
                print(f"      S3 Locations: {len(error_log_s3_locations)} files stored")
                for i, location in enumerate(error_log_s3_locations[:3], 1):  # Show first 3
                    print(f"        {i}. {location}")
                if len(error_log_s3_locations) > 3:
                    print(f"        ... and {len(error_log_s3_locations) - 3} more")
        
        # Add console link if available
        console_url = execution.get('console_url')
        if console_url:
            print(f"\n   🔗 Step Functions: {console_url}")
        
        print("-" * 80)