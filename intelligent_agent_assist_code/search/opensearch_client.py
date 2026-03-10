# opensearch_client.py
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from intelligent_agent_assist_code.config.settings import OPENSEARCH_ENDPOINT, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


def get_opensearch_client():
    """
    Create and return an OpenSearch client with AWS SigV4 authentication.
    Uses credentials from .env via config.settings.
    """
    if not OPENSEARCH_ENDPOINT:
        raise ConnectionError("❌ OPENSEARCH_ENDPOINT not configured in .env")
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise ConnectionError("❌ AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) not configured in .env")
    
    try:
        # Parse host from endpoint URL
        host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "")
        
        # Create AWS credentials explicitly from .env values
        credentials = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        ).get_credentials()
        
        if not credentials:
            raise ConnectionError("❌ Failed to initialize AWS credentials")
        
        # Create SigV4 authentication
        auth = AWSV4SignerAuth(credentials, AWS_REGION, "es")
        
        # Initialize OpenSearch client
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=10
        )
        
        # Test connection
        try:
            client.info()
        except Exception as conn_err:
            raise ConnectionError(
                f"❌ Failed to connect to OpenSearch cluster at {OPENSEARCH_ENDPOINT}. "
                f"Error: {str(conn_err)}. Check if cluster is running and credentials are valid."
            )
        
        return client
        
    except ConnectionError:
        raise
    except Exception as e:
        raise ConnectionError(f"❌ OpenSearch client creation error: {str(e)}")