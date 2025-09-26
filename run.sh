
# export AGENT_ID=agent-xxxx
# export ALIAS_ID=TSTALIASIDxxxx
# export MS_BUCKET=medimage-digitaltwin  # Use your bucket name
# export AWS_REGION=us-east-1
export API_BASE=https://<api-id>.execute-api.<region>.amazonaws.com/prod
# python scripts/invoke_agent.py

# running with api
# Preprocess only:
python scripts/invoke_agent.py preprocess --api-base $API_BASE --source-key raw/chestmnist.npz --denoise --normalize
# Synthesis only (raw):
python scripts/invoke_agent.py synthesis --api-base $API_BASE --input-key raw/chestmnist.npz
# Synthesis using preprocessed run:
python scripts/invoke_agent.py synthesis --api-base $API_BASE --run-id <RUN_ID>
# Full pipeline:
python scripts/invoke_agent.py pipeline --api-base $API_BASE --input-key raw/chestmnist.npz --denoise --normalize --export-png

#running via api
# Your HTTP backend that forwards to Lambda (e.g., API Gateway → Lambda) or any HTTP service
# invoke_api(api_base, "/synthesis", payload)
# You pay for AWS usage; still not testing Bedrock Agent tool selection or Agent payloads

#running with agent
# The Agent (model + tools) that picks actions from your OpenAPI and invokes mapped Lambdas
# Send a user message to the Agent runtime; Agent decides to call POST /synthesis//preprocess
# your client app no longer needs to call your /synthesis or /preprocess HTTP endpoints with invoke_api. The Agent will choose those actions and call the mapped Lambdas directly (via the ARNs in your OpenAPI).

#openapi.yaml file only used by the agent not api,
python scripts/invoke_agent.py agent \
  --message "Preprocess the latest run with denoise=true and zip the previews." \
  --agent-id AGENT-xxxxxxxx \
  --agent-alias-id TSTALIAS-xxxxxxxx 

#differences
# Do you need API Gateway?
# - Using invoke_api (HTTP client) → Yes, you need an HTTP endpoint:
# -EITHER API Gateway (routes like POST /preprocess → your Lambda), or
# -a Lambda Function URL (simplest for testing; one URL per function).

# -Using a Bedrock Agent → No HTTP endpoint required. You just:
# -Deploy the Lambda.
# -Put its ARN in x-amazon-bedrock-integration.uri.
# -call the Agent via bedrock-agent-runtime.invoke_agent