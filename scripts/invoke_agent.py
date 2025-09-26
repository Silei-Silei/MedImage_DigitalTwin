import os
import json
import argparse
import datetime
import boto3
import requests


def invoke_bedrock_chat(message: str, region: str):
    """
    Call the model with the text prompt and return the response text.

    Input:
    message: the raw user prompt you pass into the function
    Anthropic  only recognizes two roles:
    "user" — your side (prompts, tool results you send back)
    "assistant" — the model’s replies
    """

    #Client is the end-user application for the agent
    client = boto3.client("bedrock-runtime", region_name=region)

    resp = client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": message}
            ],
            "max_tokens": 200 #upper limit for the tokens in response
        })
    )
    return resp["body"].read().decode()


def invoke_api(base_url: str, path: str, payload: dict, headers: dict | None = None, timeout: int = 30):
    """
    call any HTTP endpoint with POST and JSON body

    Simple helper to invoke API Gateway endpoint
    Client (end-user app)
    agent is the middle layer that knows how to talk to the API.
    API (server)
    POST as the action the agent performs on the API to fulfill the client’s request

    Input:
    base_url: The root of the API
    path: The specific route under the base where you’re sending the request
    payload: The data you’re sending in the request body
    headers: Optional HTTP headers for the request
    timeout: How long to wait for a response before giving up
    """
    if not base_url:
        raise ValueError("API base URL is required. Pass --api-base or set API_BASE env var.")
    if base_url.endswith('/'):
        base = base_url[:-1]
    else:
        base = base_url

    #Construct full URL endpoit
    url = f"{base}{path}"

    #POST JSON to the API endpoint
    resp = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()

    #Return parsed JSON body
    return resp.json()


import  uuid

import uuid

def invoke_agent(message: str, *, region: str, agent_id: str, alias_id: str) -> str:
    """
    Send a message to a Bedrock Agent and return the aggregated text response.
    Expects AGENT_ID + AGENT_ALIAS_ID (or pass explicitly).
    """
    client = boto3.client("bedrock-agent-runtime", region_name=region)
    resp = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=str(uuid.uuid4()),
        inputText=message,
    )

    # Event stream: aggregate all chunk bytes into a single string
    out = []
    completion = resp.get("completion")
    if completion is not None:
        for event in completion:
            # common shape: {'chunk': {'bytes': b'...'}}
            chunk = event.get("chunk")
            if chunk and "bytes" in chunk:
                out.append(chunk["bytes"].decode("utf-8", errors="ignore"))
            # you may also see trace or other events; ignore them for now
    return "".join(out)



def save_to_s3(text: str, bucket: str, key: str, region: str):
    """Save text content to S3 as UTF-8 encoded object
     Input:
     text: The text content to save
     bucket: The S3 bucket name
     key: The S3 object key (path within the bucket)
     region: AWS region where the bucket is located
    """

    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def main():
    #source_key run from the preprocseed data
    # input_key raw data
    
    parser = argparse.ArgumentParser(description="MedSynthetic agent helper invoker")
    sub = parser.add_subparsers(dest="command", required=True)

    # Chat (original behavior)
    chat_p = sub.add_parser("chat", help="Invoke Bedrock chat model example")
    chat_p.add_argument("--message", default="请介绍一下医学合成数据。")
    chat_p.add_argument("--save-s3", action="store_true")

    # Start image generation (digital twin)
    start_p = sub.add_parser("synthesis", help="Generate digital twin from input S3 key")
    start_p.add_argument("--run-id", required=False, help="Use work/<run_id>/processed.npy as input")
    start_p.add_argument("--source-key", required=False, help="Direct S3 key for input (e.g., processed.npy)")
    start_p.add_argument("--input-key", required=False, help="S3 key of raw input .npy/.npz; defaults to MS_RAW_KEY or raw/chestmnist.npz")
    start_p.add_argument("--recipe", required=False, help="Optional JSON string for metadata")
    start_p.add_argument("--export-png", action="store_true", help="Also export preview PNGs to S3")
    start_p.add_argument("--api-base", default=os.environ.get("API_BASE"), help="Base URL like https://abc.execute-api.us-east-1.amazonaws.com/prod")

    # Preprocess
    proc_p = sub.add_parser("preprocess", help="Preprocess using run_id digital twin or a source_key")
    proc_p.add_argument("--run-id", required=False)
    proc_p.add_argument("--source-key", required=False)
    proc_p.add_argument("--denoise", action="store_true")
    proc_p.add_argument("--normalize", action="store_true")
    proc_p.add_argument("--resample", action="store_true")
    proc_p.add_argument("--export-png", action="store_true", help="Also export preview PNGs to S3")
    proc_p.add_argument("--api-base", default=os.environ.get("API_BASE"), help="Base URL like https://abc.execute-api.us-east-1.amazonaws.com/prod")

    # Pipeline: preprocess then synthesis in one command
    pipe_p = sub.add_parser("pipeline", help="Run preprocess then synthesis")
    pipe_p.add_argument("--input-key", required=True, help="Raw input .npy/.npz for preprocess step")
    pipe_p.add_argument("--denoise", action="store_true")
    pipe_p.add_argument("--normalize", action="store_true")
    pipe_p.add_argument("--resample", action="store_true")
    pipe_p.add_argument("--export-png", action="store_true")
    pipe_p.add_argument("--api-base", default=os.environ.get("API_BASE"))

    # Agent (talk to Bedrock Agent; Agent decides which Lambda/action to call)
    agent_p = sub.add_parser("agent", help="Send a message to the Bedrock Agent")
    agent_p.add_argument("--message", required=True, help="User message for the Agent")
    agent_p.add_argument("--agent-id", default=os.environ.get("AGENT_ID"), help="Bedrock Agent ID (or set AGENT_ID)")
    agent_p.add_argument("--agent-alias-id", default=os.environ.get("AGENT_ALIAS_ID"), help="Bedrock Agent alias ID (or set AGENT_ALIAS_ID)")
    agent_p.add_argument("--save-s3", action="store_true", help="Save Agent response to S3 like chat")


    args = parser.parse_args()

    region = os.environ.get("AWS_REGION", "us-east-1")
    bucket = os.environ.get("MS_BUCKET", "medsynthetic")

    if args.command == "chat":
        text = invoke_bedrock_chat(args.message, region)
        print(text)
        if args.save_s3:
            ## Use RUN_ID from the environment if present; otherwise create a timestamped run id (UTC).
            run_id = os.environ.get("RUN_ID", datetime.datetime.utcnow().strftime("run_%Y-%m-%d_%H-%M-%S"))
            key = f"output/{run_id}/claude_output.json"

            ## Save the model response text to S3 in the chosen region/bucket.
            save_to_s3(text, bucket, key, region)
            print(f"Output saved to s3://{bucket}/{key}")
        return

    if args.command == "synthesis":
        payload = {}
        if args.run_id:
            payload["run_id"] = args.run_id
        if args.source_key:
            payload["source_key"] = args.source_key
        if args.input_key:
            payload["input_key"] = args.input_key
        if args.recipe:
            try:
                payload["recipe"] = json.loads(args.recipe)
            except Exception:
                payload["recipe"] = {"raw": args.recipe}
        if args.export_png:
            payload["export_png"] = True
        
        #Call backend's /synthesis endpoint with the assembled payload.
        result = invoke_api(args.api_base, "/synthesis", payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "preprocess":
        payload = {
            "denoise": bool(args.denoise),
            "normalize": bool(args.normalize),
            "resample": bool(args.resample)
        }
        if args.run_id:
            payload["run_id"] = args.run_id
        if args.source_key:
            payload["source_key"] = args.source_key
        if args.export_png:
            payload["export_png"] = True
        result = invoke_api(args.api_base, "/preprocess", payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "pipeline":
        # Step 1: preprocess raw
        pre_payload = {
            "resample": bool(args.resample),
            "normalize": bool(args.normalize),
            "denoise": bool(args.denoise),
            "source_key": args.input_key,
        }
        if args.export_png:
            pre_payload["export_png"] = True
        pre_resp = invoke_api(args.api_base, "/preprocess", pre_payload)
        print("[preprocess]", json.dumps(pre_resp, ensure_ascii=False))

        run_id = pre_resp.get("run_id")
        # Step 2: synthesis using run_id's processed output
        syn_payload = {"run_id": run_id}
        if args.export_png:
            syn_payload["export_png"] = True
        syn_resp = invoke_api(args.api_base, "/synthesis", syn_payload)
        print("[synthesis]", json.dumps(syn_resp, ensure_ascii=False, indent=2))
        return

    if args.command == "agent":
        if not args.agent_id or not args.agent_alias_id:
            raise ValueError("Missing Agent identifiers. Provide --agent-id and --agent-alias-id or set AGENT_ID / AGENT_ALIAS_ID env vars.")
        reply = invoke_agent(
            args.message,
            region=region,
            agent_id=args.agent_id,
            alias_id=args.agent_alias_id
        )
        print(reply)

        if args.save_s3:
            run_id = os.environ.get("RUN_ID", datetime.datetime.utcnow().strftime("run_%Y-%m-%d_%H-%M-%S"))
            key = f"output/{run_id}/agent_output.txt"
            save_to_s3(reply, bucket, key, region)
            print(f"Output saved to s3://{bucket}/{key}")
        return


if __name__ == "__main__":
    main()
