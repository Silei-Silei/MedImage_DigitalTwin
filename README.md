# MedImage Digital Twin

A minimal integration to expose your S3-based medical image pipeline via an Amazon Bedrock Agent.

## Overview

This project provides a complete pipeline for medical data synthesis and preprocessing:

- **Digital Twin Generation**: Creates synthetic medical data while preserving statistical properties
- **Data Preprocessing**: Applies denoising, normalization, and resampling operations
- **Bedrock Agent Integration**: Exposes functionality through Amazon Bedrock Agent APIs
- **S3 Storage**: Handles input/output data storage with optional PNG previews

## Architecture

```
CLI (scripts/invoke_agent.py) → API Gateway/Bedrock Agent → Lambda Functions → S3 Storage
```

### Components

- **Agent API**: `agent/openapi.yaml` defines `/synthesis` and `/preprocess` endpoints
- **Lambda Functions**: 
  - `ms_start_generation.py`: Generates digital twins from input data
  - `ms_latest_outputs.py`: Applies preprocessing operations
- **CLI Tools**: `scripts/invoke_agent.py` for testing and automation
- **Infrastructure**: IAM policies and deployment configurations

## Quick Start

### Prerequisites

- AWS CLI configured
- Python 3.8+
- Required packages: `boto3`, `numpy`, `scipy`, `Pillow`, `requests`

### Setup

1. **Configure Environment**:
   ```bash
   export API_BASE=https://<api-id>.execute-api.<region>.amazonaws.com/prod
   export MS_BUCKET=your-bucket-name
   export AWS_REGION=us-east-1
   ```

2. **Deploy Lambda Functions**:
   - Update ARNs in `agent/openapi.yaml`
   - Deploy with required dependencies (`scipy`, `Pillow`)

3. **Test the Pipeline**:
   ```bash
   # Generate digital twin
   python scripts/invoke_agent.py synthesis --api-base $API_BASE --input-key raw/chestmnist.npz --export-png
   
   # Preprocess data
   python scripts/invoke_agent.py preprocess --api-base $API_BASE --run-id <RUN_ID> --denoise --normalize --export-png
   
   # Full pipeline
   python scripts/invoke_agent.py pipeline --api-base $API_BASE --input-key raw/chestmnist.npz --denoise --normalize --export-png
   ```

## API Endpoints

### `/synthesis`
Generates digital twins from input data.

**Input precedence**: `run_id` → `source_key` → `input_key` → default raw data

**Parameters**:
- `run_id` (optional): Use `work/<run_id>/processed.npy` as input
- `source_key` (optional): Direct S3 key for input
- `input_key` (optional): Raw input S3 key (.npy/.npz)
- `recipe` (optional): Metadata object
- `export_png` (optional): Generate PNG previews

### `/preprocess`
Applies preprocessing operations to data.

**Parameters**:
- `run_id` (optional): Use digital twin from this run
- `source_key` (optional): Direct S3 key to process
- `denoise` (optional): Apply mean filtering
- `normalize` (optional): Min-max normalization
- `resample` (optional): Downsample by factor
- `export_png` (optional): Generate PNG previews

## Data Flow

1. **Input**: Raw medical data (.npy/.npz) stored in S3
2. **Preprocessing** (optional): Apply denoising, normalization, resampling
3. **Synthesis**: Generate digital twin preserving statistical properties
4. **Output**: 
   - Processed data: `work/<run_id>/processed.npy`
   - Digital twin: `work/<run_id>/digital_twin.npy`
   - Status: `work/<run_id>/status.json`
   - PNG previews: `work/<run_id>/*_png/` (if enabled)

## File Structure

```
MedImage_DigitalTwin/
├── agent/
│   └── openapi.yaml              # Bedrock Agent API definition
├── lambdas/
│   ├── ms_start_generation.py    # Digital twin generation
│   └── ms_latest_outputs.py      # Data preprocessing
├── scripts/
│   └── invoke_agent.py           # CLI testing tool
├── infra/
│   ├── lambda_trust.json         # IAM trust policy
│   └── lambda_bucket_policy.json # S3 permissions
├── run.sh                        # Example usage script
└── README.md                     # This file
```

## Configuration

### Environment Variables

- `API_BASE`: Your API Gateway/Bedrock Agent endpoint
- `MS_BUCKET`: S3 bucket for data storage (default: `medimage-digitaltwin`)
- `AWS_REGION`: AWS region (default: `us-east-1`)
- `MS_RAW_KEY`: Default input data key (default: `raw/chestmnist.npz`)

### Lambda Dependencies

Ensure these packages are included in your Lambda deployment:
- `boto3` (AWS SDK)
- `numpy` (array operations)
- `scipy` (image processing)
- `Pillow` (PNG export)

## Deployment Notes

1. **Update ARNs**: Replace `${region}` and `${account-id}` in `agent/openapi.yaml`
2. **IAM Permissions**: Apply policies from `infra/` directory
3. **Bucket Configuration**: Update bucket name in policies and environment
4. **Lambda Packaging**: Include all dependencies or use Lambda Layers

## Contributing

This project follows a simple, modular structure designed for easy extension and maintenance.

## License

MIT License - see LICENSE file for details.