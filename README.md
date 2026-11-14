# found-model-eng-team-4
This is the repo for our small medical question answering model we are building as a part of the Foundation Model Engineering course

## Google Cloud GPU Infrastructure

### Setup Status

- [x] Create a shared GCP project
- [x] Configure billing and budget alerts
- [x] Obtain global GPU quota approval
- [x] Confirm regional NVIDIA L4 GPU quota in us-east4
- [x] Configure team IAM permissions
- [ ] Provision the GPU VM
- [ ] Configure shared storage and permissions
- [ ] Download the base model
- [ ] Verify team access

VM provisioning is currently pending due to GPU capacity
availability across multiple regions.

### Planned VM Configuration

| Configuration | Value |
|---|---|
| Machine type | g2-standard-4 |
| GPU | 1 x NVIDIA L4 (24 GB VRAM) |
| CPU | 4 vCPUs |
| RAM | 16 GB |
| Boot disk | 200 GB |
| OS image | Deep Learning VM |
| Provisioning model | Standard (non-Spot) |
| Region / Zone | TBD |

### Team Access

Each team member has been assigned the following IAM roles:

- Compute Viewer
- Compute OS Login
- VM Operator (custom start/stop permissions)

### Start the VM

```bash
gcloud compute instances start foundation-model-gpu \
    --project=YOUR_PROJECT_ID \
    --zone=YOUR_VM_ZONE
```

### SSH into the VM

```bash
gcloud compute ssh foundation-model-gpu \
    --project=YOUR_PROJECT_ID \
    --zone=YOUR_VM_ZONE
```

### Stop the VM

```bash
gcloud compute instances stop foundation-model-gpu \
    --project=YOUR_PROJECT_ID \
    --zone=YOUR_VM_ZONE
```

Do not stop the VM while another team member is running
an experiment. Closing an SSH session does not stop the VM.

### Shared Storage

```text
/shared/
├── models/
├── data/
└── checkpoints/
```

Environment variables:

```bash
export FME_SHARED_DIR=/shared
export HF_HOME=/shared/models
```

Base model: `Qwen/Qwen3-4B-Instruct-2507`

Planned local model path:

```text
/shared/models/Qwen3-4B-Instruct-2507
```

### Budget and Quota

- Monthly budget: $100
- Budget alerts: 50%, 75%, and 90%
- Budget and quota contact: Yuhan (via team group chat)

Budget alerts do not automatically stop the VM.

The final VM location and connection details will be
shared internally once provisioning succeeds.
