# Deploy to GCP Cloud Run (GitHub Actions)

This project includes a GitHub Actions workflow that deploys to Cloud Run on every
push to `main`, after the `Unit Tests` workflow completes successfully.

## Why Workload Identity Federation (WIF)
Do not commit or upload service account keys to GitHub. Instead, use WIF so GitHub
receives short-lived tokens for the service account listed in `dvc-credentials.json`.

## One-time GCP setup (run locally)
Replace the values in ALL_CAPS with your own.

```bash
export PROJECT_ID="mlops-484110"
export REGION="YOUR_REGION"
export REPO="YOUR_GITHUB_ORG/YOUR_GITHUB_REPO"
export SERVICE_ACCOUNT="dvc-796@mlops-484110.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"
gcloud services enable iamcredentials.googleapis.com \
  sts.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com

# Create a Workload Identity Pool + Provider
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Actions Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"

# Allow GitHub repo to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}"

# Allow deploy/push permissions for the service account
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/iam.serviceAccountUser"
```

## GitHub Actions secrets
Add these repository secrets:

- `GCP_PROJECT_ID` = `mlops-484110`
- `GCP_REGION` = `YOUR_REGION`
- `GCP_WORKLOAD_IDENTITY_PROVIDER` =
  `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
- `GCP_SERVICE_ACCOUNT` = `dvc-796@mlops-484110.iam.gserviceaccount.com`
- Optional: `GCP_RUN_SERVICE_ACCOUNT` if you want the Cloud Run service to run
  as a specific service account (for GCS/DVC access at runtime).

## What the workflow does
- Waits for `Unit Tests` to finish successfully on `main`.
- Builds and pushes a Docker image to Artifact Registry.
- Deploys the image to Cloud Run.

The workflow file is at `.github/workflows/deploy-cloud-run.yaml`.
