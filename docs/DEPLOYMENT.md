# Production deployment guide

## Delivery flow

The CD workflow builds the API image once per commit, publishes immutable commit-SHA and `latest` tags to GitHub Container Registry (GHCR), then deploys that same SHA-tagged image to EC2, Kubernetes, or both. It also builds the dedicated MLflow server image used by both targets. A model-bootstrap step verifies the `production` alias; only when the registry is empty does it train, register, and assign the first production model. Deployments never promote an existing candidate model automatically.

Select a target manually in **Actions → Continuous Deployment → Run workflow**: `ec2`, `kubernetes`, or `both`. For automatic releases on every push to `main`, create the GitHub repository/environment variable `DEPLOY_TARGET` with exactly one of those values. It defaults to `ec2`, preserving the EC2 release path.

## EC2 deployment

Use Ubuntu 22.04 LTS (x86_64), at least 2 vCPU and 4 GB memory. Create security-group inbound rules for SSH (`22/TCP`) from only fixed administrative CIDRs and the API port (`8000/TCP`, unless `API_PORT` changes it) only from a trusted load balancer, reverse proxy, or client CIDR. Do not expose port 8000 publicly; put TLS termination in an ALB or reverse proxy. Allow outbound HTTPS to GHCR and the MLflow server.

On the new instance, run this once, replacing each angle-bracketed value:

```sh
sudo mkdir -p <EC2_APP_DIR>
sudo chown <EC2_USER>:<EC2_USER> <EC2_APP_DIR>
```

The deployment script installs Docker Engine and Docker Compose v2 if absent, uploads the Compose file, writes a mode-600 `.env` file from protected GitHub secrets, removes Docker images and build cache unused for seven days before pulling, verifies Docker's data filesystem has at least 1 GiB and 10,000 inodes free, then replaces the container and checks `/health` for up to two minutes. This prevents a new image layer from exhausting overlayfs during extraction. The SSH user must have passwordless `sudo` for first-time Docker installation, or Docker must already be usable by it.

The EC2 Compose stack runs MLflow in its own persistent service. It uses one worker because its SQLite backend is a single-node store; this avoids concurrent SQLite writes and unnecessary memory pressure. The API and model-bootstrap service use `http://mlflow:5000`, where `mlflow` is Docker Compose service DNS; do not override it with `localhost`. If a Compose dependency fails, the deployment action prints the latest logs for MLflow, model bootstrap, and API services.

## GitHub Environments, secrets, and variables

Create protected GitHub Environments named `production-ec2` and `production-kubernetes`. Put the following secrets in **production-ec2**:

| Secret | Value required |
| --- | --- |
| `EC2_HOST` | EC2 public IPv4 address or DNS hostname |
| `EC2_USER` | SSH user, normally `ubuntu` on Ubuntu |
| `EC2_SSH_KEY` | Full private-key content for that EC2 key pair |
| `EC2_APP_DIR` | Absolute deployment directory, such as `/opt/qa-log-anomaly-detector` |
| `GHCR_TOKEN` | PAT with `read:packages`, required only for a private GHCR package |

Optional protected environment variable: `API_PORT` (defaults to `8000`). The workflow uses `GITHUB_TOKEN` with `packages: write` to publish GHCR images. If organization policy restricts it, permit GitHub Actions package publishing/read-write workflow permissions.

To require a larger free-space reserve before image pulls, set `MIN_DOCKER_FREE_MB` in the remote deployment environment (default: `1024`).

Put this secret in **production-kubernetes**:

| Secret | Value required |
| --- | --- |
| `KUBECONFIG` | Base64-encoded kubeconfig for a least-privilege deployer identity |

## Kubernetes and MLflow

The manifests work with EKS, Minikube, or k3s when the cluster has a default dynamic `StorageClass`. EKS needs the AWS EBS CSI driver; Minikube needs its storage provisioner enabled; k3s normally provides `local-path`. Before deploying, verify that a default class exists:

```sh
kubectl get storageclass
```

Kubernetes deploys a one-replica MLflow service backed by a 10 GiB `ReadWriteOnce` PVC, plus a two-replica API Deployment. The API uses the Kubernetes service DNS name and the approved registered-model alias:

```text
MLFLOW_TRACKING_URI=http://mlflow.qa-log-anomaly-detector.svc.cluster.local:5000
MLFLOW_MODEL_URI=models:/qa-log-anomaly-detector@production
```

The MLflow PVC uses SQLite and a filesystem artifact root. This is appropriate for a single MLflow replica and demos/small deployments. For high availability or concurrent write workloads, use an external PostgreSQL backend and object-store artifact root; do not scale the supplied SQLite MLflow Deployment beyond one replica.

If the GHCR packages are private, create this Kubernetes Secret once. Replace every angle-bracketed value:

```sh
kubectl apply -f k8s/namespace.yaml
kubectl -n qa-log-anomaly-detector create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io \
  --docker-username='<GITHUB_OWNER>' \
  --docker-password='<GHCR_READ_TOKEN>'
```

`GHCR_READ_TOKEN` needs `read:packages`. It is a Kubernetes Secret, not a GitHub Actions secret. The supplied ConfigMap holds only non-sensitive MLflow connection settings. Put future AWS, database, or object-store credentials in dedicated Kubernetes Secrets.

The model-bootstrap Job runs automatically after MLflow becomes healthy and before API rollout. It is idempotent: when `qa-log-anomaly-detector@production` already exists, it makes no changes. On an empty registry, it runs the initial pipeline, registers the first model, and assigns the `production` alias. A registry containing candidate versions but no `production` alias still fails: validate the candidate and explicitly promote it in the separate model-release process.

The Kubernetes API Deployment has labels/selectors, immutable SHA image injection, non-root security context, `readOnlyRootFilesystem`, readiness and liveness probes, resource requests/limits, a zero-unavailable rolling strategy, and a PodDisruptionBudget.

## Separate training pipeline and AWS access

Candidate-model training is a separate workflow or Job. Configure its environment with these values in the runtime environment, never in the repository or GitHub Actions logs:

| Environment variable | Required value |
| --- | --- |
| `USE_S3` | `true` when training reads source logs from S3 |
| `S3_BUCKET` | Approved training-data bucket name |
| `S3_INPUT_KEY` | Approved input object key |
| `AWS_DEFAULT_REGION` | Region containing that bucket |
| `MLFLOW_TRACKING_URI` | The MLflow registry where the candidate will be registered |
| `MLFLOW_REGISTERED_MODEL_NAME` | `qa-log-anomaly-detector` |

For EC2 training, attach an Instance IAM Role with only the required S3 read permissions (and any approved artifact permissions). The AWS SDK obtains temporary credentials automatically from instance metadata; do not add access keys to Compose, GitHub Secrets, or the repository.

For EKS training, use IAM Roles for Service Accounts (IRSA): associate a least-privilege IAM role with the dedicated training service account and let the SDK use its projected web-identity token. For Minikube or k3s demonstrations, create a Kubernetes Secret containing AWS credentials only in the local cluster and mount it only into the training Job. This repository does not require or create that local-development Secret.

After validation, promote a candidate explicitly with the MLflow client/UI by assigning its chosen version the `production` alias. The next EC2 or Kubernetes deployment will verify and reuse that alias.

## First deployment and verification

1. Create the EC2 directory and GitHub Environment secrets above.
2. For Kubernetes, create `ghcr-pull-secret` when required and verify the PVC becomes `Bound`.
3. The deployment automatically starts MLflow, performs idempotent model bootstrap, and starts the API only after bootstrap succeeds.
4. Push to `main` after setting `DEPLOY_TARGET`, or run the CD workflow and select a target.
5. Approve the relevant protected Environment.

Verify EC2 from a permitted host:

```sh
curl --fail http://<EC2_HOST>:<API_PORT>/health
```

Verify Kubernetes:

```sh
kubectl -n qa-log-anomaly-detector rollout status deployment/mlflow
kubectl -n qa-log-anomaly-detector rollout status deployment/qa-log-anomaly-api
kubectl -n qa-log-anomaly-detector get pods,pvc,service
```

## Rollback

Every release image has an immutable Git SHA. For EC2, use the prior healthy API SHA:

```sh
cd <EC2_APP_DIR>
API_IMAGE=ghcr.io/<GITHUB_OWNER>/qa-log-anomaly-detector-api:<PREVIOUS_COMMIT_SHA> \
  docker compose -f docker-compose.prod.yml up -d --no-deps --force-recreate api
curl --fail http://127.0.0.1:<API_PORT>/health
```

For Kubernetes, roll back the API independently:

```sh
kubectl -n qa-log-anomaly-detector rollout undo deployment/qa-log-anomaly-api
kubectl -n qa-log-anomaly-detector rollout status deployment/qa-log-anomaly-api
```

MLflow data is persistent and is not rolled back with the API. Promote or reassign the `production` model alias separately if model rollback is required.
