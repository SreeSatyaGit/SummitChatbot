Docker Swarm deploy + secrets

Overview
--------
This project includes a `docker-stack.yml` file and a GitHub Actions workflow (`.github/workflows/deploy.yml`) that build images, create Docker secrets and deploy a Docker Stack to a Swarm cluster. Secrets are not stored in the repo — they are supplied via GitHub Actions Secrets.

Pre-requisites
-------------
- A Docker Registry (Docker Hub, ECR, GCR, etc.) with credentials stored in GitHub Secrets: `REGISTRY`, `REGISTRY_USER`, `REGISTRY_TOKEN`.
- GitHub Secrets: `RAG_API_KEY` and `RAG_API_KEYS` (comma-separated keys)
- A Docker Swarm manager node where the GitHub runner can reach the Docker daemon (self-hosted runner recommended) or credentials to access a remote Docker endpoint.

How it works
------------
- The workflow builds and pushes images, creates swarm secrets `rag_api_key` and `rag_api_keys`, then deploys `docker-stack.yml` as a stack.
- Services read their API key(s) from `/run/secrets/rag_api_key` and `/run/secrets/rag_api_keys` (the code falls back to env vars for local development).

Notes
-----
- In production, prefer a managed secret store (AWS Secrets Manager, HashiCorp Vault) and inject secrets into the swarm with a secure CI runner; the included GitHub action will create Docker secrets on the runner's Docker daemon.
- Verify GPU scheduling and device plugin configuration for the `rag_server` service when deploying to a GPU-enabled swarm.
