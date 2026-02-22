## OmniAgent EC2 Deployment Guide (From Scratch)

This guide deploys the project on one EC2 instance using Docker Compose and gives a single shareable link.

### 0) Important security fix first

Your repo currently contains real API keys in `.env` and `.env.prod`.

1. Rotate all exposed keys now (OpenAI, Anthropic, Google, Tavily).
2. Replace them in your local `.env.prod`.
3. Never commit secrets again.

### 1) Create AWS EC2 instance

1. In AWS Console, launch `Ubuntu 24.04 LTS` (or `Ubuntu 22.04 LTS`).
2. Instance type: at least `t3.large` (recommended) for smoother Docker builds.
3. Storage: 20+ GB.
4. Create/download a key pair (`.pem`).
5. Security Group inbound rules:
   - `22` (SSH) from `My IP`
   - `3000` (App URL) from `0.0.0.0/0`
   - You do not need to expose `8000` publicly with this setup.

### 2) SSH into EC2

```bash
chmod 400 ~/Downloads/your-key.pem
ssh -i ~/Downloads/your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 3) Install Docker + Compose plugin

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
```

### 4) Clone project on EC2

```bash
git clone <your-repo-url> omniagent
cd omniagent
```

### 5) Configure production env

1. Copy the template:

```bash
cp .env.prod.example .env.prod
```

2. Edit `.env.prod` and set valid keys:
   - `OPENAI_API_KEY=...` (required for default flows + embeddings)
   - `TAVILY_API_KEY=...` (required for web/news tasks)
   - `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` optional unless you choose those providers
3. Keep:
   - `NEXT_PUBLIC_API_BASE=` (empty)
   - `BACKEND_INTERNAL_URL=http://backend:8000`

Why: frontend uses same-origin `/api` proxy to backend, so one recruiter link works and model list loads reliably.

### 6) Build and start

```bash
docker compose --env-file .env.prod up -d --build
docker compose ps
```

### 7) Verify backend + model endpoint

```bash
docker compose logs backend --tail=200
curl -sS http://localhost:8000/api/models
```

Expected: JSON with `providers`, `models`, and `default`.

### 8) Verify frontend from EC2

```bash
docker compose logs frontend --tail=200
curl -I http://localhost:3000
```

### 9) Open in browser (shareable link)

Use:

`http://<EC2_PUBLIC_IP>:3000`

### 10) If models still do not load (exact checks)

1. Browser DevTools -> Network:
   - `GET /api/models` must return `200`.
2. On EC2:

```bash
docker compose exec frontend printenv | grep -E "NEXT_PUBLIC_API_BASE|BACKEND_INTERNAL_URL"
docker compose exec frontend wget -qO- http://backend:8000/api/models
```

3. If `/api/models` fails in browser but works from container, restart frontend:

```bash
docker compose up -d --build frontend
```

### 11) Optional: domain + HTTPS (recommended for recruiters)

1. Buy/attach a domain.
2. Point DNS `A` record to EC2 public IP.
3. Put Nginx/Caddy in front for TLS on `443`.
4. Share `https://yourdomain.com` instead of raw IP.

### 12) Day-2 operations

Update code and redeploy:

```bash
git pull
docker compose --env-file .env.prod up -d --build
```

Stop:

```bash
docker compose down
```
