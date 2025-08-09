# Secrets Directory

This directory structure shows how to organize your secrets safely.

## Setup Instructions

1. **Copy this directory structure** to create a `secrets/` directory:
   ```bash
   cp -r configs/secrets.example secrets/
   ```

2. **The `secrets/` directory is git-ignored** and will not be committed.

3. **Fill in your actual secrets** in the copied files.

## Directory Structure

```
secrets/
├── api_keys/
│   ├── openai.key
│   ├── anthropic.key
│   └── huggingface.key
├── tokens/
│   ├── github_token.txt
│   └── jwt_secret.key
├── certificates/
│   ├── ssl_cert.pem
│   └── ssl_key.pem
└── database/
    ├── db_password.txt
    └── redis_auth.txt
```

## Security Best Practices

- Never commit actual secrets to git
- Use environment variables in production
- Rotate secrets regularly
- Use different secrets for different environments
- Consider using secret management tools like:
  - HashiCorp Vault
  - AWS Secrets Manager
  - Azure Key Vault
  - Google Secret Manager

## Loading Secrets in Code

```python
import os
from pathlib import Path

def load_secret(secret_name):
    """Load secret from file or environment variable."""
    # Try environment variable first
    env_value = os.getenv(secret_name)
    if env_value:
        return env_value
    
    # Fallback to secret file
    secret_file = Path("secrets") / f"{secret_name.lower()}.key"
    if secret_file.exists():
        return secret_file.read_text().strip()
    
    raise ValueError(f"Secret {secret_name} not found")
```