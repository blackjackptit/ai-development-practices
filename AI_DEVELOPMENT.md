# AI Development Workflow Guide

## Overview

This guide covers the complete development lifecycle for AI applications—from project setup through deployment. It consolidates best practices for team collaboration, code organization, testing, and deployment specific to AI/LLM applications.

**Contents:**
- Development environment setup
- Project structure and organization
- Git workflow and branching strategy
- Code review process
- Testing and quality assurance
- Deployment and release process
- Team collaboration and communication
- Development best practices

**Related Guides:**
- [Architecture](ARCHITECTURE.md) - System design patterns
- [Cost Reduction](COST_REDUCTION_RULES.md) - Cost optimization
- [Testing](TESTING.md) - Testing strategies
- [Integration](INTEGRATION.md) - API integration patterns

---

## 1. Development Environment Setup

### 1.1 Prerequisites

**Required Tools:**
```bash
# Python 3.9+ with virtual environment
python3 --version

# Git for version control
git --version

# Docker for containerization
docker --version

# Node.js (if building frontend)
node --version
```

**Recommended Tools:**
- VS Code or PyCharm with AI/LLM extensions
- Postman or Insomnia for API testing
- Redis for local caching
- PostgreSQL for local database

### 1.2 Local Development Setup

**Step 1: Clone and Setup**
```bash
# Clone repository
git clone https://github.com/your-org/your-ai-app.git
cd your-ai-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

**Step 2: Environment Variables**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# NEVER commit .env to git
```

**.env.example:**
```bash
# LLM Provider API Keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql://localhost/ai_app_dev
REDIS_URL=redis://localhost:6379/0

# Application
FLASK_ENV=development
DEBUG=True
LOG_LEVEL=DEBUG

# Cost Limits (development)
MAX_TOKENS_PER_REQUEST=4000
DAILY_COST_LIMIT_USD=10.00
```

**Step 3: Database Setup**
```bash
# Create database
createdb ai_app_dev

# Run migrations
python manage.py db upgrade

# Seed test data (optional)
python manage.py seed
```

**Step 4: Verify Setup**
```bash
# Run tests
pytest

# Start development server
python run.py

# Verify health endpoint
curl http://localhost:5000/health
```

### 1.3 IDE Configuration

**VS Code Settings (.vscode/settings.json):**
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".venv": true
  }
}
```

**Recommended Extensions:**
- Python (Microsoft)
- Pylance
- GitLens
- Thunder Client (API testing)
- Better Comments
- Error Lens

---

## 2. Project Structure

### 2.1 Recommended Directory Layout

```
your-ai-app/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── api/
│   ├── __init__.py         # Application factory
│   ├── blueprints/         # Route handlers
│   │   ├── stocks.py
│   │   ├── market.py
│   │   └── system.py
│   ├── extensions.py       # Shared extensions (db, cache)
│   ├── helpers.py          # Utility functions
│   └── middleware.py       # Request/response middleware
├── config/
│   ├── __init__.py
│   ├── development.py
│   ├── production.py
│   └── testing.py
├── database/
│   ├── migrations/         # Database migrations
│   └── seeds/              # Seed data
├── docs/
│   ├── architecture/       # Architecture docs
│   ├── features/           # Feature specifications
│   └── api/                # API documentation
├── jobs/
│   ├── __init__.py
│   └── scheduled/          # Scheduled background jobs
├── scripts/
│   └── deployment/         # Deployment scripts
├── src/
│   ├── __init__.py
│   ├── llm/                # LLM integration layer
│   │   ├── providers/      # Provider-specific code
│   │   ├── router.py       # Model routing logic
│   │   └── cache.py        # Response caching
│   ├── validation/         # Input validation
│   ├── security/           # Security utilities
│   └── monitoring/         # Metrics and logging
├── tests/
│   ├── __init__.py
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── fixtures/           # Test fixtures
│   └── mocks/              # Mock responses
├── .env.example            # Example environment variables
├── .gitignore
├── .pre-commit-config.yaml # Pre-commit hooks
├── docker-compose.yml      # Local services
├── Dockerfile
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── run.py                  # Application entry point
└── README.md
```

### 2.2 Configuration Management

**config/__init__.py:**
```python
import os
from typing import Dict, Any

class Config:
    """Base configuration"""

    # Application
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-prod')
    DEBUG = False
    TESTING = False

    # Database
    DATABASE_URL = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL_SECONDS = 3600

    # LLM Providers
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Cost Limits
    MAX_TOKENS_PER_REQUEST = int(os.getenv('MAX_TOKENS_PER_REQUEST', 4000))
    DAILY_COST_LIMIT_USD = float(os.getenv('DAILY_COST_LIMIT_USD', 100.0))

    # Rate Limiting
    RATE_LIMIT_PER_USER_MINUTE = 60
    RATE_LIMIT_PER_USER_HOUR = 1000

    # Monitoring
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    METRICS_ENABLED = True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    DAILY_COST_LIMIT_USD = 10.0

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

    # Require all secrets in production
    @classmethod
    def validate(cls):
        required = ['SECRET_KEY', 'DATABASE_URL', 'ANTHROPIC_API_KEY']
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required config: {missing}")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_URL = 'postgresql://localhost/ai_app_test'
    CACHE_TTL_SECONDS = 0  # Disable caching in tests
    DAILY_COST_LIMIT_USD = 1.0

config_by_name: Dict[str, Any] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}
```

### 2.3 Code Organization Principles

**1. Separation of Concerns:**
- API routes in `blueprints/` (HTTP handling only)
- Business logic in `src/` (reusable, testable)
- Data access in `database/` (queries, models)
- External integrations in `src/llm/`, `src/external/`

**2. Dependency Injection:**
```python
# Good: Inject dependencies
def generate_response(prompt: str, llm_client: LLMClient, cache: Cache):
    cached = cache.get(prompt)
    if cached:
        return cached

    response = llm_client.generate(prompt)
    cache.set(prompt, response)
    return response

# Bad: Hard-coded dependencies
def generate_response(prompt: str):
    from src.llm.anthropic import client  # Hard to test
    return client.generate(prompt)
```

**3. Configuration Over Hard-coding:**
```python
# Good: Use configuration
max_tokens = current_app.config['MAX_TOKENS_PER_REQUEST']

# Bad: Hard-coded values
max_tokens = 4000  # What if we need to change this?
```

---

## 3. Git Workflow

### 3.1 Branching Strategy

**Branch Types:**
```
main                    # Production-ready code
├── develop            # Integration branch
│   ├── feature/xyz    # New features
│   ├── bugfix/xyz     # Bug fixes
│   ├── hotfix/xyz     # Urgent production fixes
│   └── refactor/xyz   # Code refactoring
```

**Branch Naming:**
- `feature/add-semantic-cache` - New features
- `bugfix/fix-rate-limit-bypass` - Bug fixes
- `hotfix/fix-pii-leak` - Production hotfixes
- `refactor/split-api-server` - Refactoring
- `docs/update-readme` - Documentation

### 3.2 Commit Message Convention

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Build, dependencies, tooling

**Examples:**
```bash
# Good commit messages
feat(llm): Add confidence-based model routing
fix(security): Prevent prompt injection in user input
docs(api): Update authentication endpoint docs
refactor(cache): Extract Redis client to separate module

# Bad commit messages
"Fixed bug"
"Updated code"
"WIP"
```

**Commit with Co-author (for AI assistance):**
```bash
git commit -m "feat(api): Add investment plan endpoints

Implemented CRUD endpoints for investment plans including
filtering by status and date range.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 3.3 Pull Request Process

**Step 1: Create Feature Branch**
```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/add-prompt-cache

# Make changes, commit regularly
git add .
git commit -m "feat(cache): Add semantic caching layer"
```

**Step 2: Keep Branch Updated**
```bash
# Regularly sync with develop
git checkout develop
git pull origin develop
git checkout feature/add-prompt-cache
git rebase develop  # Or: git merge develop
```

**Step 3: Create Pull Request**

**PR Title:** Clear, descriptive summary
```
feat(cache): Add semantic caching for LLM responses
```

**PR Description Template:**
```markdown
## Summary
Implements semantic caching using embeddings to reduce duplicate LLM calls.

## Changes
- Added `SemanticCache` class with embedding-based similarity
- Integrated with existing Redis cache
- Added cache hit/miss metrics
- Updated cost tracking to include cache savings

## Testing
- Unit tests for semantic matching algorithm
- Integration tests with mock LLM responses
- Load tests showing 40% cache hit rate

## Performance Impact
- Reduced average response time from 2.1s to 1.3s
- 40% reduction in LLM API costs
- Minimal increase in Redis memory usage (estimate: +50MB)

## Checklist
- [x] Tests added and passing
- [x] Documentation updated
- [x] No security vulnerabilities introduced
- [x] Backward compatible
- [x] Environment variables documented

## Related Issues
Closes #123
```

### 3.4 Code Review Checklist

**For Reviewers:**

**Functionality:**
- [ ] Code does what PR description says
- [ ] Edge cases handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs

**AI/LLM Specific:**
- [ ] Uses cheapest capable model
- [ ] Deterministic logic checked first
- [ ] Proper token limits set
- [ ] Cost tracking implemented
- [ ] Cache strategy appropriate
- [ ] PII detection/redaction if needed

**Security:**
- [ ] Input validation present
- [ ] No prompt injection vulnerabilities
- [ ] API keys not hard-coded
- [ ] Rate limiting considered
- [ ] Output sanitization if needed

**Code Quality:**
- [ ] Follows project conventions
- [ ] Code is readable and maintainable
- [ ] No unnecessary complexity
- [ ] Proper separation of concerns
- [ ] Comments explain "why", not "what"

**Testing:**
- [ ] Unit tests cover new code
- [ ] Integration tests if needed
- [ ] Tests are meaningful, not just for coverage
- [ ] Tests use mocked LLM responses (not real API calls)

**Documentation:**
- [ ] README updated if needed
- [ ] API docs updated
- [ ] Complex logic has comments
- [ ] Environment variables documented

**Performance:**
- [ ] No obvious performance issues
- [ ] Database queries optimized
- [ ] Caching considered
- [ ] No memory leaks

---

## 4. Development Best Practices

### 4.1 Cost-Aware Development

**Always Check Cost Before Using LLM:**
```python
# Before implementing with LLM, ask:
# 1. Can I use regex/libraries?
# 2. Can I use a rule-based system?
# 3. Can I cache this?
# 4. Do I need the expensive model?

# Example: Email validation
def validate_email(email: str) -> bool:
    # ✅ FREE: Use regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,}$'
    return re.match(pattern, email) is not None

    # ❌ EXPENSIVE: Don't use LLM
    # response = llm.generate(f"Is '{email}' a valid email? Yes or No")
    # return response.strip().lower() == 'yes'
```

**Use Development Mocks:**
```python
# In development, use mocked responses
if app.config['TESTING'] or app.config['DEBUG']:
    from tests.mocks import MockLLMClient
    llm_client = MockLLMClient()
else:
    from src.llm import ProductionLLMClient
    llm_client = ProductionLLMClient()
```

**Set Cost Limits Locally:**
```bash
# .env for development
DAILY_COST_LIMIT_USD=5.00
MAX_TOKENS_PER_REQUEST=2000
```

### 4.2 Security-First Development

**Input Validation Pattern:**
```python
from src.validation import InputValidator
from src.security import PromptInjectionDetector

def process_user_input(user_input: str):
    # 1. Validate
    validator = InputValidator()
    if not validator.validate_length(user_input, max_length=1000):
        raise ValueError("Input too long")

    # 2. Check for prompt injection
    detector = PromptInjectionDetector()
    if detector.is_attack(user_input):
        logger.warning(f"Prompt injection detected: {user_input[:50]}")
        raise SecurityError("Suspicious input detected")

    # 3. Redact PII
    from src.security import PIIDetector
    sanitized = PIIDetector().redact_pii(user_input)

    # 4. Now safe to process
    return llm_client.generate(sanitized)
```

**Never Log Sensitive Data:**
```python
# Good: Log without PII
logger.info(f"Processing request {request_id} for user {user_id}")

# Bad: Logs full content (may contain PII)
logger.info(f"Request content: {user_input}")
```

**Use Environment Variables for Secrets:**
```python
# Good: Use environment variables
api_key = os.getenv('ANTHROPIC_API_KEY')

# Bad: Hard-coded secrets
api_key = "sk-ant-..."  # NEVER DO THIS
```

### 4.3 Error Handling Patterns

**Graceful Degradation:**
```python
def get_ai_recommendation(user_query: str):
    try:
        # Try expensive model first
        return llm_client.generate(user_query, model='opus')
    except RateLimitError:
        logger.warning("Rate limited, falling back to cheaper model")
        return llm_client.generate(user_query, model='haiku')
    except Exception as e:
        logger.error(f"LLM error: {e}")
        # Fall back to rule-based system
        return rule_engine.get_recommendation(user_query)
```

**Circuit Breaker Pattern:**
```python
from src.resilience import CircuitBreaker

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60
)

@circuit_breaker.protected
def call_external_api():
    # If this fails 5 times, circuit opens for 60 seconds
    return requests.get('https://external-api.com/data')
```

### 4.4 Logging Best Practices

**Structured Logging:**
```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def log_llm_request(request_data: dict):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': request_data['id'],
        'user_id': request_data['user_id'],
        'model': request_data['model'],
        'input_tokens': request_data['input_tokens'],
        'output_tokens': request_data['output_tokens'],
        'cost_usd': request_data['cost'],
        'latency_ms': request_data['latency'],
        'cache_hit': request_data.get('cache_hit', False),
    }
    logger.info(json.dumps(log_entry))
```

**Log Levels:**
```python
# DEBUG: Detailed diagnostic information
logger.debug(f"Cache key: {cache_key}")

# INFO: General informational messages
logger.info(f"Request {request_id} completed in {latency}ms")

# WARNING: Something unexpected but not an error
logger.warning(f"Cache miss for common query: {query}")

# ERROR: Error that prevented operation
logger.error(f"LLM API error: {str(e)}", exc_info=True)

# CRITICAL: Severe error, system may be unstable
logger.critical(f"Cost limit exceeded: ${daily_cost}")
```

### 4.5 Testing During Development

**Test-Driven Development:**
```python
# 1. Write test first
def test_semantic_cache_finds_similar_queries():
    cache = SemanticCache()

    # Store original
    cache.set("What is Python?", "Python is a programming language")

    # Similar query should hit cache
    result = cache.get("What's Python?")
    assert result == "Python is a programming language"

# 2. Run test (it fails)
# pytest tests/unit/test_cache.py

# 3. Implement feature
class SemanticCache:
    def get(self, query: str):
        embedding = self.get_embedding(query)
        similar = self.find_similar(embedding, threshold=0.9)
        if similar:
            return self.storage.get(similar.key)
        return None

# 4. Run test (it passes)
```

**Quick Test Script:**
```python
# scripts/test_feature.py
"""Quick manual test for new feature"""

from src.llm import LLMClient
from src.cache import SemanticCache

def main():
    cache = SemanticCache()
    client = LLMClient(model='haiku')

    # Test 1: Cache miss
    print("Test 1: First call (cache miss)")
    response1 = client.generate_with_cache("What is AI?", cache)
    print(f"Response: {response1}")
    print(f"Cache hit: {cache.last_hit}")

    # Test 2: Cache hit
    print("\nTest 2: Similar call (cache hit)")
    response2 = client.generate_with_cache("What's AI?", cache)
    print(f"Response: {response2}")
    print(f"Cache hit: {cache.last_hit}")

    assert cache.last_hit == True, "Expected cache hit"
    print("\n✓ All tests passed")

if __name__ == '__main__':
    main()
```

---

## 5. Testing and Quality Assurance

### 5.1 Testing Strategy

**Testing Pyramid:**
```
        /\
       /E2E\        (5%) - Full end-to-end tests
      /------\
     /Integr.\     (15%) - Integration tests
    /----------\
   /   Unit     \  (80%) - Unit tests
  /--------------\
```

**Test Categories:**

**Unit Tests (80% of tests):**
- Mock all external dependencies
- Fast execution (<1 second per test)
- Test business logic, validation, utilities
```python
def test_pii_detector_redacts_email():
    detector = PIIDetector()
    text = "Contact me at john@example.com"
    redacted = detector.redact_pii(text)
    assert "john@example.com" not in redacted
    assert "[EMAIL_REDACTED]" in redacted
```

**Integration Tests (15% of tests):**
- Test component interactions
- Use test database, real Redis
- Mock only external APIs (LLM providers)
```python
def test_api_endpoint_with_auth(client, auth_token):
    response = client.post(
        '/api/generate',
        json={'prompt': 'Hello'},
        headers={'Authorization': f'Bearer {auth_token}'}
    )
    assert response.status_code == 200
    assert 'response' in response.json
```

**E2E Tests (5% of tests):**
- Full user workflows
- Use staging environment
- Test critical paths only
```python
def test_full_user_workflow():
    # 1. User registers
    # 2. User authenticates
    # 3. User sends prompt
    # 4. User receives response
    # 5. Response is cached
    # 6. Second similar prompt hits cache
    pass
```

### 5.2 Pre-commit Hooks

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--ignore=E203,W503']

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/unit -v
        language: system
        pass_filenames: false
        always_run: true
```

**Install pre-commit:**
```bash
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### 5.3 Continuous Integration

**GitHub Actions Workflow (.github/workflows/ci.yml):**
```yaml
name: CI

on:
  pull_request:
    branches: [develop, main]
  push:
    branches: [develop, main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          flake8 src/ api/ tests/
          black --check src/ api/ tests/

      - name: Run unit tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/unit --cov=src --cov=api --cov-report=xml

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/integration

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=80
```

---

## 6. Deployment Process

### 6.1 Deployment Checklist

**Pre-deployment:**
- [ ] All tests passing in CI
- [ ] Code reviewed and approved
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
- [ ] Cost limits set appropriately

**Deployment:**
- [ ] Deploy to staging first
- [ ] Run smoke tests on staging
- [ ] Deploy to production
- [ ] Verify health checks
- [ ] Monitor error rates for 15 minutes
- [ ] Check cost metrics

**Post-deployment:**
- [ ] Verify all features working
- [ ] Monitor logs for errors
- [ ] Check performance metrics
- [ ] Confirm cost is within budget
- [ ] Update documentation if needed
- [ ] Notify team of deployment

### 6.2 Deployment Strategy

**Blue-Green Deployment:**
```
┌─────────────┐
│ Load        │
│ Balancer    │
└─────────────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼──┐
│Blue │ │Green│
│(Old)│ │(New)│
└─────┘ └─────┘

1. Deploy to Green
2. Test Green
3. Switch traffic to Green
4. Keep Blue for rollback
```

**Rolling Deployment:**
```
Server 1: Old → Deploy → New
Server 2: Old → Old → Deploy → New
Server 3: Old → Old → Old → Deploy → New

Gradual rollout, always some servers available
```

### 6.3 Environment-Specific Configuration

**Staging Environment:**
- Use separate database
- Lower rate limits
- Reduced cost limits
- Test data only
- Monitor closely

**Production Environment:**
- Strict rate limiting
- Cost alerts at 80%, 95%, 100%
- High availability setup
- Real-time monitoring
- Automated backups

**Environment Variables:**
```bash
# Staging
FLASK_ENV=staging
DATABASE_URL=postgresql://staging-db
DAILY_COST_LIMIT_USD=50.00
RATE_LIMIT_PER_USER_HOUR=500

# Production
FLASK_ENV=production
DATABASE_URL=postgresql://prod-db
DAILY_COST_LIMIT_USD=500.00
RATE_LIMIT_PER_USER_HOUR=1000
```

### 6.4 Deployment Scripts

**deploy.sh:**
```bash
#!/bin/bash
set -e

ENV=$1  # staging or production

if [ "$ENV" != "staging" ] && [ "$ENV" != "production" ]; then
    echo "Usage: ./deploy.sh [staging|production]"
    exit 1
fi

echo "Deploying to $ENV..."

# 1. Run pre-deployment checks
echo "Running tests..."
pytest tests/unit tests/integration

# 2. Build Docker image
echo "Building Docker image..."
docker build -t ai-app:$ENV .

# 3. Push to registry
echo "Pushing to registry..."
docker tag ai-app:$ENV registry.example.com/ai-app:$ENV
docker push registry.example.com/ai-app:$ENV

# 4. Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl set image deployment/ai-app \
    ai-app=registry.example.com/ai-app:$ENV \
    --namespace=$ENV

# 5. Wait for rollout
echo "Waiting for rollout..."
kubectl rollout status deployment/ai-app --namespace=$ENV

# 6. Run smoke tests
echo "Running smoke tests..."
./scripts/smoke-test.sh $ENV

echo "✓ Deployment complete!"
```

**smoke-test.sh:**
```bash
#!/bin/bash
set -e

ENV=$1
BASE_URL="https://$ENV.example.com"

echo "Running smoke tests on $BASE_URL..."

# Test 1: Health check
echo "1. Health check..."
curl -f $BASE_URL/health || exit 1

# Test 2: Authentication
echo "2. Authentication..."
TOKEN=$(curl -s -X POST $BASE_URL/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"test","password":"test"}' \
    | jq -r '.token')

# Test 3: Generate response
echo "3. Generate response..."
curl -f -X POST $BASE_URL/api/generate \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello"}' || exit 1

echo "✓ All smoke tests passed!"
```

### 6.5 Rollback Procedure

**Quick Rollback:**
```bash
# Rollback to previous version
kubectl rollout undo deployment/ai-app --namespace=production

# Rollback to specific revision
kubectl rollout undo deployment/ai-app --to-revision=3 --namespace=production

# Check rollout status
kubectl rollout status deployment/ai-app --namespace=production
```

**When to Rollback:**
- Error rate > 5%
- P95 latency > 10 seconds
- Cost spike > 2x expected
- Critical security vulnerability discovered
- Database corruption detected

---

## 7. Team Collaboration

### 7.1 Communication Guidelines

**Daily Standup (Async):**
```markdown
## Yesterday
- Completed semantic caching implementation
- Reduced average response time by 35%

## Today
- Add unit tests for cache layer
- Update documentation

## Blockers
- Need review on PR #123
- Clarification needed on PII detection requirements
```

**Feature Discussion Template:**
```markdown
## Feature: Semantic Caching

### Problem
Users are asking similar questions, causing redundant LLM calls and increased costs.

### Proposed Solution
Implement semantic caching using embeddings to detect similar queries.

### Alternatives Considered
1. Exact match caching (too limited)
2. Fuzzy string matching (not semantically aware)

### Implementation Plan
1. Integrate embedding model (all-MiniLM-L6-v2)
2. Store embeddings in Redis with vector search
3. Add cache hit/miss metrics
4. Set similarity threshold to 0.9

### Cost/Benefit
- Cost: 2-3 days development, +$10/month for embeddings
- Benefit: Estimated 30-40% reduction in LLM costs ($200/month savings)

### Questions
- What similarity threshold should we use?
- Should we cache negative results?
```

### 7.2 Code Review Best Practices

**For Authors:**
1. Keep PRs small (< 400 lines changed)
2. Write clear description with context
3. Add screenshots/videos for UI changes
4. Respond to feedback within 24 hours
5. Resolve all conversations before merging

**For Reviewers:**
1. Review within 24 hours
2. Be constructive, not critical
3. Ask questions, don't assume
4. Approve if minor changes needed
5. Test locally if changing critical paths

**Review Comments:**
```python
# Good: Constructive with explanation
"Consider caching this result since we call it in a loop.
This could reduce API calls from N to 1."

# Bad: Vague criticism
"This looks slow."
```

### 7.3 Knowledge Sharing

**Weekly Tech Talk (30 minutes):**
- Share learnings from past week
- Demo new features
- Discuss challenges and solutions
- Review cost/performance metrics

**Documentation Culture:**
- Document decisions in ADR (Architecture Decision Records)
- Update README when adding features
- Comment complex algorithms
- Keep runbooks updated

**Architecture Decision Record (ADR) Template:**
```markdown
# ADR 001: Use Semantic Caching

## Status
Accepted

## Context
Users frequently ask similar questions, leading to redundant LLM API calls.
Our monthly LLM costs are $600, with estimated 30-40% being duplicate queries.

## Decision
Implement semantic caching using sentence embeddings to detect similar queries.

## Consequences
**Positive:**
- 30-40% reduction in LLM costs
- Faster response times for similar queries
- Better user experience

**Negative:**
- Additional complexity in caching layer
- Small cost for embedding model ($10/month)
- Risk of serving stale/incorrect cached responses

**Mitigation:**
- Set conservative similarity threshold (0.9)
- Add cache invalidation on content updates
- Monitor cache hit rate and quality

## Implementation
- Embedding model: all-MiniLM-L6-v2 (fast, good quality)
- Storage: Redis with vector similarity search
- Threshold: 0.9 (tune based on metrics)

## Date
2026-02-06

## Authors
@john, @jane
```

---

## 8. Monitoring and Debugging

### 8.1 Development Monitoring

**Local Metrics Dashboard:**
```python
# scripts/monitor.py
"""Local development metrics"""

import redis
import json
from tabulate import tabulate

def get_local_metrics():
    r = redis.from_url('redis://localhost:6379/0')

    # Get today's metrics
    today = datetime.now().strftime('%Y-%m-%d')
    key = f"metrics:{today}"

    metrics = r.hgetall(key)

    data = [
        ['Total Requests', metrics.get('total_requests', 0)],
        ['Total Cost (USD)', f"${metrics.get('total_cost', 0):.4f}"],
        ['Cache Hit Rate', f"{metrics.get('cache_hit_rate', 0):.1f}%"],
        ['Avg Latency (ms)', metrics.get('avg_latency', 0)],
    ]

    print(tabulate(data, headers=['Metric', 'Value'], tablefmt='grid'))

if __name__ == '__main__':
    get_local_metrics()
```

### 8.2 Debugging Techniques

**Debug Logging:**
```python
import logging

# Enable debug logging in development
if app.debug:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Log LLM calls
    @app.before_request
    def log_request():
        logger.debug(f"Request: {request.method} {request.path}")
        logger.debug(f"Headers: {dict(request.headers)}")
        logger.debug(f"Body: {request.get_json()}")
```

**Interactive Debugging with pdb:**
```python
import pdb

def process_prompt(prompt: str):
    # Set breakpoint
    pdb.set_trace()

    # Debug interactively:
    # - Print variables: p prompt
    # - Next line: n
    # - Continue: c
    # - Quit: q

    response = llm_client.generate(prompt)
    return response
```

**Flask Shell for Testing:**
```bash
# Start Flask shell
flask shell

# Test components interactively
>>> from src.llm import LLMClient
>>> client = LLMClient()
>>> response = client.generate("Hello")
>>> print(response)
```

### 8.3 Common Issues and Solutions

**Issue: High LLM Costs**
```python
# Debug: Log all LLM calls
def log_llm_call(model, input_tokens, output_tokens, cost):
    logger.warning(f"LLM call: {model}, {input_tokens}+{output_tokens} tokens, ${cost:.4f}")

# Find expensive calls
# grep "LLM call" logs/app.log | sort -t'$' -k2 -n | tail -20
```

**Issue: Slow Response Times**
```python
# Debug: Add timing
import time

start = time.time()
response = llm_client.generate(prompt)
elapsed = time.time() - start

if elapsed > 5.0:
    logger.warning(f"Slow LLM call: {elapsed:.2f}s")
```

**Issue: Cache Not Working**
```python
# Debug: Check cache
def debug_cache(cache_key):
    cached = redis_client.get(cache_key)
    logger.debug(f"Cache lookup: {cache_key}")
    logger.debug(f"Cached value: {cached}")
    logger.debug(f"TTL: {redis_client.ttl(cache_key)}s")
```

---

## 9. Development Workflow Examples

### 9.1 Adding a New Feature

**Example: Add Prompt Template System**

**Step 1: Create Branch**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/prompt-templates
```

**Step 2: Write Tests First (TDD)**
```python
# tests/unit/test_prompt_templates.py
def test_template_substitution():
    template = PromptTemplate("Hello {name}, you are {age} years old")
    result = template.render(name="John", age=30)
    assert result == "Hello John, you are 30 years old"

def test_template_missing_variable():
    template = PromptTemplate("Hello {name}")
    with pytest.raises(ValueError):
        template.render(age=30)  # Missing 'name'
```

**Step 3: Implement Feature**
```python
# src/templates.py
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        self.variables = self._extract_variables()

    def _extract_variables(self):
        import re
        return set(re.findall(r'\{(\w+)\}', self.template))

    def render(self, **kwargs):
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return self.template.format(**kwargs)
```

**Step 4: Test Locally**
```bash
pytest tests/unit/test_prompt_templates.py -v
python scripts/test_templates.py  # Manual test
```

**Step 5: Update Documentation**
```markdown
# docs/features/prompt-templates.md

## Prompt Templates

Use templates to standardize prompts and reduce errors.

### Usage
\`\`\`python
from src.templates import PromptTemplate

template = PromptTemplate(
    "Analyze this text: {text}\n"
    "Focus on: {focus_area}"
)

result = template.render(
    text=user_input,
    focus_area="sentiment"
)
\`\`\`
```

**Step 6: Create Pull Request**
```bash
git add .
git commit -m "feat(templates): Add prompt template system"
git push origin feature/prompt-templates

# Create PR on GitHub with description
```

### 9.2 Fixing a Bug

**Example: Fix Rate Limiting Bypass**

**Step 1: Reproduce Bug**
```python
# tests/integration/test_rate_limit_bug.py
def test_rate_limit_bypass():
    """Bug: Users can bypass rate limit by changing user-agent"""

    # Make 100 requests with different user agents
    for i in range(100):
        response = client.post(
            '/api/generate',
            headers={'User-Agent': f'Browser-{i}'}
        )
        # Should be rate limited after 60 requests
        if i > 60:
            assert response.status_code == 429  # Currently fails
```

**Step 2: Write Fix**
```python
# Before: Rate limiting by user-agent
@rate_limiter.limit("60/minute", key=lambda: request.headers.get('User-Agent'))

# After: Rate limiting by API key or IP
@rate_limiter.limit("60/minute", key=lambda:
    request.headers.get('X-API-Key') or request.remote_addr
)
```

**Step 3: Verify Fix**
```bash
pytest tests/integration/test_rate_limit_bug.py -v
# Now passes
```

**Step 4: Add Regression Test**
```python
# Keep test in test suite to prevent regression
```

**Step 5: Create Hotfix**
```bash
git checkout -b hotfix/rate-limit-bypass
git add .
git commit -m "fix(security): Prevent rate limit bypass via user-agent"
git push origin hotfix/rate-limit-bypass

# Create PR, mark as urgent
```

---

## 10. Development Checklist

### 10.1 New Feature Checklist

**Planning:**
- [ ] Feature requirements documented
- [ ] Architecture decision made
- [ ] Cost impact estimated
- [ ] Security implications considered
- [ ] Breaking changes identified

**Development:**
- [ ] Tests written first (TDD)
- [ ] Feature implemented
- [ ] All tests passing locally
- [ ] Code follows style guide
- [ ] No hard-coded values
- [ ] Error handling implemented
- [ ] Logging added

**AI/LLM Specific:**
- [ ] Deterministic logic checked first
- [ ] Cheapest capable model used
- [ ] Token limits set
- [ ] Caching implemented
- [ ] Cost tracking added
- [ ] PII handling if needed

**Documentation:**
- [ ] Code comments added
- [ ] API documentation updated
- [ ] README updated if needed
- [ ] Migration guide if breaking changes

**Review:**
- [ ] Self-review completed
- [ ] PR created with clear description
- [ ] Tests passing in CI
- [ ] Code reviewed by peer
- [ ] Security reviewed if needed
- [ ] All comments addressed

**Deployment:**
- [ ] Deployed to staging
- [ ] Smoke tests passed
- [ ] Deployed to production
- [ ] Monitoring verified
- [ ] Team notified

### 10.2 Bug Fix Checklist

**Investigation:**
- [ ] Bug reproduced locally
- [ ] Root cause identified
- [ ] Impact assessed (severity, users affected)
- [ ] Regression test written

**Fix:**
- [ ] Minimal fix implemented (don't over-engineer)
- [ ] Regression test passes
- [ ] No new bugs introduced
- [ ] Side effects considered

**Testing:**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Edge cases tested

**Documentation:**
- [ ] Changelog updated
- [ ] Related documentation updated
- [ ] Post-mortem written if critical

**Release:**
- [ ] Hotfix or regular release decided
- [ ] Deployed and verified
- [ ] Users notified if needed

### 10.3 Code Review Checklist

**Before Requesting Review:**
- [ ] All tests passing
- [ ] Code self-reviewed
- [ ] No debug code left
- [ ] Commits are clean and logical
- [ ] PR description is clear

**For Reviewers:**
- [ ] Understand the problem being solved
- [ ] Code does what description says
- [ ] Tests are meaningful
- [ ] No security vulnerabilities
- [ ] Cost implications considered
- [ ] Code is maintainable
- [ ] Documentation is sufficient

---

## 11. Resources and References

### 11.1 Internal Documentation

- [Architecture Guide](ARCHITECTURE.md) - System design
- [Cost Reduction Rules](COST_REDUCTION_RULES.md) - Cost optimization
- [Security Guide](SECURITY.md) - Security practices
- [Testing Guide](TESTING.md) - Testing strategies
- [Integration Guide](INTEGRATION.md) - API integration
- [Observability Guide](OBSERVABILITY.md) - Monitoring and logging
- [Compliance Guide](COMPLIANCE.md) - Regulatory compliance

### 11.2 External Resources

**Python Development:**
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [pytest Documentation](https://docs.pytest.org/)

**AI/LLM:**
- [Anthropic API Docs](https://docs.anthropic.com/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)

**DevOps:**
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

**Best Practices:**
- [12 Factor App](https://12factor.net/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### 11.3 Development Tools

**Code Quality:**
- Black (code formatting)
- flake8 (linting)
- isort (import sorting)
- pylint (static analysis)
- mypy (type checking)

**Testing:**
- pytest (test framework)
- pytest-cov (coverage)
- pytest-mock (mocking)
- locust (load testing)

**Monitoring:**
- Prometheus (metrics)
- Grafana (dashboards)
- Sentry (error tracking)
- DataDog (APM)

---

## 12. Appendix

### 12.1 Common Commands Reference

```bash
# Development
python run.py                     # Start dev server
flask shell                       # Interactive shell
pytest                            # Run all tests
pytest tests/unit -v              # Run unit tests verbose
pytest --cov=src                  # Run with coverage

# Git
git checkout -b feature/xyz       # Create feature branch
git rebase develop                # Update from develop
git push origin feature/xyz       # Push branch

# Docker
docker-compose up                 # Start local services
docker-compose down               # Stop services
docker build -t app:dev .         # Build image

# Database
flask db migrate -m "message"     # Create migration
flask db upgrade                  # Run migrations
flask db downgrade                # Rollback migration

# Deployment
./scripts/deploy.sh staging       # Deploy to staging
./scripts/deploy.sh production    # Deploy to production
./scripts/smoke-test.sh staging   # Run smoke tests
```

### 12.2 Troubleshooting Guide

**Problem: Tests failing with "connection refused"**
```bash
# Solution: Start local services
docker-compose up -d postgres redis
```

**Problem: Import errors in tests**
```bash
# Solution: Install in editable mode
pip install -e .
```

**Problem: "Module not found" in production**
```bash
# Solution: Check requirements.txt includes all dependencies
pip freeze > requirements.txt
```

**Problem: High memory usage**
```bash
# Debug: Profile memory
pip install memory_profiler
python -m memory_profiler script.py
```

**Problem: Slow tests**
```bash
# Debug: Profile test time
pytest --durations=10
```

---

## Summary

This guide provides a complete development workflow for AI applications:

1. **Environment Setup** - Get started with proper configuration
2. **Project Structure** - Organize code for maintainability
3. **Git Workflow** - Collaborate effectively with team
4. **Best Practices** - Write secure, cost-efficient code
5. **Testing** - Ensure quality with comprehensive tests
6. **Deployment** - Ship safely to production
7. **Collaboration** - Work effectively as a team
8. **Monitoring** - Debug and optimize continuously

**Key Principles:**
- 🎯 Cost-aware development (deterministic logic first)
- 🔒 Security-first approach (validate, detect, protect)
- ✅ Test-driven development (write tests first)
- 📊 Monitor everything (cost, performance, quality)
- 🤝 Collaborate openly (clear communication, good documentation)

**Next Steps:**
1. Set up your development environment
2. Clone the project and run tests
3. Pick a task from the backlog
4. Create a branch and start coding
5. Submit PR and get it reviewed
6. Deploy and monitor

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Status:** Active
