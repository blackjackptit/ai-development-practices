# Security Architecture for AI Applications

## Overview

This guide provides comprehensive architectural patterns and infrastructure design for securing AI/LLM applications. It focuses on defense-in-depth strategies, secure system design, threat detection architecture, and incident response infrastructure.

**Focus Areas:**
- Security architecture patterns
- Defense in depth design
- Input validation infrastructure
- PII detection and redaction architecture
- Authentication and authorization systems
- API security architecture
- Data protection infrastructure
- Threat detection and response
- Security monitoring systems
- Zero-trust architecture

**Related Guides:**
- [Security Guide](SECURITY.md) - Security practices and implementation
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design patterns
- [Observability Architecture](OBSERVABILITY_ARCHITECTURE.md) - Monitoring infrastructure
- [Compliance Guide](COMPLIANCE.md) - Regulatory compliance

---

## 1. Security Architecture Patterns

### 1.1 Defense in Depth Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Layer 7: Monitoring                     │
│  (SIEM, Security Analytics, Threat Intelligence)            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Layer 6: Response                         │
│  (Automated Response, Incident Management)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Layer 5: Application Security                │
│  (WAF, Input Validation, Output Sanitization)              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Layer 4: Access Control                      │
│  (Authentication, Authorization, RBAC)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Layer 3: Data Protection                     │
│  (Encryption, PII Redaction, DLP)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Layer 2: Network Security                    │
│  (TLS, Firewall, VPC, Private Subnets)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Layer 1: Infrastructure                      │
│  (Hardened OS, Security Patches, IDS/IPS)                  │
└─────────────────────────────────────────────────────────────┘
```

**Key Principles:**

1. **Multiple Layers** - No single point of failure
2. **Fail Secure** - Default to deny/block
3. **Least Privilege** - Minimum necessary access
4. **Separation of Duties** - No single person has complete control
5. **Audit Everything** - Comprehensive logging

### 1.2 Zero Trust Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User/Client                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Identity Provider                          │
│  (Authentication: MFA, SSO)                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Policy Engine                              │
│  (Risk Assessment, Context-based Authorization)             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Policy Enforcement                         │
│  (API Gateway with Security Policies)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│   Service   │  │   Service   │  │   Service    │
│  (Minimal   │  │  (Minimal   │  │  (Minimal    │
│   Trust)    │  │   Trust)    │  │   Trust)     │
└─────────────┘  └─────────────┘  └──────────────┘
```

**Core Tenets:**

1. **Verify explicitly** - Always authenticate and authorize
2. **Use least privilege** - Just-in-time and just-enough access
3. **Assume breach** - Minimize blast radius, segment access
4. **Inspect and log** - All traffic, internal and external

### 1.3 AI-Specific Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Request                            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Security Gateway                           │
│  - Rate limiting                                            │
│  - API key validation                                       │
│  - Request size limits                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Input Validation Layer                       │
│  - Length checks                                            │
│  - Format validation                                        │
│  - Prompt injection detection                               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  PII Detection Layer                        │
│  - Email, phone, SSN detection                             │
│  - Credit card detection                                    │
│  - Automatic redaction                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Content Filter                             │
│  - Toxic content detection                                  │
│  - Inappropriate requests                                   │
│  - Safety checks                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    LLM Processing                           │
│  - Isolated execution                                       │
│  - Resource limits                                          │
│  - Cost tracking                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Output Validation Layer                      │
│  - PII check in response                                    │
│  - Toxic content filter                                     │
│  - Sanitization                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Security Audit Log                         │
│  - All security events logged                               │
│  - SIEM integration                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Input Validation Architecture

### 2.1 Multi-Layer Validation Pipeline

```python
# src/security/validation_pipeline.py
"""
Multi-layer input validation pipeline
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    error_message: Optional[str] = None
    sanitized_input: Optional[str] = None
    security_flags: Dict[str, Any] = None

class ValidationLayer(ABC):
    """Base class for validation layers"""

    @abstractmethod
    def validate(self, input_text: str) -> ValidationResult:
        """Validate input"""
        pass

class LengthValidator(ValidationLayer):
    """Validates input length"""

    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

    def validate(self, input_text: str) -> ValidationResult:
        if len(input_text) > self.max_length:
            return ValidationResult(
                valid=False,
                error_message=f"Input exceeds maximum length of {self.max_length}"
            )
        return ValidationResult(valid=True, sanitized_input=input_text)

class FormatValidator(ValidationLayer):
    """Validates input format"""

    def validate(self, input_text: str) -> ValidationResult:
        # Check for null bytes
        if '\x00' in input_text:
            return ValidationResult(
                valid=False,
                error_message="Input contains null bytes"
            )

        # Check for control characters (except newline, tab, carriage return)
        invalid_chars = [c for c in input_text if ord(c) < 32 and c not in '\n\t\r']
        if invalid_chars:
            return ValidationResult(
                valid=False,
                error_message="Input contains invalid control characters"
            )

        return ValidationResult(valid=True, sanitized_input=input_text)

class PromptInjectionDetector(ValidationLayer):
    """Detects prompt injection attempts"""

    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+(instructions|rules|prompts)",
        r"system\s*:\s*you\s+are",
        r"(new|different)\s+(role|character|personality)",
        r"forget\s+(everything|all|previous)",
        r"(admin|root|sudo)\s+mode",
        r"<\|im_start\|>",
        r"###\s*instruction",
        r"act\s+as\s+(if|though)",
        r"(override|bypass)\s+(security|safety)",
    ]

    def __init__(self):
        import re
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def validate(self, input_text: str) -> ValidationResult:
        for pattern in self.patterns:
            if pattern.search(input_text):
                return ValidationResult(
                    valid=False,
                    error_message="Potential prompt injection detected",
                    security_flags={'prompt_injection': True, 'pattern': pattern.pattern}
                )

        return ValidationResult(valid=True, sanitized_input=input_text)

class PIIDetector(ValidationLayer):
    """Detects and redacts PII"""

    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    }

    def __init__(self, redact: bool = True):
        import re
        self.redact = redact
        self.patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PII_PATTERNS.items()
        }

    def validate(self, input_text: str) -> ValidationResult:
        pii_found = {}
        sanitized = input_text

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(input_text)
            if matches:
                pii_found[pii_type] = len(matches)
                if self.redact:
                    sanitized = pattern.sub(f'[{pii_type.upper()}_REDACTED]', sanitized)

        if pii_found:
            return ValidationResult(
                valid=True,  # Valid but flagged
                sanitized_input=sanitized,
                security_flags={'pii_detected': pii_found}
            )

        return ValidationResult(valid=True, sanitized_input=input_text)

class ContentSafetyValidator(ValidationLayer):
    """Validates content safety"""

    UNSAFE_PATTERNS = [
        r'\b(kill|murder|suicide|self[\s-]?harm)\b',
        r'\b(bomb|weapon|explosive)\b',
        r'\b(illegal|hack|exploit|crack)\b',
    ]

    def __init__(self):
        import re
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.UNSAFE_PATTERNS]

    def validate(self, input_text: str) -> ValidationResult:
        for pattern in self.patterns:
            if pattern.search(input_text):
                return ValidationResult(
                    valid=False,
                    error_message="Content safety violation detected",
                    security_flags={'unsafe_content': True}
                )

        return ValidationResult(valid=True, sanitized_input=input_text)

class ValidationPipeline:
    """Orchestrates validation layers"""

    def __init__(self, layers: list[ValidationLayer] = None):
        self.layers = layers or [
            LengthValidator(),
            FormatValidator(),
            PromptInjectionDetector(),
            PIIDetector(redact=True),
            ContentSafetyValidator(),
        ]

    def validate(self, input_text: str) -> ValidationResult:
        """Run input through all validation layers"""
        current_input = input_text
        all_flags = {}

        for layer in self.layers:
            result = layer.validate(current_input)

            if not result.valid:
                # Validation failed, return immediately
                return result

            # Update input with sanitized version
            if result.sanitized_input:
                current_input = result.sanitized_input

            # Collect security flags
            if result.security_flags:
                all_flags.update(result.security_flags)

        return ValidationResult(
            valid=True,
            sanitized_input=current_input,
            security_flags=all_flags if all_flags else None
        )

# Global pipeline instance
validation_pipeline = ValidationPipeline()
```

**Usage:**
```python
from src.security.validation_pipeline import validation_pipeline

def process_user_input(user_input: str):
    """Process and validate user input"""
    result = validation_pipeline.validate(user_input)

    if not result.valid:
        # Log security event
        security_logger.warning(
            f"Input validation failed: {result.error_message}",
            extra={'flags': result.security_flags}
        )
        raise ValidationError(result.error_message)

    # Use sanitized input
    if result.security_flags:
        # Log but allow (e.g., PII was redacted)
        security_logger.info(
            "Security flags raised during validation",
            extra={'flags': result.security_flags}
        )

    return result.sanitized_input
```

### 2.2 Rate Limiting Architecture

```python
# src/security/rate_limiter.py
"""
Multi-tier rate limiting architecture
"""

from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
from enum import Enum

class RateLimitTier(Enum):
    """Rate limit tiers"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    cost_limit_usd: float

class RateLimiter:
    """Multi-tier rate limiter with Redis backend"""

    TIER_CONFIGS = {
        RateLimitTier.FREE: RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            cost_limit_usd=1.0
        ),
        RateLimitTier.PRO: RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            cost_limit_usd=50.0
        ),
        RateLimitTier.ENTERPRISE: RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
            requests_per_day=500000,
            cost_limit_usd=1000.0
        ),
    }

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def check_rate_limit(
        self,
        user_id: str,
        tier: RateLimitTier = RateLimitTier.FREE
    ) -> Dict[str, any]:
        """Check if user is within rate limits"""
        config = self.TIER_CONFIGS[tier]
        now = datetime.utcnow()

        # Check each time window
        checks = {
            'minute': (60, config.requests_per_minute),
            'hour': (3600, config.requests_per_hour),
            'day': (86400, config.requests_per_day),
        }

        for window, (seconds, limit) in checks.items():
            key = f"rate_limit:{user_id}:{window}:{now.strftime('%Y%m%d%H%M' if window == 'minute' else '%Y%m%d%H' if window == 'hour' else '%Y%m%d')}"

            current = self.redis.get(key)
            current = int(current) if current else 0

            if current >= limit:
                return {
                    'allowed': False,
                    'limit': limit,
                    'current': current,
                    'window': window,
                    'retry_after': seconds
                }

        # Check cost limit
        cost_key = f"cost:daily:{now.strftime('%Y%m%d')}:user:{user_id}"
        daily_cost = float(self.redis.get(cost_key) or 0)

        if daily_cost >= config.cost_limit_usd:
            return {
                'allowed': False,
                'limit': config.cost_limit_usd,
                'current': daily_cost,
                'window': 'cost',
                'retry_after': 86400  # Retry tomorrow
            }

        # All checks passed, increment counters
        for window, (seconds, limit) in checks.items():
            key = f"rate_limit:{user_id}:{window}:{now.strftime('%Y%m%d%H%M' if window == 'minute' else '%Y%m%d%H' if window == 'hour' else '%Y%m%d')}"

            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, seconds * 2)  # TTL = 2x window
            pipe.execute()

        return {
            'allowed': True,
            'remaining': {
                'minute': config.requests_per_minute - current,
                'hour': config.requests_per_hour - current,
                'day': config.requests_per_day - current,
                'cost_usd': config.cost_limit_usd - daily_cost
            }
        }

    def get_usage(self, user_id: str) -> Dict[str, int]:
        """Get current usage for user"""
        now = datetime.utcnow()

        usage = {}
        for window in ['minute', 'hour', 'day']:
            key = f"rate_limit:{user_id}:{window}:{now.strftime('%Y%m%d%H%M' if window == 'minute' else '%Y%m%d%H' if window == 'hour' else '%Y%m%d')}"
            usage[window] = int(self.redis.get(key) or 0)

        cost_key = f"cost:daily:{now.strftime('%Y%m%d')}:user:{user_id}"
        usage['cost_usd'] = float(self.redis.get(cost_key) or 0)

        return usage
```

**Integration with API:**
```python
from flask import Flask, request, jsonify
from src.security.rate_limiter import RateLimiter, RateLimitTier

app = Flask(__name__)
rate_limiter = RateLimiter(redis_client)

@app.before_request
def check_rate_limit():
    """Check rate limit before processing request"""
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({'error': 'Missing user ID'}), 401

    # Get user tier from database/cache
    user_tier = get_user_tier(user_id)

    # Check rate limit
    result = rate_limiter.check_rate_limit(user_id, user_tier)

    if not result['allowed']:
        return jsonify({
            'error': 'Rate limit exceeded',
            'limit': result['limit'],
            'current': result['current'],
            'window': result['window'],
            'retry_after': result['retry_after']
        }), 429

    # Store remaining limits in response headers
    g.rate_limit_remaining = result['remaining']
```

---

## 3. Authentication and Authorization Architecture

### 3.1 JWT-Based Authentication

```python
# src/security/auth.py
"""
JWT-based authentication system
"""

import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import secrets

@dataclass
class TokenPair:
    """Access and refresh tokens"""
    access_token: str
    refresh_token: str
    expires_in: int  # seconds

class JWTAuthManager:
    """Manages JWT authentication"""

    def __init__(
        self,
        secret_key: str,
        access_token_expiry: int = 3600,  # 1 hour
        refresh_token_expiry: int = 604800  # 7 days
    ):
        self.secret_key = secret_key
        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        self.algorithm = 'HS256'

    def generate_tokens(
        self,
        user_id: str,
        user_tier: str,
        permissions: list[str]
    ) -> TokenPair:
        """Generate access and refresh tokens"""
        now = datetime.utcnow()

        # Access token (short-lived)
        access_payload = {
            'user_id': user_id,
            'user_tier': user_tier,
            'permissions': permissions,
            'type': 'access',
            'iat': now,
            'exp': now + timedelta(seconds=self.access_token_expiry),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }

        access_token = jwt.encode(
            access_payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        # Refresh token (long-lived)
        refresh_payload = {
            'user_id': user_id,
            'type': 'refresh',
            'iat': now,
            'exp': now + timedelta(seconds=self.refresh_token_expiry),
            'jti': secrets.token_urlsafe(16)
        }

        refresh_token = jwt.encode(
            refresh_payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expiry
        )

    def verify_token(self, token: str, token_type: str = 'access') -> Dict:
        """Verify and decode token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            if payload.get('type') != token_type:
                raise ValueError(f"Invalid token type: expected {token_type}")

            return payload

        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    def refresh_access_token(self, refresh_token: str) -> TokenPair:
        """Generate new access token from refresh token"""
        # Verify refresh token
        payload = self.verify_token(refresh_token, token_type='refresh')

        user_id = payload['user_id']

        # Get current user data (tier, permissions)
        user_data = get_user_data(user_id)

        # Generate new token pair
        return self.generate_tokens(
            user_id=user_id,
            user_tier=user_data['tier'],
            permissions=user_data['permissions']
        )

    def revoke_token(self, token: str):
        """Revoke token (add to blacklist)"""
        payload = self.verify_token(token)
        jti = payload['jti']
        exp = payload['exp']

        # Store token ID in Redis blacklist until expiry
        ttl = exp - datetime.utcnow().timestamp()
        if ttl > 0:
            redis_client.setex(
                f"blacklist:{jti}",
                int(ttl),
                "1"
            )

    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Don't verify expiry for blacklist check
            )
            jti = payload['jti']
            return redis_client.exists(f"blacklist:{jti}")
        except:
            return True  # Invalid tokens are considered blacklisted
```

### 3.2 API Key Management

```python
# src/security/api_keys.py
"""
API key management system
"""

import secrets
import hashlib
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class APIKey:
    """API key data"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: list[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]

class APIKeyManager:
    """Manages API keys"""

    KEY_PREFIX = "sk_"

    def __init__(self, db_connection):
        self.db = db_connection

    def generate_key(
        self,
        user_id: str,
        name: str,
        permissions: list[str],
        expires_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """Generate new API key"""
        # Generate random key
        random_bytes = secrets.token_bytes(32)
        key = f"{self.KEY_PREFIX}{secrets.token_urlsafe(32)}"

        # Hash key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Generate key ID
        key_id = f"key_{secrets.token_urlsafe(16)}"

        # Calculate expiry
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        # Store in database
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_used=None
        )

        self._store_key(api_key)

        # Return plaintext key (only time it's visible)
        return key, api_key

    def verify_key(self, key: str) -> Optional[APIKey]:
        """Verify and retrieve API key"""
        if not key.startswith(self.KEY_PREFIX):
            return None

        # Hash provided key
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Lookup in database
        api_key = self._get_key_by_hash(key_hash)

        if not api_key:
            return None

        # Check if expired
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update last used timestamp
        self._update_last_used(api_key.key_id)

        return api_key

    def revoke_key(self, key_id: str):
        """Revoke API key"""
        cursor = self.db.cursor()
        cursor.execute(
            "DELETE FROM api_keys WHERE key_id = %s",
            (key_id,)
        )
        self.db.commit()

    def list_keys(self, user_id: str) -> list[APIKey]:
        """List all keys for user"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT key_id, key_hash, user_id, name, permissions,
                   created_at, expires_at, last_used
            FROM api_keys
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))

        return [self._row_to_api_key(row) for row in cursor.fetchall()]

    def rotate_key(self, old_key_id: str) -> tuple[str, APIKey]:
        """Rotate API key (generate new, revoke old)"""
        # Get old key details
        old_key = self._get_key_by_id(old_key_id)

        # Generate new key with same permissions
        new_key, api_key = self.generate_key(
            user_id=old_key.user_id,
            name=f"{old_key.name} (rotated)",
            permissions=old_key.permissions,
            expires_days=(old_key.expires_at - datetime.utcnow()).days if old_key.expires_at else None
        )

        # Revoke old key
        self.revoke_key(old_key_id)

        return new_key, api_key

    def _store_key(self, api_key: APIKey):
        """Store key in database"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO api_keys (
                key_id, key_hash, user_id, name, permissions,
                created_at, expires_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            api_key.key_id,
            api_key.key_hash,
            api_key.user_id,
            api_key.name,
            api_key.permissions,
            api_key.created_at,
            api_key.expires_at
        ))
        self.db.commit()

    def _get_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get key by hash"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT key_id, key_hash, user_id, name, permissions,
                   created_at, expires_at, last_used
            FROM api_keys
            WHERE key_hash = %s
        """, (key_hash,))

        row = cursor.fetchone()
        return self._row_to_api_key(row) if row else None

    def _update_last_used(self, key_id: str):
        """Update last used timestamp"""
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE api_keys
            SET last_used = %s
            WHERE key_id = %s
        """, (datetime.utcnow(), key_id))
        self.db.commit()
```

### 3.3 Role-Based Access Control (RBAC)

```python
# src/security/rbac.py
"""
Role-Based Access Control system
"""

from typing import Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class Permission(Enum):
    """System permissions"""
    # LLM operations
    LLM_GENERATE = "llm:generate"
    LLM_STREAM = "llm:stream"
    LLM_HAIKU = "llm:model:haiku"
    LLM_SONNET = "llm:model:sonnet"
    LLM_OPUS = "llm:model:opus"

    # Data operations
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"

    # Admin operations
    ADMIN_USERS = "admin:users"
    ADMIN_KEYS = "admin:keys"
    ADMIN_SETTINGS = "admin:settings"

    # Cost operations
    COST_VIEW = "cost:view"
    COST_UNLIMITED = "cost:unlimited"

@dataclass
class Role:
    """User role with permissions"""
    name: str
    permissions: Set[Permission]
    description: str

class RBACManager:
    """Manages roles and permissions"""

    ROLES = {
        'free': Role(
            name='free',
            permissions={
                Permission.LLM_GENERATE,
                Permission.LLM_HAIKU,
                Permission.DATA_READ,
                Permission.COST_VIEW,
            },
            description='Free tier user'
        ),
        'pro': Role(
            name='pro',
            permissions={
                Permission.LLM_GENERATE,
                Permission.LLM_STREAM,
                Permission.LLM_HAIKU,
                Permission.LLM_SONNET,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.COST_VIEW,
            },
            description='Pro tier user'
        ),
        'enterprise': Role(
            name='enterprise',
            permissions={
                Permission.LLM_GENERATE,
                Permission.LLM_STREAM,
                Permission.LLM_HAIKU,
                Permission.LLM_SONNET,
                Permission.LLM_OPUS,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_DELETE,
                Permission.COST_VIEW,
                Permission.COST_UNLIMITED,
            },
            description='Enterprise user'
        ),
        'admin': Role(
            name='admin',
            permissions=set(Permission),  # All permissions
            description='System administrator'
        ),
    }

    def __init__(self):
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> roles

    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        if role_name not in self.ROLES:
            raise ValueError(f"Unknown role: {role_name}")

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role_name)

    def revoke_role(self, user_id: str, role_name: str):
        """Revoke role from user"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user"""
        if user_id not in self.user_roles:
            return set()

        permissions = set()
        for role_name in self.user_roles[user_id]:
            role = self.ROLES.get(role_name)
            if role:
                permissions.update(role.permissions)

        return permissions

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission"""
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions

    def require_permission(self, permission: Permission):
        """Decorator to require permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                user_id = get_current_user_id()  # From request context

                if not self.has_permission(user_id, permission):
                    raise PermissionError(
                        f"User {user_id} does not have permission: {permission.value}"
                    )

                return func(*args, **kwargs)
            return wrapper
        return decorator

# Global RBAC manager
rbac = RBACManager()

# Usage example
@rbac.require_permission(Permission.LLM_OPUS)
def generate_with_opus(prompt: str):
    """Generate response with Opus model (enterprise only)"""
    return llm_client.generate(prompt, model='opus')
```

---

## 4. Data Protection Architecture

### 4.1 Encryption at Rest

```python
# src/security/encryption.py
"""
Data encryption at rest
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import os

class DataEncryption:
    """Handles data encryption/decryption"""

    def __init__(self, master_key: str = None):
        """Initialize with master key"""
        if not master_key:
            master_key = os.getenv('MASTER_KEY')

        if not master_key:
            raise ValueError("Master key not provided")

        # Derive encryption key from master key
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ai_app_salt',  # Should be random and stored securely
            iterations=100000,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt data"""
        if not plaintext:
            return plaintext

        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt data"""
        if not ciphertext:
            return ciphertext

        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypt file"""
        with open(input_path, 'rb') as f:
            plaintext = f.read()

        encrypted = self.cipher.encrypt(plaintext)

        with open(output_path, 'wb') as f:
            f.write(encrypted)

    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypt file"""
        with open(input_path, 'rb') as f:
            ciphertext = f.read()

        decrypted = self.cipher.decrypt(ciphertext)

        with open(output_path, 'wb') as f:
            f.write(decrypted)

# Global encryption instance
encryption = DataEncryption()

# Usage for sensitive fields in database
class User:
    def __init__(self, email: str, phone: str):
        self.email = email
        self.phone_encrypted = encryption.encrypt(phone)

    @property
    def phone(self):
        return encryption.decrypt(self.phone_encrypted)
```

### 4.2 Secrets Management

```python
# src/security/secrets_manager.py
"""
Secrets management with external secret stores
"""

from typing import Optional, Dict
import boto3
from abc import ABC, abstractmethod

class SecretsBackend(ABC):
    """Abstract secrets backend"""

    @abstractmethod
    def get_secret(self, secret_name: str) -> str:
        pass

    @abstractmethod
    def set_secret(self, secret_name: str, secret_value: str):
        pass

class AWSSecretsManager(SecretsBackend):
    """AWS Secrets Manager backend"""

    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client('secretsmanager', region_name=region)

    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from AWS"""
        response = self.client.get_secret_value(SecretId=secret_name)
        return response['SecretString']

    def set_secret(self, secret_name: str, secret_value: str):
        """Store secret in AWS"""
        try:
            self.client.create_secret(
                Name=secret_name,
                SecretString=secret_value
            )
        except self.client.exceptions.ResourceExistsException:
            self.client.update_secret(
                SecretId=secret_name,
                SecretString=secret_value
            )

class HashiCorpVault(SecretsBackend):
    """HashiCorp Vault backend"""

    def __init__(self, url: str, token: str):
        import hvac
        self.client = hvac.Client(url=url, token=token)

    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Vault"""
        secret = self.client.secrets.kv.v2.read_secret_version(path=secret_name)
        return secret['data']['data']['value']

    def set_secret(self, secret_name: str, secret_value: str):
        """Store secret in Vault"""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=secret_name,
            secret={'value': secret_value}
        )

class SecretsManager:
    """Manages application secrets"""

    def __init__(self, backend: SecretsBackend):
        self.backend = backend
        self._cache: Dict[str, str] = {}

    def get(self, secret_name: str, use_cache: bool = True) -> str:
        """Get secret value"""
        if use_cache and secret_name in self._cache:
            return self._cache[secret_name]

        value = self.backend.get_secret(secret_name)

        if use_cache:
            self._cache[secret_name] = value

        return value

    def set(self, secret_name: str, secret_value: str):
        """Set secret value"""
        self.backend.set_secret(secret_name, secret_value)

        # Invalidate cache
        if secret_name in self._cache:
            del self._cache[secret_name]

    def get_api_key(self, provider: str) -> str:
        """Get API key for LLM provider"""
        return self.get(f"api_keys/{provider}")

    def get_database_url(self) -> str:
        """Get database connection string"""
        return self.get("database/url")

    def rotate_secret(self, secret_name: str, new_value: str):
        """Rotate secret"""
        # Set new value
        self.set(secret_name, new_value)

        # Clear cache to force reload
        if secret_name in self._cache:
            del self._cache[secret_name]

# Initialize secrets manager
secrets_backend = AWSSecretsManager()  # or HashiCorpVault()
secrets_manager = SecretsManager(secrets_backend)
```

---

## 5. Threat Detection Architecture

### 5.1 Anomaly Detection

```python
# src/security/anomaly_detection.py
"""
Anomaly detection for security threats
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

@dataclass
class AnomalyScore:
    """Anomaly detection result"""
    score: float  # 0-1, higher = more anomalous
    is_anomaly: bool
    reasons: List[str]
    timestamp: datetime

class AnomalyDetector:
    """Detects anomalous behavior"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.threshold = 0.7  # Anomaly threshold

    def detect_user_anomaly(self, user_id: str, current_request: Dict) -> AnomalyScore:
        """Detect anomalies in user behavior"""
        scores = []
        reasons = []

        # 1. Request rate anomaly
        rate_score = self._check_request_rate(user_id)
        if rate_score > 0.5:
            scores.append(rate_score)
            reasons.append(f"Unusual request rate: {rate_score:.2f}")

        # 2. Request size anomaly
        size_score = self._check_request_size(user_id, current_request)
        if size_score > 0.5:
            scores.append(size_score)
            reasons.append(f"Unusual request size: {size_score:.2f}")

        # 3. Time of day anomaly
        time_score = self._check_time_of_day(user_id)
        if time_score > 0.5:
            scores.append(time_score)
            reasons.append(f"Unusual time of access: {time_score:.2f}")

        # 4. Geographic anomaly
        geo_score = self._check_geographic(user_id, current_request)
        if geo_score > 0.5:
            scores.append(geo_score)
            reasons.append(f"Unusual geographic location: {geo_score:.2f}")

        # 5. Model usage anomaly
        model_score = self._check_model_usage(user_id, current_request)
        if model_score > 0.5:
            scores.append(model_score)
            reasons.append(f"Unusual model usage: {model_score:.2f}")

        # Calculate overall anomaly score
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0.0

        return AnomalyScore(
            score=overall_score,
            is_anomaly=overall_score > self.threshold,
            reasons=reasons,
            timestamp=datetime.utcnow()
        )

    def _check_request_rate(self, user_id: str) -> float:
        """Check if request rate is unusual"""
        # Get request counts for last hour
        now = datetime.utcnow()
        key = f"requests:hourly:{user_id}"

        current_hour_count = int(self.redis.get(f"{key}:{now.strftime('%Y%m%d%H')}") or 0)

        # Get historical average
        historical_counts = []
        for i in range(1, 25):  # Last 24 hours
            past_hour = now - timedelta(hours=i)
            count = int(self.redis.get(f"{key}:{past_hour.strftime('%Y%m%d%H')}") or 0)
            historical_counts.append(count)

        if not historical_counts:
            return 0.0

        avg = np.mean(historical_counts)
        std = np.std(historical_counts)

        if std == 0:
            return 0.0

        # Z-score
        z_score = (current_hour_count - avg) / std

        # Convert to 0-1 score (3 std devs = 1.0)
        score = min(abs(z_score) / 3.0, 1.0)

        return score if z_score > 0 else 0.0  # Only flag increases

    def _check_request_size(self, user_id: str, current_request: Dict) -> float:
        """Check if request size is unusual"""
        current_size = len(current_request.get('prompt', ''))

        # Get historical average request size
        key = f"request_sizes:{user_id}"
        sizes_str = self.redis.lrange(key, 0, 99)  # Last 100 requests

        if not sizes_str:
            return 0.0

        sizes = [int(s) for s in sizes_str]
        avg = np.mean(sizes)
        std = np.std(sizes)

        if std == 0:
            return 0.0

        z_score = (current_size - avg) / std
        score = min(abs(z_score) / 3.0, 1.0)

        return score if z_score > 0 else 0.0

    def _check_time_of_day(self, user_id: str) -> float:
        """Check if time of day is unusual"""
        now = datetime.utcnow()
        current_hour = now.hour

        # Get historical hour distribution
        key = f"access_hours:{user_id}"
        hour_counts = {}

        for hour in range(24):
            count = int(self.redis.get(f"{key}:{hour}") or 0)
            hour_counts[hour] = count

        total = sum(hour_counts.values())
        if total == 0:
            return 0.0

        # Calculate expected frequency for current hour
        expected_freq = hour_counts[current_hour] / total

        # Score based on rarity (invert frequency)
        score = 1.0 - expected_freq

        return score if score > 0.5 else 0.0  # Only flag rare times

    def _check_geographic(self, user_id: str, current_request: Dict) -> float:
        """Check if geographic location is unusual"""
        current_ip = current_request.get('ip_address')
        if not current_ip:
            return 0.0

        # Get country from IP (using GeoIP)
        current_country = get_country_from_ip(current_ip)

        # Get historical countries
        key = f"access_countries:{user_id}"
        countries = self.redis.smembers(key)

        if not countries:
            # First access, store and allow
            self.redis.sadd(key, current_country)
            return 0.0

        if current_country not in countries:
            # New country
            return 1.0

        return 0.0

    def _check_model_usage(self, user_id: str, current_request: Dict) -> float:
        """Check if model usage is unusual"""
        current_model = current_request.get('model')

        # Get historical model distribution
        key = f"model_usage:{user_id}"
        model_counts = {}

        for model in ['haiku', 'sonnet', 'opus']:
            count = int(self.redis.get(f"{key}:{model}") or 0)
            model_counts[model] = count

        total = sum(model_counts.values())
        if total == 0:
            return 0.0

        # Check if using expensive model unusually
        if current_model == 'opus':
            opus_freq = model_counts['opus'] / total
            if opus_freq < 0.1:  # Less than 10% historical usage
                return 0.8

        return 0.0

# Global anomaly detector
anomaly_detector = AnomalyDetector(redis_client)
```

### 5.2 Intrusion Detection System (IDS)

```python
# src/security/ids.py
"""
Intrusion Detection System
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event detected by IDS"""
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    evidence: Dict
    timestamp: datetime
    blocked: bool

class IntrusionDetectionSystem:
    """Detects security intrusions"""

    ATTACK_PATTERNS = {
        'sql_injection': [
            r"'\s*OR\s+'1'\s*=\s*'1",
            r";\s*DROP\s+TABLE",
            r"UNION\s+SELECT",
        ],
        'xss': [
            r"<script",
            r"javascript:",
            r"onerror\s*=",
        ],
        'command_injection': [
            r";\s*rm\s+-rf",
            r"\|\s*sh",
            r"&&\s*cat\s+/etc/passwd",
        ],
        'path_traversal': [
            r"\.\./\.\./",
            r"\.\.\\\.\.\\",
        ],
    }

    def __init__(self):
        import re
        self.patterns = {}
        for attack_type, patterns in self.ATTACK_PATTERNS.items():
            self.patterns[attack_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def scan_request(
        self,
        request_data: Dict,
        source_ip: str,
        user_id: Optional[str] = None
    ) -> List[SecurityEvent]:
        """Scan request for security threats"""
        events = []

        # 1. Check for known attack patterns
        for field, value in request_data.items():
            if isinstance(value, str):
                attack_events = self._check_attack_patterns(
                    value, field, source_ip, user_id
                )
                events.extend(attack_events)

        # 2. Check for credential stuffing
        if self._detect_credential_stuffing(source_ip):
            events.append(SecurityEvent(
                event_type='credential_stuffing',
                threat_level=ThreatLevel.HIGH,
                source_ip=source_ip,
                user_id=user_id,
                description='Multiple authentication failures detected',
                evidence={'ip': source_ip},
                timestamp=datetime.utcnow(),
                blocked=False
            ))

        # 3. Check for DDoS
        if self._detect_ddos(source_ip):
            events.append(SecurityEvent(
                event_type='ddos',
                threat_level=ThreatLevel.CRITICAL,
                source_ip=source_ip,
                user_id=user_id,
                description='Potential DDoS attack detected',
                evidence={'ip': source_ip},
                timestamp=datetime.utcnow(),
                blocked=True
            ))

        # 4. Check for data exfiltration
        if self._detect_data_exfiltration(request_data, user_id):
            events.append(SecurityEvent(
                event_type='data_exfiltration',
                threat_level=ThreatLevel.CRITICAL,
                source_ip=source_ip,
                user_id=user_id,
                description='Potential data exfiltration detected',
                evidence={'user_id': user_id},
                timestamp=datetime.utcnow(),
                blocked=True
            ))

        return events

    def _check_attack_patterns(
        self,
        value: str,
        field: str,
        source_ip: str,
        user_id: Optional[str]
    ) -> List[SecurityEvent]:
        """Check for known attack patterns"""
        events = []

        for attack_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(value):
                    events.append(SecurityEvent(
                        event_type=attack_type,
                        threat_level=ThreatLevel.HIGH,
                        source_ip=source_ip,
                        user_id=user_id,
                        description=f'{attack_type} detected in {field}',
                        evidence={
                            'field': field,
                            'pattern': pattern.pattern,
                            'value_snippet': value[:100]
                        },
                        timestamp=datetime.utcnow(),
                        blocked=True
                    ))

        return events

    def _detect_credential_stuffing(self, source_ip: str) -> bool:
        """Detect credential stuffing attempts"""
        # Check auth failures from IP in last 5 minutes
        key = f"auth_failures:{source_ip}"
        failures = int(redis_client.get(key) or 0)

        return failures > 10

    def _detect_ddos(self, source_ip: str) -> bool:
        """Detect DDoS attacks"""
        # Check request rate from IP
        key = f"requests_per_min:{source_ip}"
        rate = int(redis_client.get(key) or 0)

        return rate > 100  # More than 100 req/min from single IP

    def _detect_data_exfiltration(
        self,
        request_data: Dict,
        user_id: Optional[str]
    ) -> bool:
        """Detect potential data exfiltration"""
        if not user_id:
            return False

        # Check if user is requesting large amounts of data
        key = f"data_requests:{user_id}"
        recent_requests = int(redis_client.get(key) or 0)

        # Check response size
        response_size = request_data.get('max_tokens', 0)

        return recent_requests > 1000 and response_size > 4000

# Global IDS
ids = IntrusionDetectionSystem()
```

---

## 6. Incident Response Architecture

### 6.1 Automated Incident Response

```python
# src/security/incident_response.py
"""
Automated incident response system
"""

from typing import List, Dict, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class IncidentStatus(Enum):
    """Incident status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class Incident:
    """Security incident"""
    incident_id: str
    event_type: str
    threat_level: str
    source_ip: str
    user_id: str
    description: str
    detected_at: datetime
    status: IncidentStatus
    actions_taken: List[str]

class IncidentResponder:
    """Automated incident response"""

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default incident handlers"""
        self.register_handler('prompt_injection', self._handle_prompt_injection)
        self.register_handler('credential_stuffing', self._handle_credential_stuffing)
        self.register_handler('ddos', self._handle_ddos)
        self.register_handler('data_exfiltration', self._handle_data_exfiltration)
        self.register_handler('sql_injection', self._handle_sql_injection)

    def register_handler(self, event_type: str, handler: Callable):
        """Register custom handler"""
        self.handlers[event_type] = handler

    def respond(self, event: SecurityEvent) -> Incident:
        """Respond to security event"""
        # Create incident
        incident = Incident(
            incident_id=generate_incident_id(),
            event_type=event.event_type,
            threat_level=event.threat_level.value,
            source_ip=event.source_ip,
            user_id=event.user_id or 'unknown',
            description=event.description,
            detected_at=event.timestamp,
            status=IncidentStatus.DETECTED,
            actions_taken=[]
        )

        # Get handler for event type
        handler = self.handlers.get(event.event_type)

        if handler:
            try:
                # Execute automated response
                handler(incident, event)
                incident.status = IncidentStatus.CONTAINED
            except Exception as e:
                print(f"Error handling incident: {e}")
                # Escalate to manual review
                self._escalate(incident)
        else:
            # No handler, escalate
            self._escalate(incident)

        # Log incident
        self._log_incident(incident)

        # Alert if critical
        if event.threat_level == ThreatLevel.CRITICAL:
            self._alert_security_team(incident)

        return incident

    def _handle_prompt_injection(self, incident: Incident, event: SecurityEvent):
        """Handle prompt injection attempt"""
        # 1. Block request
        incident.actions_taken.append('Blocked request')

        # 2. Increment violation counter for user
        if event.user_id:
            key = f"violations:{event.user_id}"
            violations = redis_client.incr(key)
            redis_client.expire(key, 86400)  # 24 hour window

            incident.actions_taken.append(f'Recorded violation (total: {violations})')

            # 3. Temporary ban if multiple violations
            if violations >= 3:
                self._ban_user(event.user_id, duration_hours=24)
                incident.actions_taken.append('Temporary ban applied (24 hours)')

        # 4. Block IP temporarily
        self._block_ip(event.source_ip, duration_minutes=15)
        incident.actions_taken.append('IP blocked temporarily (15 minutes)')

    def _handle_credential_stuffing(self, incident: Incident, event: SecurityEvent):
        """Handle credential stuffing attack"""
        # 1. Block IP
        self._block_ip(event.source_ip, duration_minutes=60)
        incident.actions_taken.append('IP blocked (60 minutes)')

        # 2. Add to threat intelligence
        self._add_to_threat_intel(event.source_ip, 'credential_stuffing')
        incident.actions_taken.append('Added to threat intelligence')

        # 3. Notify affected users if successful login
        if event.user_id:
            self._notify_user_suspicious_activity(event.user_id)
            incident.actions_taken.append('User notified of suspicious activity')

    def _handle_ddos(self, incident: Incident, event: SecurityEvent):
        """Handle DDoS attack"""
        # 1. Block IP
        self._block_ip(event.source_ip, duration_minutes=120)
        incident.actions_taken.append('IP blocked (120 minutes)')

        # 2. Enable aggressive rate limiting
        self._enable_aggressive_rate_limiting()
        incident.actions_taken.append('Aggressive rate limiting enabled')

        # 3. Scale up infrastructure if needed
        if self._check_if_scaling_needed():
            self._trigger_auto_scaling()
            incident.actions_taken.append('Auto-scaling triggered')

        # 4. Alert infrastructure team
        self._alert_team('infrastructure', incident)
        incident.actions_taken.append('Infrastructure team alerted')

    def _handle_data_exfiltration(self, incident: Incident, event: SecurityEvent):
        """Handle data exfiltration attempt"""
        # 1. Immediately suspend user account
        if event.user_id:
            self._suspend_user(event.user_id)
            incident.actions_taken.append('User account suspended')

        # 2. Block IP
        self._block_ip(event.source_ip, duration_minutes=1440)  # 24 hours
        incident.actions_taken.append('IP blocked (24 hours)')

        # 3. Audit user's recent activity
        self._audit_user_activity(event.user_id)
        incident.actions_taken.append('User activity audit initiated')

        # 4. Alert security team immediately
        self._alert_team('security', incident, urgent=True)
        incident.actions_taken.append('Security team alerted (urgent)')

    def _handle_sql_injection(self, incident: Incident, event: SecurityEvent):
        """Handle SQL injection attempt"""
        # 1. Block request
        incident.actions_taken.append('Request blocked')

        # 2. Block IP
        self._block_ip(event.source_ip, duration_minutes=1440)  # 24 hours
        incident.actions_taken.append('IP blocked (24 hours)')

        # 3. Alert security team
        self._alert_team('security', incident)
        incident.actions_taken.append('Security team alerted')

    # Helper methods
    def _block_ip(self, ip_address: str, duration_minutes: int):
        """Block IP address"""
        redis_client.setex(
            f"blocked_ip:{ip_address}",
            duration_minutes * 60,
            "1"
        )

    def _ban_user(self, user_id: str, duration_hours: int):
        """Temporarily ban user"""
        redis_client.setex(
            f"banned_user:{user_id}",
            duration_hours * 3600,
            "1"
        )

    def _suspend_user(self, user_id: str):
        """Permanently suspend user account"""
        # Update database
        db.execute(
            "UPDATE users SET status = 'suspended', suspended_at = %s WHERE user_id = %s",
            (datetime.utcnow(), user_id)
        )

    def _escalate(self, incident: Incident):
        """Escalate incident to manual review"""
        # Add to review queue
        redis_client.lpush('incident_review_queue', incident.incident_id)

        # Alert security team
        self._alert_team('security', incident)

    def _alert_security_team(self, incident: Incident):
        """Alert security team"""
        # Send to PagerDuty/Slack/Email
        pass

    def _log_incident(self, incident: Incident):
        """Log incident to SIEM"""
        # Send to security information and event management system
        pass

# Global incident responder
incident_responder = IncidentResponder()
```

---

## 7. Security Monitoring Dashboard

### 7.1 Real-time Security Dashboard

**Grafana Dashboard JSON (Security Overview):**
```json
{
  "dashboard": {
    "title": "Security Monitoring Dashboard",
    "panels": [
      {
        "id": 1,
        "title": "Security Events (Last Hour)",
        "type": "stat",
        "targets": [{
          "expr": "sum(rate(security_events_total[1h])) * 3600"
        }],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 10, "color": "yellow"},
                {"value": 50, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Security Events by Type",
        "type": "piechart",
        "targets": [{
          "expr": "sum by (event_type) (rate(security_events_total[1h]))"
        }]
      },
      {
        "id": 3,
        "title": "Blocked IPs",
        "type": "stat",
        "targets": [{
          "expr": "count(blocked_ips)"
        }]
      },
      {
        "id": 4,
        "title": "Failed Authentication Attempts",
        "type": "graph",
        "targets": [{
          "expr": "rate(auth_failures_total[5m])",
          "legendFormat": "Auth Failures/sec"
        }]
      },
      {
        "id": 5,
        "title": "Top Attack Sources",
        "type": "table",
        "targets": [{
          "expr": "topk(10, sum by (source_ip) (security_events_total))"
        }]
      },
      {
        "id": 6,
        "title": "PII Detections",
        "type": "graph",
        "targets": [{
          "expr": "rate(pii_detections_total[5m])",
          "legendFormat": "PII Detections/sec"
        }]
      },
      {
        "id": 7,
        "title": "Prompt Injection Attempts",
        "type": "stat",
        "targets": [{
          "expr": "sum(increase(security_events_total{event_type=\"prompt_injection\"}[1h]))"
        }],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 1, "color": "yellow"},
                {"value": 5, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 8,
        "title": "Active Incidents",
        "type": "stat",
        "targets": [{
          "expr": "count(incidents{status!=\"closed\"})"
        }]
      }
    ]
  }
}
```

---

## 8. Security Architecture Checklist

**Infrastructure Security:**
- [ ] TLS 1.3 for all connections
- [ ] Network segmentation (VPC, private subnets)
- [ ] Firewall rules (allowlist only necessary ports)
- [ ] DDoS protection (CloudFlare, AWS Shield)
- [ ] IDS/IPS deployed

**Application Security:**
- [ ] Input validation on all endpoints
- [ ] Prompt injection detection
- [ ] PII detection and redaction
- [ ] Output sanitization
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (CSP headers)
- [ ] CSRF protection

**Authentication & Authorization:**
- [ ] JWT-based authentication
- [ ] API key management system
- [ ] RBAC implemented
- [ ] MFA for admin accounts
- [ ] Password hashing (bcrypt/argon2)
- [ ] Session management (expiry, invalidation)
- [ ] OAuth 2.0 for third-party integrations

**Data Protection:**
- [ ] Encryption at rest (AES-256)
- [ ] Encryption in transit (TLS 1.3)
- [ ] Secrets management (AWS Secrets Manager/Vault)
- [ ] Database encryption
- [ ] Backup encryption
- [ ] Data retention policies

**Monitoring & Response:**
- [ ] Security event logging (SIEM)
- [ ] Anomaly detection
- [ ] Intrusion detection system (IDS)
- [ ] Automated incident response
- [ ] Security dashboards
- [ ] Alert configuration
- [ ] Incident response playbooks

**Compliance:**
- [ ] GDPR compliance
- [ ] HIPAA compliance (if applicable)
- [ ] SOC 2 compliance (if applicable)
- [ ] Regular security audits
- [ ] Penetration testing
- [ ] Vulnerability scanning

---

## 9. Summary

This guide provides comprehensive security architecture for AI applications:

**Key Architectures:**
1. **Defense in Depth** - Multiple security layers
2. **Zero Trust** - Verify explicitly, least privilege
3. **Input Validation** - Multi-layer validation pipeline
4. **PII Detection** - Automatic detection and redaction
5. **Authentication** - JWT + API keys + RBAC
6. **Data Protection** - Encryption, secrets management
7. **Threat Detection** - Anomaly detection, IDS
8. **Incident Response** - Automated response system

**Core Principles:**
- 🛡️ **Defense in depth** (multiple layers)
- 🔒 **Zero trust** (never trust, always verify)
- 🎯 **Least privilege** (minimum necessary access)
- 📊 **Monitor everything** (detect threats early)
- ⚡ **Automate response** (contain threats quickly)

**Related Documentation:**
- [Security Guide](SECURITY.md) - Security practices
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design
- [Observability Architecture](OBSERVABILITY_ARCHITECTURE.md) - Monitoring
- [Compliance Guide](COMPLIANCE.md) - Regulatory compliance

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Status:** Active
