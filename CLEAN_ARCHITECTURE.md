# Clean Architecture for AI Applications

## Overview

Clean Architecture (by Robert C. Martin) provides a framework for building maintainable, testable, and scalable AI applications by separating concerns into layers with clear dependency rules. This guide adapts Clean Architecture principles specifically for AI/LLM applications.

**Core Principle:** Dependencies flow inward. Inner layers know nothing about outer layers.

```
┌────────────────────────────────────────────────────────┐
│              External Interfaces                       │
│  (Web, CLI, API, Database, LLM Providers)             │
└───────────────────────┬────────────────────────────────┘
                        │ (depends on)
┌───────────────────────▼────────────────────────────────┐
│           Interface Adapters                           │
│  (Controllers, Presenters, Gateways)                   │
└───────────────────────┬────────────────────────────────┘
                        │ (depends on)
┌───────────────────────▼────────────────────────────────┐
│              Use Cases                                 │
│  (Application Business Rules)                          │
└───────────────────────┬────────────────────────────────┘
                        │ (depends on)
┌───────────────────────▼────────────────────────────────┐
│              Entities                                  │
│  (Enterprise Business Rules)                           │
└────────────────────────────────────────────────────────┘
```

**Key Benefits for AI Applications:**
- Easy to swap LLM providers (Anthropic, OpenAI, Azure)
- Testable without calling real APIs
- Business logic independent of frameworks
- Clear separation of AI-specific code from business logic
- Simplified migration and refactoring

---

## Table of Contents

1. [Clean Architecture Layers](#1-clean-architecture-layers)
2. [Dependency Rule](#2-dependency-rule)
3. [Layer Details for AI Applications](#3-layer-details-for-ai-applications)
4. [Complete Implementation Example](#4-complete-implementation-example)
5. [AI-Specific Patterns](#5-ai-specific-patterns)
6. [Testing Strategy](#6-testing-strategy)
7. [Integration with Other Architectures](#7-integration-with-other-architectures)
8. [Migration Guide](#8-migration-guide)
9. [Best Practices](#9-best-practices)
10. [Anti-Patterns to Avoid](#10-anti-patterns-to-avoid)

---

## 1. Clean Architecture Layers

### 1.1 Four Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4: FRAMEWORKS & DRIVERS                │
│                                                                 │
│  External Dependencies:                                         │
│  • Web Framework (Flask, FastAPI, Express)                     │
│  • Database (PostgreSQL, MongoDB)                              │
│  • LLM SDKs (anthropic, openai libraries)                      │
│  • Cache (Redis)                                               │
│  • Message Queue (RabbitMQ, Kafka)                            │
│  • External APIs                                               │
│                                                                 │
│  ▼ Dependency Direction                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 3: INTERFACE ADAPTERS                     │
│                                                                 │
│  Converts data between use cases and external world:           │
│  • Controllers (HTTP handlers)                                 │
│  • Presenters (Format responses)                               │
│  • Gateways (Implement repository interfaces)                 │
│  • LLM Provider Adapters (Anthropic, OpenAI implementations)   │
│                                                                 │
│  ▼ Dependency Direction                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LAYER 2: USE CASES                         │
│                                                                 │
│  Application-specific business rules:                           │
│  • Generate Summary Use Case                                    │
│  • Classify Intent Use Case                                     │
│  • Extract Entities Use Case                                    │
│  • Answer Question Use Case                                     │
│  • Cost Optimization Logic                                      │
│  • Input Validation Orchestration                              │
│                                                                 │
│  ▼ Dependency Direction                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       LAYER 1: ENTITIES                         │
│                                                                 │
│  Enterprise-wide business rules:                                │
│  • Domain Models (User, Conversation, Message)                 │
│  • Business Rules (Token limits, Cost calculations)            │
│  • Value Objects (PromptTemplate, TokenCount)                  │
│  • Domain Events (MessageCreated, CostExceeded)                │
│                                                                 │
│  ▲ NO DEPENDENCIES (pure business logic)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Dependency Rule

**The Golden Rule:** Source code dependencies must point ONLY INWARD.

### 2.1 What Each Layer Can Know About

| Layer | Can Depend On | Cannot Depend On |
|-------|---------------|------------------|
| **Entities** | Nothing | Everything |
| **Use Cases** | Entities only | Interface Adapters, Frameworks |
| **Interface Adapters** | Use Cases, Entities | Frameworks (only through interfaces) |
| **Frameworks** | Everything | Nothing (it's the outermost) |

### 2.2 Dependency Inversion Principle

Outer layers depend on **interfaces** defined by inner layers:

```python
# ❌ BAD: Use Case depends on concrete LLM provider
class GenerateSummaryUseCase:
    def __init__(self):
        self.llm = AnthropicClient()  # Direct dependency on framework

    def execute(self, text: str):
        return self.llm.generate(text)

# ✅ GOOD: Use Case depends on abstraction
class GenerateSummaryUseCase:
    def __init__(self, llm_gateway: LLMGatewayInterface):
        self.llm_gateway = llm_gateway  # Depends on interface

    def execute(self, text: str):
        return self.llm_gateway.generate(text)

# Interface defined in Use Case layer
class LLMGatewayInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

# Implementation in Interface Adapter layer
class AnthropicGateway(LLMGatewayInterface):
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(...)
        return response.content[0].text
```

---

## 3. Layer Details for AI Applications

### 3.1 Layer 1: Entities (Domain Models)

Pure business logic with no external dependencies.

```python
# entities/conversation.py
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
from decimal import Decimal

@dataclass
class TokenCount:
    """Value object for token counting."""
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def calculate_cost(self, pricing: 'ModelPricing') -> Decimal:
        """Business rule: Calculate cost based on pricing."""
        input_cost = (self.input_tokens / 1_000_000) * pricing.input_price
        output_cost = (self.output_tokens / 1_000_000) * pricing.output_price
        return Decimal(str(input_cost + output_cost))

@dataclass
class ModelPricing:
    """Entity for model pricing."""
    model_name: str
    input_price: Decimal  # per 1M tokens
    output_price: Decimal  # per 1M tokens

    @classmethod
    def for_model(cls, model_name: str) -> 'ModelPricing':
        """Business rule: Get pricing for model."""
        pricing_table = {
            "claude-3-haiku-20240307": (Decimal("0.25"), Decimal("1.25")),
            "claude-3-sonnet-20240229": (Decimal("3.0"), Decimal("15.0")),
            "claude-3-opus-20240229": (Decimal("15.0"), Decimal("75.0")),
        }
        input_p, output_p = pricing_table.get(
            model_name,
            (Decimal("3.0"), Decimal("15.0"))
        )
        return cls(model_name, input_p, output_p)

@dataclass
class Message:
    """Entity representing a message in a conversation."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    tokens: TokenCount
    cost: Decimal = field(default=Decimal("0"))

    def calculate_cost(self, pricing: ModelPricing) -> None:
        """Business rule: Calculate and store cost."""
        self.cost = self.tokens.calculate_cost(pricing)

@dataclass
class Conversation:
    """Entity representing a conversation."""
    id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_cost: Decimal = field(default=Decimal("0"))

    def add_message(self, message: Message) -> None:
        """Business rule: Add message and update total cost."""
        self.messages.append(message)
        self.total_cost += message.cost

    def get_context_window(self, max_messages: int = 10) -> List[Message]:
        """Business rule: Get recent messages for context."""
        return self.messages[-max_messages:]

    def exceeds_cost_limit(self, limit: Decimal) -> bool:
        """Business rule: Check if conversation exceeds cost limit."""
        return self.total_cost > limit

@dataclass
class User:
    """Entity representing a user."""
    id: str
    tier: str  # "free", "pro", "enterprise"
    daily_budget: Decimal
    daily_spend: Decimal = field(default=Decimal("0"))

    def can_afford(self, cost: Decimal) -> bool:
        """Business rule: Check if user can afford operation."""
        return (self.daily_spend + cost) <= self.daily_budget

    def charge(self, cost: Decimal) -> None:
        """Business rule: Charge user for operation."""
        if not self.can_afford(cost):
            raise ValueError(f"User {self.id} exceeds daily budget")
        self.daily_spend += cost
```

### 3.2 Layer 2: Use Cases (Application Business Rules)

Application-specific logic that orchestrates entities.

```python
# use_cases/generate_summary.py
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from entities.conversation import Message, TokenCount, ModelPricing

# ============================================================================
# INPUT/OUTPUT DATA STRUCTURES (DTOs)
# ============================================================================

@dataclass
class GenerateSummaryRequest:
    """Use case input."""
    text: str
    user_id: str
    max_length: Optional[int] = 200
    model: str = "claude-3-haiku-20240307"

@dataclass
class GenerateSummaryResponse:
    """Use case output."""
    summary: str
    tokens_used: TokenCount
    cost: float
    model_used: str
    cached: bool

# ============================================================================
# GATEWAY INTERFACES (defined by use case, implemented by adapters)
# ============================================================================

class LLMGatewayInterface(ABC):
    """Interface for LLM providers (implemented in adapter layer)."""

    @abstractmethod
    def generate(self, prompt: str, model: str, max_tokens: int) -> tuple[str, TokenCount]:
        """Generate text using LLM."""
        pass

class UserRepositoryInterface(ABC):
    """Interface for user data access."""

    @abstractmethod
    def get_user(self, user_id: str) -> 'User':
        pass

    @abstractmethod
    def update_user(self, user: 'User') -> None:
        pass

class CacheGatewayInterface(ABC):
    """Interface for caching."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: int) -> None:
        pass

# ============================================================================
# USE CASE
# ============================================================================

class GenerateSummaryUseCase:
    """
    Use case: Generate a summary of text.

    Business rules:
    1. Check cache first
    2. Validate user can afford operation
    3. Generate summary using cheapest appropriate model
    4. Track cost and update user budget
    5. Cache result
    """

    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        user_repository: UserRepositoryInterface,
        cache_gateway: CacheGatewayInterface,
    ):
        self.llm_gateway = llm_gateway
        self.user_repository = user_repository
        self.cache_gateway = cache_gateway

    def execute(self, request: GenerateSummaryRequest) -> GenerateSummaryResponse:
        """Execute the use case."""

        # 1. Check cache
        cache_key = self._generate_cache_key(request.text, request.model)
        cached_result = self.cache_gateway.get(cache_key)

        if cached_result:
            return GenerateSummaryResponse(
                summary=cached_result,
                tokens_used=TokenCount(0, 0),
                cost=0.0,
                model_used=request.model,
                cached=True
            )

        # 2. Get user and check budget
        user = self.user_repository.get_user(request.user_id)

        # 3. Estimate cost (rough estimate before generation)
        pricing = ModelPricing.for_model(request.model)
        estimated_tokens = TokenCount(
            input_tokens=len(request.text) // 4,  # rough estimate
            output_tokens=request.max_length
        )
        estimated_cost = estimated_tokens.calculate_cost(pricing)

        if not user.can_afford(estimated_cost):
            raise ValueError(
                f"Insufficient budget. Need ${estimated_cost}, "
                f"have ${user.daily_budget - user.daily_spend} remaining"
            )

        # 4. Generate summary
        prompt = self._build_prompt(request.text, request.max_length)
        summary, actual_tokens = self.llm_gateway.generate(
            prompt=prompt,
            model=request.model,
            max_tokens=request.max_length
        )

        # 5. Calculate actual cost
        actual_cost = actual_tokens.calculate_cost(pricing)

        # 6. Update user budget
        user.charge(actual_cost)
        self.user_repository.update_user(user)

        # 7. Cache result
        self.cache_gateway.set(cache_key, summary, ttl=86400)  # 24 hours

        return GenerateSummaryResponse(
            summary=summary,
            tokens_used=actual_tokens,
            cost=float(actual_cost),
            model_used=request.model,
            cached=False
        )

    def _build_prompt(self, text: str, max_length: int) -> str:
        """Build prompt for summarization."""
        return f"""Summarize the following text in {max_length} words or less:

{text}

Summary:"""

    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"summary:{model}:{text_hash}"
```

### 3.3 Layer 3: Interface Adapters

Convert data between use cases and external interfaces.

```python
# adapters/llm_gateways/anthropic_gateway.py
import anthropic
from use_cases.interfaces import LLMGatewayInterface
from entities.conversation import TokenCount

class AnthropicGateway(LLMGatewayInterface):
    """Adapter for Anthropic Claude API."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, model: str, max_tokens: int) -> tuple[str, TokenCount]:
        """Generate text using Anthropic API."""
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            text = message.content[0].text
            tokens = TokenCount(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens
            )

            return text, tokens

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")

# adapters/llm_gateways/openai_gateway.py
import openai
from use_cases.interfaces import LLMGatewayInterface
from entities.conversation import TokenCount

class OpenAIGateway(LLMGatewayInterface):
    """Adapter for OpenAI API."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, model: str, max_tokens: int) -> tuple[str, TokenCount]:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.choices[0].message.content
            tokens = TokenCount(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

            return text, tokens

        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

# adapters/repositories/user_repository.py
import psycopg2
from use_cases.interfaces import UserRepositoryInterface
from entities.conversation import User
from decimal import Decimal

class PostgresUserRepository(UserRepositoryInterface):
    """Adapter for user data in PostgreSQL."""

    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)

    def get_user(self, user_id: str) -> User:
        """Get user from database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, tier, daily_budget, daily_spend FROM users WHERE id = %s",
            (user_id,)
        )
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"User {user_id} not found")

        return User(
            id=row[0],
            tier=row[1],
            daily_budget=Decimal(str(row[2])),
            daily_spend=Decimal(str(row[3]))
        )

    def update_user(self, user: User) -> None:
        """Update user in database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET daily_spend = %s WHERE id = %s",
            (float(user.daily_spend), user.id)
        )
        self.conn.commit()

# adapters/cache/redis_cache_gateway.py
import redis
from typing import Optional
from use_cases.interfaces import CacheGatewayInterface

class RedisCacheGateway(CacheGatewayInterface):
    """Adapter for Redis cache."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        value = self.redis.get(key)
        return value.decode() if value else None

    def set(self, key: str, value: str, ttl: int) -> None:
        """Set value in cache with TTL."""
        self.redis.setex(key, ttl, value)

# adapters/controllers/summary_controller.py
from flask import Flask, request, jsonify
from use_cases.generate_summary import (
    GenerateSummaryUseCase,
    GenerateSummaryRequest,
    GenerateSummaryResponse
)

class SummaryController:
    """HTTP controller for summary generation."""

    def __init__(self, use_case: GenerateSummaryUseCase):
        self.use_case = use_case

    def handle_generate_summary(self):
        """Handle POST /api/summary request."""
        # 1. Parse request (convert HTTP to DTO)
        data = request.get_json()
        use_case_request = GenerateSummaryRequest(
            text=data["text"],
            user_id=request.headers.get("X-User-ID", "anonymous"),
            max_length=data.get("max_length", 200),
            model=data.get("model", "claude-3-haiku-20240307")
        )

        # 2. Execute use case
        try:
            response = self.use_case.execute(use_case_request)

            # 3. Format response (convert DTO to HTTP)
            return jsonify({
                "summary": response.summary,
                "tokens_used": {
                    "input": response.tokens_used.input_tokens,
                    "output": response.tokens_used.output_tokens,
                    "total": response.tokens_used.total
                },
                "cost": response.cost,
                "model": response.model_used,
                "cached": response.cached
            }), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error"}), 500

# adapters/presenters/summary_presenter.py
from use_cases.generate_summary import GenerateSummaryResponse

class SummaryPresenter:
    """Presenter for formatting summary responses."""

    def present(self, response: GenerateSummaryResponse) -> dict:
        """Format response for CLI output."""
        return {
            "summary": response.summary,
            "tokens": f"{response.tokens_used.total:,}",
            "cost": f"${response.cost:.4f}",
            "model": response.model_used,
            "cached": "Yes" if response.cached else "No"
        }
```

### 3.4 Layer 4: Frameworks & Drivers

External dependencies and framework setup.

```python
# main.py - Application entry point
from flask import Flask
from adapters.llm_gateways.anthropic_gateway import AnthropicGateway
from adapters.llm_gateways.openai_gateway import OpenAIGateway
from adapters.repositories.user_repository import PostgresUserRepository
from adapters.cache.redis_cache_gateway import RedisCacheGateway
from adapters.controllers.summary_controller import SummaryController
from use_cases.generate_summary import GenerateSummaryUseCase
import os

# Dependency Injection / Composition Root
def create_app() -> Flask:
    """Create and configure Flask app with dependencies."""
    app = Flask(__name__)

    # 1. Create gateways (outermost layer)
    llm_gateway = AnthropicGateway(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Or: llm_gateway = OpenAIGateway(api_key=os.getenv("OPENAI_API_KEY"))

    user_repository = PostgresUserRepository(
        connection_string=os.getenv("DATABASE_URL")
    )

    cache_gateway = RedisCacheGateway(
        redis_url=os.getenv("REDIS_URL", "redis://localhost")
    )

    # 2. Create use cases (inject gateways)
    generate_summary_use_case = GenerateSummaryUseCase(
        llm_gateway=llm_gateway,
        user_repository=user_repository,
        cache_gateway=cache_gateway
    )

    # 3. Create controllers (inject use cases)
    summary_controller = SummaryController(generate_summary_use_case)

    # 4. Register routes
    @app.route("/api/summary", methods=["POST"])
    def generate_summary():
        return summary_controller.handle_generate_summary()

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "healthy"}, 200

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
```

---

## 4. Complete Implementation Example

### 4.1 Project Structure

```
ai-app/
├── entities/                    # Layer 1: Enterprise Business Rules
│   ├── __init__.py
│   ├── conversation.py          # Conversation, Message entities
│   ├── user.py                  # User entity
│   └── value_objects.py         # TokenCount, ModelPricing
│
├── use_cases/                   # Layer 2: Application Business Rules
│   ├── __init__.py
│   ├── interfaces.py            # Gateway interfaces (defined here!)
│   ├── generate_summary.py      # Generate summary use case
│   ├── classify_intent.py       # Classify intent use case
│   ├── extract_entities.py      # Extract entities use case
│   └── answer_question.py       # Answer question use case
│
├── adapters/                    # Layer 3: Interface Adapters
│   ├── __init__.py
│   ├── llm_gateways/
│   │   ├── __init__.py
│   │   ├── anthropic_gateway.py
│   │   ├── openai_gateway.py
│   │   └── mock_gateway.py      # For testing
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── user_repository.py
│   │   ├── conversation_repository.py
│   │   └── memory_repository.py # For testing
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── redis_cache_gateway.py
│   │   └── memory_cache_gateway.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── summary_controller.py
│   │   └── chat_controller.py
│   └── presenters/
│       ├── __init__.py
│       └── summary_presenter.py
│
├── frameworks/                  # Layer 4: Frameworks & Drivers
│   ├── __init__.py
│   ├── web/
│   │   ├── __init__.py
│   │   └── flask_app.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── cli_app.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
│
├── tests/
│   ├── unit/                    # Test entities and use cases
│   │   ├── test_entities.py
│   │   └── test_use_cases.py
│   ├── integration/             # Test adapters
│   │   └── test_gateways.py
│   └── e2e/                     # Test full system
│       └── test_api.py
│
├── main.py                      # Application entry point
├── requirements.txt
└── README.md
```

### 4.2 Example: Multi-Provider LLM Gateway

```python
# use_cases/interfaces.py
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    cached: bool = False

class LLMGatewayInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Generate text using LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass

# adapters/llm_gateways/multi_provider_gateway.py
from typing import List
from use_cases.interfaces import LLMGatewayInterface, LLMResponse

class MultiProviderGateway(LLMGatewayInterface):
    """
    Gateway that tries multiple providers with fallback.
    Implements the adapter pattern and circuit breaker.
    """

    def __init__(self, providers: List[LLMGatewayInterface]):
        self.providers = providers
        self.current_provider_index = 0

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Try providers in order until one succeeds."""
        errors = []

        for i in range(len(self.providers)):
            provider_index = (self.current_provider_index + i) % len(self.providers)
            provider = self.providers[provider_index]

            if not provider.is_available():
                errors.append(f"{provider.__class__.__name__} not available")
                continue

            try:
                response = provider.generate(prompt, model, max_tokens, temperature)

                # Update current provider on success
                self.current_provider_index = provider_index

                return response

            except Exception as e:
                errors.append(f"{provider.__class__.__name__}: {str(e)}")

        # All providers failed
        raise RuntimeError(
            f"All LLM providers failed. Errors: {'; '.join(errors)}"
        )

    def is_available(self) -> bool:
        """Check if any provider is available."""
        return any(p.is_available() for p in self.providers)
```

---

## 5. AI-Specific Patterns

### 5.1 Cost-Aware Use Case

```python
# use_cases/cost_aware_generation.py
from abc import abstractmethod
from use_cases.interfaces import LLMGatewayInterface
from entities.conversation import ModelPricing, TokenCount
from decimal import Decimal

class CostAwareGenerationUseCase:
    """
    Use case that selects cheapest appropriate model.
    Implements cascade pattern: Haiku → Sonnet → Opus
    """

    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        max_cost_per_request: Decimal = Decimal("0.05")
    ):
        self.llm_gateway = llm_gateway
        self.max_cost = max_cost_per_request

        # Model cascade: cheapest to most expensive
        self.model_cascade = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ]

    def execute(
        self,
        prompt: str,
        max_tokens: int = 1024,
        quality_threshold: float = 0.8
    ) -> tuple[str, Decimal]:
        """
        Generate with cheapest model that meets quality threshold.
        """

        for model in self.model_cascade:
            # Estimate cost
            pricing = ModelPricing.for_model(model)
            estimated_tokens = TokenCount(
                input_tokens=len(prompt) // 4,
                output_tokens=max_tokens
            )
            estimated_cost = estimated_tokens.calculate_cost(pricing)

            # Check if within budget
            if estimated_cost > self.max_cost:
                continue

            # Try generation
            response = self.llm_gateway.generate(prompt, model, max_tokens)

            # Check quality (simplified - could use more sophisticated checks)
            quality = self._assess_quality(response.text)

            if quality >= quality_threshold:
                actual_tokens = TokenCount(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens
                )
                actual_cost = actual_tokens.calculate_cost(pricing)
                return response.text, actual_cost

        raise ValueError("No model met quality threshold within budget")

    def _assess_quality(self, text: str) -> float:
        """Assess response quality (0-1)."""
        # Simplified quality check
        # In practice: check for hallucination, coherence, completeness
        if len(text) < 10:
            return 0.0
        if "I don't know" in text.lower():
            return 0.5
        return 1.0
```

### 5.2 Prompt Template Entity

```python
# entities/prompt_template.py
from dataclasses import dataclass
from typing import Dict, Any
from string import Template

@dataclass
class PromptTemplate:
    """Value object for prompt templates."""

    template: str
    variables: list[str]

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        # Validate all required variables provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return Template(self.template).safe_substitute(**kwargs)

    @classmethod
    def summarization(cls) -> 'PromptTemplate':
        """Create summarization template."""
        return cls(
            template="""Summarize the following text in $max_words words or less:

$text

Summary:""",
            variables=["text", "max_words"]
        )

    @classmethod
    def classification(cls, categories: list[str]) -> 'PromptTemplate':
        """Create classification template."""
        categories_str = ", ".join(categories)
        return cls(
            template=f"""Classify the following text into one of these categories: {categories_str}

Text: $text

Category:""",
            variables=["text"]
        )
```

### 5.3 Deterministic-First Use Case

```python
# use_cases/extract_email.py
import re
from typing import Optional

class ExtractEmailUseCase:
    """
    Extract email from text.
    Uses deterministic logic (regex) instead of LLM.
    """

    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def __init__(self, llm_gateway: Optional[LLMGatewayInterface] = None):
        self.llm_gateway = llm_gateway

    def execute(self, text: str) -> Optional[str]:
        """
        Extract email using regex first, LLM as fallback.
        Cost: $0 for 95% of cases.
        """

        # Try deterministic approach first (FREE)
        emails = re.findall(self.EMAIL_PATTERN, text)
        if emails:
            return emails[0]  # Return first email found

        # Fallback to LLM only if no match (5% of cases)
        if self.llm_gateway:
            prompt = f"Extract the email address from this text: {text}"
            response = self.llm_gateway.generate(prompt, "claude-3-haiku-20240307", 50)

            # Validate LLM response with regex
            extracted = re.findall(self.EMAIL_PATTERN, response.text)
            if extracted:
                return extracted[0]

        return None
```

---

## 6. Testing Strategy

### 6.1 Testing Each Layer

```python
# tests/unit/test_entities.py
import pytest
from entities.conversation import TokenCount, ModelPricing, Message
from decimal import Decimal
from datetime import datetime

def test_token_count_total():
    """Test entity business rule: token total."""
    tokens = TokenCount(input_tokens=100, output_tokens=50)
    assert tokens.total == 150

def test_token_count_calculate_cost():
    """Test entity business rule: cost calculation."""
    tokens = TokenCount(input_tokens=1_000_000, output_tokens=1_000_000)
    pricing = ModelPricing("test", Decimal("1.0"), Decimal("2.0"))

    cost = tokens.calculate_cost(pricing)

    # (1M / 1M * 1.0) + (1M / 1M * 2.0) = 1.0 + 2.0 = 3.0
    assert cost == Decimal("3.0")

def test_message_calculate_cost():
    """Test entity business rule: message cost."""
    message = Message(
        id="1",
        role="user",
        content="Hello",
        timestamp=datetime.utcnow(),
        tokens=TokenCount(100, 50)
    )
    pricing = ModelPricing.for_model("claude-3-haiku-20240307")

    message.calculate_cost(pricing)

    assert message.cost > Decimal("0")

# tests/unit/test_use_cases.py
import pytest
from use_cases.generate_summary import (
    GenerateSummaryUseCase,
    GenerateSummaryRequest
)
from use_cases.interfaces import (
    LLMGatewayInterface,
    UserRepositoryInterface,
    CacheGatewayInterface
)
from entities.conversation import User, TokenCount
from decimal import Decimal

# Mock implementations for testing
class MockLLMGateway(LLMGatewayInterface):
    def generate(self, prompt: str, model: str, max_tokens: int):
        return "Test summary", TokenCount(100, 50)

    def is_available(self):
        return True

class MockUserRepository(UserRepositoryInterface):
    def __init__(self):
        self.users = {
            "user1": User("user1", "pro", Decimal("10.0"), Decimal("0.0"))
        }

    def get_user(self, user_id: str):
        return self.users.get(user_id)

    def update_user(self, user: User):
        self.users[user.id] = user

class MockCacheGateway(CacheGatewayInterface):
    def __init__(self):
        self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value: str, ttl: int):
        self.cache[key] = value

def test_generate_summary_use_case():
    """Test use case with mocked dependencies."""
    # Arrange
    llm_gateway = MockLLMGateway()
    user_repo = MockUserRepository()
    cache_gateway = MockCacheGateway()

    use_case = GenerateSummaryUseCase(llm_gateway, user_repo, cache_gateway)
    request = GenerateSummaryRequest(
        text="Long text to summarize",
        user_id="user1",
        max_length=100
    )

    # Act
    response = use_case.execute(request)

    # Assert
    assert response.summary == "Test summary"
    assert response.cost > 0
    assert not response.cached

    # Check user was charged
    user = user_repo.get_user("user1")
    assert user.daily_spend > Decimal("0")

def test_generate_summary_uses_cache():
    """Test use case uses cache on second call."""
    # Arrange
    llm_gateway = MockLLMGateway()
    user_repo = MockUserRepository()
    cache_gateway = MockCacheGateway()

    use_case = GenerateSummaryUseCase(llm_gateway, user_repo, cache_gateway)
    request = GenerateSummaryRequest(
        text="Same text",
        user_id="user1"
    )

    # Act
    response1 = use_case.execute(request)
    response2 = use_case.execute(request)

    # Assert
    assert not response1.cached
    assert response2.cached
    assert response2.cost == 0  # No cost for cached

def test_generate_summary_rejects_over_budget():
    """Test use case rejects request if user over budget."""
    # Arrange
    llm_gateway = MockLLMGateway()
    user_repo = MockUserRepository()
    user_repo.users["user1"].daily_spend = Decimal("9.99")  # Almost at limit
    cache_gateway = MockCacheGateway()

    use_case = GenerateSummaryUseCase(llm_gateway, user_repo, cache_gateway)
    request = GenerateSummaryRequest(
        text="Very long text" * 1000,  # Will exceed budget
        user_id="user1"
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Insufficient budget"):
        use_case.execute(request)

# tests/integration/test_anthropic_gateway.py
import pytest
from adapters.llm_gateways.anthropic_gateway import AnthropicGateway
import os

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
def test_anthropic_gateway_real_api():
    """Integration test with real Anthropic API."""
    gateway = AnthropicGateway(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = gateway.generate(
        prompt="Say 'Hello World'",
        model="claude-3-haiku-20240307",
        max_tokens=20
    )

    assert "hello" in response.text.lower()
    assert response.input_tokens > 0
    assert response.output_tokens > 0
```

### 6.2 Test Doubles

```python
# adapters/llm_gateways/mock_gateway.py
from use_cases.interfaces import LLMGatewayInterface, LLMResponse
from typing import Dict, Callable

class MockLLMGateway(LLMGatewayInterface):
    """Mock LLM gateway for testing."""

    def __init__(self):
        self.responses: Dict[str, str] = {}
        self.call_count = 0
        self.last_prompt = None

    def set_response(self, prompt_pattern: str, response: str):
        """Configure mock response."""
        self.responses[prompt_pattern] = response

    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float = 0.7) -> LLMResponse:
        """Return mock response."""
        self.call_count += 1
        self.last_prompt = prompt

        # Find matching response
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return LLMResponse(
                    text=response,
                    input_tokens=len(prompt) // 4,
                    output_tokens=len(response) // 4,
                    model=model,
                    provider="mock"
                )

        return LLMResponse(
            text="Mock response",
            input_tokens=100,
            output_tokens=50,
            model=model,
            provider="mock"
        )

    def is_available(self) -> bool:
        return True
```

---

## 7. Integration with Other Architectures

### 7.1 Clean + Cost-Efficient Architecture

```python
# use_cases/cost_optimized_pipeline.py
class CostOptimizedPipelineUseCase:
    """
    Implements cost-aware pipeline from Cost-Efficient Architecture.

    Pipeline: Deterministic → Rules → Cache → Cheap LLM → Expensive LLM
    """

    def __init__(
        self,
        deterministic_processors: list,  # Regex, libraries
        rule_engine,                      # Rule-based system
        cache_gateway: CacheGatewayInterface,
        cheap_llm_gateway: LLMGatewayInterface,  # Haiku
        expensive_llm_gateway: LLMGatewayInterface  # Opus
    ):
        self.deterministic = deterministic_processors
        self.rules = rule_engine
        self.cache = cache_gateway
        self.cheap_llm = cheap_llm_gateway
        self.expensive_llm = expensive_llm_gateway

    def execute(self, input_text: str) -> tuple[str, Decimal, str]:
        """
        Process through pipeline, stopping at first successful layer.
        Returns: (result, cost, layer_used)
        """

        # Layer 1: Deterministic logic (FREE)
        for processor in self.deterministic:
            result = processor.process(input_text)
            if result:
                return result, Decimal("0"), "deterministic"

        # Layer 2: Rule-based system (FREE)
        result = self.rules.evaluate(input_text)
        if result:
            return result, Decimal("0"), "rules"

        # Layer 3: Cache (CHEAP)
        cache_key = self._hash(input_text)
        cached = self.cache.get(cache_key)
        if cached:
            return cached, Decimal("0.0001"), "cache"

        # Layer 4: Cheap LLM ($ - Haiku)
        try:
            response = self.cheap_llm.generate(input_text, "claude-3-haiku-20240307", 1024)
            cost = self._calculate_cost(response)

            # Cache for future
            self.cache.set(cache_key, response.text, 86400)

            return response.text, cost, "cheap_llm"
        except Exception:
            pass

        # Layer 5: Expensive LLM ($$$ - Opus, last resort)
        response = self.expensive_llm.generate(input_text, "claude-3-opus-20240229", 1024)
        cost = self._calculate_cost(response)
        return response.text, cost, "expensive_llm"
```

### 7.2 Clean + Security Architecture

```python
# use_cases/secure_generation.py
class SecureGenerationUseCase:
    """
    Implements security validation from Security Architecture.
    Wraps generation with validation pipeline.
    """

    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        validation_pipeline: 'ValidationPipeline',  # From Security Architecture
        pii_detector: 'PIIDetector',
        output_sanitizer: 'OutputSanitizer'
    ):
        self.llm = llm_gateway
        self.validator = validation_pipeline
        self.pii_detector = pii_detector
        self.sanitizer = output_sanitizer

    def execute(self, user_input: str) -> str:
        """
        Secure generation with validation.

        Flow: Input Validation → Generation → Output Sanitization
        """

        # 1. Input validation (Security Architecture)
        validation_result = self.validator.validate(user_input)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid input: {validation_result.errors}")

        # 2. PII detection and redaction
        sanitized_input, pii_found = self.pii_detector.detect_and_redact(user_input)
        if pii_found:
            # Log PII detection event for compliance
            self._log_pii_detection(pii_found)

        # 3. Generate with sanitized input
        response = self.llm.generate(sanitized_input, "claude-3-haiku-20240307", 1024)

        # 4. Output sanitization
        safe_output = self.sanitizer.sanitize(response.text)

        return safe_output
```

### 7.3 Clean + Compliance Architecture

```python
# use_cases/compliant_data_access.py
class CompliantDataAccessUseCase:
    """
    Implements compliance checks from Compliance Architecture.
    Ensures user consent before processing.
    """

    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        consent_manager: 'ConsentManager',  # From Compliance Architecture
        audit_logger: 'AuditLogger'
    ):
        self.llm = llm_gateway
        self.consent = consent_manager
        self.audit = audit_logger

    def execute(self, user_id: str, data: str, purpose: str) -> str:
        """
        Process data with consent verification.

        GDPR compliance:
        1. Check user consent for purpose
        2. Log processing in audit trail
        3. Process only if consented
        """

        # 1. Verify consent (GDPR requirement)
        if not self.consent.check_consent(user_id, purpose):
            raise ValueError(f"User {user_id} has not consented to {purpose}")

        # 2. Log data access (compliance requirement)
        self.audit.log_data_access(
            user_id=user_id,
            purpose=purpose,
            data_type="llm_processing"
        )

        # 3. Process data
        response = self.llm.generate(data, "claude-3-haiku-20240307", 1024)

        # 4. Log processing completion
        self.audit.log_processing_complete(
            user_id=user_id,
            purpose=purpose,
            tokens_processed=response.input_tokens + response.output_tokens
        )

        return response.text
```

---

## 8. Migration Guide

### 8.1 Migrating Existing Code to Clean Architecture

**Step 1: Identify Entities**

Extract pure business logic from existing code:

```python
# Before: Mixed concerns
class SummaryService:
    def summarize(self, text: str, user_id: str):
        # Database access mixed with business logic
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT budget FROM users WHERE id = %s", (user_id,))
        budget = cursor.fetchone()[0]

        # LLM call mixed with cost calculation
        client = anthropic.Anthropic(api_key=API_KEY)
        response = client.messages.create(...)

        cost = (response.usage.input_tokens * 0.25 +
                response.usage.output_tokens * 1.25) / 1_000_000

        if cost > budget:
            raise ValueError("Over budget")

        return response.content[0].text

# After: Separated into entities and use case
# entities/user.py
class User:
    def can_afford(self, cost: Decimal) -> bool:
        return cost <= self.daily_budget - self.daily_spend

# use_cases/summarize.py
class SummarizeUseCase:
    def execute(self, request):
        user = self.user_repo.get_user(request.user_id)
        estimated_cost = self._estimate_cost(request.text)

        if not user.can_afford(estimated_cost):
            raise ValueError("Over budget")

        return self.llm_gateway.generate(...)
```

**Step 2: Define Interfaces**

Create abstractions for external dependencies:

```python
# use_cases/interfaces.py
class LLMGatewayInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class UserRepositoryInterface(ABC):
    @abstractmethod
    def get_user(self, user_id: str) -> User:
        pass
```

**Step 3: Implement Adapters**

Move framework-specific code to adapters:

```python
# adapters/anthropic_gateway.py
class AnthropicGateway(LLMGatewayInterface):
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(...)
        return response.content[0].text
```

**Step 4: Wire Dependencies**

Use dependency injection in main:

```python
# main.py
def create_app():
    # Create adapters
    llm_gateway = AnthropicGateway(anthropic.Anthropic(api_key=...))
    user_repo = PostgresUserRepository(conn_string=...)

    # Create use case with dependencies
    summarize_use_case = SummarizeUseCase(llm_gateway, user_repo)

    # Create controller
    summary_controller = SummaryController(summarize_use_case)

    return app
```

### 8.2 Incremental Migration Strategy

1. **Start with new features** - Implement new functionality using Clean Architecture
2. **Extract entities** - Move business logic from existing code to entities
3. **Define interfaces** - Create abstractions for external dependencies
4. **Implement adapters** - Wrap existing framework code in adapters
5. **Refactor use cases** - Extract application logic into use cases
6. **Wire dependencies** - Set up dependency injection
7. **Test thoroughly** - Ensure behavior unchanged

---

## 9. Best Practices

### 9.1 Dos

✅ **DO** define interfaces in use case layer
✅ **DO** use dependency injection
✅ **DO** keep entities pure (no external dependencies)
✅ **DO** make use cases testable without frameworks
✅ **DO** use DTOs for use case input/output
✅ **DO** put business rules in entities
✅ **DO** put application logic in use cases
✅ **DO** make adapters thin (just translation)
✅ **DO** use factories for complex object creation
✅ **DO** keep dependencies flowing inward

### 9.2 Don'ts

❌ **DON'T** let entities depend on frameworks
❌ **DON'T** put business logic in controllers
❌ **DON'T** let use cases know about HTTP/database details
❌ **DON'T** pass framework objects (Request, Response) to use cases
❌ **DON'T** use concrete classes in use case constructors
❌ **DON'T** violate the dependency rule
❌ **DON'T** skip the interface layer
❌ **DON'T** make entities aware of persistence
❌ **DON'T** couple use cases to specific frameworks
❌ **DON'T** test through frameworks (test use cases directly)

### 9.3 Code Quality Checklist

**Entities:**
- [ ] No external dependencies (no imports of frameworks)
- [ ] Pure business logic only
- [ ] Immutable where possible
- [ ] Well-tested with unit tests

**Use Cases:**
- [ ] Depend only on interfaces (not concrete classes)
- [ ] Use DTOs for input/output
- [ ] Single responsibility
- [ ] Framework-agnostic
- [ ] Testable with mocks

**Adapters:**
- [ ] Implement interfaces defined by use cases
- [ ] Thin translation layer only
- [ ] No business logic
- [ ] Handle framework-specific concerns

**Dependency Injection:**
- [ ] All dependencies injected through constructor
- [ ] No service locator pattern
- [ ] Composition root in main.py
- [ ] Easy to swap implementations

---

## 10. Anti-Patterns to Avoid

### 10.1 Leaky Abstraction

❌ **Wrong:** Interface exposes framework details

```python
# Bad: Interface leaks HTTP concerns
class UserGatewayInterface(ABC):
    @abstractmethod
    def get_user(self, request: flask.Request) -> flask.Response:
        pass
```

✅ **Correct:** Interface uses domain objects

```python
# Good: Interface uses domain types
class UserRepositoryInterface(ABC):
    @abstractmethod
    def get_user(self, user_id: str) -> User:
        pass
```

### 10.2 God Use Case

❌ **Wrong:** Single use case does everything

```python
# Bad: God use case
class AIServiceUseCase:
    def summarize(self, text): ...
    def classify(self, text): ...
    def extract(self, text): ...
    def answer(self, question): ...
    def analyze(self, data): ...
```

✅ **Correct:** Focused, single-purpose use cases

```python
# Good: Focused use cases
class GenerateSummaryUseCase: ...
class ClassifyIntentUseCase: ...
class ExtractEntitiesUseCase: ...
class AnswerQuestionUseCase: ...
```

### 10.3 Anemic Domain Model

❌ **Wrong:** Entities with no behavior

```python
# Bad: Just data, no behavior
@dataclass
class User:
    id: str
    budget: Decimal
    spend: Decimal

# Business logic in use case
class UseCase:
    def execute(self, user: User, cost: Decimal):
        if user.spend + cost > user.budget:  # Business logic outside entity
            raise ValueError("Over budget")
```

✅ **Correct:** Entities with behavior

```python
# Good: Business logic in entity
@dataclass
class User:
    id: str
    budget: Decimal
    spend: Decimal

    def can_afford(self, cost: Decimal) -> bool:  # Business logic in entity
        return self.spend + cost <= self.budget

    def charge(self, cost: Decimal):
        if not self.can_afford(cost):
            raise ValueError("Over budget")
        self.spend += cost

# Use case uses entity behavior
class UseCase:
    def execute(self, user: User, cost: Decimal):
        user.charge(cost)  # Delegate to entity
```

### 10.4 Service Locator

❌ **Wrong:** Use case finds its own dependencies

```python
# Bad: Service locator anti-pattern
class GenerateSummaryUseCase:
    def execute(self, text: str):
        llm_gateway = ServiceLocator.get("llm_gateway")  # ❌ Hidden dependency
        user_repo = ServiceLocator.get("user_repository")  # ❌ Hard to test
        return llm_gateway.generate(text)
```

✅ **Correct:** Dependency injection

```python
# Good: Dependencies explicit in constructor
class GenerateSummaryUseCase:
    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        user_repository: UserRepositoryInterface
    ):
        self.llm_gateway = llm_gateway
        self.user_repository = user_repository

    def execute(self, text: str):
        return self.llm_gateway.generate(text)
```

---

## References

### Books
- **"Clean Architecture"** by Robert C. Martin
- **"Domain-Driven Design"** by Eric Evans
- **"Implementing Domain-Driven Design"** by Vaughn Vernon

### Related Architecture Documents
- [System Architecture](SYSTEM_ARCHITECTURE.md) - Overall system design
- [Cost-Efficient Architecture](COST_EFFICIENT_ARCHITECTURE.md) - Cost optimization patterns
- [Security Architecture](SECURITY_ARCHITECTURE.md) - Security infrastructure
- [Testing Architecture](AI_TESTING_ARCHITECTURE.md) - Testing strategies

### External Resources
- [Clean Coder Blog](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Clean Architecture in Python](https://www.cosmicpython.com/)

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Total:** 2,100+ lines of Clean Architecture guidance
**Status:** Active
