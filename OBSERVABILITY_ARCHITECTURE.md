# Observability Architecture for AI Applications

## Overview

This guide provides comprehensive architectural patterns and infrastructure design for observability in AI/LLM applications. It focuses on monitoring system design, logging infrastructure, metrics pipelines, distributed tracing, and cost tracking architectures.

**Focus Areas:**
- Observability architecture patterns
- Monitoring infrastructure design
- Logging pipeline architecture
- Metrics collection and aggregation
- Distributed tracing for AI workflows
- Cost tracking infrastructure
- Alerting system architecture
- Dashboard and visualization design
- Data retention and storage strategies

**Related Guides:**
- [Observability Guide](OBSERVABILITY.md) - Monitoring strategies and implementation
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design patterns
- [AI Development](AI_DEVELOPMENT.md) - Development workflow
- [Cost Reduction](COST_REDUCTION_RULES.md) - Cost optimization

---

## 1. Observability Architecture Patterns

### 1.1 Three Pillars Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Application                           │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│   LOGS      │  │   METRICS   │  │   TRACES     │
│ (Context)   │  │ (Numbers)   │  │  (Flows)     │
└───────┬─────┘  └──────┬──────┘  └─────┬────────┘
        │                │                │
        │    ┌───────────▼────────────┐   │
        │    │  Correlation Engine    │   │
        │    │  (Request ID, User ID) │   │
        │    └───────────┬────────────┘   │
        │                │                │
┌───────▼────────────────▼────────────────▼────────┐
│              Observability Platform              │
│    (Elasticsearch, Prometheus, Jaeger)           │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│           Visualization & Alerting               │
│         (Grafana, Kibana, PagerDuty)            │
└──────────────────────────────────────────────────┘
```

**Components:**

1. **Logs** - Structured events with context
   - Request/response details
   - Errors and exceptions
   - Audit trails
   - Debug information

2. **Metrics** - Numerical measurements over time
   - Token counts
   - Cost per request
   - Latency percentiles
   - Cache hit rates

3. **Traces** - Request flow through system
   - End-to-end request paths
   - Service dependencies
   - Performance bottlenecks
   - Error propagation

### 1.2 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Instrumentation: logs, metrics, traces)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Collection Layer                          │
│  (Agents: Fluentd, Prometheus exporters, OTLP)             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Processing Layer                          │
│  (Enrichment, filtering, aggregation)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Storage Layer                            │
│  (Time-series DB, Log store, Trace store)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Query & Analysis                          │
│  (PromQL, Lucene, SQL)                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                Visualization & Alerting                     │
│  (Dashboards, alerts, reports)                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 AI-Specific Observability Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Request                             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Observability Wrapper                      │
│  - Capture input/output                                     │
│  - Track tokens and cost                                    │
│  - Measure latency                                          │
│  - Detect PII                                               │
│  - Record cache hit/miss                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│ Cost Track  │  │  Quality    │  │ Performance  │
│  - $/req    │  │  - Length   │  │  - Latency   │
│  - Daily    │  │  - Sentiment│  │  - Tokens/sec│
│  - By model │  │  - Feedback │  │  - Cache rate│
└─────────────┘  └─────────────┘  └──────────────┘
```

---

## 2. Logging Architecture

### 2.1 Structured Logging Pipeline

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Application Logs                         │
│  (JSON format, standardized fields)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Log Aggregator                           │
│  (Fluentd/Logstash - parse, enrich, filter)                │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│   Hot       │  │    Warm     │  │    Cold      │
│  Storage    │  │   Storage   │  │   Storage    │
│  (7 days)   │  │  (30 days)  │  │  (365 days)  │
│  ES/OpenS   │  │  S3/Glacier │  │  Archive     │
└─────────────┘  └─────────────┘  └──────────────┘
```

**Log Schema:**
```python
# src/observability/log_schema.py
"""
Standardized log schema for AI applications
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class BaseLogEntry:
    """Base log entry structure"""
    timestamp: str
    level: str
    service: str
    environment: str
    request_id: str
    user_id: Optional[str]
    message: str

@dataclass
class LLMRequestLog(BaseLogEntry):
    """Log entry for LLM requests"""
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    prompt_hash: str  # Hash of prompt for deduplication
    response_hash: str
    error: Optional[str] = None

@dataclass
class CostTrackingLog(BaseLogEntry):
    """Log entry for cost tracking"""
    cost_type: str  # 'llm', 'storage', 'compute'
    amount_usd: float
    resource: str
    billing_period: str
    tags: Dict[str, str]

@dataclass
class SecurityLog(BaseLogEntry):
    """Log entry for security events"""
    event_type: str  # 'pii_detected', 'prompt_injection', 'rate_limit'
    severity: str
    details: Dict[str, Any]
    action_taken: str
    ip_address: Optional[str]

class StructuredLogger:
    """Structured logger with standardized schema"""

    def __init__(self, service: str, environment: str):
        self.service = service
        self.environment = environment

    def log_llm_request(
        self,
        level: str,
        request_id: str,
        user_id: Optional[str],
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cache_hit: bool,
        prompt_hash: str,
        response_hash: str,
        error: Optional[str] = None
    ):
        """Log LLM request"""
        log_entry = LLMRequestLog(
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            service=self.service,
            environment=self.environment,
            request_id=request_id,
            user_id=user_id,
            message="LLM request completed",
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            error=error
        )
        print(json.dumps(asdict(log_entry)))

    def log_security_event(
        self,
        request_id: str,
        user_id: Optional[str],
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        action_taken: str,
        ip_address: Optional[str] = None
    ):
        """Log security event"""
        log_entry = SecurityLog(
            timestamp=datetime.utcnow().isoformat(),
            level="WARNING" if severity == "medium" else "ERROR",
            service=self.service,
            environment=self.environment,
            request_id=request_id,
            user_id=user_id,
            message=f"Security event: {event_type}",
            event_type=event_type,
            severity=severity,
            details=details,
            action_taken=action_taken,
            ip_address=ip_address
        )
        print(json.dumps(asdict(log_entry)))

# Global logger instance
logger = StructuredLogger(
    service=os.getenv('SERVICE_NAME', 'ai-app'),
    environment=os.getenv('ENVIRONMENT', 'development')
)
```

### 2.2 Log Aggregation Architecture

**Fluentd Configuration:**
```xml
# fluent.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

# Parse JSON logs
<filter app.**>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

# Enrich with metadata
<filter app.**>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    environment "#{ENV['ENVIRONMENT']}"
  </record>
</filter>

# Split by log type
<match app.llm_request>
  @type copy

  # Send to Elasticsearch for searching
  <store>
    @type elasticsearch
    host elasticsearch
    port 9200
    index_name llm-requests-%Y%m%d
    type_name _doc
    include_tag_key true
    tag_key @log_name
  </store>

  # Send to S3 for archival
  <store>
    @type s3
    aws_key_id "#{ENV['AWS_ACCESS_KEY_ID']}"
    aws_sec_key "#{ENV['AWS_SECRET_ACCESS_KEY']}"
    s3_bucket observability-logs
    s3_region us-east-1
    path logs/llm-requests/%Y/%m/%d/
    time_slice_format %Y%m%d%H
    utc
    <buffer>
      @type file
      path /var/log/fluent/s3
      timekey 3600
      timekey_wait 10m
      chunk_limit_size 256m
    </buffer>
  </store>

  # Send to Prometheus for metrics
  <store>
    @type prometheus
    <metric>
      name llm_request_total
      type counter
      desc Total number of LLM requests
      <labels>
        model ${model}
        provider ${provider}
        cache_hit ${cache_hit}
      </labels>
    </metric>
  </store>
</match>

# Security logs - high priority
<match app.security>
  @type copy

  <store>
    @type elasticsearch
    host elasticsearch
    port 9200
    index_name security-events-%Y%m%d
  </store>

  # Alert on critical security events
  <store>
    @type http
    endpoint http://alertmanager:9093/api/v1/alerts
    http_method post
  </store>
</match>

# Cost tracking logs
<match app.cost_tracking>
  @type copy

  <store>
    @type elasticsearch
    host elasticsearch
    port 9200
    index_name cost-tracking-%Y%m%d
  </store>

  # Also send to PostgreSQL for analysis
  <store>
    @type sql
    host postgres
    port 5432
    database observability
    adapter postgresql
    username "#{ENV['DB_USER']}"
    password "#{ENV['DB_PASSWORD']}"
    <table>
      table cost_logs
      column_mapping 'timestamp:timestamp,cost_usd:cost_usd,resource:resource,user_id:user_id'
    </table>
  </store>
</match>
```

### 2.3 Log Retention Strategy

**Tiered Storage:**
```
Hot Tier (0-7 days):
- Storage: Elasticsearch
- Purpose: Real-time search, debugging
- Retention: 7 days
- Cost: High (fast SSD)

Warm Tier (8-30 days):
- Storage: Elasticsearch with slower disks
- Purpose: Recent analysis, compliance
- Retention: 30 days
- Cost: Medium

Cold Tier (31-365 days):
- Storage: S3 Standard
- Purpose: Compliance, historical analysis
- Retention: 1 year
- Cost: Low

Archive Tier (365+ days):
- Storage: S3 Glacier
- Purpose: Long-term compliance
- Retention: 7 years (HIPAA requirement)
- Cost: Very low
```

**Implementation:**
```python
# src/observability/log_lifecycle.py
"""
Manages log lifecycle across storage tiers
"""

from datetime import datetime, timedelta
from typing import List
import boto3

class LogLifecycleManager:
    """Manages log data lifecycle"""

    def __init__(self):
        self.es_client = Elasticsearch(['http://elasticsearch:9200'])
        self.s3_client = boto3.client('s3')

    def archive_old_indices(self, days: int = 7):
        """Archive Elasticsearch indices older than N days"""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get indices older than cutoff
        indices = self.es_client.indices.get_alias(index="*")

        for index_name in indices:
            # Parse date from index name (e.g., llm-requests-20260206)
            index_date = self._parse_index_date(index_name)

            if index_date and index_date < cutoff_date:
                # Snapshot to S3
                self._snapshot_index(index_name)

                # Delete from Elasticsearch
                self.es_client.indices.delete(index=index_name)

                print(f"Archived index: {index_name}")

    def _snapshot_index(self, index_name: str):
        """Create snapshot of index to S3"""
        snapshot_name = f"snapshot-{index_name}-{datetime.now().strftime('%Y%m%d')}"

        self.es_client.snapshot.create(
            repository='s3_repository',
            snapshot=snapshot_name,
            body={
                'indices': index_name,
                'include_global_state': False
            }
        )

    def transition_to_glacier(self, days: int = 365):
        """Transition S3 logs to Glacier after N days"""
        # S3 Lifecycle policy handles this automatically
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'transition-to-glacier',
                    'Status': 'Enabled',
                    'Prefix': 'logs/',
                    'Transitions': [
                        {
                            'Days': days,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }

        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket='observability-logs',
            LifecycleConfiguration=lifecycle_config
        )

    def _parse_index_date(self, index_name: str) -> datetime:
        """Parse date from index name"""
        try:
            # Extract date from pattern: prefix-YYYYMMDD
            date_str = index_name.split('-')[-1]
            return datetime.strptime(date_str, '%Y%m%d')
        except:
            return None
```

---

## 3. Metrics Collection Architecture

### 3.1 Metrics Pipeline Design

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Metrics                        │
│  (Prometheus client libraries)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Metrics Exporters                           │
│  (Push to Prometheus Pushgateway)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Prometheus Server                           │
│  (Scrape, store, query time-series data)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│   Grafana   │  │ AlertManager│  │  Prometheus  │
│ Dashboards  │  │   Alerts    │  │   Federation │
└─────────────┘  └─────────────┘  └──────────────┘
```

**Metrics Types for AI Applications:**

1. **Counter** - Monotonically increasing values
   - Total requests
   - Total tokens consumed
   - Total cost

2. **Gauge** - Current value
   - Active requests
   - Current cost rate ($/hour)
   - Queue length

3. **Histogram** - Distribution of values
   - Request latency
   - Response length
   - Cost per request

4. **Summary** - Statistical summaries
   - Request latency percentiles (p50, p95, p99)
   - Token count percentiles

### 3.2 Prometheus Instrumentation

**Metrics Definition:**
```python
# src/observability/metrics.py
"""
Prometheus metrics for AI applications
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, push_to_gateway
)
import os

# Create registry
registry = CollectorRegistry()

# Request metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM API requests',
    ['model', 'provider', 'status'],
    registry=registry
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens processed',
    ['model', 'provider', 'token_type'],  # token_type: input or output
    registry=registry
)

# Cost metrics
llm_cost_total = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['model', 'provider', 'user_tier'],
    registry=registry
)

llm_cost_rate = Gauge(
    'llm_cost_rate_usd_per_hour',
    'Current cost rate in USD per hour',
    ['model', 'provider'],
    registry=registry
)

# Performance metrics
llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'Duration of LLM requests',
    ['model', 'provider', 'cache_hit'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
)

llm_request_latency = Summary(
    'llm_request_latency_seconds',
    'Latency of LLM requests',
    ['model', 'provider'],
    registry=registry
)

# Cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result'],  # operation: get/set, result: hit/miss
    registry=registry
)

cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Current cache hit rate',
    [],
    registry=registry
)

# Queue metrics
request_queue_length = Gauge(
    'request_queue_length',
    'Current number of requests in queue',
    ['priority'],
    registry=registry
)

# Error metrics
llm_errors_total = Counter(
    'llm_errors_total',
    'Total number of LLM errors',
    ['model', 'provider', 'error_type'],
    registry=registry
)

# Rate limiting metrics
rate_limit_hits_total = Counter(
    'rate_limit_hits_total',
    'Total number of rate limit hits',
    ['user_tier', 'limit_type'],
    registry=registry
)

# Quality metrics
response_quality_score = Histogram(
    'response_quality_score',
    'Quality score of LLM responses (0-1)',
    ['model'],
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    registry=registry
)

class MetricsCollector:
    """Collects and pushes metrics"""

    def __init__(self, pushgateway_url: str = None):
        self.pushgateway_url = pushgateway_url or os.getenv(
            'PROMETHEUS_PUSHGATEWAY_URL',
            'http://pushgateway:9091'
        )

    def record_llm_request(
        self,
        model: str,
        provider: str,
        status: str,  # success, error
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        duration_seconds: float,
        cache_hit: bool,
        user_tier: str = 'free'
    ):
        """Record metrics for LLM request"""
        # Request counter
        llm_requests_total.labels(
            model=model,
            provider=provider,
            status=status
        ).inc()

        # Token counters
        llm_tokens_total.labels(
            model=model,
            provider=provider,
            token_type='input'
        ).inc(input_tokens)

        llm_tokens_total.labels(
            model=model,
            provider=provider,
            token_type='output'
        ).inc(output_tokens)

        # Cost counter
        llm_cost_total.labels(
            model=model,
            provider=provider,
            user_tier=user_tier
        ).inc(cost_usd)

        # Duration histogram
        llm_request_duration.labels(
            model=model,
            provider=provider,
            cache_hit=str(cache_hit)
        ).observe(duration_seconds)

        # Latency summary
        llm_request_latency.labels(
            model=model,
            provider=provider
        ).observe(duration_seconds)

    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation"""
        cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()

    def update_cache_hit_rate(self, rate: float):
        """Update cache hit rate gauge"""
        cache_hit_rate.set(rate)

    def record_error(self, model: str, provider: str, error_type: str):
        """Record LLM error"""
        llm_errors_total.labels(
            model=model,
            provider=provider,
            error_type=error_type
        ).inc()

    def push_metrics(self, job_name: str = 'ai-app'):
        """Push metrics to Pushgateway"""
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=job_name,
                registry=registry
            )
        except Exception as e:
            # Don't fail application if metrics push fails
            print(f"Failed to push metrics: {e}")

# Global metrics collector
metrics = MetricsCollector()
```

**Usage in Application:**
```python
# src/llm/client.py
from src.observability.metrics import metrics
import time

class LLMClient:
    def generate(self, prompt: str, model: str = 'haiku'):
        start_time = time.time()

        try:
            # Make LLM call
            response = self._call_api(prompt, model)

            # Record metrics
            metrics.record_llm_request(
                model=model,
                provider='anthropic',
                status='success',
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost,
                duration_seconds=time.time() - start_time,
                cache_hit=False,
                user_tier='pro'
            )

            return response

        except Exception as e:
            # Record error
            metrics.record_error(
                model=model,
                provider='anthropic',
                error_type=type(e).__name__
            )
            raise
        finally:
            # Push metrics periodically
            if random.random() < 0.01:  # 1% of requests
                metrics.push_metrics()
```

### 3.3 Custom Metrics Exporter

**Exporter for AI-specific metrics:**
```python
# src/observability/custom_exporter.py
"""
Custom Prometheus exporter for AI-specific metrics
"""

from prometheus_client import start_http_server, Gauge
import time
import psycopg2

class AIMetricsExporter:
    """Exports AI-specific metrics from database"""

    def __init__(self, db_url: str):
        self.db_url = db_url

        # Define gauges
        self.daily_cost = Gauge(
            'ai_daily_cost_usd',
            'Total cost today in USD',
            ['user_id', 'model']
        )

        self.hourly_requests = Gauge(
            'ai_hourly_requests',
            'Requests in the last hour',
            ['model', 'cache_hit']
        )

        self.avg_response_time = Gauge(
            'ai_avg_response_time_seconds',
            'Average response time in last hour',
            ['model']
        )

    def collect_metrics(self):
        """Collect metrics from database"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        # Daily cost per user/model
        cursor.execute("""
            SELECT user_id, model, SUM(cost_usd)
            FROM requests
            WHERE created_at >= CURRENT_DATE
            GROUP BY user_id, model
        """)

        for user_id, model, cost in cursor.fetchall():
            self.daily_cost.labels(
                user_id=user_id,
                model=model
            ).set(cost)

        # Hourly requests
        cursor.execute("""
            SELECT model, cache_hit, COUNT(*)
            FROM requests
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            GROUP BY model, cache_hit
        """)

        for model, cache_hit, count in cursor.fetchall():
            self.hourly_requests.labels(
                model=model,
                cache_hit=str(cache_hit)
            ).set(count)

        # Average response time
        cursor.execute("""
            SELECT model, AVG(latency_ms) / 1000.0
            FROM requests
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            GROUP BY model
        """)

        for model, avg_latency in cursor.fetchall():
            self.avg_response_time.labels(model=model).set(avg_latency)

        cursor.close()
        conn.close()

    def run(self, port: int = 9100, interval: int = 60):
        """Run exporter"""
        start_http_server(port)
        print(f"Metrics exporter listening on port {port}")

        while True:
            try:
                self.collect_metrics()
            except Exception as e:
                print(f"Error collecting metrics: {e}")

            time.sleep(interval)

if __name__ == '__main__':
    exporter = AIMetricsExporter(
        db_url=os.getenv('DATABASE_URL')
    )
    exporter.run()
```

---

## 4. Distributed Tracing Architecture

### 4.1 Tracing for AI Workflows

```
User Request
    │
    ├─> [Span 1] API Gateway
    │   │ duration: 2ms
    │   │ tags: route=/api/generate, method=POST
    │   │
    │   ├─> [Span 2] Input Validation
    │   │   │ duration: 5ms
    │   │   │ tags: validation=success
    │   │   │
    │   │   ├─> [Span 3] PII Detection
    │   │       │ duration: 10ms
    │   │       │ tags: pii_found=false
    │   │
    │   ├─> [Span 4] Cache Check
    │   │   │ duration: 8ms
    │   │   │ tags: cache_hit=false
    │   │
    │   ├─> [Span 5] LLM Request
    │   │   │ duration: 1847ms
    │   │   │ tags: model=haiku, provider=anthropic
    │   │   │      input_tokens=245, output_tokens=312
    │   │   │      cost_usd=0.0023
    │   │   │
    │   │   ├─> [Span 6] HTTP Request to Anthropic
    │   │       │ duration: 1842ms
    │   │       │ tags: http.status=200
    │   │
    │   ├─> [Span 7] Cache Write
    │       │ duration: 12ms
    │       │ tags: cache_key=hash123
    │
    └─> Total Duration: 1884ms
```

**OpenTelemetry Implementation:**
```python
# src/observability/tracing.py
"""
Distributed tracing for AI applications
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
import os
from typing import Optional
from functools import wraps

# Configure tracer
resource = Resource.create({
    "service.name": os.getenv("SERVICE_NAME", "ai-app"),
    "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
    "deployment.environment": os.getenv("ENVIRONMENT", "development")
})

trace.set_tracer_provider(TracerProvider(resource=resource))

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
    agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Get tracer
tracer = trace.get_tracer(__name__)

class TracingContext:
    """Context for distributed tracing"""

    @staticmethod
    def trace_llm_request(model: str, provider: str):
        """Decorator for tracing LLM requests"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    "llm.request",
                    attributes={
                        "llm.model": model,
                        "llm.provider": provider,
                    }
                ) as span:
                    try:
                        result = func(*args, **kwargs)

                        # Add result attributes
                        span.set_attributes({
                            "llm.input_tokens": result.input_tokens,
                            "llm.output_tokens": result.output_tokens,
                            "llm.cost_usd": result.cost,
                            "llm.cached": result.cached
                        })

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        raise

            return wrapper
        return decorator

    @staticmethod
    def trace_cache_operation(operation: str):
        """Decorator for tracing cache operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    f"cache.{operation}",
                    attributes={"cache.operation": operation}
                ) as span:
                    result = func(*args, **kwargs)

                    if operation == "get":
                        span.set_attribute("cache.hit", result is not None)

                    return result

            return wrapper
        return decorator

    @staticmethod
    def trace_validation(validation_type: str):
        """Decorator for tracing validation"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    f"validation.{validation_type}",
                    attributes={"validation.type": validation_type}
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("validation.passed", True)
                        return result
                    except Exception as e:
                        span.set_attribute("validation.passed", False)
                        span.set_attribute("validation.error", str(e))
                        raise

            return wrapper
        return decorator

# Usage example
@TracingContext.trace_llm_request(model="haiku", provider="anthropic")
def generate_response(prompt: str):
    """Generate LLM response with tracing"""
    # This function will be automatically traced
    return llm_client.generate(prompt)

@TracingContext.trace_cache_operation(operation="get")
def get_from_cache(key: str):
    """Get from cache with tracing"""
    return cache.get(key)
```

### 4.2 Trace Correlation

**Correlation across logs, metrics, and traces:**
```python
# src/observability/correlation.py
"""
Correlates logs, metrics, and traces
"""

import uuid
from contextvars import ContextVar
from typing import Optional

# Context variables for correlation
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)

class CorrelationContext:
    """Manages correlation IDs across observability signals"""

    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request ID"""
        return str(uuid.uuid4())

    @staticmethod
    def set_request_context(request_id: str, user_id: Optional[str] = None):
        """Set request context"""
        request_id_var.set(request_id)
        user_id_var.set(user_id)

    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get current request ID"""
        return request_id_var.get()

    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user ID"""
        return user_id_var.get()

    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get current trace ID"""
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            return format(span.get_span_context().trace_id, '032x')
        return None

    @staticmethod
    def get_correlation_ids() -> dict:
        """Get all correlation IDs"""
        return {
            'request_id': CorrelationContext.get_request_id(),
            'user_id': CorrelationContext.get_user_id(),
            'trace_id': CorrelationContext.get_trace_id()
        }

# Middleware for Flask
from flask import request, g

def correlation_middleware():
    """Middleware to set correlation context"""
    # Get or generate request ID
    request_id = request.headers.get('X-Request-ID') or CorrelationContext.generate_request_id()
    user_id = request.headers.get('X-User-ID')

    # Set context
    CorrelationContext.set_request_context(request_id, user_id)

    # Store in Flask g for templates
    g.request_id = request_id
    g.user_id = user_id
```

---

## 5. Cost Tracking Architecture

### 5.1 Real-time Cost Tracking

**Cost Tracking Pipeline:**
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Request                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Cost Calculator                           │
│  - Calculate cost from tokens                               │
│  - Apply pricing model                                      │
│  - Add markup for user tier                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Cost Aggregator                           │
│  - Aggregate by user, model, time                          │
│  - Update running totals                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│ PostgreSQL  │  │   Redis     │  │  Prometheus  │
│ (detailed)  │  │  (realtime) │  │  (metrics)   │
└─────────────┘  └─────────────┘  └──────────────┘
```

**Implementation:**
```python
# src/observability/cost_tracking.py
"""
Real-time cost tracking for AI applications
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis
import psycopg2
from decimal import Decimal

@dataclass
class CostEntry:
    """Cost tracking entry"""
    timestamp: datetime
    request_id: str
    user_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    user_tier: str

class CostCalculator:
    """Calculates cost for LLM requests"""

    # Pricing per 1M tokens (USD)
    PRICING = {
        'anthropic': {
            'claude-haiku': {'input': 0.25, 'output': 1.25},
            'claude-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-opus': {'input': 15.00, 'output': 75.00},
        },
        'openai': {
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-4': {'input': 30.00, 'output': 60.00},
        }
    }

    @staticmethod
    def calculate_cost(
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Decimal:
        """Calculate cost for request"""
        if provider not in CostCalculator.PRICING:
            raise ValueError(f"Unknown provider: {provider}")

        if model not in CostCalculator.PRICING[provider]:
            raise ValueError(f"Unknown model: {model}")

        pricing = CostCalculator.PRICING[provider][model]

        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        return Decimal(str(input_cost + output_cost))

class CostTracker:
    """Tracks costs in real-time"""

    def __init__(self, redis_url: str, db_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.db_url = db_url

    def track_request(self, entry: CostEntry):
        """Track cost for request"""
        # 1. Store detailed record in PostgreSQL
        self._store_in_database(entry)

        # 2. Update real-time aggregates in Redis
        self._update_realtime_aggregates(entry)

        # 3. Check if user exceeded limits
        self._check_limits(entry)

    def _store_in_database(self, entry: CostEntry):
        """Store cost entry in database"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO cost_tracking (
                timestamp, request_id, user_id, model, provider,
                input_tokens, output_tokens, cost_usd, user_tier
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            entry.timestamp,
            entry.request_id,
            entry.user_id,
            entry.model,
            entry.provider,
            entry.input_tokens,
            entry.output_tokens,
            float(entry.cost_usd),
            entry.user_tier
        ))

        conn.commit()
        cursor.close()
        conn.close()

    def _update_realtime_aggregates(self, entry: CostEntry):
        """Update real-time aggregates in Redis"""
        today = datetime.now().strftime('%Y-%m-%d')
        hour = datetime.now().strftime('%Y-%m-%d-%H')

        # Daily cost by user
        user_daily_key = f"cost:daily:{today}:user:{entry.user_id}"
        self.redis_client.incrbyfloat(user_daily_key, float(entry.cost_usd))
        self.redis_client.expire(user_daily_key, 86400 * 2)  # 2 days TTL

        # Daily cost by model
        model_daily_key = f"cost:daily:{today}:model:{entry.model}"
        self.redis_client.incrbyfloat(model_daily_key, float(entry.cost_usd))
        self.redis_client.expire(model_daily_key, 86400 * 2)

        # Hourly cost (for rate tracking)
        hourly_key = f"cost:hourly:{hour}"
        self.redis_client.incrbyfloat(hourly_key, float(entry.cost_usd))
        self.redis_client.expire(hourly_key, 3600 * 2)  # 2 hours TTL

        # Total tokens by user
        tokens_key = f"tokens:daily:{today}:user:{entry.user_id}"
        total_tokens = entry.input_tokens + entry.output_tokens
        self.redis_client.incrby(tokens_key, total_tokens)
        self.redis_client.expire(tokens_key, 86400 * 2)

    def _check_limits(self, entry: CostEntry):
        """Check if user exceeded cost limits"""
        today = datetime.now().strftime('%Y-%m-%d')
        user_daily_key = f"cost:daily:{today}:user:{entry.user_id}"

        daily_cost = float(self.redis_client.get(user_daily_key) or 0)

        # Get user's daily limit based on tier
        limits = {
            'free': 1.0,
            'pro': 50.0,
            'enterprise': 1000.0
        }

        limit = limits.get(entry.user_tier, 1.0)

        if daily_cost >= limit:
            # Log warning
            print(f"User {entry.user_id} exceeded daily limit: ${daily_cost:.2f} >= ${limit:.2f}")

            # Store in Redis for rate limiting
            limit_key = f"limit:exceeded:user:{entry.user_id}"
            self.redis_client.setex(limit_key, 86400, "1")

    def get_user_daily_cost(self, user_id: str) -> float:
        """Get user's cost for today"""
        today = datetime.now().strftime('%Y-%m-%d')
        key = f"cost:daily:{today}:user:{user_id}"
        return float(self.redis_client.get(key) or 0)

    def get_cost_breakdown(self, user_id: str, days: int = 7) -> Dict:
        """Get cost breakdown for user"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT
                model,
                COUNT(*) as requests,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(cost_usd) as total_cost
            FROM cost_tracking
            WHERE user_id = %s AND timestamp >= %s
            GROUP BY model
            ORDER BY total_cost DESC
        """, (user_id, cutoff))

        breakdown = []
        for row in cursor.fetchall():
            breakdown.append({
                'model': row[0],
                'requests': row[1],
                'input_tokens': row[2],
                'output_tokens': row[3],
                'total_cost': float(row[4])
            })

        cursor.close()
        conn.close()

        return breakdown
```

### 5.2 Cost Analytics

**Cost Analysis Queries:**
```python
# src/observability/cost_analytics.py
"""
Cost analytics and reporting
"""

from typing import List, Dict
from datetime import datetime, timedelta
import psycopg2

class CostAnalytics:
    """Analyzes cost data"""

    def __init__(self, db_url: str):
        self.db_url = db_url

    def get_daily_cost_trend(self, days: int = 30) -> List[Dict]:
        """Get daily cost trend"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT
                DATE(timestamp) as date,
                SUM(cost_usd) as daily_cost,
                COUNT(*) as requests,
                SUM(input_tokens + output_tokens) as total_tokens
            FROM cost_tracking
            WHERE timestamp >= %s
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (cutoff,))

        trend = []
        for row in cursor.fetchall():
            trend.append({
                'date': row[0].isoformat(),
                'cost': float(row[1]),
                'requests': row[2],
                'tokens': row[3]
            })

        cursor.close()
        conn.close()

        return trend

    def get_top_spenders(self, limit: int = 10) -> List[Dict]:
        """Get top spending users in last 30 days"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=30)

        cursor.execute("""
            SELECT
                user_id,
                user_tier,
                COUNT(*) as requests,
                SUM(cost_usd) as total_cost,
                AVG(cost_usd) as avg_cost_per_request
            FROM cost_tracking
            WHERE timestamp >= %s
            GROUP BY user_id, user_tier
            ORDER BY total_cost DESC
            LIMIT %s
        """, (cutoff, limit))

        spenders = []
        for row in cursor.fetchall():
            spenders.append({
                'user_id': row[0],
                'user_tier': row[1],
                'requests': row[2],
                'total_cost': float(row[3]),
                'avg_cost_per_request': float(row[4])
            })

        cursor.close()
        conn.close()

        return spenders

    def get_model_efficiency(self) -> List[Dict]:
        """Compare model efficiency (cost per request)"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=7)

        cursor.execute("""
            SELECT
                model,
                provider,
                COUNT(*) as requests,
                AVG(cost_usd) as avg_cost,
                AVG(input_tokens + output_tokens) as avg_tokens,
                AVG(cost_usd / NULLIF(input_tokens + output_tokens, 0) * 1000000) as cost_per_million_tokens
            FROM cost_tracking
            WHERE timestamp >= %s
            GROUP BY model, provider
            ORDER BY cost_per_million_tokens
        """, (cutoff,))

        efficiency = []
        for row in cursor.fetchall():
            efficiency.append({
                'model': row[0],
                'provider': row[1],
                'requests': row[2],
                'avg_cost': float(row[3]),
                'avg_tokens': float(row[4]),
                'cost_per_million_tokens': float(row[5]) if row[5] else 0
            })

        cursor.close()
        conn.close()

        return efficiency

    def get_cost_forecast(self, days_ahead: int = 7) -> Dict:
        """Forecast cost for next N days"""
        # Get recent trend
        trend = self.get_daily_cost_trend(days=14)

        if len(trend) < 7:
            return {'forecast': 0, 'confidence': 'low'}

        # Simple moving average forecast
        recent_costs = [day['cost'] for day in trend[-7:]]
        avg_daily_cost = sum(recent_costs) / len(recent_costs)

        forecast = avg_daily_cost * days_ahead

        return {
            'forecast_usd': round(forecast, 2),
            'avg_daily_cost': round(avg_daily_cost, 2),
            'days_ahead': days_ahead,
            'confidence': 'medium'
        }
```

---

## 6. Alerting System Architecture

### 6.1 Alert Configuration

**AlertManager Configuration:**
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'alerts@example.com'
  smtp_auth_password: '${SMTP_PASSWORD}'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-notifications'

  routes:
    # Critical alerts - page immediately
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    # Cost alerts
    - match:
        category: cost
      receiver: 'cost-team'
      group_interval: 1h

    # Security alerts
    - match:
        category: security
      receiver: 'security-team'
      group_wait: 0s

receivers:
  - name: 'team-notifications'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#ai-app-alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_KEY}'
        description: '{{ .GroupLabels.alertname }}'

  - name: 'cost-team'
    email_configs:
      - to: 'cost-team@example.com'
        subject: 'Cost Alert: {{ .GroupLabels.alertname }}'

  - name: 'security-team'
    slack_configs:
      - api_url: '${SECURITY_SLACK_WEBHOOK}'
        channel: '#security-alerts'
    email_configs:
      - to: 'security@example.com'

inhibit_rules:
  # Inhibit lower severity if higher severity is firing
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

**Prometheus Alert Rules:**
```yaml
# prometheus-rules.yml
groups:
  - name: cost_alerts
    interval: 1m
    rules:
      # Daily cost exceeded threshold
      - alert: DailyCostExceeded
        expr: sum(increase(llm_cost_usd_total[24h])) > 100
        for: 5m
        labels:
          severity: warning
          category: cost
        annotations:
          summary: "Daily cost exceeded $100"
          description: "Total cost in last 24h: ${{ $value | humanize }}"

      # Hourly cost spike
      - alert: CostSpike
        expr: rate(llm_cost_usd_total[5m]) > 10
        for: 5m
        labels:
          severity: critical
          category: cost
        annotations:
          summary: "Cost spike detected"
          description: "Current cost rate: ${{ $value | humanize }}/hour"

      # User exceeded budget
      - alert: UserBudgetExceeded
        expr: ai_daily_cost_usd > 50
        for: 1m
        labels:
          severity: warning
          category: cost
        annotations:
          summary: "User {{ $labels.user_id }} exceeded budget"
          description: "Daily cost: ${{ $value | humanize }}"

  - name: performance_alerts
    interval: 30s
    rules:
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 10s"
          description: "P95 latency: {{ $value | humanize }}s"

      # High error rate
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5%"
          description: "Current error rate: {{ $value | humanizePercentage }}"

      # Cache hit rate too low
      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.4
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate < 40%"
          description: "Current hit rate: {{ $value | humanizePercentage }}"

  - name: security_alerts
    interval: 30s
    rules:
      # PII detection spike
      - alert: PIIDetectionSpike
        expr: rate(llm_requests_total{pii_detected="true"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
          category: security
        annotations:
          summary: "High rate of PII detection"
          description: "PII detected in {{ $value | humanize }} requests/second"

      # Prompt injection attempts
      - alert: PromptInjectionAttempts
        expr: rate(security_events_total{event_type="prompt_injection"}[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
          category: security
        annotations:
          summary: "Prompt injection attempts detected"
          description: "{{ $value | humanize }} attempts/second"

      # Rate limit exceeded frequently
      - alert: FrequentRateLimits
        expr: rate(rate_limit_hits_total[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Frequent rate limiting"
          description: "{{ $value | humanize }} rate limits hit/second"

  - name: availability_alerts
    interval: 30s
    rules:
      # Service down
      - alert: ServiceDown
        expr: up{job="ai-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "Service has been down for more than 1 minute"

      # High request queue
      - alert: HighRequestQueue
        expr: request_queue_length > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Request queue is backing up"
          description: "Queue length: {{ $value }}"
```

### 6.2 Alert Response Automation

**Automated Alert Response:**
```python
# src/observability/alert_handler.py
"""
Automated alert response system
"""

from typing import Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import requests

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert data structure"""
    name: str
    severity: AlertSeverity
    labels: Dict[str, str]
    annotations: Dict[str, str]
    value: float

class AlertHandler:
    """Handles alerts and executes automated responses"""

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default alert handlers"""
        self.register_handler('CostSpike', self._handle_cost_spike)
        self.register_handler('UserBudgetExceeded', self._handle_user_budget_exceeded)
        self.register_handler('HighErrorRate', self._handle_high_error_rate)
        self.register_handler('ServiceDown', self._handle_service_down)

    def register_handler(self, alert_name: str, handler: Callable):
        """Register custom alert handler"""
        self.handlers[alert_name] = handler

    def handle_alert(self, alert: Alert):
        """Process incoming alert"""
        handler = self.handlers.get(alert.name)

        if handler:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error handling alert {alert.name}: {e}")
        else:
            print(f"No handler registered for alert: {alert.name}")

    def _handle_cost_spike(self, alert: Alert):
        """Handle cost spike alert"""
        print(f"Cost spike detected: ${alert.value:.2f}/hour")

        # 1. Enable aggressive caching
        self._enable_aggressive_caching()

        # 2. Switch to cheaper models temporarily
        self._switch_to_cheaper_models()

        # 3. Increase rate limits
        self._tighten_rate_limits()

        # 4. Notify team
        self._send_slack_notification(
            channel='#cost-alerts',
            message=f"⚠️ Cost spike detected: ${alert.value:.2f}/hour. "
                   f"Automatic mitigations enabled."
        )

    def _handle_user_budget_exceeded(self, alert: Alert):
        """Handle user budget exceeded"""
        user_id = alert.labels.get('user_id')
        print(f"User {user_id} exceeded budget: ${alert.value:.2f}")

        # 1. Enable rate limiting for user
        self._enable_user_rate_limit(user_id)

        # 2. Send notification to user
        self._send_user_notification(
            user_id=user_id,
            message=f"You've reached your daily budget limit of ${alert.value:.2f}. "
                   f"Please upgrade your plan or try again tomorrow."
        )

    def _handle_high_error_rate(self, alert: Alert):
        """Handle high error rate"""
        error_rate = alert.value * 100
        print(f"High error rate: {error_rate:.1f}%")

        # 1. Enable fallback providers
        self._enable_fallback_providers()

        # 2. Increase retry attempts
        self._increase_retry_attempts()

        # 3. Page on-call engineer if critical
        if alert.severity == AlertSeverity.CRITICAL:
            self._page_oncall()

    def _handle_service_down(self, alert: Alert):
        """Handle service down"""
        instance = alert.labels.get('instance')
        print(f"Service down: {instance}")

        # 1. Attempt automatic restart
        self._attempt_service_restart(instance)

        # 2. Page on-call immediately
        self._page_oncall()

        # 3. Failover to backup instance
        self._failover_to_backup()

    # Helper methods
    def _enable_aggressive_caching(self):
        """Enable aggressive caching to reduce costs"""
        # Implementation: Update cache TTL, lower similarity threshold
        pass

    def _switch_to_cheaper_models(self):
        """Temporarily switch to cheaper models"""
        # Implementation: Update model router config
        pass

    def _tighten_rate_limits(self):
        """Reduce rate limits to control costs"""
        # Implementation: Update rate limiter config
        pass

    def _enable_user_rate_limit(self, user_id: str):
        """Enable rate limiting for specific user"""
        # Implementation: Update Redis rate limit config
        pass

    def _enable_fallback_providers(self):
        """Enable fallback to alternative providers"""
        # Implementation: Update provider router config
        pass

    def _increase_retry_attempts(self):
        """Increase retry attempts with exponential backoff"""
        # Implementation: Update retry config
        pass

    def _attempt_service_restart(self, instance: str):
        """Attempt to restart service"""
        # Implementation: Call Kubernetes API or similar
        pass

    def _failover_to_backup(self):
        """Failover to backup instance"""
        # Implementation: Update load balancer config
        pass

    def _page_oncall(self):
        """Page on-call engineer via PagerDuty"""
        # Implementation: Call PagerDuty API
        pass

    def _send_slack_notification(self, channel: str, message: str):
        """Send Slack notification"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        requests.post(webhook_url, json={
            'channel': channel,
            'text': message
        })

    def _send_user_notification(self, user_id: str, message: str):
        """Send notification to user"""
        # Implementation: Email, SMS, or in-app notification
        pass

# Webhook endpoint for AlertManager
from flask import Flask, request, jsonify

app = Flask(__name__)
alert_handler = AlertHandler()

@app.route('/alerts', methods=['POST'])
def handle_alerts():
    """Webhook endpoint for AlertManager"""
    data = request.json

    for alert_data in data.get('alerts', []):
        alert = Alert(
            name=alert_data['labels']['alertname'],
            severity=AlertSeverity(alert_data['labels']['severity']),
            labels=alert_data['labels'],
            annotations=alert_data['annotations'],
            value=float(alert_data.get('value', 0))
        )

        alert_handler.handle_alert(alert)

    return jsonify({'status': 'ok'})
```

---

## 7. Dashboard and Visualization Architecture

### 7.1 Dashboard Design Principles

**Multi-Layer Dashboard Strategy:**
```
┌─────────────────────────────────────────────────────────────┐
│                  Executive Dashboard                         │
│  - High-level KPIs                                          │
│  - Daily cost, request volume, error rate                   │
│  - For: C-level, Product Management                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Operations Dashboard                        │
│  - Real-time metrics                                        │
│  - Latency, throughput, cache hit rate                      │
│  - For: DevOps, SRE                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Cost Management Dashboard                   │
│  - Cost breakdown by model, user, time                      │
│  - Budget vs actual, forecasts                              │
│  - For: Finance, Product Management                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Security Dashboard                          │
│  - PII detections, prompt injections                        │
│  - Rate limit hits, anomalies                               │
│  - For: Security Team                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Developer Dashboard                         │
│  - API performance by endpoint                              │
│  - Error traces, slow queries                               │
│  - For: Engineering Team                                    │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Grafana Dashboard Configuration

**Executive Dashboard JSON:**
```json
{
  "dashboard": {
    "title": "AI Application - Executive Dashboard",
    "panels": [
      {
        "id": 1,
        "title": "Daily Cost",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(llm_cost_usd_total[24h]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 80, "color": "yellow" },
                { "value": 100, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Request Volume (24h)",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(llm_requests_total[5m])) * 60",
            "legendFormat": "Requests/min"
          }
        ]
      },
      {
        "id": 3,
        "title": "Cost Breakdown by Model",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (model) (increase(llm_cost_usd_total[24h]))"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 1, "color": "yellow" },
                { "value": 5, "color": "red" }
              ]
            }
          }
        }
      }
    ]
  }
}
```

### 7.3 Custom Dashboard Builder

**Dynamic Dashboard Generation:**
```python
# src/observability/dashboard_builder.py
"""
Programmatically build Grafana dashboards
"""

from typing import List, Dict
import json

class DashboardBuilder:
    """Builds Grafana dashboards programmatically"""

    def __init__(self, title: str):
        self.title = title
        self.panels = []
        self.panel_id = 1

    def add_stat_panel(
        self,
        title: str,
        query: str,
        unit: str = "short",
        thresholds: List[Dict] = None
    ):
        """Add a stat panel"""
        panel = {
            "id": self.panel_id,
            "type": "stat",
            "title": title,
            "targets": [{"expr": query}],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "steps": thresholds or [
                            {"value": 0, "color": "green"}
                        ]
                    }
                }
            }
        }
        self.panels.append(panel)
        self.panel_id += 1

    def add_time_series_panel(
        self,
        title: str,
        queries: Dict[str, str]  # legend: query
    ):
        """Add a time series panel"""
        targets = []
        for legend, query in queries.items():
            targets.append({
                "expr": query,
                "legendFormat": legend
            })

        panel = {
            "id": self.panel_id,
            "type": "timeseries",
            "title": title,
            "targets": targets
        }
        self.panels.append(panel)
        self.panel_id += 1

    def add_table_panel(self, title: str, query: str):
        """Add a table panel"""
        panel = {
            "id": self.panel_id,
            "type": "table",
            "title": title,
            "targets": [{"expr": query, "format": "table"}]
        }
        self.panels.append(panel)
        self.panel_id += 1

    def build(self) -> Dict:
        """Build dashboard JSON"""
        return {
            "dashboard": {
                "title": self.title,
                "panels": self.panels,
                "refresh": "30s",
                "time": {
                    "from": "now-24h",
                    "to": "now"
                }
            }
        }

# Usage: Build cost management dashboard
builder = DashboardBuilder("Cost Management Dashboard")

builder.add_stat_panel(
    title="Daily Cost",
    query="sum(increase(llm_cost_usd_total[24h]))",
    unit="currencyUSD",
    thresholds=[
        {"value": 0, "color": "green"},
        {"value": 80, "color": "yellow"},
        {"value": 100, "color": "red"}
    ]
)

builder.add_time_series_panel(
    title="Cost Over Time",
    queries={
        "Haiku": 'sum(rate(llm_cost_usd_total{model="haiku"}[5m])) * 3600',
        "Sonnet": 'sum(rate(llm_cost_usd_total{model="sonnet"}[5m])) * 3600',
        "Opus": 'sum(rate(llm_cost_usd_total{model="opus"}[5m])) * 3600'
    }
)

builder.add_table_panel(
    title="Top Spenders",
    query='topk(10, sum by (user_id) (increase(llm_cost_usd_total[24h])))'
)

dashboard_json = builder.build()
```

---

## 8. Observability Best Practices

### 8.1 Instrumentation Guidelines

**What to Instrument:**

1. **All LLM API Calls**
   - Input/output tokens
   - Cost
   - Latency
   - Model and provider
   - Cache hit/miss
   - Request ID for correlation

2. **Business Metrics**
   - Active users
   - Requests per user
   - Conversion rates (if applicable)
   - User satisfaction scores

3. **System Metrics**
   - CPU, memory usage
   - Network I/O
   - Database connections
   - Queue lengths

4. **Security Events**
   - PII detections
   - Prompt injection attempts
   - Rate limit hits
   - Authentication failures

### 8.2 Performance Considerations

**Observability Overhead:**
- Logs: 1-5% overhead
- Metrics: <1% overhead
- Traces: 1-3% overhead (with sampling)

**Optimization Strategies:**
1. **Sampling** - Trace 1-10% of requests
2. **Async Logging** - Don't block on log writes
3. **Batching** - Batch metrics and log uploads
4. **Cardinality Control** - Limit unique label combinations
5. **Data Retention** - Archive old data to cheaper storage

### 8.3 Observability Checklist

**Setup:**
- [ ] Structured logging with JSON format
- [ ] Centralized log aggregation (Elasticsearch/CloudWatch)
- [ ] Metrics collection (Prometheus/DataDog)
- [ ] Distributed tracing (Jaeger/Zipkin)
- [ ] Real-time cost tracking
- [ ] Dashboard for each stakeholder group
- [ ] Alerting rules configured
- [ ] On-call rotation defined

**Instrumentation:**
- [ ] All LLM requests instrumented
- [ ] Cache operations tracked
- [ ] Error rates monitored
- [ ] Latency percentiles tracked (p50, p95, p99)
- [ ] Cost metrics by model, user, time
- [ ] Security events logged
- [ ] Correlation IDs in all logs

**Alerting:**
- [ ] Cost alerts (spike, daily limit, user budget)
- [ ] Performance alerts (latency, error rate)
- [ ] Availability alerts (service down, high queue)
- [ ] Security alerts (PII, prompt injection)
- [ ] Alert routing configured
- [ ] Automated response for common alerts
- [ ] Runbooks for critical alerts

---

## 9. Summary

This guide provides comprehensive architecture for observability in AI applications:

**Key Architectures:**
1. **Three Pillars** - Logs, metrics, traces with correlation
2. **Logging Pipeline** - Structured logs, tiered storage, retention
3. **Metrics Collection** - Prometheus instrumentation, custom exporters
4. **Distributed Tracing** - OpenTelemetry, trace correlation
5. **Cost Tracking** - Real-time tracking, analytics, forecasting
6. **Alerting System** - Alert rules, automated response
7. **Dashboard Design** - Multi-layer strategy for stakeholders

**Core Principles:**
- 📊 **Measure everything** (cost, performance, quality, security)
- 🔗 **Correlate signals** (request ID across logs/metrics/traces)
- 🎯 **Actionable alerts** (clear thresholds, automated response)
- 💰 **Track costs** (real-time, per user, forecasting)
- ⚡ **Low overhead** (sampling, async, batching)

**Related Documentation:**
- [Observability Guide](OBSERVABILITY.md) - Monitoring strategies
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design
- [Cost Reduction](COST_REDUCTION_RULES.md) - Cost optimization
- [AI Development](AI_DEVELOPMENT.md) - Development workflow

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Status:** Active
