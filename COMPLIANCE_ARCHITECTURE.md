# Compliance Architecture for AI Applications

## Overview

This guide provides comprehensive architectural patterns and infrastructure design for regulatory compliance in AI/LLM applications. It focuses on data governance, consent management, privacy by design, and automated compliance monitoring.

**Focus Areas:**
- Compliance architecture patterns
- Data governance infrastructure
- Consent management systems
- Data Subject Rights (DSR) automation
- Audit logging and trails
- Data retention and deletion
- Privacy by design patterns
- Cross-border data transfers
- Compliance monitoring and reporting
- Record keeping systems

**Related Guides:**
- [Compliance Guide](COMPLIANCE.md) - Regulatory requirements and practices
- [Security Architecture](SECURITY_ARCHITECTURE.md) - Security infrastructure
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design patterns
- [Observability Architecture](OBSERVABILITY_ARCHITECTURE.md) - Monitoring infrastructure

---

## 1. Compliance Architecture Patterns

### 1.1 Privacy by Design Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Request                            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Consent Validator                          │
│  - Check user consent for data processing                   │
│  - Verify purpose matches consent                           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Data Minimization Layer                       │
│  - Collect only necessary data                              │
│  - Redact PII where possible                                │
│  - Anonymize when retention > 90 days                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Purpose Limitation                         │
│  - Ensure data used only for stated purpose                 │
│  - Block secondary use without consent                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Processing Engine                          │
│  - Process with privacy controls                            │
│  - Log all data access                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Retention Management                          │
│  - Auto-delete based on retention policy                    │
│  - Anonymize after retention period                         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Audit Log                                 │
│  - Immutable record of all data operations                  │
│  - 7-year retention for compliance                          │
└─────────────────────────────────────────────────────────────┘
```

**Core Principles:**

1. **Proactive not Reactive** - Privacy embedded from design
2. **Privacy as Default** - Maximum privacy settings by default
3. **Privacy in Functionality** - Full functionality with privacy
4. **End-to-End Security** - Complete lifecycle protection
5. **Visibility and Transparency** - Open about data practices
6. **User-Centric** - Keep user interests central

### 1.2 Compliance-Driven Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection                          │
│  - Consent obtained                                         │
│  - Purpose documented                                       │
│  - Legal basis recorded                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Data Processing                          │
│  - Purpose limitation enforced                              │
│  - Access controls applied                                  │
│  - Processing logged                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐  ┌──────▼──────┐  ┌─────▼────────┐
│  Storage    │  │   Transfer  │  │   Sharing    │
│ (Encrypted) │  │ (Controlled)│  │ (Contractual)│
└───────┬─────┘  └──────┬──────┘  └─────┬────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Data Retention                           │
│  - Retention policy applied                                 │
│  - Auto-deletion scheduled                                  │
│  - Anonymization triggered                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Data Deletion                            │
│  - Secure erasure                                           │
│  - Deletion verified                                        │
│  - Deletion certificate issued                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Consent Management Architecture

### 2.1 Consent Management System

```python
# src/compliance/consent_manager.py
"""
Consent management system for GDPR/CCPA compliance
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

class ConsentPurpose(Enum):
    """Data processing purposes"""
    ESSENTIAL = "essential"  # Required for service
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    PERSONALIZATION = "personalization"
    THIRD_PARTY_SHARING = "third_party_sharing"
    AI_TRAINING = "ai_training"

class ConsentStatus(Enum):
    """Consent status"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

@dataclass
class ConsentRecord:
    """Individual consent record"""
    user_id: str
    purpose: ConsentPurpose
    status: ConsentStatus
    granted_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    expires_at: Optional[datetime]
    method: str  # 'explicit', 'opt_in', 'opt_out'
    ip_address: str
    user_agent: str
    version: str  # Policy version

class ConsentManager:
    """Manages user consent for data processing"""

    def __init__(self, db_connection):
        self.db = db_connection

    def record_consent(
        self,
        user_id: str,
        purposes: List[ConsentPurpose],
        status: ConsentStatus,
        ip_address: str,
        user_agent: str,
        method: str = 'explicit',
        expires_days: Optional[int] = None
    ) -> List[ConsentRecord]:
        """Record user consent for purposes"""
        records = []
        now = datetime.utcnow()

        for purpose in purposes:
            # Calculate expiry
            expires_at = None
            if expires_days:
                expires_at = now + timedelta(days=expires_days)

            record = ConsentRecord(
                user_id=user_id,
                purpose=purpose,
                status=status,
                granted_at=now if status == ConsentStatus.GRANTED else None,
                withdrawn_at=None,
                expires_at=expires_at,
                method=method,
                ip_address=ip_address,
                user_agent=user_agent,
                version=self.get_current_policy_version()
            )

            # Store in database
            self._store_consent(record)
            records.append(record)

            # Emit event for audit
            self._emit_consent_event(record)

        return records

    def check_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose
    ) -> bool:
        """Check if user has consented to purpose"""
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT status, expires_at
            FROM user_consents
            WHERE user_id = %s
              AND purpose = %s
            ORDER BY granted_at DESC
            LIMIT 1
        """, (user_id, purpose.value))

        row = cursor.fetchone()

        if not row:
            # No consent record
            # Essential purposes are implicitly consented
            return purpose == ConsentPurpose.ESSENTIAL

        status, expires_at = row

        if status != ConsentStatus.GRANTED.value:
            return False

        # Check expiry
        if expires_at and datetime.utcnow() > expires_at:
            # Mark as expired
            self._mark_consent_expired(user_id, purpose)
            return False

        return True

    def withdraw_consent(
        self,
        user_id: str,
        purposes: List[ConsentPurpose]
    ):
        """Withdraw user consent"""
        now = datetime.utcnow()

        cursor = self.db.cursor()

        for purpose in purposes:
            # Update status
            cursor.execute("""
                UPDATE user_consents
                SET status = %s, withdrawn_at = %s
                WHERE user_id = %s
                  AND purpose = %s
                  AND status = %s
            """, (
                ConsentStatus.WITHDRAWN.value,
                now,
                user_id,
                purpose.value,
                ConsentStatus.GRANTED.value
            ))

            # Emit event
            self._emit_consent_event(ConsentRecord(
                user_id=user_id,
                purpose=purpose,
                status=ConsentStatus.WITHDRAWN,
                granted_at=None,
                withdrawn_at=now,
                expires_at=None,
                method='explicit',
                ip_address='',
                user_agent='',
                version=self.get_current_policy_version()
            ))

            # Trigger data deletion workflow if required
            if purpose == ConsentPurpose.ESSENTIAL:
                self._trigger_account_deletion(user_id)
            else:
                self._trigger_purpose_data_deletion(user_id, purpose)

        self.db.commit()

    def get_user_consents(self, user_id: str) -> Dict[ConsentPurpose, ConsentRecord]:
        """Get all consents for user"""
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT purpose, status, granted_at, withdrawn_at, expires_at,
                   method, ip_address, user_agent, version
            FROM user_consents
            WHERE user_id = %s
            ORDER BY purpose, granted_at DESC
        """, (user_id,))

        consents = {}
        for row in cursor.fetchall():
            purpose = ConsentPurpose(row[0])
            if purpose not in consents:  # Get most recent
                consents[purpose] = ConsentRecord(
                    user_id=user_id,
                    purpose=purpose,
                    status=ConsentStatus(row[1]),
                    granted_at=row[2],
                    withdrawn_at=row[3],
                    expires_at=row[4],
                    method=row[5],
                    ip_address=row[6],
                    user_agent=row[7],
                    version=row[8]
                )

        return consents

    def export_consent_history(self, user_id: str) -> str:
        """Export consent history for user (GDPR Article 15)"""
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT purpose, status, granted_at, withdrawn_at, expires_at,
                   method, ip_address, user_agent, version, created_at
            FROM user_consents
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))

        history = []
        for row in cursor.fetchall():
            history.append({
                'purpose': row[0],
                'status': row[1],
                'granted_at': row[2].isoformat() if row[2] else None,
                'withdrawn_at': row[3].isoformat() if row[3] else None,
                'expires_at': row[4].isoformat() if row[4] else None,
                'method': row[5],
                'ip_address': row[6],
                'user_agent': row[7],
                'policy_version': row[8],
                'recorded_at': row[9].isoformat()
            })

        return json.dumps({
            'user_id': user_id,
            'consent_history': history,
            'exported_at': datetime.utcnow().isoformat()
        }, indent=2)

    def _store_consent(self, record: ConsentRecord):
        """Store consent in database"""
        cursor = self.db.cursor()

        cursor.execute("""
            INSERT INTO user_consents (
                user_id, purpose, status, granted_at, withdrawn_at,
                expires_at, method, ip_address, user_agent, version,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            record.user_id,
            record.purpose.value,
            record.status.value,
            record.granted_at,
            record.withdrawn_at,
            record.expires_at,
            record.method,
            record.ip_address,
            record.user_agent,
            record.version,
            datetime.utcnow()
        ))

        self.db.commit()

    def _emit_consent_event(self, record: ConsentRecord):
        """Emit consent event for audit trail"""
        from src.compliance.audit_logger import audit_logger

        audit_logger.log_event(
            event_type='consent_change',
            user_id=record.user_id,
            details={
                'purpose': record.purpose.value,
                'status': record.status.value,
                'method': record.method,
                'ip_address': record.ip_address
            }
        )

    def get_current_policy_version(self) -> str:
        """Get current privacy policy version"""
        # Could be from database, config, or file
        return "2.0"

    def _mark_consent_expired(self, user_id: str, purpose: ConsentPurpose):
        """Mark consent as expired"""
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE user_consents
            SET status = %s
            WHERE user_id = %s AND purpose = %s AND status = %s
        """, (
            ConsentStatus.EXPIRED.value,
            user_id,
            purpose.value,
            ConsentStatus.GRANTED.value
        ))
        self.db.commit()

    def _trigger_account_deletion(self, user_id: str):
        """Trigger full account deletion workflow"""
        from src.compliance.data_deletion import deletion_manager
        deletion_manager.schedule_account_deletion(user_id)

    def _trigger_purpose_data_deletion(self, user_id: str, purpose: ConsentPurpose):
        """Trigger deletion of data for specific purpose"""
        from src.compliance.data_deletion import deletion_manager
        deletion_manager.schedule_purpose_deletion(user_id, purpose)

# Global consent manager
consent_manager = ConsentManager(db_connection)
```

### 2.2 Consent Verification Middleware

```python
# src/compliance/consent_middleware.py
"""
Middleware to verify consent before processing
"""

from flask import request, jsonify, g
from functools import wraps

def require_consent(*purposes: ConsentPurpose):
    """Decorator to require consent for purposes"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = g.user_id  # From authentication

            # Check consent for all required purposes
            for purpose in purposes:
                if not consent_manager.check_consent(user_id, purpose):
                    return jsonify({
                        'error': 'Consent required',
                        'purpose': purpose.value,
                        'message': f'You must consent to {purpose.value} to use this feature'
                    }), 403

            return func(*args, **kwargs)

        return wrapper
    return decorator

# Usage example
@app.route('/api/personalized-recommendations')
@require_consent(ConsentPurpose.PERSONALIZATION, ConsentPurpose.ANALYTICS)
def get_recommendations():
    """Get personalized recommendations (requires consent)"""
    return jsonify({'recommendations': [...]})
```

---

## 3. Data Subject Rights (DSR) Architecture

### 3.1 Automated DSR Handler

```python
# src/compliance/dsr_handler.py
"""
Data Subject Rights (DSR) request handler
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import zipfile
import io

class DSRType(Enum):
    """Types of data subject requests"""
    ACCESS = "access"  # GDPR Article 15, CCPA Right to Know
    RECTIFICATION = "rectification"  # GDPR Article 16
    ERASURE = "erasure"  # GDPR Article 17, CCPA Right to Delete
    PORTABILITY = "portability"  # GDPR Article 20
    RESTRICTION = "restriction"  # GDPR Article 18
    OBJECTION = "objection"  # GDPR Article 21, CCPA Right to Opt-Out
    AUTOMATED_DECISION = "automated_decision"  # GDPR Article 22

class DSRStatus(Enum):
    """DSR request status"""
    RECEIVED = "received"
    VERIFIED = "verified"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"

@dataclass
class DSRRequest:
    """Data subject rights request"""
    request_id: str
    user_id: str
    request_type: DSRType
    status: DSRStatus
    created_at: datetime
    verified_at: Optional[datetime]
    completed_at: Optional[datetime]
    details: Dict[str, Any]
    response_data: Optional[str]

class DSRHandler:
    """Handles data subject rights requests"""

    # GDPR requires response within 30 days
    RESPONSE_DEADLINE_DAYS = 30

    def __init__(self, db_connection):
        self.db = db_connection

    def submit_request(
        self,
        user_id: str,
        request_type: DSRType,
        details: Dict[str, Any]
    ) -> DSRRequest:
        """Submit new DSR request"""
        request_id = self._generate_request_id()

        request = DSRRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            status=DSRStatus.RECEIVED,
            created_at=datetime.utcnow(),
            verified_at=None,
            completed_at=None,
            details=details,
            response_data=None
        )

        # Store request
        self._store_request(request)

        # Send verification email
        self._send_verification_email(request)

        # Log in audit trail
        self._log_dsr_event(request, 'request_submitted')

        return request

    def verify_request(self, request_id: str, verification_token: str) -> bool:
        """Verify DSR request identity"""
        # Verify token
        if not self._verify_token(request_id, verification_token):
            return False

        # Update request status
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE dsr_requests
            SET status = %s, verified_at = %s
            WHERE request_id = %s
        """, (DSRStatus.VERIFIED.value, datetime.utcnow(), request_id))
        self.db.commit()

        # Process request asynchronously
        request = self.get_request(request_id)
        self._process_request_async(request)

        return True

    def process_request(self, request: DSRRequest) -> DSRRequest:
        """Process verified DSR request"""
        # Update status
        self._update_status(request.request_id, DSRStatus.PROCESSING)

        try:
            # Route to appropriate handler
            if request.request_type == DSRType.ACCESS:
                response_data = self._handle_access_request(request)
            elif request.request_type == DSRType.ERASURE:
                response_data = self._handle_erasure_request(request)
            elif request.request_type == DSRType.PORTABILITY:
                response_data = self._handle_portability_request(request)
            elif request.request_type == DSRType.RECTIFICATION:
                response_data = self._handle_rectification_request(request)
            elif request.request_type == DSRType.RESTRICTION:
                response_data = self._handle_restriction_request(request)
            elif request.request_type == DSRType.OBJECTION:
                response_data = self._handle_objection_request(request)
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")

            # Mark as completed
            self._complete_request(request.request_id, response_data)

            # Send notification
            self._send_completion_email(request)

            request.status = DSRStatus.COMPLETED
            request.completed_at = datetime.utcnow()
            request.response_data = response_data

        except Exception as e:
            # Log error
            print(f"Error processing DSR request {request.request_id}: {e}")

            # Mark as rejected
            self._update_status(request.request_id, DSRStatus.REJECTED)
            request.status = DSRStatus.REJECTED

        return request

    def _handle_access_request(self, request: DSRRequest) -> str:
        """Handle right to access (GDPR Article 15)"""
        user_id = request.user_id

        # Collect all user data
        user_data = {
            'user_profile': self._get_user_profile(user_id),
            'llm_requests': self._get_llm_request_history(user_id),
            'consent_history': self._get_consent_history(user_id),
            'api_keys': self._get_api_keys(user_id),
            'billing_info': self._get_billing_info(user_id),
            'support_tickets': self._get_support_tickets(user_id),
        }

        # Create data package
        package = self._create_data_package(user_id, user_data)

        # Store package
        package_path = self._store_data_package(request.request_id, package)

        return package_path

    def _handle_erasure_request(self, request: DSRRequest) -> str:
        """Handle right to erasure (GDPR Article 17)"""
        user_id = request.user_id

        # Create deletion record before erasing
        deletion_record = self._create_deletion_record(user_id)

        # Delete user data from all systems
        deleted_items = []

        # 1. User profile
        self._delete_user_profile(user_id)
        deleted_items.append('user_profile')

        # 2. LLM request history
        self._delete_llm_requests(user_id)
        deleted_items.append('llm_requests')

        # 3. API keys
        self._delete_api_keys(user_id)
        deleted_items.append('api_keys')

        # 4. Billing info (keep for legal requirement)
        # Note: Some data must be retained for legal/accounting
        deleted_items.append('billing_info_anonymized')

        # 5. Support tickets (anonymize)
        self._anonymize_support_tickets(user_id)
        deleted_items.append('support_tickets_anonymized')

        # Generate deletion certificate
        certificate = {
            'request_id': request.request_id,
            'user_id': user_id,
            'deleted_at': datetime.utcnow().isoformat(),
            'deleted_items': deleted_items,
            'deletion_record_id': deletion_record
        }

        return json.dumps(certificate, indent=2)

    def _handle_portability_request(self, request: DSRRequest) -> str:
        """Handle right to data portability (GDPR Article 20)"""
        user_id = request.user_id

        # Export data in machine-readable format (JSON)
        user_data = {
            'user_profile': self._get_user_profile(user_id),
            'llm_requests': self._get_llm_request_history(user_id, format='json'),
            'preferences': self._get_user_preferences(user_id),
        }

        # Create ZIP archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add JSON files
            zip_file.writestr('profile.json', json.dumps(user_data['user_profile'], indent=2))
            zip_file.writestr('requests.json', json.dumps(user_data['llm_requests'], indent=2))
            zip_file.writestr('preferences.json', json.dumps(user_data['preferences'], indent=2))

            # Add README
            zip_file.writestr('README.txt', self._generate_portability_readme())

        # Store archive
        archive_path = self._store_archive(request.request_id, zip_buffer.getvalue())

        return archive_path

    def _handle_rectification_request(self, request: DSRRequest) -> str:
        """Handle right to rectification (GDPR Article 16)"""
        user_id = request.user_id
        corrections = request.details.get('corrections', {})

        # Apply corrections
        corrected_fields = []
        for field, new_value in corrections.items():
            if self._update_user_field(user_id, field, new_value):
                corrected_fields.append(field)

        # Log changes
        self._log_rectification(user_id, corrected_fields)

        return json.dumps({
            'corrected_fields': corrected_fields,
            'corrected_at': datetime.utcnow().isoformat()
        })

    def _handle_restriction_request(self, request: DSRRequest) -> str:
        """Handle right to restriction (GDPR Article 18)"""
        user_id = request.user_id

        # Mark account as restricted
        self._restrict_account(user_id)

        return json.dumps({
            'status': 'restricted',
            'restricted_at': datetime.utcnow().isoformat()
        })

    def _handle_objection_request(self, request: DSRRequest) -> str:
        """Handle right to object (GDPR Article 21)"""
        user_id = request.user_id
        purpose = request.details.get('purpose')

        # Withdraw consent for purpose
        consent_manager.withdraw_consent(user_id, [ConsentPurpose(purpose)])

        return json.dumps({
            'objection_recorded': True,
            'purpose': purpose,
            'recorded_at': datetime.utcnow().isoformat()
        })

    def get_request(self, request_id: str) -> Optional[DSRRequest]:
        """Get DSR request by ID"""
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT request_id, user_id, request_type, status,
                   created_at, verified_at, completed_at, details, response_data
            FROM dsr_requests
            WHERE request_id = %s
        """, (request_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return DSRRequest(
            request_id=row[0],
            user_id=row[1],
            request_type=DSRType(row[2]),
            status=DSRStatus(row[3]),
            created_at=row[4],
            verified_at=row[5],
            completed_at=row[6],
            details=json.loads(row[7]),
            response_data=row[8]
        )

    def check_deadline(self, request: DSRRequest) -> Dict[str, Any]:
        """Check if request is within deadline"""
        deadline = request.created_at + timedelta(days=self.RESPONSE_DEADLINE_DAYS)
        days_remaining = (deadline - datetime.utcnow()).days

        return {
            'deadline': deadline.isoformat(),
            'days_remaining': days_remaining,
            'overdue': days_remaining < 0
        }

    # Helper methods
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import secrets
        return f"dsr_{secrets.token_urlsafe(16)}"

    def _store_request(self, request: DSRRequest):
        """Store DSR request in database"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO dsr_requests (
                request_id, user_id, request_type, status,
                created_at, details
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            request.request_id,
            request.user_id,
            request.request_type.value,
            request.status.value,
            request.created_at,
            json.dumps(request.details)
        ))
        self.db.commit()

    def _update_status(self, request_id: str, status: DSRStatus):
        """Update request status"""
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE dsr_requests
            SET status = %s
            WHERE request_id = %s
        """, (status.value, request_id))
        self.db.commit()

    def _complete_request(self, request_id: str, response_data: str):
        """Mark request as completed"""
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE dsr_requests
            SET status = %s, completed_at = %s, response_data = %s
            WHERE request_id = %s
        """, (
            DSRStatus.COMPLETED.value,
            datetime.utcnow(),
            response_data,
            request_id
        ))
        self.db.commit()

# Global DSR handler
dsr_handler = DSRHandler(db_connection)
```

---

## 4. Audit Logging Architecture

### 4.1 Immutable Audit Trail

```python
# src/compliance/audit_logger.py
"""
Immutable audit logging for compliance
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

@dataclass
class AuditEvent:
    """Audit log event"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    previous_hash: Optional[str]
    current_hash: str

class ImmutableAuditLogger:
    """Blockchain-style immutable audit logger"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.last_hash = self._get_last_hash()

    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Dict[str, Any] = None
    ) -> AuditEvent:
        """Log audit event"""
        import secrets

        event = AuditEvent(
            event_id=f"audit_{secrets.token_urlsafe(16)}",
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            previous_hash=self.last_hash,
            current_hash=None  # Will be calculated
        )

        # Calculate hash (blockchain-style)
        event.current_hash = self._calculate_hash(event)

        # Store event
        self._store_event(event)

        # Update last hash
        self.last_hash = event.current_hash

        return event

    def verify_chain(self, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify integrity of audit chain"""
        cursor = self.db.cursor()

        if start_date:
            cursor.execute("""
                SELECT event_id, timestamp, event_type, user_id, ip_address,
                       details, previous_hash, current_hash
                FROM audit_log
                WHERE timestamp >= %s
                ORDER BY timestamp ASC
            """, (start_date,))
        else:
            cursor.execute("""
                SELECT event_id, timestamp, event_type, user_id, ip_address,
                       details, previous_hash, current_hash
                FROM audit_log
                ORDER BY timestamp ASC
            """)

        events = []
        for row in cursor.fetchall():
            events.append(AuditEvent(
                event_id=row[0],
                timestamp=row[1],
                event_type=row[2],
                user_id=row[3],
                ip_address=row[4],
                details=json.loads(row[5]),
                previous_hash=row[6],
                current_hash=row[7]
            ))

        # Verify chain
        tampered = []
        for i, event in enumerate(events):
            # Recalculate hash
            calculated_hash = self._calculate_hash(event)

            if calculated_hash != event.current_hash:
                tampered.append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'expected_hash': event.current_hash,
                    'calculated_hash': calculated_hash
                })

            # Check chain link
            if i > 0:
                if event.previous_hash != events[i-1].current_hash:
                    tampered.append({
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'error': 'Broken chain link'
                    })

        return {
            'verified': len(tampered) == 0,
            'total_events': len(events),
            'tampered_events': tampered
        }

    def export_audit_trail(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """Export audit trail for compliance reporting"""
        cursor = self.db.cursor()

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)

        events = []
        for row in cursor.fetchall():
            events.append({
                'event_id': row[0],
                'timestamp': row[1].isoformat(),
                'event_type': row[2],
                'user_id': row[3],
                'ip_address': row[4],
                'details': json.loads(row[5]),
                'hash': row[7]
            })

        return json.dumps({
            'export_date': datetime.utcnow().isoformat(),
            'total_events': len(events),
            'events': events
        }, indent=2)

    def _calculate_hash(self, event: AuditEvent) -> str:
        """Calculate hash for event"""
        data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'details': event.details,
            'previous_hash': event.previous_hash
        }

        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _store_event(self, event: AuditEvent):
        """Store event in database"""
        cursor = self.db.cursor()

        cursor.execute("""
            INSERT INTO audit_log (
                event_id, timestamp, event_type, user_id, ip_address,
                details, previous_hash, current_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event.event_id,
            event.timestamp,
            event.event_type,
            event.user_id,
            event.ip_address,
            json.dumps(event.details),
            event.previous_hash,
            event.current_hash
        ))

        self.db.commit()

    def _get_last_hash(self) -> Optional[str]:
        """Get last hash in chain"""
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT current_hash
            FROM audit_log
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        return row[0] if row else None

# Global audit logger
audit_logger = ImmutableAuditLogger(db_connection)
```

---

## 5. Data Retention and Deletion Architecture

### 5.1 Automated Data Lifecycle Management

```python
# src/compliance/data_lifecycle.py
"""
Automated data lifecycle management
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class DataCategory(Enum):
    """Data categories with different retention periods"""
    USER_PROFILE = "user_profile"  # As long as account active
    LLM_REQUESTS = "llm_requests"  # 90 days
    LOGS = "logs"  # 365 days
    AUDIT_LOGS = "audit_logs"  # 7 years (HIPAA)
    BILLING = "billing"  # 7 years (tax law)
    CONSENT_RECORDS = "consent_records"  # 7 years
    METRICS = "metrics"  # 2 years

@dataclass
class RetentionPolicy:
    """Data retention policy"""
    category: DataCategory
    retention_days: int
    deletion_method: str  # 'hard_delete', 'soft_delete', 'anonymize'
    legal_basis: str

class DataLifecycleManager:
    """Manages data lifecycle and retention"""

    RETENTION_POLICIES = {
        DataCategory.USER_PROFILE: RetentionPolicy(
            category=DataCategory.USER_PROFILE,
            retention_days=0,  # Keep while account active
            deletion_method='hard_delete',
            legal_basis='User account closure'
        ),
        DataCategory.LLM_REQUESTS: RetentionPolicy(
            category=DataCategory.LLM_REQUESTS,
            retention_days=90,
            deletion_method='anonymize',
            legal_basis='Service improvement'
        ),
        DataCategory.LOGS: RetentionPolicy(
            category=DataCategory.LOGS,
            retention_days=365,
            deletion_method='hard_delete',
            legal_basis='Operational requirements'
        ),
        DataCategory.AUDIT_LOGS: RetentionPolicy(
            category=DataCategory.AUDIT_LOGS,
            retention_days=2557,  # 7 years
            deletion_method='archive',
            legal_basis='HIPAA/Legal compliance'
        ),
        DataCategory.BILLING: RetentionPolicy(
            category=DataCategory.BILLING,
            retention_days=2557,  # 7 years
            deletion_method='archive',
            legal_basis='Tax law'
        ),
    }

    def __init__(self, db_connection):
        self.db = db_connection

    def schedule_deletion(self, user_id: str, category: DataCategory):
        """Schedule data for deletion"""
        policy = self.RETENTION_POLICIES[category]

        deletion_date = datetime.utcnow() + timedelta(days=policy.retention_days)

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO scheduled_deletions (
                user_id, category, deletion_date, deletion_method, scheduled_at
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            user_id,
            category.value,
            deletion_date,
            policy.deletion_method,
            datetime.utcnow()
        ))
        self.db.commit()

    def process_scheduled_deletions(self):
        """Process all due deletions"""
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT id, user_id, category, deletion_method
            FROM scheduled_deletions
            WHERE deletion_date <= %s
              AND status = 'pending'
        """, (datetime.utcnow(),))

        for row in cursor.fetchall():
            deletion_id, user_id, category, method = row

            try:
                # Execute deletion
                if method == 'hard_delete':
                    self._hard_delete(user_id, DataCategory(category))
                elif method == 'soft_delete':
                    self._soft_delete(user_id, DataCategory(category))
                elif method == 'anonymize':
                    self._anonymize(user_id, DataCategory(category))
                elif method == 'archive':
                    self._archive(user_id, DataCategory(category))

                # Mark as completed
                cursor.execute("""
                    UPDATE scheduled_deletions
                    SET status = 'completed', completed_at = %s
                    WHERE id = %s
                """, (datetime.utcnow(), deletion_id))

                # Log in audit trail
                audit_logger.log_event(
                    event_type='data_deleted',
                    user_id=user_id,
                    details={
                        'category': category,
                        'method': method
                    }
                )

            except Exception as e:
                print(f"Error deleting data for {user_id}/{category}: {e}")

                # Mark as failed
                cursor.execute("""
                    UPDATE scheduled_deletions
                    SET status = 'failed', error = %s
                    WHERE id = %s
                """, (str(e), deletion_id))

        self.db.commit()

    def _hard_delete(self, user_id: str, category: DataCategory):
        """Permanently delete data"""
        cursor = self.db.cursor()

        if category == DataCategory.USER_PROFILE:
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
        elif category == DataCategory.LLM_REQUESTS:
            cursor.execute("DELETE FROM llm_requests WHERE user_id = %s", (user_id,))
        elif category == DataCategory.LOGS:
            cursor.execute("DELETE FROM logs WHERE user_id = %s", (user_id,))

        self.db.commit()

    def _soft_delete(self, user_id: str, category: DataCategory):
        """Mark as deleted (keep for recovery period)"""
        cursor = self.db.cursor()

        cursor.execute("""
            UPDATE users
            SET deleted = TRUE, deleted_at = %s
            WHERE user_id = %s
        """, (datetime.utcnow(), user_id))

        self.db.commit()

    def _anonymize(self, user_id: str, category: DataCategory):
        """Anonymize data (remove PII but keep for analytics)"""
        cursor = self.db.cursor()

        if category == DataCategory.LLM_REQUESTS:
            # Replace user_id with anonymous ID
            anon_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]

            cursor.execute("""
                UPDATE llm_requests
                SET user_id = %s, anonymized = TRUE
                WHERE user_id = %s
            """, (f"anon_{anon_id}", user_id))

        self.db.commit()

    def _archive(self, user_id: str, category: DataCategory):
        """Archive data to cold storage"""
        # Export to S3 Glacier or similar
        cursor = self.db.cursor()

        if category == DataCategory.AUDIT_LOGS:
            cursor.execute("""
                SELECT * FROM audit_log
                WHERE user_id = %s
            """, (user_id,))

            data = cursor.fetchall()

            # Upload to archive storage
            self._upload_to_archive(user_id, category, data)

            # Delete from primary storage
            cursor.execute("""
                DELETE FROM audit_log
                WHERE user_id = %s
            """, (user_id,))

        self.db.commit()

    def _upload_to_archive(self, user_id: str, category: DataCategory, data: List):
        """Upload data to archive storage (S3 Glacier)"""
        import boto3

        s3 = boto3.client('s3')

        # Create archive file
        archive_data = json.dumps(data, default=str)
        filename = f"archive/{user_id}/{category.value}/{datetime.utcnow().strftime('%Y%m%d')}.json"

        s3.put_object(
            Bucket='compliance-archives',
            Key=filename,
            Body=archive_data,
            StorageClass='GLACIER'
        )

# Global lifecycle manager
lifecycle_manager = DataLifecycleManager(db_connection)
```

---

## 6. Cross-Border Data Transfer Architecture

### 6.1 Data Residency Manager

```python
# src/compliance/data_residency.py
"""
Manages data residency for cross-border transfers
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class DataRegion(Enum):
    """Data storage regions"""
    EU = "eu"  # European Union
    US = "us"  # United States
    UK = "uk"  # United Kingdom
    CA = "ca"  # Canada
    JP = "jp"  # Japan
    AU = "au"  # Australia

class TransferMechanism(Enum):
    """GDPR transfer mechanisms"""
    ADEQUACY_DECISION = "adequacy_decision"  # Article 45
    STANDARD_CONTRACTUAL_CLAUSES = "scc"  # Article 46
    BINDING_CORPORATE_RULES = "bcr"  # Article 47
    CONSENT = "consent"  # Article 49

@dataclass
class DataResidencyRule:
    """Data residency rules"""
    user_region: DataRegion
    storage_region: DataRegion
    allowed: bool
    transfer_mechanism: Optional[TransferMechanism]
    legal_basis: str

class DataResidencyManager:
    """Manages data residency and cross-border transfers"""

    # GDPR adequacy decisions (countries with adequate protection)
    ADEQUATE_COUNTRIES = [
        DataRegion.EU,
        DataRegion.UK,
        DataRegion.CA,
        DataRegion.JP,
    ]

    RESIDENCY_RULES = {
        # EU users: data must stay in EU or adequate countries
        (DataRegion.EU, DataRegion.EU): DataResidencyRule(
            user_region=DataRegion.EU,
            storage_region=DataRegion.EU,
            allowed=True,
            transfer_mechanism=None,
            legal_basis='Same region'
        ),
        (DataRegion.EU, DataRegion.US): DataResidencyRule(
            user_region=DataRegion.EU,
            storage_region=DataRegion.US,
            allowed=True,
            transfer_mechanism=TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES,
            legal_basis='SCCs in place'
        ),
        # Add more rules...
    }

    def __init__(self):
        self.region_databases = {
            DataRegion.EU: 'postgresql://eu-db',
            DataRegion.US: 'postgresql://us-db',
            DataRegion.UK: 'postgresql://uk-db',
        }

    def get_storage_region(self, user_region: DataRegion) -> DataRegion:
        """Determine where to store user data"""
        # EU users must have data in EU
        if user_region == DataRegion.EU:
            return DataRegion.EU

        # UK users can be in EU or UK
        if user_region == DataRegion.UK:
            return DataRegion.EU  # or DataRegion.UK

        # US users in US
        if user_region == DataRegion.US:
            return DataRegion.US

        # Default to US with consent
        return DataRegion.US

    def check_transfer_allowed(
        self,
        from_region: DataRegion,
        to_region: DataRegion
    ) -> Dict:
        """Check if data transfer is allowed"""
        rule = self.RESIDENCY_RULES.get((from_region, to_region))

        if not rule:
            return {
                'allowed': False,
                'reason': 'No transfer rule defined'
            }

        if not rule.allowed:
            return {
                'allowed': False,
                'reason': 'Transfer not permitted'
            }

        return {
            'allowed': True,
            'transfer_mechanism': rule.transfer_mechanism.value if rule.transfer_mechanism else None,
            'legal_basis': rule.legal_basis
        }

    def get_database_connection(self, user_region: DataRegion):
        """Get database connection for user's region"""
        storage_region = self.get_storage_region(user_region)
        return self.region_databases[storage_region]

# Global residency manager
residency_manager = DataResidencyManager()
```

---

## 7. Compliance Monitoring and Reporting

### 7.1 Compliance Dashboard

```python
# src/compliance/compliance_monitor.py
"""
Monitors compliance metrics and generates reports
"""

from typing import Dict, List
from datetime import datetime, timedelta

class ComplianceMonitor:
    """Monitors compliance metrics"""

    def __init__(self, db_connection):
        self.db = db_connection

    def get_compliance_metrics(self) -> Dict:
        """Get current compliance metrics"""
        cursor = self.db.cursor()

        # DSR response time
        cursor.execute("""
            SELECT AVG(EXTRACT(EPOCH FROM (completed_at - created_at)) / 86400)
            FROM dsr_requests
            WHERE status = 'completed'
              AND created_at >= NOW() - INTERVAL '30 days'
        """)
        avg_dsr_response_days = cursor.fetchone()[0] or 0

        # Consent rate
        cursor.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'granted') * 100.0 / COUNT(*)
            FROM user_consents
            WHERE created_at >= NOW() - INTERVAL '30 days'
        """)
        consent_rate = cursor.fetchone()[0] or 0

        # Data breaches
        cursor.execute("""
            SELECT COUNT(*)
            FROM security_incidents
            WHERE incident_type = 'data_breach'
              AND created_at >= NOW() - INTERVAL '30 days'
        """)
        data_breaches = cursor.fetchone()[0] or 0

        # Audit log integrity
        integrity_result = audit_logger.verify_chain(
            start_date=datetime.utcnow() - timedelta(days=30)
        )

        return {
            'dsr_response_time_days': round(avg_dsr_response_days, 1),
            'dsr_response_target_days': 30,
            'dsr_compliance': avg_dsr_response_days <= 30,
            'consent_rate_percent': round(consent_rate, 1),
            'data_breaches_30days': data_breaches,
            'audit_log_integrity': integrity_result['verified'],
            'timestamp': datetime.utcnow().isoformat()
        }

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate compliance report for period"""
        cursor = self.db.cursor()

        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.utcnow().isoformat()
        }

        # DSR statistics
        cursor.execute("""
            SELECT
                request_type,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                AVG(EXTRACT(EPOCH FROM (completed_at - created_at)) / 86400) as avg_days
            FROM dsr_requests
            WHERE created_at BETWEEN %s AND %s
            GROUP BY request_type
        """, (start_date, end_date))

        dsr_stats = []
        for row in cursor.fetchall():
            dsr_stats.append({
                'type': row[0],
                'total': row[1],
                'completed': row[2],
                'avg_response_days': round(row[3], 1) if row[3] else 0
            })

        report['dsr_statistics'] = dsr_stats

        # Consent statistics
        cursor.execute("""
            SELECT
                purpose,
                COUNT(*) FILTER (WHERE status = 'granted') as granted,
                COUNT(*) FILTER (WHERE status = 'denied') as denied,
                COUNT(*) FILTER (WHERE status = 'withdrawn') as withdrawn
            FROM user_consents
            WHERE created_at BETWEEN %s AND %s
            GROUP BY purpose
        """, (start_date, end_date))

        consent_stats = []
        for row in cursor.fetchall():
            consent_stats.append({
                'purpose': row[0],
                'granted': row[1],
                'denied': row[2],
                'withdrawn': row[3]
            })

        report['consent_statistics'] = consent_stats

        # Data breaches
        cursor.execute("""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE notified_within_72h = TRUE)
            FROM data_breaches
            WHERE occurred_at BETWEEN %s AND %s
        """, (start_date, end_date))

        breach_count, notified_on_time = cursor.fetchone()
        report['data_breaches'] = {
            'total': breach_count,
            'notified_within_72h': notified_on_time,
            'gdpr_compliant': breach_count == notified_on_time
        }

        return report

# Global compliance monitor
compliance_monitor = ComplianceMonitor(db_connection)
```

---

## 8. Compliance Architecture Checklist

**Data Governance:**
- [ ] Data inventory and classification
- [ ] Data flow mapping
- [ ] Privacy impact assessments (PIAs)
- [ ] Data protection officer (DPO) designated
- [ ] Data processing records maintained (Article 30)

**Consent Management:**
- [ ] Consent collection mechanism
- [ ] Purpose specification
- [ ] Consent withdrawal process
- [ ] Consent versioning
- [ ] Consent audit trail

**Data Subject Rights:**
- [ ] Access request automation (Article 15)
- [ ] Rectification process (Article 16)
- [ ] Erasure process (Article 17)
- [ ] Data portability (Article 20)
- [ ] 30-day response SLA
- [ ] Identity verification

**Audit and Logging:**
- [ ] Immutable audit logs
- [ ] 7-year retention for audit logs
- [ ] Chain-of-custody for data
- [ ] Tamper detection
- [ ] Export capability

**Data Retention:**
- [ ] Retention policies defined
- [ ] Automated deletion schedules
- [ ] Legal hold process
- [ ] Anonymization procedures
- [ ] Archive storage (cold)

**Cross-Border Transfers:**
- [ ] Data residency rules
- [ ] Standard Contractual Clauses (SCCs)
- [ ] Transfer impact assessments
- [ ] User region detection
- [ ] Regional data storage

**Monitoring and Reporting:**
- [ ] Compliance dashboard
- [ ] DSR response time tracking
- [ ] Consent rate monitoring
- [ ] Breach notification system
- [ ] Regular compliance reports

**Documentation:**
- [ ] Privacy policy published
- [ ] Cookie policy published
- [ ] Data processing agreements (DPAs)
- [ ] Records of processing activities (ROPA)
- [ ] Incident response plan

---

## 9. Summary

This guide provides comprehensive compliance architecture for AI applications:

**Key Architectures:**
1. **Privacy by Design** - Embedded privacy from ground up
2. **Consent Management** - Granular consent with audit trail
3. **DSR Automation** - Automated handling of data subject requests
4. **Immutable Audit Logs** - Blockchain-style tamper-proof logging
5. **Data Lifecycle Management** - Automated retention and deletion
6. **Data Residency** - Region-aware data storage
7. **Compliance Monitoring** - Real-time metrics and reporting

**Core Principles:**
- 🔒 **Privacy by design** (proactive not reactive)
- ✅ **Consent-based** (granular, revocable)
- ⚡ **Automated DSR** (30-day SLA compliance)
- 📊 **Immutable audit** (tamper-proof records)
- 🌍 **Data residency** (GDPR-compliant transfers)
- 📈 **Continuous monitoring** (compliance metrics)

**Related Documentation:**
- [Compliance Guide](COMPLIANCE.md) - Regulatory requirements
- [Security Architecture](SECURITY_ARCHITECTURE.md) - Security infrastructure
- [Observability Architecture](OBSERVABILITY_ARCHITECTURE.md) - Monitoring
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Status:** Active
