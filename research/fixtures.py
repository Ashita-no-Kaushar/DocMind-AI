"""Deterministic offline corpus, ground truth, and lexical retriever.

The ablation study runs without Ollama, embeddings, or any network access so
that every reported number is reproducible from this file alone. The retrieval
backend is a plain BM25 scorer that mirrors the ``HybridRetriever`` interface
(``corpus``, ``docstore``, ``retrieve(query, allowed_node_ids=None)``) so the
map builder and the map agent exercise their production code paths unchanged.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from llama_index.core.schema import NodeWithScore, TextNode

CHUNKS_PER_SECTION = 2
WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "may",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)

BM25_K1 = 1.2
BM25_B = 0.75

DOCUMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "refund_policy.txt",
        (
            "Refund Policy Overview. Customers may request a full refund within 30 days "
            "of the original purchase date. Refund eligibility is evaluated by the "
            "billing system against the order record and the payment capture timestamp.",
            "Refund Window Tiers. Standard merchandise is refundable for 30 days. "
            "Electronics and perishable goods are refundable for 15 days only. "
            "Digital licences become non refundable immediately after first download.",
            "Refund Processing Timeline. Approved refunds are submitted to the payment "
            "processor within 2 business days. The processor settles refunds within 5 "
            "business days. Settlement delays are tracked by the finance reconciliation "
            "queue and escalated to the payments on call rotation.",
            "Refund Eligibility Exceptions. Promotional orders, gift card purchases, "
            "and duplicate charges are handled by the exceptions desk. Chargeback "
            "responses are prepared by the disputes team within the issuer deadline.",
            "Refund Request Evidence. A refund request requires the order identifier, "
            "the purchase date, and the reason for the request. Photographic evidence "
            "is required for damaged physical goods claims.",
            "Refund Communication. Refund confirmations are emailed to the billing "
            "address on the order. Support agents must not promise a settlement date "
            "that differs from the published refund processing timeline.",
        ),
    ),
    (
        "shipping_times.txt",
        (
            "Shipping Time Overview. Domestic ground shipping completes in 3 to 5 "
            "business days after the carrier receives the parcel. Orders ship from the "
            "fulfilment centre closest to the delivery postcode.",
            "Expedited Shipping Options. Expedited delivery completes in 2 business "
            "days and requires an order placed before 14:00 local time. Overnight "
            "delivery is offered on a subset of metropolitan postcodes.",
            "International Shipping. International parcels clear customs within 7 to "
            "12 business days. Import duties are assessed by the destination country "
            "and are not included in the shipping quote.",
            "Carrier Handoff and Scanning. The carrier scan event marks the parcel as "
            "in transit. Delivery exceptions such as address corrections require a "
            "carrier appointment and may add one additional business day.",
            "Delayed Shipment Handling. A shipment is considered delayed when the "
            "carrier scan shows no movement for 72 hours. The logistics operations "
            "team reviews delayed shipments every weekday morning.",
            "Packaging Standards. Parcels must use the standard box sizes and must "
            "include the packing slip. Fragile items require the padded insert and the "
            "fragile label on the outer carton.",
        ),
    ),
    (
        "warranty_terms.txt",
        (
            "Warranty Coverage Overview. The manufacturer warranty covers defects in "
            "materials and workmanship for 24 months from the delivery date. Cosmetic "
            "wear and customer induced damage are excluded from coverage.",
            "Warranty Claim Process. A warranty claim requires the serial number, the "
            "delivery date, and a description of the fault. Claims are triaged by the "
            "service desk and assigned a claim identifier within 3 business days.",
            "Warranty Repair and Replacement. Products are repaired in house when the "
            "repair is completed within 10 business days. Slow repairs are replaced "
            "instead of repaired to avoid extended customer downtime.",
            "Warranty Exclusions. Water damage, unauthorised disassembly, and "
            "consumable batteries are not covered. Software configuration errors are "
            "not warranty defects and are handled by the support team.",
            "Warranty Transfer. Warranty coverage transfers with the product for its "
            "full remaining term. Resale does not create a new warranty period and "
            "does not extend the original coverage end date.",
            "Warranty Documentation. The warranty certificate is issued at purchase and "
            "is available from the account portal. The certificate records the model, "
            "the serial number, and the coverage end date.",
        ),
    ),
    (
        "incident_runbook.txt",
        (
            "Incident Severity Definitions. A severity 1 incident is a full outage "
            "affecting all customers. A severity 2 incident is a partial degradation "
            "affecting a large fraction of traffic. A severity 3 incident is limited "
            "in scope with a documented workaround available.",
            "Incident Commander Responsibilities. The incident commander owns the "
            "response, assigns roles, and approves the customer communication. The "
            "commander is never the engineer performing the mitigation directly.",
            "Detection and Alerting. Alerting covers availability, latency, and error "
            "rate. Alerts route to the on call rotation through the paging provider. "
            "Flapping alerts are suppressed for 10 minutes before escalation.",
            "Mitigation Strategy. Mitigation prefers a rollback of the most recent "
            "deployment over a forward fix during a severity 1 incident. Feature flags "
            "are used to disable the affected code path while the fix is validated.",
            "Communication Cadence. A severity 1 incident posts an internal update "
            "every 30 minutes and a public status update every 60 minutes. The public "
            "update states impact, mitigation, and the next update time.",
            "Postmortem Process. Every severity 1 and severity 2 incident produces a "
            "written postmortem within 5 business days. The postmortem is blameless "
            "and lists corrective actions with named owners and due dates.",
        ),
    ),
    (
        "release_checklist.txt",
        (
            "Release Checklist Overview. A release is a versioned build promoted "
            "through the staging and production environments. Every release requires a "
            "completed checklist attached to the change record.",
            "Pre Release Verification. The verification step runs the automated test "
            "suite, the dependency audit, and the migration dry run. A failing "
            "verification step blocks the release without an exception.",
            "Staging Promotion. Promotion to staging requires the release manager "
            "approval. Staging soak time is 30 minutes with error rate monitoring "
            "before production promotion is allowed.",
            "Database Migration Review. Schema changes require backward compatible "
            "migrations. Destructive changes are split into a separate follow up "
            "release after the application code has been deployed and observed.",
            "Production Promotion. Production promotion uses a staged rollout with 5 "
            "percent traffic for 15 minutes, then 25 percent, then full traffic. The "
            "on call engineer observes dashboards during every stage.",
            "Rollback Plan. The rollback plan names the previous version identifier and "
            "the trigger conditions for rollback. Rollback is executed by the on call "
            "engineer without further approval when a trigger condition is met.",
        ),
    ),
    (
        "payroll_policy.txt",
        (
            "Payroll Calendar. Payroll runs on the last working day of each month. "
            "Payments for leavers in the month of their final working day are included "
            "in that run when the submission cutoff allows.",
            "Timesheet Submission. Employees submit timesheets by the third working "
            "day of the following month. Late or missing timesheets are escalated to "
            "the line manager before payroll is calculated.",
            "Overtime Calculation. Overtime is paid at the contractual multiplier for "
            "hours worked beyond the contracted weekly hours. Unapproved overtime is "
            "not payable and requires manager approval recorded before the cutoff.",
            "Salary Review Cycle. Salary reviews happen annually with effect from the "
            "first of July. Off cycle adjustments are approved by the department head "
            "and the people operations team.",
            "Payslip Delivery. Payslips are delivered to the employee portal and are "
            "not emailed. A payslip correction reissues the document with a revision "
            "number and a note describing the change.",
            "Statutory Reporting. Payroll exports feed the statutory reporting return. "
            "The export is reconciled against the payroll run total before it is "
            "submitted to the reporting authority.",
        ),
    ),
    (
        "api_rate_limits.txt",
        (
            "API Rate Limit Policy. The API applies a per token request quota measured "
            "over a rolling 60 second window. The default quota is 600 requests per "
            "minute for the standard tier.",
            "Burst Behaviour. A burst allowance permits short spikes above the steady "
            "state quota. When the burst allowance is exhausted the API returns HTTP "
            "429 with a Retry-After header indicating the next permitted attempt.",
            "Rate Limit Headers. Every response includes the remaining quota in the "
            "rate limit headers. Clients should read the header rather than assuming "
            "the steady state quota applies to their endpoint.",
            "Concurrency Limits. The concurrency limit caps the number of in flight "
            "requests per token. Requests above the concurrency limit are queued for a "
            "short grace period and then rejected with HTTP 503.",
            "Quota Increase Requests. A quota increase requires a written justification "
            "describing the workload and the expected peak rate. Increases are granted "
            "for a fixed period and are reviewed before renewal.",
            "Client Retry Guidance. Clients should use exponential backoff with "
            "jitter. A retry loop without backoff can exhaust the burst allowance and "
            "extend the throttling window for the whole token.",
        ),
    ),
    (
        "access_control.txt",
        (
            "Access Control Principles. Access follows least privilege and is granted "
            "per role rather than per person. Access is reviewed quarterly and is "
            "removed automatically when the role no longer requires it.",
            "Role Definitions. Roles are defined in the identity provider and are "
            "versioned. A role change requires an approval from the system owner and "
            "is recorded in the access register with a change ticket reference.",
            "Multi Factor Authentication. Multi factor authentication is mandatory for "
            "administrative roles and for any access to production data. Hardware keys "
            "are required for break glass accounts.",
            "Break Glass Access. Break glass accounts are disabled by default and are "
            "enabled through a time limited approval. Every break glass session is "
            "logged and reviewed by the security team within one business day.",
            "Service Account Governance. Service accounts use short lived credentials "
            "issued by the secrets manager. Long lived static credentials for service "
            "accounts are prohibited and are removed by the automated rotation job.",
            "Access Review Evidence. The quarterly access review produces a signed "
            "attestation per system owner. Unattested systems are reported to the risk "
            "register with a remediation due date.",
        ),
    ),
)

QUERIES: tuple[dict, ...] = (
    {
        "id": "q01",
        "question": "Within how many days may a customer request a full refund?",
        "doc": "refund_policy.txt",
        "chunk": 0,
    },
    {
        "id": "q02",
        "question": "What is the refund window for electronics?",
        "doc": "refund_policy.txt",
        "chunk": 1,
    },
    {
        "id": "q03",
        "question": "How many business days does refund settlement take?",
        "doc": "refund_policy.txt",
        "chunk": 2,
    },
    {
        "id": "q04",
        "question": "How long does domestic ground shipping take?",
        "doc": "shipping_times.txt",
        "chunk": 0,
    },
    {
        "id": "q05",
        "question": "When is expedited delivery available?",
        "doc": "shipping_times.txt",
        "chunk": 1,
    },
    {
        "id": "q06",
        "question": "How long does international customs clearance take?",
        "doc": "shipping_times.txt",
        "chunk": 2,
    },
    {
        "id": "q07",
        "question": "How long is the manufacturer warranty?",
        "doc": "warranty_terms.txt",
        "chunk": 0,
    },
    {
        "id": "q08",
        "question": "What do I need to submit a warranty claim?",
        "doc": "warranty_terms.txt",
        "chunk": 1,
    },
    {
        "id": "q09",
        "question": "Is water damage covered by warranty?",
        "doc": "warranty_terms.txt",
        "chunk": 3,
    },
    {
        "id": "q10",
        "question": "What defines a severity 1 incident?",
        "doc": "incident_runbook.txt",
        "chunk": 0,
    },
    {
        "id": "q11",
        "question": "Who owns the incident response?",
        "doc": "incident_runbook.txt",
        "chunk": 1,
    },
    {
        "id": "q12",
        "question": "What is the public status update cadence for an outage?",
        "doc": "incident_runbook.txt",
        "chunk": 4,
    },
    {
        "id": "q13",
        "question": "What blocks a release?",
        "doc": "release_checklist.txt",
        "chunk": 1,
    },
    {
        "id": "q14",
        "question": "How long is the staging soak time?",
        "doc": "release_checklist.txt",
        "chunk": 2,
    },
    {
        "id": "q15",
        "question": "What is the staged rollout plan for production?",
        "doc": "release_checklist.txt",
        "chunk": 4,
    },
    {
        "id": "q16",
        "question": "On which working day of the month does the payroll run?",
        "doc": "payroll_policy.txt",
        "chunk": 0,
    },
    {
        "id": "q17",
        "question": "When are timesheets due?",
        "doc": "payroll_policy.txt",
        "chunk": 1,
    },
    {
        "id": "q18",
        "question": "How is overtime paid?",
        "doc": "payroll_policy.txt",
        "chunk": 2,
    },
    {
        "id": "q19",
        "question": "What is the default API rate limit quota?",
        "doc": "api_rate_limits.txt",
        "chunk": 0,
    },
    {
        "id": "q20",
        "question": "What happens when I exceed the burst allowance?",
        "doc": "api_rate_limits.txt",
        "chunk": 1,
    },
    {
        "id": "q21",
        "question": "How should clients handle throttling responses?",
        "doc": "api_rate_limits.txt",
        "chunk": 5,
    },
    {
        "id": "q22",
        "question": "Which principle governs access grants?",
        "doc": "access_control.txt",
        "chunk": 0,
    },
    {
        "id": "q23",
        "question": "Is multi factor authentication mandatory?",
        "doc": "access_control.txt",
        "chunk": 2,
    },
    {
        "id": "q24",
        "question": "How are break glass accounts reviewed?",
        "doc": "access_control.txt",
        "chunk": 3,
    },
    {
        "id": "q25",
        "question": "What is the quarterly access review cadence?",
        "doc": "access_control.txt",
        "chunk": 5,
    },
    {
        "id": "n01",
        "question": "What is the office wifi password?",
        "doc": None,
        "chunk": None,
    },
    {
        "id": "n02",
        "question": "Who won the 2022 world cup final?",
        "doc": None,
        "chunk": None,
    },
    {
        "id": "n03",
        "question": "What is the capital city of Portugal?",
        "doc": None,
        "chunk": None,
    },
    {
        "id": "n04",
        "question": "Can you book me a flight to Tokyo?",
        "doc": None,
        "chunk": None,
    },
    {
        "id": "n05",
        "question": "What is the recommended daily calorie intake for adults?",
        "doc": None,
        "chunk": None,
    },
    {
        "id": "n06",
        "question": "Explain quantum chromodynamics lattice gauge theory.",
        "doc": None,
        "chunk": None,
    },
)


def tokenize(text: str) -> list[str]:
    return [
        term
        for term in WORD_RE.findall(str(text or "").lower())
        if term not in STOPWORDS and len(term) > 1
    ]


DISTRACTOR_DOCUMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "gift_card_terms.txt",
        (
            "Gift Card Terms. Gift cards are valid for 24 months from the date of "
            "issue. The refund window does not apply to gift card balances, which are "
            "non refundable after activation.",
            "Gift Card Activation. A gift card is activated on first use at a "
            "checkout. An unactivated card can be cancelled for a full refund within "
            "30 days of issue.",
            "Gift Card Partial Refund. A partial refund for an activated gift card is "
            "issued as store credit rather than a payment card refund. Store credit "
            "expires 12 months after issue.",
            "Gift Card Balance Inquiry. The remaining balance is visible in the account "
            "portal. Balances are not transferable and cannot be exchanged for cash at "
            "a retail branch.",
            "Gift Card Fraud. A gift card reported stolen is frozen immediately. Frozen "
            "cards are not refunded and the freeze is reported to the fraud team.",
            "Gift Card Expiry Notice. A reminder is sent 60 days before expiry. An "
            "expired card balance is donated to charity and is not refundable.",
        ),
    ),
    (
        "subscription_cancellation.txt",
        (
            "Subscription Cancellation. A subscription may be cancelled at any time "
            "from the account portal. Cancellation stops the next renewal charge and "
            "does not refund the current period.",
            "Subscription Renewal. A subscription renews automatically each month until "
            "it is cancelled. A renewal reminder is sent 3 days before the charge.",
            "Subscription Refund Exception. A refund within the first 14 days of a "
            "subscription is available once per customer as a goodwill exception "
            "approved by support.",
            "Subscription Plan Changes. A plan change takes effect at the next renewal. "
            "A downgrade does not trigger a payment for the difference.",
            "Subscription Pause. A subscription may be paused for up to 3 months. The "
            "pause does not extend the billing anniversary date.",
            "Subscription Data Retention. Cancellation stops collection of usage data "
            "after the retention window of 30 days expires.",
        ),
    ),
    (
        "store_credit_policy.txt",
        (
            "Store Credit Policy. Store credit is issued for approved returns and for "
            "goodwill adjustments. Credit is applied at checkout before a payment card "
            "charge.",
            "Store Credit Expiry. Store credit expires 12 months from the date of "
            "issue. The expiry date is shown on the credit record in the account "
            "portal.",
            "Store Credit Refund. A purchase paid entirely with store credit is "
            "refunded as store credit. A purchase paid with a payment card is refunded "
            "to the original card.",
            "Store Credit Transfer. Store credit cannot be transferred to another "
            "account. A transfer request is refused and logged by the support team.",
            "Store Credit Balance. The available credit balance is the sum of unexpired "
            "credit records. Expired records are archived, not deleted.",
            "Store Credit Adjustment. A support agent may issue up to the approved "
            "goodwill limit in a single adjustment without a second approval.",
        ),
    ),
    (
        "incident_communications.txt",
        (
            "Customer Communication During An Outage. A public status page is updated "
            "every 60 minutes during a severity 1 incident. Each update states the "
            "impact, the mitigation in progress, and the time of the next update.",
            "Status Page Ownership. The incident commander owns the public status page "
            "during a severity 1 incident. A communications lead may draft updates but "
            "does not publish them independently.",
            "Support Contact During An Outage. Support requests during an outage are "
            "triaged by severity. Requests about a known incident are answered with the "
            "status page link rather than an individual investigation.",
            "Post Incident Customer Notice. A customer notice is published within 3 "
            "business days of a severity 1 incident and links to the written "
            "postmortem when it is available.",
            "Communication Templates. Approved templates exist for acknowledgement, "
            "in progress, monitoring, and resolution messages. Templates are reviewed "
            "each quarter by the communications lead.",
            "Notification Channel Policy. Transactional notifications are sent by email "
            "and are not duplicated in the in product inbox unless the customer opts "
            "in.",
        ),
    ),
    (
        "api_authentication.txt",
        (
            "API Authentication. Requests are authenticated with a bearer token issued "
            "to a project. Tokens are scoped to a single workspace and are not shared "
            "between environments.",
            "Token Rotation. Tokens are rotated every 90 days. Rotation issues a new "
            "token and revokes the previous token after a 24 hour overlap window.",
            "Token Storage. Tokens must be stored in a secrets manager. A token "
            "committed to a repository is revoked and reissued without exception.",
            "Service Account Authentication. A service account authenticates with a "
            "short lived credential issued by the secrets manager. Static credentials "
            "are prohibited.",
            "Authentication Failures. A rejected token returns HTTP 401 with a generic "
            "message. The response does not disclose whether the token exists.",
            "Scope Management. A token carries an explicit scope list. A request outside "
            "the granted scopes returns HTTP 403 and is recorded in the access log.",
        ),
    ),
    (
        "release_rollback.txt",
        (
            "Rollback Trigger Conditions. Rollback is triggered by an error rate above "
            "2 percent for 10 minutes, a latency regression above 30 percent, or a "
            "data integrity alert.",
            "Rollback Execution. The on call engineer executes the rollback without "
            "further approval when a trigger condition is met. The rollback is recorded "
            "in the incident timeline.",
            "Rollback Version Selection. The rollback target is the previous version "
            "identifier recorded at promotion time. A version that never served traffic "
            "is not a valid target.",
            "Database Compatibility. A rollback is blocked when the deployed schema is "
            "not backward compatible with the target version. A forward fix is required "
            "in that case.",
            "Rollback Verification. After a rollback the on call engineer confirms the "
            "error rate and latency dashboards have recovered before closing the "
            "incident.",
            "Rollback Rehearsal. Rollback is rehearsed each quarter in staging. The "
            "rehearsal records the elapsed time from trigger to completed rollback.",
        ),
    ),
    (
        "payroll_tax.txt",
        (
            "Tax Code Assignment. Each employee has a tax code assigned at hire. A tax "
            "code change requires the employee declaration form before the next payroll "
            "run.",
            "Taxable Benefits. Company car benefit and health insurance are treated as "
            "taxable benefits and are reported in the year end return.",
            "Statutory Reporting Deadline. The statutory reporting return is submitted "
            "within the deadline set by the reporting authority. A late submission "
            "attracts a penalty.",
            "Payroll Reconciliation. The payroll run total is reconciled against the "
            "journal entry before the statutory return is produced.",
            "Pension Contributions. Pension contributions are deducted at the "
            "contractual rate and reported separately from income tax in the export.",
            "Year End Adjustment. A year end adjustment corrects rounding differences "
            "and is itemised for employee review before submission.",
        ),
    ),
    (
        "warranty_service_levels.txt",
        (
            "Service Level Definitions. A next business day service level applies to "
            "hardware faults. A standard service level applies to software faults and "
            "completes within 10 business days.",
            "Service Level Credits. A service level credit is applied when a repair "
            "exceeds the agreed service level. The credit is calculated from the "
            "affected contract value.",
            "On Site Support. On site support is available for contracted enterprise "
            "customers. The engineer books an appointment through the service "
            "desk.",
            "Spare Parts. A spare part is held for 24 months after a product is "
            "withdrawn from sale. A part request outside that window is refused.",
            "Return Authorisation. A return authorisation number is required before any "
            "hardware is shipped back. Shipments without a number are not accepted.",
            "Service Reporting. A service report is issued for every repair and lists "
            "the fault, the parts replaced, and the service level achieved.",
        ),
    ),
)

MULTIHOP_QUERIES: tuple[dict, ...] = (
    {
        "id": "h01",
        "question": "A customer wants a refund for a gift card and also wants to know how long refunds take.",
        "docs": (("gift_card_terms.txt", 1), ("refund_policy.txt", 2)),
    },
    {
        "id": "h02",
        "question": "What refund window applies to a subscription, and how long do refunds settle?",
        "docs": (("subscription_cancellation.txt", 2), ("refund_policy.txt", 2)),
    },
    {
        "id": "h03",
        "question": "How long do I have to request a refund, and what happens to store credit issued for that refund?",
        "docs": (("refund_policy.txt", 0), ("store_credit_policy.txt", 2)),
    },
    {
        "id": "h04",
        "question": "When should the public status page be updated during an outage, and who owns that update?",
        "docs": (("incident_communications.txt", 0), ("incident_runbook.txt", 3)),
    },
    {
        "id": "h05",
        "question": "What triggers a rollback, and how long do I have to wait before a retry is allowed?",
        "docs": (("release_rollback.txt", 0), ("api_rate_limits.txt", 5)),
    },
    {
        "id": "h06",
        "question": "How long is a token valid, and what happens to a token that was committed to a repository?",
        "docs": (("api_authentication.txt", 1), ("api_authentication.txt", 2)),
    },
)

SCALING_QUERIES: tuple[dict, ...] = tuple(
    query
    for query in QUERIES
    if query.get("doc") and query["doc"] in {name for name, _ in DOCUMENTS}
)


def all_documents() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return DOCUMENTS + DISTRACTOR_DOCUMENTS


def build_nodes(documents=None) -> list[TextNode]:
    nodes: list[TextNode] = []
    for file_name, chunks in documents or DOCUMENTS:
        for index, chunk in enumerate(chunks):
            nodes.append(
                TextNode(
                    text=chunk,
                    id_=f"{file_name}#{index}",
                    metadata={
                        "file_name": file_name,
                        "start_char_idx": index * 512,
                        "source_id": file_name,
                    },
                )
            )
    return nodes


def node_key(file_name: str, index: int) -> str:
    return f"{file_name}#{index}"


class _Docstore:
    def __init__(self, nodes):
        self._nodes = {node.node_id: node for node in nodes}

    def get_node(self, node_id):
        return self._nodes.get(node_id)


class FixtureRetriever:
    """A deterministic BM25 retriever exposing the ``HybridRetriever`` surface."""

    def __init__(self, nodes, top_k: int = 8):
        self.docstore = _Docstore(nodes)
        self.corpus = [node.node_id for node in nodes]
        self.top_k = max(1, int(top_k))
        self._nodes = {node.node_id: node for node in nodes}
        self._terms = {
            node_id: tokenize(node.get_content())
            for node_id, node in self._nodes.items()
        }
        self._lengths = {node_id: len(terms) for node_id, terms in self._terms.items()}
        self._avg_length = (
            sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0
        )
        self._frequency: Counter = Counter()
        for terms in self._terms.values():
            self._frequency.update(set(terms))
        self._document_count = max(1, len(self._terms))
        self.calls: list[dict] = []

    def _idf(self, term: str) -> float:
        count = self._frequency.get(term, 0)
        return math.log(1.0 + (self._document_count - count + 0.5) / (count + 0.5))

    def score(self, query: str, node_id: str) -> float:
        terms = self._terms.get(node_id) or []
        if not terms:
            return 0.0
        counts = Counter(terms)
        length = self._lengths.get(node_id, 0)
        normalization = (
            1.0
            - BM25_B
            + BM25_B * (length / self._avg_length if self._avg_length else 1.0)
        )
        score = 0.0
        for term in tokenize(query):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            score += (
                self._idf(term)
                * (frequency * (BM25_K1 + 1.0))
                / (frequency + BM25_K1 * (normalization or 1.0))
            )
        return score

    def retrieve(self, query: str, allowed_node_ids=None, top_k: int | None = None):
        limit = max(1, int(top_k or self.top_k))
        candidates = (
            [node_id for node_id in self.corpus if node_id in set(allowed_node_ids)]
            if allowed_node_ids is not None
            else list(self.corpus)
        )
        self.calls.append(
            {
                "query": str(query or ""),
                "restricted": allowed_node_ids is not None,
                "candidates": len(candidates),
            }
        )
        scored = [(node_id, self.score(query, node_id)) for node_id in candidates]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            NodeWithScore(node=self._nodes[node_id], score=round(score, 6))
            for node_id, score in scored[:limit]
            if score > 0.0
        ]


def build_retriever(top_k: int = 8) -> FixtureRetriever:
    return FixtureRetriever(build_nodes(), top_k=top_k)


FILLER_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("regional_office_handbook", "regional office"),
    ("supplier_agreement_terms", "supplier agreement"),
    ("quality_assurance_procedure", "quality assurance"),
    ("field_service_manual", "field service"),
    ("records_retention_policy", "records retention"),
    ("vendor_onboarding_guide", "vendor onboarding"),
    ("capacity_planning_notes", "capacity planning"),
    ("customer_escalation_matrix", "customer escalation"),
    ("asset_lifecycle_policy", "asset lifecycle"),
    ("training_competency_matrix", "training competency"),
    ("procurement_thresholds", "procurement thresholds"),
    ("inventory_control_notes", "inventory control"),
)

FILLER_TEMPLATES: tuple[str, ...] = (
    "{subject} overview. This {subject} document defines the review cycle, the "
    "approval path, and the record kept for each decision made under it.",
    "Review cadence. The {subject} is reviewed every quarter by the owning team. "
    "Findings are recorded and tracked until the next review.",
    "Approval path. A decision under the {subject} needs one named approver. The "
    "approver is recorded with the decision and the date it was taken.",
    "Exceptions. An exception to the {subject} requires a written justification and "
    "a review date. Exceptions are listed in the quarterly report.",
    "Retention. Records produced under the {subject} are kept for the retention "
    "period and then archived. Archived records are not deleted on request.",
    "Escalation. A question about the {subject} that is not answered by this "
    "document is escalated to the owning team, which responds within 5 business days.",
    "Budget. Work under the {subject} is charged to the cost centre named in the "
    "purchase record. Overruns are reported in the monthly summary.",
    "Training. Staff applying the {subject} complete the recorded training module "
    "before their first decision and refresh it annually.",
    "Reporting. A monthly summary of activity under the {subject} is produced for "
    "the operations review and stored with the supporting records.",
    "Related documents. This {subject} references the standard operating procedure "
    "and the risk register. Neither is superseded by this document.",
    "Change control. A change to the {subject} is versioned and approved before it "
    "takes effect. The previous version is retained for audit.",
    "Definitions. A decision under the {subject} means a recorded choice made by a "
    "named approver, not an informal agreement or a preliminary discussion.",
)


def build_filler_documents(count: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return deterministic distractor documents for the scaling study.

    These documents deliberately reuse the vocabulary of the real corpus so that
    they compete for the same query terms. They are never ground-truth answers;
    their only job is to make routing harder as the corpus grows.
    """
    documents = []
    for index in range(max(0, int(count))):
        subject_key, subject = FILLER_SUBJECTS[index % len(FILLER_SUBJECTS)]
        round_index = index // len(FILLER_SUBJECTS)
        name = (
            f"{subject_key}_{round_index}.txt" if round_index else f"{subject_key}.txt"
        )
        chunks = tuple(
            template.format(subject=subject) for template in FILLER_TEMPLATES[:6]
        )
        documents.append((name, chunks))
    return tuple(documents)


HASH_EMBEDDING_DIMS = 256


def hash_embedding(text: str, dims: int = HASH_EMBEDDING_DIMS) -> list[float]:
    """Return a deterministic bag-of-terms embedding hashed into a fixed width.

    This is a reproducible stand-in for a dense encoder, not a trained model.
    It gives the naive-vector baseline a real dense-similarity retrieval path
    without downloading an embedding model, and the same query always produces
    the same vector.
    """
    vector = [0.0] * dims
    for term in tokenize(text):
        bucket = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(bucket[:4], "big") % dims
        sign = 1.0 if bucket[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine(left, right) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


class DenseFixtureRetriever(FixtureRetriever):
    """Dense cosine retrieval over hash embeddings, mirroring the same surface."""

    def __init__(self, nodes, top_k: int = 8, dims: int = HASH_EMBEDDING_DIMS):
        super().__init__(nodes, top_k=top_k)
        self.dims = dims
        self._vectors = {
            node_id: hash_embedding(self._nodes[node_id].get_content(), dims)
            for node_id in self.corpus
        }

    def score(self, query: str, node_id: str) -> float:
        vector = self._vectors.get(node_id)
        if vector is None:
            return 0.0
        return max(0.0, cosine(hash_embedding(query, self.dims), vector))


def build_dense_retriever(top_k: int = 8) -> DenseFixtureRetriever:
    return DenseFixtureRetriever(build_nodes(), top_k=top_k)


class FlatTextAgent:
    """Agentic retrieval over flat chunk cards with no document structure.

    This is the "text-based agent, no map" ablation baseline. It keeps the
    observe -> plan -> act loop and a deterministic keyword planner, but scores
    raw chunk cards instead of section cards and has no document grouping,
    section hierarchy, or adjacency edges to navigate. It exists only in the
    research harness; the production application always uses the map agent.
    """

    def __init__(self, retriever, tokenizer=None, select=4, max_results=8):
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.select = max(1, int(select))
        self.max_results = max(1, int(max_results))

    def _cards(self, query: str) -> list[dict]:
        scores = [
            (node_id, self.retriever.score(query, node_id))
            for node_id in self.retriever.corpus
        ]
        scores.sort(key=lambda item: (-item[1], item[0]))
        cards = []
        for node_id, score in scores:
            node = self.retriever.docstore.get_node(node_id)
            text = node.get_content() if node is not None else ""
            cards.append(
                {
                    "id": node_id,
                    "title": str(node_id),
                    "summary": text[:160],
                    "score": score,
                }
            )
        return cards

    def retrieve(self, query: str):
        cards = self._cards(query)
        planned = [card["id"] for card in cards if card["score"] > 0.0][: self.select]
        actions = [{"step": 0, "action": "plan", "reason": "flat_keyword_scores"}]
        if not planned:
            actions.append(
                {"step": 1, "action": "retrieve_global", "reason": "no_planned_chunks"}
            )
            return [], self._trace(cards, (), actions, 1, [])
        allowed = set(planned)
        results = list(self.retriever.retrieve(query, allowed_node_ids=allowed))[
            : self.max_results
        ]
        actions.append(
            {
                "step": 1,
                "action": "retrieve_chunks",
                "reason": "routed_evidence",
                "allowed_section_ids": list(planned),
                "result_count": len(results),
            }
        )
        if not results:
            actions.append({"step": 1, "action": "reflect", "reason": "weak_evidence"})
            results = self.retriever.retrieve(query)[: self.max_results]
            actions.append(
                {
                    "step": 2,
                    "action": "global_fallback",
                    "reason": "routed_evidence_weak",
                    "result_count": len(results),
                }
            )
        return results, self._trace(cards, planned, actions, 2, results)

    def _trace(self, cards, planned, actions, step_count, results) -> dict:
        return {
            "considered_section_ids": [card["id"] for card in cards],
            "planned_section_ids": list(planned),
            "selected_section_ids": list(planned),
            "actions": actions,
            "step_count": step_count,
            "reflection_flag": any(
                action.get("action") == "reflect" for action in actions
            ),
            "planner_mode": "flat_deterministic",
            "metrics": {},
            "result_metadata": [],
        }
