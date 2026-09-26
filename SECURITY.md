# Security Policy

## Scope and reporting

DocMind accepts untrusted documents, public GitHub content, and public website content, then sends selected prompts and retrieved context to the configured model or R2R service. Security-sensitive areas include:

- Upload validation, parser invocation, archive handling, and path containment.
- GitHub normalization, metadata requests, subprocess execution, checkout handling, and source loading.
- Website URL validation, DNS resolution, redirects, TLS connection pinning, and response processing.
- Provider endpoints, chat/embedding credentials, R2R credentials, and remote document ownership.
- Local indexes, work directories, browser persistence, logs, exports, and reset behavior.
- Runtime binding, Docker/Compose isolation, and the lack of application authentication.

If GitHub private vulnerability reporting is available, use it. Otherwise open an issue without exploit details, credentials, private documents, sensitive local paths, or logs, and request a private contact method.

## Deployment assumptions

DocMind has no built-in authentication or user authorization. Its default runtime bind address is `127.0.0.1`, and the default Compose port publication is `127.0.0.1:8501:8501`. It is designed for local, single-user use.

A non-loopback or wildcard bind requires the explicit `DOCMIND_ALLOW_REMOTE_BIND=true` opt-in and a user-configured authenticated reverse proxy. The opt-in is a deployment switch, not authentication. Do not expose the Streamlit port directly to an untrusted network. An authenticated proxy must provide authentication, authorization, transport security, request limits, and an appropriate network boundary.

## Threat model and controls

### Local files, uploads, and parsers

- The upload allowlist contains 25 extensions and is not a guarantee that every accepted file has a reliable parser.
- Uploads are limited to 10 files per batch, 25 MiB per file, and 100 MiB total.
- Filenames are restricted, path separators and control characters are rejected, and resolved destinations must remain under the operation directory.
- The local loader uses the current explicit upload list rather than scanning arbitrary stale files.
- Hidden paths, configured binary/archive/environment patterns, symlinks/reparse points, special files, and paths outside the selected source root are rejected.
- The parser layer bounds source bytes, extracted records, extracted characters, JSON depth, MIME parts/depth, table dimensions, and PDF pages.
- ZIP-based formats are preflighted for member count, compressed size, total and per-member uncompressed size, compression ratio, path traversal, duplicate paths, symlink/special members, and encrypted members.
- `.doc` and `.ppt` are explicitly unsupported without conversion. Image-only PDF, DOCX, and PPTX content is not OCRed.
- Extraction reports contain source names, status, counts, and sanitized warnings/errors, not document text. They are tagged with the active source ID and index generation so stale reports are not shown for a replacement source.

These are application guardrails, not a parser sandbox. A vulnerable or unexpectedly expensive third-party parser can still consume CPU or memory within the limits of the host and container.

### GitHub repositories and subprocesses

- Inputs are normalized to exactly `owner/repo` or an `https://github.com/owner/repo` URL. Credentials, ports, query strings, fragments, extra path segments, and dot-segment names are rejected.
- Only public repositories are supported; private authentication is not implemented.
- Git is invoked without a shell, with terminal prompts disabled, using a shallow single-branch clone.
- The clone has a 120-second timeout, bounded output, and bounded metadata handling.
- The checkout is rejected when it contains a reparse/symlink path, special file, too many files, an oversized file, or excessive unpacked/download data.
- The checkout is installed with an atomic candidate-to-target replacement and a rollback attempt. Failed operation directories are cleaned when safe.
- GitHub metadata and clone output are not treated as trusted content. Repository text remains untrusted input to the parser and prompt layers.

The defaults are 100 MiB for repository download, 10,000 checkout files, 10 MiB per checkout file, 250 MiB unpacked checkout data, 64 KiB clone output, 10 seconds for metadata, and 120 seconds for clone. A metadata failure can fall back to a bounded clone. These controls reduce accidental resource use but do not make `git` or repository content harmless.

### Websites and SSRF resistance

Website ingestion is limited to public HTTPS pages:

- Embedded URL credentials, malformed hosts/ports, blocked hostnames, and non-HTTPS URLs are rejected.
- Hostnames are checked against local/metadata names, and every returned A and AAAA address is checked against private, loopback, link-local, multicast, reserved, unspecified, and other non-global ranges. Known metadata addresses are also rejected.
- The request uses a validated numeric address while retaining the original hostname for SNI and certificate verification. Redirect destinations are resolved and revalidated before use, with a maximum of three redirects.
- Only HTML and plain-text content is accepted. HTML is converted to text, responses are streamed with a 5 MiB cap, and connect/read/total deadlines are enforced.
- Responses and pools are closed on success and failure. Anti-bot challenges, empty pages, unsupported content, DNS failures, HTTP failures, and timeouts are categorized without exposing raw remote responses.

DNS resolution, network routing, and provider callbacks are still operating-system and external-service dependencies. The application cannot forcibly interrupt a callback after it has entered an operating-system or third-party call. The pinned-address design reduces DNS-rebinding exposure but is not a complete network sandbox.

### Provider and credential handling

- Ollama and chat/embedding provider profiles default to safe local behavior. Non-loopback chat/embedding endpoints must use HTTPS; URL credentials, query strings, fragments, and control characters are rejected. The R2R client accepts absolute HTTP or HTTPS URLs, so a remote R2R deployment must be protected with HTTPS and an appropriate network boundary.
- OpenAI, LM Studio, TabbyAPI, and other OpenAI-compatible profiles are isolated from one another. Model catalogs are endpoint- and credential-fingerprint aware, and a changed endpoint invalidates the associated model list.
- Official OpenAI requires an explicit key and is restricted to the official OpenAI host. Other profiles cannot receive official OpenAI credentials through the provider-specific binding checks.
- Browser persistence excludes all provider and embedding API-key fields. Keys are not stored in the R2R ownership registry.
- The R2R registry stores endpoint and credential fingerprints, document IDs, source metadata, and status, not the credential value. Registry writes are serialized, validated, and atomically replaced.
- Error messages and logs are redacted before being emitted. Do not paste credentials, document contents, prompts, or sensitive paths into issue reports.

Provider and R2R servers receive the prompts, context, and files required by the selected workflow. The user is responsible for trusting those endpoints and for applying network and authentication controls outside DocMind.

### Prompt and context boundaries

Retrieved document text is untrusted reference data, not an instruction channel. Local RAG prompts delimit the context, tell the model to ignore commands and role changes inside it, and remove attempts to inject the delimiter markers. The current query is kept separate from the context block.

These boundaries reduce accidental prompt injection and marker confusion. They cannot guarantee that a model will ignore sophisticated instructions, hidden content, or adversarial source text. Generated claims are not checked for factual entailment, and a citation marker is not proof of support.

### Lifecycle locking and atomic state

Ingestion, cache operations, GitHub checkout replacement, R2R transactions, and project reset use a cross-process lifecycle lock. The default lock path is `.docmind-ingestion.lock`; `DOCMIND_INGESTION_LOCK_PATH` selects another path, and `DOCMIND_INGESTION_LOCK_TIMEOUT_SECONDS` controls its wait.

A local source is committed only after a complete index candidate is ready. Failed ingestion restores the prior session source/query state when one existed. Owned work directories are operation-scoped and retried/removed where possible. This reduces cross-process races but does not provide per-browser filesystem isolation.

### Index cache

The local index cache is not encrypted. It can contain extracted source text, embeddings, and source metadata. The default is at most five entries and 512 MiB total, with count/byte pruning and symlink/reparse-point checks.

Persistence writes to a candidate directory, validates that the candidate can be loaded, atomically replaces the target, and attempts to restore the prior entry if commit fails. Supported overrides include `DOCMIND_INDEX_CACHE_MAX_ENTRIES` and `DOCMIND_INDEX_CACHE_MAX_BYTES`. Cache contents are shared process/filesystem state and should be protected with operating-system permissions.

### Logs and browser storage

Logs use a redaction filter for credential-shaped values, authorization material, sensitive query parameters, provider-token patterns, and exception text. File logging rotates at a default 1 MiB with three backups. `DOCMIND_LOG_FILE`, `DOCMIND_LOG_MAX_BYTES`, and `DOCMIND_LOG_BACKUP_COUNT` control supported settings. If the file is not writable, logging falls back to stdout.

Redaction is a defense-in-depth measure, not a guarantee that arbitrary sensitive text will never be logged. Review logs and exported DOCX transcripts before sharing them. Browser `localStorage` contains only the selected non-secret settings; it is not an access-control boundary and is not suitable for secrets.

### R2R ownership and reset

R2R is an external service integration. The client uses a versioned v3 contract, validates response shapes and sizes, polls document status, rolls back newly uploaded documents on transaction failure, and deletes replaced owned documents only after new documents are ready.

The local ownership registry is keyed by project/workspace and records endpoint/credential fingerprints. Reset defaults to deleting documents owned by the current identity, but it does not claim ownership of documents associated with another endpoint or credential. A failed deletion leaves the registry record so cleanup can be retried. The local-only reset option does not contact R2R.

Reset clears current local source state and owned work directories, while retaining the registry, browser settings, session-only keys, shared caches, logs, and other sessions. Users must separately delete files and remote documents that are not owned by the current workspace.

## Important residual risks

- There is no built-in authentication, authorization, or multi-user isolation.
- LlamaIndex `Settings`, adapter caches, logging, and some paths are process-global; the lifecycle lock serializes operations but does not create per-browser index isolation.
- External LLM, embedding, GitHub, website, and R2R services have their own availability, privacy, retention, TLS, and administrative risks.
- A citation or retrieval score does not verify factual support.
- Parser and GitHub controls are bounded application checks, not operating-system sandboxes.
- Local caches, logs, work files, browser settings, and exports are not encrypted by DocMind.
- A remote reverse proxy, TLS configuration, network policy, and provider/R2R credential policy must be supplied by the operator.

## Dependency and deployment status

`Pipfile` is directly pinned, `Pipfile.lock` is committed with hashes and a Python 3.13 declaration, and CI installs the locked dependency set with hash enforcement. The local gate passed `pipenv verify` and `pip check`.

Docker and Compose are statically validated and CI is build-only. A local Docker build, runtime, health check, read-only-volume exercise, and authenticated-proxy deployment were not available in the current environment.
