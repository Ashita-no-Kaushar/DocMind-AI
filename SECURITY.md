# Security Policy

## Reporting a Vulnerability

If GitHub private vulnerability reporting is available for this repository, use it. Otherwise open a public issue without exploit details, credentials, private documents, or sensitive local paths and request a private contact method.

Security-sensitive areas include:

- Upload validation and path containment
- Website URL validation, redirects, and SSRF resistance
- GitHub cloning and repository file handling
- External LLM, embedding, and R2R endpoints
- API-key handling
- Local index/cache contents and logs
- Docker filesystem and network exposure

## Deployment Assumptions

DocMind currently has no application authentication and uses process-wide temporary/cache paths. It is designed for local, single-user use. Do not expose it directly to an untrusted network without an authentication and isolation layer.

## Implemented Guardrails

### Local Files

- 25-extension allowlist
- 10-file, 25 MiB/file, and 100 MiB total limits
- Restricted filename characters
- Resolved destination containment
- No executable/archive extensions in the upload allowlist
- Explicit current-upload loading rather than scanning arbitrary stale files

### Directory and GitHub Content

- Hidden paths and configured binary/archive/environment patterns excluded
- File symlinks rejected
- Resolved files required to remain under the selected source root
- GitHub input restricted to exactly `owner/repo` on `github.com`
- Dot-segment repository names rejected
- Git invoked without a shell and with a 120-second timeout

### Websites

- HTTPS only
- Embedded credentials rejected
- Blocked hostnames
- DNS resolution checks every returned A and AAAA address against private, loopback, link-local, multicast, reserved, unspecified, and other non-global ranges
- Known cloud metadata hostnames and addresses rejected
- HTTPS connections pinned to a validated numeric address while preserving hostname certificate verification
- Redirect targets revalidated, up to three redirects
- Per-request timeouts, a 180-second total ingestion deadline, and categorized failures
- HTML/plain-text content types only
- 5 MiB streamed response cap
- Responses and connection pools closed on success and failure

### Provider and Secret Handling

- Browser persistence excludes OpenAI and R2R API keys
- R2R documents are uploaded only when the user enables R2R and uses that path
- The default Ollama endpoint is local, but users can configure remote endpoints
- Logging falls back to console if the configured log file is not writable

## Important Risks

- External OpenAI-compatible and R2R servers receive data from the configured workflow.
- Local index caches and logs can contain extracted document text.
- GitHub content is cloned with the permissions of the application user; `subprocess.run` is not a sandbox.
- Clone and parser resource limits are applied after or incompletely relative to ingestion.
- The app has no user authentication or per-session filesystem isolation.
- Docker has a read-only root filesystem and dropped capabilities, but runtime verification is still required.
- A citation marker does not verify factual support.

## Dependency and Deployment Status

The project currently uses wildcard dependency versions and does not commit a lockfile. Docker configuration was statically reviewed but not built in the latest local environment.
