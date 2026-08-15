# Changelog

Release notes for DocMind are published with GitHub Releases:

https://github.com/Ashita-no-Kaushar/DocMind-AI/releases

## Vulnerability Disclosure in Release Notes

When a release fixes a publicly known runtime vulnerability in DocMind that already has a CVE or similar public identifier at release time, the release notes identify that vulnerability and summarize the upgrade impact.

If a release has no such vulnerability fixes, the release notes may omit this section.

## Versioning

DocMind release tags use semantic versioning, for example `v1.5.0`.

## [Unreleased]

## [v1.5.0] - 2026-08-15

- Fix GitHub repository ingestion failures caused by stale checkouts and transient Windows file locks.
- Purge stale `__pycache__` on startup so the app never runs outdated bytecode.
- Only load files that were just uploaded during ingestion (no stale-leftover pollution).
- Remove the Local Hugging Face embedding backend (Ollama embeddings only).
- Apply Top K / similarity threshold settings live; remove the dead Chat Mode setting.
- Retrieve document introductions for summary/about-the-document questions.
- Export chat history as a Word (.docx) document; remove the Custom answer-style preset.
- General reliability and security hardening.
