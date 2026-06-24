# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in `codebase-indexer`, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities by opening a [GitHub Security Advisory](https://github.com/julianstephens/codebase-indexer/security/advisories/new) in this repository.

### What to include

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The version(s) affected
- Any suggested mitigations, if known

### Response timeline

- You will receive an acknowledgment within **48 hours**
- A fix or mitigation plan will be communicated within **7 days** for critical issues

## Security Considerations

`codebase-indexer` reads source files from disk and stores extracted symbols in a local SQLite database. Keep the following in mind:

- **Untrusted repositories**: Parsing untrusted source trees may expose sensitive information embedded in source files (e.g. hardcoded secrets). Review the repository before indexing it.
- **Database file**: The generated `.db` file contains extracted source code snippets. Treat it with the same sensitivity as the source repository itself.
- **`.cbmignore` / `.gitignore`**: Ensure sensitive files are properly excluded via ignore rules before indexing.
