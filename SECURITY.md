# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on the repository page. Do not open a public issue for them.
We aim to acknowledge reports within a few days.

## What the SDK does with secrets

- The API key is sent only in the `Authorization` header over the address you configured, never in
  URLs or bodies, and it is removed from every message the SDK prints.
- Redirects are not followed, so a key cannot be replayed to another address.
- The credentials file is created with mode `0600` in a `0700` directory.
- Environment variables are never sent wholesale. Config values whose key looks like a secret are
  masked, and credentials in a git remote URL are removed before it is sent.
- TLS certificates are always verified. Use `SSL_CERT_FILE` for a private certificate authority.
- Spooled runs contain your metrics and config but never the key.

Plain `http://` addresses send the key unencrypted. `airon login` and `airon doctor` warn about
them. Use `https://` except on localhost.
