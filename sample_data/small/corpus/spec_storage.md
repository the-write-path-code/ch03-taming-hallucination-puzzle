# Archive Vault Storage Specification

| Tier | Retrieval target | Minimum retention | Encryption |
|---|---:|---:|---|
| Vault-A | under 4 hours | 30 days | AES-256 |
| Vault-B | under 24 hours | 365 days | AES-256 |
| Vault-Legal | under 24 hours | legal hold duration | AES-256 |

Vault-B is appropriate for support attachments retained under DR-44 unless a legal hold changes the retention requirement.
