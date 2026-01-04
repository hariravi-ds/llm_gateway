import uuid
import numpy as np
from redis import Redis
from app.services.embed import embed_text

r = Redis(host="localhost", port=6379, decode_responses=False)


def ingest(tenant_id="acme", doc_version="v1"):
    # Example “knowledge base”
    docs = [
        ("password_reset", "To reset your password: go to Settings → Security → Reset Password. If locked out, contact IT at it@acme.com."),
        ("vpn_setup", "VPN setup: install AcmeVPN, sign in with SSO, choose region 'US-East'. For issues, open a ticket."),
        ("expense_policy", "Expense policy: meals reimbursed up to $50/day with itemized receipt. Alcohol is not reimbursable."),
    ]

    for name, text in docs:
        doc_id = name
        chunk_id = "0"
        vec = embed_text(text)
        key = f"doc:{tenant_id}:{doc_id}:{chunk_id}:{uuid.uuid4().hex}".encode()

        r.hset(key, mapping={
            b"tenant_id": tenant_id.encode(),
            b"doc_id": doc_id.encode(),
            b"chunk_id": chunk_id.encode(),
            b"doc_version": doc_version.encode(),
            b"text": text.encode(),
            b"vec": vec.tobytes(),
        })
        print("✅ Ingested", doc_id)


if __name__ == "__main__":
    ingest()
