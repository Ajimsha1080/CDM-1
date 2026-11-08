#!/usr/bin/env python3
"""CLI utility to create, list, and revoke CDM-1 API keys.
Stores key hashes in the persistent SQLite database.
"""

import sys
import os
import argparse
import sqlite3
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.auth import PersistentApiKeyManager


def main():
    parser = argparse.ArgumentParser(description="CDM-1 API Key Management Tool")
    parser.add_argument("--db", default=os.environ.get("CDM_KEY_DB", "experiments/api_keys.sqlite"), help="Path to SQLite key database")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Generate and register a new API key")
    group.add_argument("--list", action="store_true", help="List all registered API keys (hashes only)")
    group.add_argument("--revoke", type=str, metavar="KEY_OR_HASH", help="Revoke an existing API key by token or hash")

    parser.add_argument("--tenant", type=str, help="Tenant ID for new key (required for --create)")
    parser.add_argument("--tier", type=str, default="standard", choices=["standard", "developer", "enterprise"], help="Key tier")
    parser.add_argument("--rpm", type=int, default=120, help="Rate limit in requests per minute")
    parser.add_argument("--label", type=str, default=None, help="Optional human-readable label")

    args = parser.parse_args()
    key_manager = PersistentApiKeyManager(db_path=args.db)

    if args.create:
        if not args.tenant:
            print("Error: --tenant is required when creating a new API key.", file=sys.stderr)
            sys.exit(1)
        raw_key, key_hash = key_manager.generate_key(
            tenant_id=args.tenant,
            tier=args.tier,
            rate_limit_rpm=args.rpm,
            label=args.label,
        )
        print("\n=== CDM-1 API Key Created Successfully ===")
        print(f"Tenant ID:   {args.tenant}")
        print(f"Tier:        {args.tier}")
        print(f"Rate Limit:  {args.rpm} RPM")
        if args.label:
            print(f"Label:       {args.label}")
        print(f"Key Hash:    {key_hash}")
        print(f"\nAPI Secret Token (save this securely, it will NOT be shown again):\n{raw_key}\n")

    elif args.list:
        conn = sqlite3.connect(args.db)
        try:
            cur = conn.cursor()
            cur.execute("SELECT key_hash, tenant_id, tier, rate_limit_rpm, created_at, revoked, label FROM api_keys ORDER BY created_at DESC;")
            rows = cur.fetchall()
            print(f"\nRegistered API Keys in {args.db}: ({len(rows)} total)")
            print(f"{'Key Hash':<66} {'Tenant':<20} {'Tier':<12} {'RPM':<6} {'Status':<10} {'Label'}")
            print("-" * 125)
            for r in rows:
                k_hash, tenant, tier, rpm, created_at, revoked, label = r
                status = "REVOKED" if revoked else "ACTIVE"
                lbl = label or "-"
                print(f"{k_hash:<66} {tenant:<20} {tier:<12} {rpm:<6} {status:<10} {lbl}")
            print()
        finally:
            conn.close()

    elif args.revoke:
        revoked = key_manager.revoke_key(args.revoke)
        if not revoked:
            # Also try matching hash directly
            conn = sqlite3.connect(args.db)
            try:
                cur = conn.cursor()
                cur.execute("UPDATE api_keys SET revoked = 1 WHERE key_hash = ?;", (args.revoke.strip(),))
                conn.commit()
                revoked = cur.rowcount > 0
            finally:
                conn.close()

        if revoked:
            print(f"Success: Key '{args.revoke[:10]}...' has been revoked.")
        else:
            print(f"Error: Key '{args.revoke[:10]}...' not found in database.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
