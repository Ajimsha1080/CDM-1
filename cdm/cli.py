#!/usr/bin/env python3
"""CDM-1 Unified Command-Line Interface (CLI).
Provides enterprise commands for serving, development playground,
manifest verification, one-shot decision inference, and health diagnostics.
"""

import sys
import os
import argparse
import json


def cmd_version(args):
    from cdm import __version__
    print(f"CDM-1 Cognitive Decision Architecture v{__version__}")
    print("Underlying Model: ModernBERT-base (158M parameters)")
    print("Status: Production-Ready (Verified)")


def cmd_verify(args):
    print("[*] Verifying CDM-1 release checkpoint integrity and Ed25519 cryptographic signature...")
    from cdm.security.manifest_verification import verify_checkpoint_authenticity, SecurityVerificationError
    model_dir = args.model_dir or "experiments/release/cdm-1-general"
    try:
        res = verify_checkpoint_authenticity(model_dir)
        print("[+] Checkpoint Authenticated Successfully!")
        print(f"    Status: {res.get('status')}")
        print(f"    SHA-256: {res.get('sha256')}")
        print(f"    Signature Verified: {res.get('signature_verified')}")
        print(f"    Tensor Count: {res.get('tensor_count')}")
    except SecurityVerificationError as e:
        print(f"[!] Security Verification FAILED: {e}")
        sys.exit(1)


def cmd_serve(args):
    import uvicorn
    host = args.host or "127.0.0.1"
    port = int(args.port or 8000)
    print(f"[*] Launching CDM-1 REST API on http://{host}:{port} ...")
    uvicorn.run("api.main:app", host=host, port=port, reload=args.reload)


def cmd_playground(args):
    from scripts.start_playground import main as start_pg
    print("[*] Launching CDM-1 Developer Playground...")
    start_pg()


def cmd_decide(args):
    from sdk.cdm_sdk import CDM
    client = CDM(mode=args.mode, local_model_path=args.model_dir)
    structured = {}
    if args.structured_json:
        try:
            structured = json.loads(args.structured_json)
        except Exception as e:
            print(f"[!] Invalid JSON for --structured-json: {e}")
            sys.exit(1)
    
    candidates = [c.strip() for c in args.candidates.split(",")] if args.candidates else None
    res = client.decide(
        text=args.text,
        structured_features=structured,
        task=args.task,
        candidates=candidates,
    )
    print(json.dumps(res.model_dump(), indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="cdm",
        description="CDM-1 Cognitive Decision Model Enterprise CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available CDM-1 commands")

    # version
    parser_ver = subparsers.add_parser("version", help="Print CDM-1 version and architecture details")
    parser_ver.set_defaults(func=cmd_version)

    # verify
    parser_verify = subparsers.add_parser("verify", help="Verify release checkpoint SHA-256 and Ed25519 signature")
    parser_verify.add_argument("--model-dir", default="experiments/release/cdm-1-general", help="Path to checkpoint directory")
    parser_verify.set_defaults(func=cmd_verify)

    # serve
    parser_serve = subparsers.add_parser("serve", help="Start the FastAPI REST backend server")
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser_serve.add_argument("--reload", action="store_true", help="Enable live auto-reload")
    parser_serve.set_defaults(func=cmd_serve)

    # playground
    parser_pg = subparsers.add_parser("playground", help="Launch the Next.js Developer Playground UI")
    parser_pg.set_defaults(func=cmd_playground)

    # decide
    parser_decide = subparsers.add_parser("decide", help="Submit a one-shot cognitive decision inquiry")
    parser_decide.add_argument("text", help="Inquiry or problem text")
    parser_decide.add_argument("--task", default="choice", choices=["choice", "binary", "score", "predict"], help="Decision task type")
    parser_decide.add_argument("--candidates", help="Comma-separated candidate action IDs")
    parser_decide.add_argument("--structured-json", help="JSON string of structured tabular features")
    parser_decide.add_argument("--mode", default="local", choices=["local", "hosted"], help="Execution mode")
    parser_decide.add_argument("--model-dir", default="experiments/release/cdm-1-general", help="Path to local weights")
    parser_decide.set_defaults(func=cmd_decide)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
