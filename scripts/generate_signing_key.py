#!/usr/bin/env python3
"""Generates a cryptographically secure Ed25519 keypair for signing CDM-1 release manifests.
Private keys must NEVER be stored in the repository or written to disk.
"""

from cryptography.hazmat.primitives.asymmetric import ed25519


def main():
    private_key = ed25519.Ed25519PrivateKey.generate()
    priv_hex = private_key.private_bytes_raw().hex()
    pub_hex = private_key.public_key().public_bytes_raw().hex()

    print("=" * 70)
    print(" CDM-1: Generated Ed25519 Release Manifest Signing Keypair")
    print("=" * 70)
    print(f"Public Key (hex) : {pub_hex}")
    print(f"Private Key (hex): {priv_hex}")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Set the private key in your environment (never commit to git or write to disk):")
    print(f"   Linux/macOS: export CDM_RELEASE_PRIVATE_KEY={priv_hex}")
    print(f"   Windows PS : $env:CDM_RELEASE_PRIVATE_KEY=\"{priv_hex}\"")
    print("\n2. Update the embedded public key in cdm/security/manifest_verification.py:")
    print(f"   TRUSTED_RELEASE_PUBLIC_KEY = \"{pub_hex}\"")
    print("\n3. Regenerate and sign the release manifest:")
    print("   python scripts/generate_and_sign_manifest.py")
    print("   (or 'make manifest')")
    print("=" * 70)


if __name__ == "__main__":
    main()
