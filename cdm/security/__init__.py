"""Security, Authentication, and Artifact Cryptographic Verification."""

from cdm.security.manifest_verification import (
    sign_manifest,
    verify_manifest_signature,
    verify_checkpoint_authenticity,
    SecurityVerificationError,
    compute_file_sha256,
    TRUSTED_RELEASE_PUBLIC_KEY,
)

__all__ = [
    "sign_manifest",
    "verify_manifest_signature",
    "verify_checkpoint_authenticity",
    "SecurityVerificationError",
    "compute_file_sha256",
    "TRUSTED_RELEASE_PUBLIC_KEY",
]
