#!/usr/bin/env python3
"""
Generate RSA keypair for issue-based encrypted submissions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evaluator RSA keypair")
    parser.add_argument(
        "--public-key-out",
        default=".github/eval/eval_public_key.pem",
        help="Path to write public key PEM",
    )
    parser.add_argument(
        "--private-key-out",
        default=".secrets/eval_private_key.pem",
        help="Path to write private key PEM (DO NOT COMMIT)",
    )
    args = parser.parse_args()

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    public_path = Path(args.public_key_out).resolve()
    private_path = Path(args.private_key_out).resolve()
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)

    public_path.write_bytes(public_pem)
    private_path.write_bytes(private_pem)
    print(f"Public key written to: {public_path}")
    print(f"Private key written to: {private_path}")
    print("Next:")
    print(f"  gh secret set EVAL_PRIVATE_KEY_PEM < {private_path}")
    print("  python .tools/encrypt_submission.py --input-dir <your_gt_images_dir> --public-key .github/eval/eval_public_key.pem --output .github/eval/gt.enc.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
