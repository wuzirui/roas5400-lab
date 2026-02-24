#!/usr/bin/env python3
"""
Encrypt a directory of prediction images into a JSON payload for issue-based eval.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CRYPTO_AAD = b"roas5400-issue-eval-v1"


def _collect_files(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise ValueError(f"No image files found in {input_dir}")
    return files


def _build_tar_bytes(input_dir: Path, files: list[Path]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path in files:
            arcname = path.relative_to(input_dir).as_posix()
            tf.add(path, arcname=arcname, recursive=False)
    return buf.getvalue()


def _encrypt_tar_bytes(tar_bytes: bytes, public_key_pem: bytes) -> dict[str, object]:
    public_key = serialization.load_pem_public_key(public_key_pem)
    aes_key = AESGCM.generate_key(bit_length=256)
    aes = AESGCM(aes_key)
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, tar_bytes, CRYPTO_AAD)

    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "version": 1,
        "alg": "RSA-OAEP-SHA256+AES-256-GCM",
        "encrypted_key": base64.b64encode(encrypted_key).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Encrypt prediction images for issue-based evaluator")
    parser.add_argument("--input-dir", required=True, help="Directory with prediction images")
    parser.add_argument("--public-key", required=True, help="Public key PEM path")
    parser.add_argument("--output", required=True, help="Output encrypted payload JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    public_key_path = Path(args.public_key).resolve()
    output_path = Path(args.output).resolve()

    files = _collect_files(input_dir)
    tar_bytes = _build_tar_bytes(input_dir, files)
    payload = _encrypt_tar_bytes(tar_bytes, public_key_path.read_bytes())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote encrypted submission: {output_path}")
    print(f"Packed {len(files)} images from: {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
