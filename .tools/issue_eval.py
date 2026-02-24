#!/usr/bin/env python3
"""
Issue-comment triggered PSNR evaluator.

Flow:
1) Parse /eval command for student id (and optional inline image URLs)
2) Decrypt encrypted GT archive stored in repo using private key from secret
3) Download student images from issue comment URLs
4) Compute PSNR and post markdown result
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

COMMAND_PATTERN = re.compile(r"^/eval\s+([A-Za-z0-9._-]{1,64})(?:\s+(.*))?$")
URL_PATTERN = re.compile(r"https://[^\s<>)\]\"']+")
IMAGE_MARKDOWN_URL_PATTERN = re.compile(r"!\[[^\]]*\]\((https://[^)]+)\)")
IMAGE_HTML_SRC_URL_PATTERN = re.compile(r"<img[^>]+src=[\"'](https://[^\"']+)[\"']", re.IGNORECASE)
POSTIMG_DIRECT_URL_PATTERN = re.compile(r"https://i\.postimg\.cc/[^\s\"'<>]+", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
    "media.githubusercontent.com",
    "user-images.githubusercontent.com",
    "githubusercontent.com",
    "postimg.cc",
    "i.postimg.cc",
}
MAX_DOWNLOAD_BYTES_DEFAULT = 30 * 1024 * 1024
MAX_TAR_FILES_DEFAULT = 200
MAX_TAR_BYTES_DEFAULT = 100 * 1024 * 1024
CRYPTO_AAD = b"roas5400-issue-eval-v1"


def parse_eval_command(comment_body: str) -> tuple[str, list[str]]:
    """
    Parse first /eval command line:
      /eval <student_id> [url1 url2 ...]
    """
    for line in comment_body.splitlines():
        text = line.strip()
        if not text.startswith("/eval"):
            continue
        match = COMMAND_PATTERN.match(text)
        if not match:
            raise ValueError("Invalid /eval command format. Use: /eval <student_id> [image_url ...]")

        student_id = match.group(1)
        tail = (match.group(2) or "").strip()
        inline_urls = URL_PATTERN.findall(tail)
        return student_id, inline_urls
    raise ValueError("No /eval command found in comment.")


def compute_psnr(pred_img: np.ndarray, gt_img: np.ndarray) -> float:
    if pred_img.shape != gt_img.shape:
        raise ValueError(f"Image shape mismatch: pred={pred_img.shape}, gt={gt_img.shape}")
    pred = pred_img.astype(np.float64)
    gt = gt_img.astype(np.float64)
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def validate_and_list_tar_members(tar_path: Path, max_files: int, max_total_bytes: int) -> list[str]:
    file_count = 0
    total_bytes = 0
    names: list[str] = []

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Tar contains unsupported member type: {member.name}")

            normalized = member.name.replace("\\", "/")
            rel = PurePosixPath(normalized)
            if rel.is_absolute() or ".." in rel.parts or "." in rel.parts:
                raise ValueError(f"Unsafe tar path: {member.name}")
            if not rel.parts:
                raise ValueError(f"Invalid tar member path: {member.name}")

            file_count += 1
            total_bytes += member.size
            if file_count > max_files:
                raise ValueError(f"Tar has too many files: {file_count} > {max_files}")
            if total_bytes > max_total_bytes:
                raise ValueError(f"Tar uncompressed size too large: {total_bytes} > {max_total_bytes}")
            names.append(str(rel))

    if not names:
        raise ValueError("Tar archive is empty.")
    return names


def _extract_tar_safely(tar_path: Path, out_dir: Path, max_files: int, max_total_bytes: int) -> list[Path]:
    names = validate_and_list_tar_members(tar_path, max_files=max_files, max_total_bytes=max_total_bytes)
    extracted_paths: list[Path] = []

    with tarfile.open(tar_path, "r:*") as tf:
        for name in names:
            member = tf.getmember(name)
            dst = out_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_file = tf.extractfile(member)
            if src_file is None:
                raise ValueError(f"Unable to extract tar member: {name}")
            with src_file, dst.open("wb") as out:
                shutil.copyfileobj(src_file, out)
            extracted_paths.append(dst)
    return extracted_paths


def _validate_https_url(source: str) -> None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "https":
        raise ValueError("Submission URL must be HTTPS.")
    if not parsed.hostname:
        raise ValueError("Submission URL has no hostname.")
    host = parsed.hostname.lower()
    if host not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"Submission host is not allowed: {host}")


def _fetch_https_bytes(source: str, max_bytes: int) -> bytes:
    _validate_https_url(source)
    parsed = urllib.parse.urlparse(source)
    host = (parsed.hostname or "").lower()
    headers = {
        "User-Agent": "roas5400-issue-eval/1.0",
        "Accept": "application/octet-stream,*/*;q=0.8",
    }
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if gh_token and host in {"github.com", "api.github.com"}:
        # github.com web endpoints are more compatible with `token` auth style.
        headers["Authorization"] = f"token {gh_token}"

    if host == "postimg.cc":
        # postimg.cc links are HTML pages; resolve to the actual CDN image URL first.
        html = _http_get_bytes(source=source, headers=headers, max_bytes=max_bytes).decode(
            "utf-8", errors="replace"
        )
        direct_url = _extract_postimg_direct_url(html)
        if not direct_url:
            raise ValueError(f"Unable to resolve direct image URL from postimg page: {source}")
        return _http_get_bytes(source=direct_url, headers=headers, max_bytes=max_bytes)

    return _http_get_bytes(source=source, headers=headers, max_bytes=max_bytes)


def _http_get_bytes(source: str, headers: dict[str, str], max_bytes: int) -> bytes:
    req = urllib.request.Request(source, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = 0
        chunks: list[bytes] = []
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Submission download too large: {total} > {max_bytes}")
            chunks.append(chunk)
    return b"".join(chunks)


def _extract_postimg_direct_url(html_text: str) -> str | None:
    match = POSTIMG_DIRECT_URL_PATTERN.search(html_text)
    if not match:
        return None
    return match.group(0)


def _decrypt_payload_to_tar_bytes(payload_path: Path, private_key_pem: str) -> bytes:
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Encrypted payload is not valid JSON.") from exc

    required = {"version", "alg", "encrypted_key", "nonce", "ciphertext"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Encrypted payload missing keys: {sorted(missing)}")
    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version: {payload['version']}")
    if payload["alg"] != "RSA-OAEP-SHA256+AES-256-GCM":
        raise ValueError(f"Unsupported algorithm: {payload['alg']}")

    try:
        encrypted_key = base64.b64decode(payload["encrypted_key"])
        nonce = base64.b64decode(payload["nonce"])
        ciphertext = base64.b64decode(payload["ciphertext"])
    except Exception as exc:
        raise ValueError("Encrypted payload fields are not valid base64.") from exc

    private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    tar_bytes = AESGCM(aes_key).decrypt(nonce, ciphertext, CRYPTO_AAD)
    return tar_bytes


def _iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        name = path.name
        if rel.startswith("__MACOSX/") or name.startswith("._") or name == ".DS_Store":
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _align_gt_to_pred_shape(gt_arr: np.ndarray, pred_arr: np.ndarray) -> np.ndarray:
    target_h, target_w = int(pred_arr.shape[0]), int(pred_arr.shape[1])
    pil = Image.fromarray(gt_arr, mode="RGB")
    aligned = ImageOps.fit(
        pil,
        (target_w, target_h),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    return np.asarray(aligned, dtype=np.uint8)


def evaluate_predictions(
    pred_dir: Path,
    gt_dir: Path,
    shape_policy: str = "align_gt_to_pred",
) -> dict[str, object]:
    pred_map = {p.relative_to(pred_dir).as_posix(): p for p in _iter_image_files(pred_dir)}
    gt_map = {p.relative_to(gt_dir).as_posix(): p for p in _iter_image_files(gt_dir)}

    if not gt_map:
        raise ValueError("Ground-truth archive contains no images.")
    if not pred_map:
        raise ValueError("Submission archive contains no images.")

    missing = sorted(set(gt_map) - set(pred_map))
    extra = sorted(set(pred_map) - set(gt_map))
    if missing:
        raise ValueError(f"Submission is missing files: {missing[:5]}")
    if extra:
        raise ValueError(f"Submission has unexpected files: {extra[:5]}")

    per_image: list[dict[str, object]] = []
    psnr_values: list[float] = []
    aligned_count = 0
    for rel in sorted(gt_map):
        pred_arr = _load_rgb_uint8(pred_map[rel])
        gt_arr = _load_rgb_uint8(gt_map[rel])
        if pred_arr.shape != gt_arr.shape:
            if shape_policy == "strict":
                raise ValueError(f"Image shape mismatch: pred={pred_arr.shape}, gt={gt_arr.shape}")
            if shape_policy == "align_gt_to_pred":
                gt_arr = _align_gt_to_pred_shape(gt_arr, pred_arr)
                aligned_count += 1
            else:
                raise ValueError(f"Unknown shape policy: {shape_policy}")
        score = compute_psnr(pred_arr, gt_arr)
        psnr_values.append(score)
        per_image.append({"file": rel, "psnr_db": score})

    mean_psnr = float(np.mean(psnr_values))
    return {
        "num_images": len(psnr_values),
        "mean_psnr_db": mean_psnr,
        "per_image": per_image,
        "shape_policy": shape_policy,
        "aligned_images": aligned_count,
    }


def collect_submission_urls(comment_body: str, inline_urls: list[str]) -> list[str]:
    urls: list[str] = []

    # 1) explicit URLs on the /eval line have highest priority
    urls.extend(inline_urls)

    # 2) markdown image links anywhere in comment
    urls.extend(IMAGE_MARKDOWN_URL_PATTERN.findall(comment_body))

    # 3) HTML img src links
    urls.extend(IMAGE_HTML_SRC_URL_PATTERN.findall(comment_body))

    # 4) plain HTTPS URLs on non-command lines (for pasted attachment links)
    for line in comment_body.splitlines():
        text = line.strip()
        if not text or text.startswith("/eval"):
            continue
        urls.extend(URL_PATTERN.findall(text))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        _validate_https_url(url)
        seen.add(url)
        deduped.append(url)
    return deduped


def _write_submission_images_by_gt_order(
    comment_body: str,
    pred_dir: Path,
    gt_dir: Path,
    max_download_bytes: int,
) -> tuple[list[str], list[str]]:
    _student_id, inline_urls = parse_eval_command(comment_body)
    submission_urls = collect_submission_urls(comment_body, inline_urls=inline_urls)
    if not submission_urls:
        raise ValueError("No submission image URLs found. Upload images in comment and include /eval.")

    gt_rel_paths = [p.relative_to(gt_dir).as_posix() for p in _iter_image_files(gt_dir)]
    if not gt_rel_paths:
        raise ValueError("Ground-truth archive contains no images.")

    if len(submission_urls) != len(gt_rel_paths):
        raise ValueError(
            f"URL/image count mismatch: got {len(submission_urls)} URLs, expected {len(gt_rel_paths)} images."
        )

    for url, rel in zip(submission_urls, gt_rel_paths):
        raw = _fetch_https_bytes(url, max_bytes=max_download_bytes)
        try:
            arr = np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ValueError(f"Downloaded file is not a valid image: {url}") from exc

        out_path = pred_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="RGB").save(out_path)

    return gt_rel_paths, submission_urls


def _format_success_markdown(student_id: str, submission_urls: list[str], result: dict[str, object]) -> str:
    lines = [
        "### PSNR Evaluation Result",
        f"- Student: `{student_id}`",
        f"- Submitted Images: `{len(submission_urls)}`",
        f"- Images Evaluated: `{result['num_images']}`",
        f"- Mean PSNR: **`{result['mean_psnr_db']:.4f} dB`**",
        f"- Shape Policy: `{result.get('shape_policy', 'unknown')}` (aligned `{result.get('aligned_images', 0)}` image(s))",
        "",
        "| Image | PSNR (dB) |",
        "| --- | ---: |",
    ]
    for idx, item in enumerate(result["per_image"], start=1):
        lines.append(f"| `{idx}` | `{item['psnr_db']:.4f}` |")
    return "\n".join(lines)


def _format_failure_markdown(student_id: str | None, message: str) -> str:
    student_text = f"`{student_id}`" if student_id else "`(unknown)`"
    return "\n".join(
        [
            "### PSNR Evaluation Failed",
            f"- Student: {student_text}",
            f"- Reason: `{message}`",
            "",
            "Expected command:",
            "`/eval <student_id> [image_url ...]`",
        ]
    )


def _run_issue_eval(
    comment_body: str,
    result_md_path: Path,
    private_key_env: str,
    gt_encrypted_path: Path,
    max_download_bytes: int,
    max_tar_files: int,
    max_tar_bytes: int,
    shape_policy: str,
) -> int:
    student_id: str | None = None
    try:
        student_id, _inline_urls = parse_eval_command(comment_body)

        private_key_pem = os.environ.get(private_key_env, "")
        if not private_key_pem.strip():
            raise ValueError(f"Missing secret env `{private_key_env}`.")

        if not gt_encrypted_path.is_file():
            raise ValueError(f"Encrypted GT file not found: {gt_encrypted_path}")

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            gt_tar = tmp_dir / "gt.tar"
            pred_extract_dir = tmp_dir / "pred"
            gt_extract_dir = tmp_dir / "gt"

            gt_tar.write_bytes(_decrypt_payload_to_tar_bytes(gt_encrypted_path, private_key_pem))

            pred_extract_dir.mkdir(parents=True, exist_ok=True)
            gt_extract_dir.mkdir(parents=True, exist_ok=True)
            _extract_tar_safely(
                gt_tar,
                gt_extract_dir,
                max_files=max_tar_files,
                max_total_bytes=max_tar_bytes,
            )

            _gt_rel, submission_urls = _write_submission_images_by_gt_order(
                comment_body=comment_body,
                pred_dir=pred_extract_dir,
                gt_dir=gt_extract_dir,
                max_download_bytes=max_download_bytes,
            )

            result = evaluate_predictions(
                pred_extract_dir,
                gt_extract_dir,
                shape_policy=shape_policy,
            )
            result_md_path.write_text(
                _format_success_markdown(student_id, submission_urls, result), encoding="utf-8"
            )
            return 0
    except Exception as exc:
        result_md_path.write_text(_format_failure_markdown(student_id, str(exc)), encoding="utf-8")
        return 1


def _handle_parse_comment(args: argparse.Namespace) -> int:
    if args.comment_file:
        body = Path(args.comment_file).read_text(encoding="utf-8")
    else:
        body = args.comment_body
    student_id, inline_urls = parse_eval_command(body)
    all_urls = collect_submission_urls(body, inline_urls=inline_urls)

    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as out:
            out.write(f"student_id={student_id}\n")
            out.write(f"submission_url_count={len(all_urls)}\n")
    else:
        print(json.dumps({"student_id": student_id, "submission_urls": all_urls}, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue-comment PSNR evaluation helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_p = subparsers.add_parser("parse-comment", help="Parse /eval command from issue comment")
    parse_p.add_argument("--comment-body", default="", help="Raw comment body text")
    parse_p.add_argument("--comment-file", help="Path to file containing comment body")
    parse_p.add_argument("--github-output", help="Optional GITHUB_OUTPUT path for key=value output")
    parse_p.set_defaults(func=_handle_parse_comment)

    run_p = subparsers.add_parser("issue-run", help="Run end-to-end issue comment evaluation")
    run_p.add_argument("--comment-body", required=True, help="Raw issue comment body")
    run_p.add_argument("--result-md", required=True, help="Where to write markdown result")
    run_p.add_argument("--private-key-env", default="EVAL_PRIVATE_KEY_PEM")
    run_p.add_argument("--gt-encrypted", default=".github/eval/gt.enc.json")
    run_p.add_argument("--max-download-bytes", type=int, default=MAX_DOWNLOAD_BYTES_DEFAULT)
    run_p.add_argument("--max-tar-files", type=int, default=MAX_TAR_FILES_DEFAULT)
    run_p.add_argument("--max-tar-bytes", type=int, default=MAX_TAR_BYTES_DEFAULT)
    run_p.add_argument(
        "--shape-policy",
        choices=["strict", "align_gt_to_pred"],
        default="align_gt_to_pred",
        help="How to handle pred/gt size mismatch.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "issue-run":
        return _run_issue_eval(
            comment_body=args.comment_body,
            result_md_path=Path(args.result_md),
            private_key_env=args.private_key_env,
            gt_encrypted_path=Path(args.gt_encrypted),
            max_download_bytes=args.max_download_bytes,
            max_tar_files=args.max_tar_files,
            max_tar_bytes=args.max_tar_bytes,
            shape_policy=args.shape_policy,
        )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
