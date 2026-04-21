"""Hashing helpers used for reproducibility and artifact integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_sha256_text(content: str) -> str:
    """Return the SHA-256 hexadecimal digest of UTF-8 text.

    Args:
        content: Input text to hash.

    Returns:
        SHA-256 digest as a lowercase hexadecimal string.

    Examples:
        >>> stable_sha256_text("abc")
        'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    """

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hexadecimal digest of a file.

    Args:
        path: Path to the file.

    Returns:
        SHA-256 digest as a lowercase hexadecimal string.

    Examples:
        >>> digest = sha256_file(Path("README.md"))
        >>> len(digest)
        64
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
