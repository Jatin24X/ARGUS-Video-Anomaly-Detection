"""Stable inference surface shared by local and hosted applications."""

from .runtime import ENGINE, PROFILES, preload, profile_payload

__all__ = ["ENGINE", "PROFILES", "preload", "profile_payload"]
