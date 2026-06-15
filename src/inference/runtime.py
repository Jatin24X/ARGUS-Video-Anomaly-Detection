from __future__ import annotations

from .engine import ENGINE, PROFILES, _profile_payload


def profile_payload(profile: object) -> dict[str, object]:
    """Return the public profile contract without exposing demo internals."""
    return _profile_payload(profile)


def preload(*, include_extractor: bool = True) -> None:
    ENGINE.preload(
        include_extractor=include_extractor,
        profile_labels=list(PROFILES.keys()),
    )
