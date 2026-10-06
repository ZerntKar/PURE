"""Feature normalization and Sentires-Guide extraction for evaluation."""

from data.sentires import SentiresGuideExtractor, normalize_feature


def normalize_text(text: str) -> str:
    return normalize_feature(text)
