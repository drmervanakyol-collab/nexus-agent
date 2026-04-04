"""
nexus/perception/matcher/matcher.py
Matcher — assigns semantic labels and affordances to detected UI elements.

This is a rule-based (no ML) V1 implementation.

SemanticLabel
-------------
  element_id      : ElementId
  primary_label   : str          human-readable name for the element
  secondary_labels: list[str]    alternative or supplementary labels
  confidence      : float        [0.0, 1.0]
  affordance      : Affordance   what the user can do with this element
  is_destructive  : bool         True when the action can cause data loss

Affordance
----------
  CLICKABLE   — button, link, icon, tab, menu
  TYPEABLE    — text input, search box
  SCROLLABLE  — scrollbar, list
  SELECTABLE  — checkbox, radio, dropdown item
  READ_ONLY   — label, static text, image
  UNKNOWN     — could not determine

Rules (applied in order)
------------------------
  1. Map ElementType → base affordance.
  2. Normalise element text (lower-case, strip).
  3. Lookup text in the label dictionary (Turkish + English).
  4. If text matches a known label, use it as primary_label and
     refine the affordance if needed.
  5. Mark is_destructive when text or element matches a destructive set.
  6. Compute confidence from (type quality, text match quality).

Confidence tiers
----------------
  0.90 — known type  + text matches known label
  0.70 — known type  + text present but unrecognised
  0.60 — UNKNOWN type + text matches known label
  0.50 — known type  + no text
  0.30 — UNKNOWN type + no/unrecognised text
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from nexus.core.types import ElementId
from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import ElementType, UIElement

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Affordance
# ---------------------------------------------------------------------------


class Affordance(Enum):
    CLICKABLE = auto()
    TYPEABLE = auto()
    SCROLLABLE = auto()
    SELECTABLE = auto()
    READ_ONLY = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# SemanticLabel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticLabel:
    """
    Semantic interpretation of a single UI element.

    Attributes
    ----------
    element_id:
        Refers back to ``UIElement.id``.
    primary_label:
        Best human-readable name for this element.
    secondary_labels:
        Additional interpretations (e.g. element-type name when a text
        match overrides it, or related synonyms).
    confidence:
        Heuristic confidence in [0.0, 1.0].
    affordance:
        What the user can do with this element.
    is_destructive:
        True when activating this element may cause irreversible data loss
        (delete, discard, close without saving, etc.).
    """

    element_id: ElementId
    primary_label: str
    secondary_labels: tuple[str, ...]
    confidence: float
    affordance: Affordance
    is_destructive: bool


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

# Base affordance per ElementType
_TYPE_AFFORDANCE: dict[ElementType, Affordance] = {
    ElementType.BUTTON:    Affordance.CLICKABLE,
    ElementType.INPUT:     Affordance.TYPEABLE,
    ElementType.LABEL:     Affordance.READ_ONLY,
    ElementType.IMAGE:     Affordance.READ_ONLY,
    ElementType.ICON:      Affordance.CLICKABLE,
    ElementType.CONTAINER: Affordance.UNKNOWN,
    ElementType.PANEL:     Affordance.UNKNOWN,
    ElementType.MENU:      Affordance.CLICKABLE,
    ElementType.DROPDOWN:  Affordance.SELECTABLE,
    ElementType.CHECKBOX:  Affordance.SELECTABLE,
    ElementType.RADIO:     Affordance.SELECTABLE,
    ElementType.LINK:      Affordance.CLICKABLE,
    ElementType.SCROLLBAR: Affordance.SCROLLABLE,
    ElementType.TAB:       Affordance.CLICKABLE,
    ElementType.DIALOG:    Affordance.UNKNOWN,
    ElementType.TOOLTIP:   Affordance.READ_ONLY,
    ElementType.UNKNOWN:   Affordance.UNKNOWN,
}

# "Known" types: anything other than UNKNOWN improves confidence
_KNOWN_TYPES: frozenset[ElementType] = frozenset(
    t for t in ElementType if t is not ElementType.UNKNOWN
)

# label_key (lower-case normalised text) → (display_label, affordance_override | None)
# affordance_override=None means "keep the type-based affordance"
_LABEL_MAP: dict[str, tuple[str, Affordance | None]] = {
    # ── Turkish ──────────────────────────────────────────────────────────
    "kaydet":    ("Kaydet",   Affordance.CLICKABLE),
    "iptal":     ("İptal",    Affordance.CLICKABLE),
    "sil":       ("Sil",      Affordance.CLICKABLE),
    "düzenle":   ("Düzenle",  Affordance.CLICKABLE),
    "ara":       ("Ara",      Affordance.CLICKABLE),
    "filtrele":  ("Filtrele", Affordance.CLICKABLE),
    "ekle":      ("Ekle",     Affordance.CLICKABLE),
    "kaldır":    ("Kaldır",   Affordance.CLICKABLE),
    "tamam":     ("Tamam",    Affordance.CLICKABLE),
    "evet":      ("Evet",     Affordance.CLICKABLE),
    "hayır":     ("Hayır",    Affordance.CLICKABLE),
    "gönder":    ("Gönder",   Affordance.CLICKABLE),
    "oluştur":   ("Oluştur",  Affordance.CLICKABLE),
    "güncelle":  ("Güncelle", Affordance.CLICKABLE),
    "onayla":    ("Onayla",   Affordance.CLICKABLE),
    "vazgeç":    ("Vazgeç",   Affordance.CLICKABLE),
    "kapat":     ("Kapat",    Affordance.CLICKABLE),
    "geri":      ("Geri",     Affordance.CLICKABLE),
    "ileri":     ("İleri",    Affordance.CLICKABLE),
    "yükle":     ("Yükle",    Affordance.CLICKABLE),
    "indir":     ("İndir",    Affordance.CLICKABLE),
    "görüntüle": ("Görüntüle", Affordance.CLICKABLE),
    "paylaş":    ("Paylaş",   Affordance.CLICKABLE),
    # ── English ──────────────────────────────────────────────────────────
    "save":      ("Save",     Affordance.CLICKABLE),
    "cancel":    ("Cancel",   Affordance.CLICKABLE),
    "delete":    ("Delete",   Affordance.CLICKABLE),
    "remove":    ("Remove",   Affordance.CLICKABLE),
    "edit":      ("Edit",     Affordance.CLICKABLE),
    "search":    ("Search",   Affordance.CLICKABLE),
    "filter":    ("Filter",   Affordance.CLICKABLE),
    "add":       ("Add",      Affordance.CLICKABLE),
    "ok":        ("OK",       Affordance.CLICKABLE),
    "yes":       ("Yes",      Affordance.CLICKABLE),
    "no":        ("No",       Affordance.CLICKABLE),
    "send":      ("Send",     Affordance.CLICKABLE),
    "submit":    ("Submit",   Affordance.CLICKABLE),
    "create":    ("Create",   Affordance.CLICKABLE),
    "update":    ("Update",   Affordance.CLICKABLE),
    "confirm":   ("Confirm",  Affordance.CLICKABLE),
    "close":     ("Close",    Affordance.CLICKABLE),
    "discard":   ("Discard",  Affordance.CLICKABLE),
    "back":      ("Back",     Affordance.CLICKABLE),
    "next":      ("Next",     Affordance.CLICKABLE),
    "upload":    ("Upload",   Affordance.CLICKABLE),
    "download":  ("Download", Affordance.CLICKABLE),
    "view":      ("View",     Affordance.CLICKABLE),
    "share":     ("Share",    Affordance.CLICKABLE),
    "open":      ("Open",     Affordance.CLICKABLE),
    "login":     ("Login",    Affordance.CLICKABLE),
    "logout":    ("Logout",   Affordance.CLICKABLE),
    "sign in":   ("Sign In",  Affordance.CLICKABLE),
    "sign up":   ("Sign Up",  Affordance.CLICKABLE),
    "sign out":  ("Sign Out", Affordance.CLICKABLE),
    "refresh":   ("Refresh",  Affordance.CLICKABLE),
    "reset":     ("Reset",    Affordance.CLICKABLE),
    "apply":     ("Apply",    Affordance.CLICKABLE),
    "clear":     ("Clear",    Affordance.CLICKABLE),
    "select":    ("Select",   Affordance.SELECTABLE),
    "check":     ("Check",    Affordance.SELECTABLE),
    "uncheck":   ("Uncheck",  Affordance.SELECTABLE),
}

# Words (lower-case) that make is_destructive=True
_DESTRUCTIVE_WORDS: frozenset[str] = frozenset([
    # Turkish
    "sil", "kaldır", "vazgeç", "iptal", "kapat",
    # English
    "delete", "remove", "discard", "cancel", "close",
    # Compound forms matched as substrings
])

# Regex for destructive: match whole-word to avoid false positives
_DESTRUCTIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_DESTRUCTIVE_WORDS)) + r")\b",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


class Matcher:
    """
    Assigns semantic labels and affordances to UI elements using rule-based
    heuristics.

    This is a stateless, deterministic class — no internal mutable state.
    """

    def match(
        self,
        elements: Sequence[UIElement],
        element_texts: dict[ElementId, str],
    ) -> list[SemanticLabel]:
        """
        Produce a SemanticLabel for each element in *elements*.

        Parameters
        ----------
        elements:
            Detected UI elements (from Locator).
        element_texts:
            OCR text per element id (from Reader / ReaderOutput.element_texts).
            Missing entries are treated as empty text.

        Returns
        -------
        list[SemanticLabel]
            One label per element, in the same order as *elements*.
        """
        labels: list[SemanticLabel] = []
        for el in elements:
            text = element_texts.get(el.id, "").strip()
            label = self._label_element(el, text)
            labels.append(label)
            _log.debug(
                "matcher_labeled",
                element_id=str(el.id)[:8],
                primary=label.primary_label,
                affordance=label.affordance.name,
                destructive=label.is_destructive,
                confidence=label.confidence,
            )
        return labels

    # ------------------------------------------------------------------
    # Core labelling logic
    # ------------------------------------------------------------------

    def _label_element(self, el: UIElement, text: str) -> SemanticLabel:
        el_type = el.element_type
        type_is_known = el_type in _KNOWN_TYPES

        # Base affordance from element type
        base_affordance = _TYPE_AFFORDANCE.get(el_type, Affordance.UNKNOWN)

        # Normalised text key for lookup
        norm = _normalise(text)
        entry = _LABEL_MAP.get(norm)

        if entry is not None:
            display_label, affordance_override = entry
            affordance = (
                affordance_override
                if affordance_override is not None
                else base_affordance
            )
            is_destructive = _is_destructive(norm)
            # Secondary: include element type name if it differs from the label
            secondary = _secondary_labels(el_type, display_label, text)
            confidence = 0.90 if type_is_known else 0.60
        else:
            affordance = base_affordance
            is_destructive = _is_destructive(norm)
            display_label = text if text else el_type.name.capitalize()
            secondary = _secondary_labels(el_type, display_label, text)
            confidence = (0.70 if text else 0.50) if type_is_known else 0.30

        return SemanticLabel(
            element_id=el.id,
            primary_label=display_label,
            secondary_labels=tuple(secondary),
            confidence=round(confidence, 3),
            affordance=affordance,
            is_destructive=is_destructive,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """
    Lower-case, collapse whitespace, strip punctuation at the edges.

    Handles Turkish İ (U+0130) explicitly before Python's locale-unaware
    str.lower(), which would otherwise produce "i\u0307" (i + combining dot).
    """
    text = text.replace("\u0130", "i")   # İ → i  (Turkish capital I with dot above)
    return re.sub(r"\s+", " ", text.lower().strip(" .,!?;:\"'"))


def _is_destructive(norm: str) -> bool:
    """Return True if *norm* (already lower-cased) contains a destructive word."""
    return bool(_DESTRUCTIVE_RE.search(norm))


def _secondary_labels(
    el_type: ElementType,
    primary: str,
    raw_text: str,
) -> list[str]:
    """
    Build the secondary_labels list.

    - Include the element type name when it adds information beyond the primary.
    - Include the raw text when it differs from the primary (e.g. mixed-case).
    """
    secondaries: list[str] = []
    type_name = el_type.name.capitalize()
    if type_name.lower() != primary.lower():
        secondaries.append(type_name)
    if raw_text and raw_text != primary and raw_text.lower() != primary.lower():
        secondaries.append(raw_text)
    return secondaries
