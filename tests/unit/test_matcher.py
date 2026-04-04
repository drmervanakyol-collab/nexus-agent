"""
tests/unit/test_matcher.py
Unit tests for nexus/perception/matcher/matcher.py — Faz 26.

Sections:
  1.  SemanticLabel value object
  2.  Affordance enum completeness
  3.  _normalise helper
  4.  _is_destructive helper
  5.  Matcher.match — Turkish patterns → CLICKABLE
  6.  Matcher.match — destructive labels
  7.  Matcher.match — element type → affordance mapping
  8.  Matcher.match — UNKNOWN type, low confidence
  9.  Matcher.match — known type + unrecognised text
  10. Matcher.match — no text → type-based fallback
  11. Matcher.match — secondary_labels populated
  12. Matcher.match — empty element list
  13. Matcher.match — missing element_texts entry
  14. Matcher.match — confidence tiers
"""
from __future__ import annotations

import uuid

import pytest

from nexus.core.types import ElementId, Rect
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import (
    Affordance,
    Matcher,
    SemanticLabel,
    _is_destructive,
    _normalise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _el(
    el_type: ElementType,
    x: int = 0,
    y: int = 0,
    w: int = 80,
    h: int = 30,
) -> UIElement:
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=el_type,
        bounding_box=Rect(x, y, w, h),
        confidence=0.9,
        is_visible=True,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=0,
    )


def _match_one(
    el: UIElement,
    text: str = "",
) -> SemanticLabel:
    """Run the matcher for a single element and return its label."""
    matcher = Matcher()
    results = matcher.match([el], {el.id: text})
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Section 1 — SemanticLabel value object
# ---------------------------------------------------------------------------


class TestSemanticLabel:
    def test_frozen(self):
        sl = SemanticLabel(
            element_id=ElementId("x"),
            primary_label="Kaydet",
            secondary_labels=("Button",),
            confidence=0.9,
            affordance=Affordance.CLICKABLE,
            is_destructive=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            sl.primary_label = "changed"  # type: ignore[misc]

    def test_fields_accessible(self):
        sl = SemanticLabel(
            element_id=ElementId("y"),
            primary_label="Sil",
            secondary_labels=(),
            confidence=0.9,
            affordance=Affordance.CLICKABLE,
            is_destructive=True,
        )
        assert sl.primary_label == "Sil"
        assert sl.is_destructive is True
        assert sl.affordance is Affordance.CLICKABLE


# ---------------------------------------------------------------------------
# Section 2 — Affordance enum completeness
# ---------------------------------------------------------------------------


class TestAffordanceEnum:
    _REQUIRED = {"CLICKABLE", "TYPEABLE", "SCROLLABLE", "SELECTABLE", "READ_ONLY", "UNKNOWN"}

    def test_all_required_members(self):
        names = {m.name for m in Affordance}
        assert self._REQUIRED <= names


# ---------------------------------------------------------------------------
# Section 3 — _normalise helper
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_lower_case(self):
        assert _normalise("KAYDET") == "kaydet"

    def test_strip_whitespace(self):
        assert _normalise("  Kaydet  ") == "kaydet"

    def test_collapse_internal_spaces(self):
        assert _normalise("Sign   In") == "sign in"

    def test_strip_punctuation(self):
        assert _normalise("Kaydet!") == "kaydet"

    def test_empty_string(self):
        assert _normalise("") == ""

    def test_turkish_lower_case(self):
        # Turkish İ → i handled by Python's str.lower()
        assert "iptal" in _normalise("İptal")


# ---------------------------------------------------------------------------
# Section 4 — _is_destructive helper
# ---------------------------------------------------------------------------


class TestIsDestructive:
    @pytest.mark.parametrize("word", [
        "sil", "delete", "remove", "discard", "cancel", "close",
        "kaldır", "vazgeç", "iptal", "kapat",
    ])
    def test_destructive_words(self, word: str):
        assert _is_destructive(word) is True

    @pytest.mark.parametrize("word", [
        "kaydet", "güncelle", "oluştur", "tamam", "evet", "save", "ok", "submit",
    ])
    def test_non_destructive_words(self, word: str):
        assert _is_destructive(word) is False

    def test_destructive_word_embedded_in_sentence(self):
        assert _is_destructive("dosyayı sil") is True

    def test_non_destructive_containing_substring(self):
        # "silme" contains "sil" but as a prefix, not a standalone word
        # The regex uses \b so "silme" should NOT match
        assert _is_destructive("silme işlemi") is False

    def test_case_insensitive(self):
        assert _is_destructive("DELETE") is True
        assert _is_destructive("Cancel") is True


# ---------------------------------------------------------------------------
# Section 5 — Turkish patterns → CLICKABLE
# ---------------------------------------------------------------------------


class TestTurkishPatterns:
    @pytest.mark.parametrize("text,expected_label", [
        ("Kaydet",    "Kaydet"),
        ("İptal",     "İptal"),
        ("Sil",       "Sil"),
        ("Düzenle",   "Düzenle"),
        ("Ara",       "Ara"),
        ("Filtrele",  "Filtrele"),
        ("Ekle",      "Ekle"),
        ("Kaldır",    "Kaldır"),
        ("Tamam",     "Tamam"),
        ("Evet",      "Evet"),
        ("Hayır",     "Hayır"),
        ("Gönder",    "Gönder"),
        ("Oluştur",   "Oluştur"),
        ("Güncelle",  "Güncelle"),
        ("Onayla",    "Onayla"),
        ("Vazgeç",    "Vazgeç"),
    ])
    def test_turkish_label_recognised(self, text: str, expected_label: str):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, text)
        assert label.primary_label == expected_label

    @pytest.mark.parametrize("text", [
        "Kaydet", "Ekle", "Tamam", "Evet", "Gönder",
        "Oluştur", "Güncelle", "Onayla",
    ])
    def test_turkish_buttons_are_clickable(self, text: str):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, text)
        assert label.affordance is Affordance.CLICKABLE

    def test_kaydet_not_destructive(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Kaydet")
        assert label.is_destructive is False

    def test_case_insensitive_match(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "KAYDET")
        assert label.primary_label == "Kaydet"


# ---------------------------------------------------------------------------
# Section 6 — destructive labels
# ---------------------------------------------------------------------------


class TestDestructiveLabels:
    @pytest.mark.parametrize("text", [
        "Sil", "Delete", "Remove", "Close", "İptal", "Cancel", "Discard",
        "Kapat", "Vazgeç", "Kaldır",
    ])
    def test_destructive_flag_set(self, text: str):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, text)
        assert label.is_destructive is True, f"Expected is_destructive=True for '{text}'"

    def test_sil_destructive_and_clickable(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Sil")
        assert label.is_destructive is True
        assert label.affordance is Affordance.CLICKABLE

    def test_tamam_not_destructive(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Tamam")
        assert label.is_destructive is False

    def test_kaydet_not_destructive(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Kaydet")
        assert label.is_destructive is False

    def test_english_cancel_destructive(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Cancel")
        assert label.is_destructive is True


# ---------------------------------------------------------------------------
# Section 7 — element type → affordance mapping
# ---------------------------------------------------------------------------


class TestTypeAffordanceMapping:
    @pytest.mark.parametrize("el_type,expected", [
        (ElementType.BUTTON,    Affordance.CLICKABLE),
        (ElementType.LINK,      Affordance.CLICKABLE),
        (ElementType.TAB,       Affordance.CLICKABLE),
        (ElementType.MENU,      Affordance.CLICKABLE),
        (ElementType.ICON,      Affordance.CLICKABLE),
        (ElementType.INPUT,     Affordance.TYPEABLE),
        (ElementType.SCROLLBAR, Affordance.SCROLLABLE),
        (ElementType.CHECKBOX,  Affordance.SELECTABLE),
        (ElementType.RADIO,     Affordance.SELECTABLE),
        (ElementType.DROPDOWN,  Affordance.SELECTABLE),
        (ElementType.LABEL,     Affordance.READ_ONLY),
        (ElementType.IMAGE,     Affordance.READ_ONLY),
        (ElementType.TOOLTIP,   Affordance.READ_ONLY),
    ])
    def test_type_to_affordance(self, el_type: ElementType, expected: Affordance):
        el = _el(el_type)
        label = _match_one(el, "")  # no text → type-driven
        assert label.affordance is expected

    def test_input_is_typeable(self):
        el = _el(ElementType.INPUT)
        label = _match_one(el, "E-posta")
        assert label.affordance is Affordance.TYPEABLE

    def test_scrollbar_is_scrollable(self):
        el = _el(ElementType.SCROLLBAR, w=12, h=200)
        label = _match_one(el, "")
        assert label.affordance is Affordance.SCROLLABLE

    def test_checkbox_is_selectable(self):
        el = _el(ElementType.CHECKBOX, w=20, h=20)
        label = _match_one(el, "")
        assert label.affordance is Affordance.SELECTABLE


# ---------------------------------------------------------------------------
# Section 8 — UNKNOWN type, low confidence
# ---------------------------------------------------------------------------


class TestUnknownType:
    def test_unknown_type_no_text_low_confidence(self):
        el = _el(ElementType.UNKNOWN)
        label = _match_one(el, "")
        assert label.confidence <= 0.35

    def test_unknown_type_with_known_text_medium_confidence(self):
        el = _el(ElementType.UNKNOWN)
        label = _match_one(el, "Kaydet")
        assert label.confidence >= 0.55
        assert label.confidence <= 0.65

    def test_unknown_type_affordance_unknown_without_text(self):
        el = _el(ElementType.UNKNOWN)
        label = _match_one(el, "")
        assert label.affordance is Affordance.UNKNOWN

    def test_unknown_type_with_known_label_gets_correct_affordance(self):
        el = _el(ElementType.UNKNOWN)
        label = _match_one(el, "Kaydet")
        assert label.affordance is Affordance.CLICKABLE


# ---------------------------------------------------------------------------
# Section 9 — known type + unrecognised text
# ---------------------------------------------------------------------------


class TestKnownTypeUnrecognisedText:
    def test_confidence_seventy(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Xyzzy")  # not in label map
        assert label.confidence == pytest.approx(0.70)

    def test_text_used_as_primary_label(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Xyzzy")
        assert label.primary_label == "Xyzzy"

    def test_not_destructive_by_default(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Xyzzy")
        assert label.is_destructive is False


# ---------------------------------------------------------------------------
# Section 10 — no text → type-based fallback
# ---------------------------------------------------------------------------


class TestNoText:
    def test_no_text_button_confidence_fifty(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "")
        assert label.confidence == pytest.approx(0.50)

    def test_no_text_primary_label_is_type_name(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "")
        assert label.primary_label.lower() == "button"

    def test_no_text_input_primary_label(self):
        el = _el(ElementType.INPUT)
        label = _match_one(el, "")
        assert label.primary_label.lower() == "input"


# ---------------------------------------------------------------------------
# Section 11 — secondary_labels populated
# ---------------------------------------------------------------------------


class TestSecondaryLabels:
    def test_secondary_includes_type_when_label_differs(self):
        # Text matches "Kaydet" → primary="Kaydet"; type=BUTTON → secondary has "Button"
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Kaydet")
        assert "Button" in label.secondary_labels

    def test_secondary_excludes_type_when_same_as_primary(self):
        # ElementType.BUTTON + no text → primary="Button"; type name == primary
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "")
        # primary is "Button", secondary should NOT duplicate it
        assert "Button" not in label.secondary_labels

    def test_secondary_is_tuple(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Kaydet")
        assert isinstance(label.secondary_labels, tuple)


# ---------------------------------------------------------------------------
# Section 12 — empty element list
# ---------------------------------------------------------------------------


class TestEmptyElements:
    def test_empty_returns_empty(self):
        matcher = Matcher()
        result = matcher.match([], {})
        assert result == []


# ---------------------------------------------------------------------------
# Section 13 — missing element_texts entry
# ---------------------------------------------------------------------------


class TestMissingText:
    def test_missing_text_treated_as_empty(self):
        el = _el(ElementType.BUTTON)
        matcher = Matcher()
        # element_texts does not have el.id
        result = matcher.match([el], {})
        assert len(result) == 1
        # Should not raise; primary label falls back to type name
        assert result[0].primary_label.lower() == "button"

    def test_missing_text_confidence_fifty_for_known_type(self):
        el = _el(ElementType.INPUT)
        matcher = Matcher()
        result = matcher.match([el], {})
        assert result[0].confidence == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Section 14 — confidence tiers
# ---------------------------------------------------------------------------


class TestConfidenceTiers:
    def test_known_type_known_label_is_ninety(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "Kaydet")
        assert label.confidence == pytest.approx(0.90)

    def test_known_type_no_text_is_fifty(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "")
        assert label.confidence == pytest.approx(0.50)

    def test_known_type_unknown_text_is_seventy(self):
        el = _el(ElementType.BUTTON)
        label = _match_one(el, "RandomText123")
        assert label.confidence == pytest.approx(0.70)

    def test_unknown_type_known_label_is_sixty(self):
        el = _el(ElementType.UNKNOWN)
        label = _match_one(el, "Kaydet")
        assert label.confidence == pytest.approx(0.60)

    def test_unknown_type_no_text_is_thirty(self):
        el = _el(ElementType.UNKNOWN)
        label = _match_one(el, "")
        assert label.confidence == pytest.approx(0.30)

    def test_confidence_in_range(self):
        for el_type in ElementType:
            el = _el(el_type)
            label = _match_one(el, "Kaydet")
            assert 0.0 <= label.confidence <= 1.0, (
                f"Confidence out of range for {el_type}: {label.confidence}"
            )
