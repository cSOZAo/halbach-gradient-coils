from string import Formatter

import pytest

from halbach_coils.gui.i18n import DEFAULT_LANGUAGE, LANGUAGES, TRANSLATIONS, Localizer


def _fields(template):
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def test_gui_defaults_to_english_and_falls_back_to_source_text():
    localizer = Localizer()

    assert DEFAULT_LANGUAGE == "en"
    assert localizer.translate("Settings") == "Settings"
    assert localizer.translate("An untranslated source string") == "An untranslated source string"


def test_spanish_translation_supports_dynamic_values():
    localizer = Localizer("es")

    assert localizer.translate("Settings") == "Configuración"
    assert localizer.translate(
        "Done.\nResults in:\n{path}", path="C:/results"
    ) == "Listo.\nResultados en:\nC:/results"


def test_all_translations_preserve_format_fields():
    for language, translations in TRANSLATIONS.items():
        assert language in LANGUAGES
        for source, translated in translations.items():
            assert _fields(translated) == _fields(source), (language, source)


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported GUI language"):
        Localizer().set_language("fr")
