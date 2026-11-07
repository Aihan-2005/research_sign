from persian_sea.alignment.semantic_dp import (
    SemanticDPResult,
    SemanticDPSettings,
    align_semantic_dp,
)
from persian_sea.alignment.sementic_db import (
    SemanticDPResult as LegacyResult,
    SemanticDPSettings as LegacySettings,
    align_semantic_dp as legacy_align,
)


def test_correct_module_name_and_legacy_shim_resolve_same_symbols():
    assert SemanticDPSettings is LegacySettings
    assert SemanticDPResult is LegacyResult
    assert align_semantic_dp is legacy_align
