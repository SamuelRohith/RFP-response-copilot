import pytest
from pydantic import ValidationError
from rfp_copilot.extraction.schemas import Requirement


def test_requirement_category_is_validated():
    requirement = Requirement(id="REQ-1", text="Use SSO", category="security")
    assert requirement.category.value == "security"
    with pytest.raises(ValidationError):
        Requirement(id="REQ-2", text="Unknown", category="made_up")
