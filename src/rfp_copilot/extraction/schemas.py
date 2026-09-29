"""
schemas.py — structured output contracts for RFP extraction.

These are the shapes an LLM's output gets validated against. Kept as
separate models (not one generic "ExtractedItem") because each type is
used differently downstream:
  - Requirement    -> needs a proposal response describing how it's met
  - Question       -> needs a direct written answer
  - ComplianceItem -> gates eligibility (pass/fail, not a narrative response)
  - EvaluationCriterion -> tells the response-generation step what to emphasize
"""

from enum import Enum
from pydantic import BaseModel, Field


class Priority(str, Enum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"
    UNSPECIFIED = "unspecified"


class Requirement(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'REQ-1' or a generated slug if the source has no explicit numbering")
    text: str = Field(description="The requirement, as close to verbatim as possible")
    category: str = Field(description="e.g. 'technical', 'functional', 'security', 'staffing' — inferred from context")
    priority: Priority = Field(description="mandatory if the text uses 'must'/'shall', optional if 'should'/'may', unspecified otherwise")
    source_section: str | None = Field(default=None, description="The heading of the section this was extracted from")


class Question(BaseModel):
    id: str
    text: str = Field(description="The question the vendor is expected to answer")
    source_section: str | None = None


class ComplianceItem(BaseModel):
    id: str
    text: str = Field(description="The mandatory condition, as stated")
    consequence: str | None = Field(default=None, description="What happens if unmet, if stated (e.g. 'disqualified')")
    source_section: str | None = None


class EvaluationCriterion(BaseModel):
    id: str
    name: str = Field(description="Short name of the criterion, e.g. 'Technical approach'")
    weight: str | None = Field(default=None, description="Weight/points as stated, e.g. '35%' or '35 points'")
    source_section: str | None = None


class ExtractionResult(BaseModel):
    """The full structured output for one chunk (or one document)."""
    requirements: list[Requirement] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    compliance_items: list[ComplianceItem] = Field(default_factory=list)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Combine results from multiple chunks into one document-level result."""
        return ExtractionResult(
            requirements=self.requirements + other.requirements,
            questions=self.questions + other.questions,
            compliance_items=self.compliance_items + other.compliance_items,
            evaluation_criteria=self.evaluation_criteria + other.evaluation_criteria,
        )
