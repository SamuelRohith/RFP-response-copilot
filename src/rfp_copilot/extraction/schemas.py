"""Validated Week 5 extraction contracts for RFP requirements and instructions."""

from enum import Enum
from pydantic import BaseModel, Field


class RequirementCategory(str, Enum):
    FUNCTIONAL = "functional"
    TECHNICAL = "technical"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    BUSINESS = "business"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    DELIVERY = "delivery"
    OTHER = "other"


class Priority(str, Enum):
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    UNKNOWN = "unknown"


class Requirement(BaseModel):
    id: str = Field(description="Short unique id, e.g. REQ-1")
    text: str
    category: RequirementCategory
    priority: Priority = Priority.UNKNOWN
    source_section: str | None = None


class Question(BaseModel):
    id: str
    text: str
    source_section: str | None = None


class ComplianceItem(BaseModel):
    id: str
    text: str
    consequence: str | None = None
    source_section: str | None = None


class EvaluationCriterion(BaseModel):
    id: str
    name: str
    weight: str | None = None
    source_section: str | None = None


class SubmissionInstruction(BaseModel):
    id: str
    text: str
    source_section: str | None = None


class ExtractionResult(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    compliance_items: list[ComplianceItem] = Field(default_factory=list)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)
    submission_instructions: list[SubmissionInstruction] = Field(default_factory=list)

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        return ExtractionResult(
            requirements=self.requirements + other.requirements,
            questions=self.questions + other.questions,
            compliance_items=self.compliance_items + other.compliance_items,
            evaluation_criteria=self.evaluation_criteria + other.evaluation_criteria,
            submission_instructions=self.submission_instructions + other.submission_instructions,
        )
