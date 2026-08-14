from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NumericRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    end: int

    @model_validator(mode="after")
    def ordered(self) -> "NumericRange":
        if self.start > self.end:
            self.start, self.end = self.end, self.start
        return self


class ZoneElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    start: int | None = None
    end: int | None = None
    qualifier: str = ""
    role: str = ""

    @model_validator(mode="after")
    def normalize(self) -> "ZoneElement":
        self.kind = self.kind.strip().casefold()
        self.qualifier = self.qualifier.strip().casefold()
        role = self.role.strip().casefold()
        self.role = role if role in {"target", "boundary", "reference"} else ""
        if self.start is None and self.end is not None:
            self.start = self.end
        if self.end is None and self.start is not None:
            self.end = self.start
        if self.start is not None and self.end is not None and self.start > self.end:
            self.start, self.end = self.end, self.start
        return self

    @property
    def numeric_range(self) -> NumericRange | None:
        if self.start is None or self.end is None:
            return None
        return NumericRange(start=self.start, end=self.end)


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[ZoneElement] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    structure: str = ""
    system: str = ""
    region: str = ""
    side: str = ""
    surface: str = ""
    zone_text: str = ""

    @model_validator(mode="after")
    def normalize(self) -> "Zone":
        self.components = _unique_normalized(self.components)
        for field_name in ("structure", "system", "region", "side", "surface"):
            setattr(self, field_name, str(getattr(self, field_name)).strip().casefold())
        unique: list[ZoneElement] = []
        seen: set[tuple[object, ...]] = set()
        for element in self.elements:
            key = (element.kind, element.start, element.end, element.qualifier, element.role)
            if key not in seen:
                seen.add(key)
                unique.append(element)
        self.elements = unique
        return self

    def element(self, kind: str) -> ZoneElement | None:
        normalized = kind.casefold()
        return next((item for item in self.elements if item.kind == normalized), None)

    @property
    def frames(self) -> NumericRange | None:
        item = self.element("frame")
        return item.numeric_range if item else None

    @property
    def stringers(self) -> NumericRange | None:
        item = self.element("stringer")
        return item.numeric_range if item else None

    @property
    def ribs(self) -> NumericRange | None:
        item = self.element("rib")
        return item.numeric_range if item else None

    @property
    def component(self) -> str:
        return self.components[0] if self.components else ""


class QueryExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defect_type: str = ""
    zone: Zone = Field(default_factory=Zone)


class Repair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_id: str = ""
    evidence_text: str = Field(min_length=1)
    defect_type: str = ""
    section_heading: str = "Описание ремонта"
    zone_text: str = ""
    zones: list[Zone] = Field(default_factory=list)


class AnswerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    answer: str = Field(min_length=1)
    supporting_source_indexes: list[int] = Field(default_factory=list)


class SearchSource(BaseModel):
    document_id: str
    source_path: str
    section_heading: str
    heading_path: list[str] = Field(default_factory=list)
    evidence_text: str
    defect_type: str = ""
    repair_description: str = ""
    zone: Zone = Field(default_factory=Zone)
    zone_status: str = "UNKNOWN"
    zone_details: list[str] = Field(default_factory=list)
    retrieval_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    extraction_status: str = "ok"


def _unique_normalized(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))
