from __future__ import annotations

from dataclasses import dataclass, field

from esg.models import NumericRange, Zone, ZoneElement


@dataclass(frozen=True, slots=True)
class ZoneMatch:
    status: str
    details: list[str] = field(default_factory=list)
    zone_index: int = -1


def compare_zones(query: Zone, candidate: Zone) -> ZoneMatch:
    details: list[str] = []
    unknown = False

    for field_name, label in (
        ("structure", "конструкция"),
        ("system", "система"),
        ("region", "область"),
        ("side", "сторона"),
        ("surface", "поверхность"),
    ):
        expected = getattr(query, field_name)
        actual = getattr(candidate, field_name)
        if not expected:
            continue
        if not actual:
            unknown = True
        elif expected != actual:
            return ZoneMatch("CONFLICT", [f"{label}: {expected} != {actual}"])

    if query.components:
        if not candidate.components:
            unknown = True
        elif not set(query.components).intersection(candidate.components):
            return ZoneMatch(
                "CONFLICT",
                [f"компоненты: {', '.join(query.components)} != {', '.join(candidate.components)}"],
            )

    for expected in query.elements:
        compatible, element_unknown, detail = _match_element(expected, candidate.elements)
        if not compatible:
            return ZoneMatch("CONFLICT", [detail])
        unknown = unknown or element_unknown
        if detail:
            details.append(detail)

    if not query.elements and not any(
        (query.structure, query.system, query.region, query.side, query.surface, query.components)
    ):
        return ZoneMatch("UNKNOWN", [])
    return ZoneMatch("UNKNOWN" if unknown else "MATCH", details)


def best_zone_match(query: Zone, candidates: list[Zone]) -> ZoneMatch:
    if not candidates:
        return ZoneMatch("UNKNOWN", [])
    rank = {"MATCH": 0, "UNKNOWN": 1, "CONFLICT": 2}
    results = [
        (ZoneMatch(result.status, result.details, index), candidate)
        for index, candidate in enumerate(candidates)
        for result in [compare_zones(query, candidate)]
    ]
    return min(
        results,
        key=lambda item: (rank[item[0].status], -_zone_specificity(item[1])),
    )[0]


def _zone_specificity(zone: Zone) -> int:
    categorical = sum(bool(value) for value in (zone.structure, zone.system, zone.region, zone.side, zone.surface))
    numbered_elements = sum(item.start is not None and item.end is not None for item in zone.elements)
    return len(zone.elements) + numbered_elements + len(zone.components) + categorical


def ranges_compatible(query: NumericRange, start: object, end: object) -> bool:
    if start is None or end is None:
        return True
    return int(start) <= query.end and int(end) >= query.start


def interval_intersection(left: ZoneElement, right: ZoneElement) -> tuple[int, int] | None:
    if left.start is None or left.end is None or right.start is None or right.end is None:
        return None
    start = max(left.start, right.start)
    end = min(left.end, right.end)
    return (start, end) if start <= end else None


def _match_element(expected: ZoneElement, candidates: list[ZoneElement]) -> tuple[bool, bool, str]:
    same_kind = [item for item in candidates if item.kind == expected.kind]
    if not same_kind:
        return True, True, f"{expected.kind}: нет данных в документе"

    compatible_qualifier = [
        item for item in same_kind
        if not expected.qualifier or not item.qualifier or item.qualifier == expected.qualifier
    ]
    if not compatible_qualifier:
        return False, False, f"{expected.kind}: несовместимый квалификатор {expected.qualifier}"

    role_unknown = False
    if expected.role:
        matching_role = [item for item in compatible_qualifier if item.role == expected.role]
        if matching_role:
            compatible_qualifier = matching_role
        else:
            role_unknown = True

    if expected.start is None or expected.end is None:
        detail = f"{expected.kind}: элемент найден"
        if role_unknown:
            detail += f", роль {expected.role} не подтверждена"
        return True, role_unknown, detail

    numbered = [item for item in compatible_qualifier if item.start is not None and item.end is not None]
    if not numbered:
        return True, True, f"{expected.kind}: номер не указан в документе"
    for item in numbered:
        intersection = interval_intersection(expected, item)
        if intersection:
            start, end = intersection
            value = str(start) if start == end else f"{start}-{end}"
            detail = f"{expected.kind}: пересечение {value}"
            if role_unknown:
                detail += f", роль {expected.role} не подтверждена"
            return True, role_unknown, detail
    return False, False, f"{expected.kind}: интервалы не пересекаются"
