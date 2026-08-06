from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from deerui.persistence.models import DeerProfileRow


DEER_CODE_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def validate_deer_code(deer_code: str) -> str:
    normalized = deer_code.strip()
    if not DEER_CODE_PATTERN.match(normalized):
        raise ValueError("Deer ID must use the standard format, for example 2.1.6")
    return normalized


class ProfileService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_profile(self, deer_code: str) -> DeerProfileRow:
        normalized = validate_deer_code(deer_code)
        existing = self.session.scalar(
            select(DeerProfileRow).where(DeerProfileRow.deer_code == normalized)
        )
        if existing is not None:
            existing.is_active = True
            self.session.flush()
            return existing

        row = DeerProfileRow(deer_code=normalized)
        self.session.add(row)
        self.session.flush()
        return row

    def list_active_deer_codes(self) -> list[str]:
        return list(
            self.session.scalars(
                select(DeerProfileRow.deer_code)
                .where(DeerProfileRow.is_active.is_(True))
                .order_by(DeerProfileRow.deer_code)
            )
        )

    def deactivate_profile(self, deer_code: str) -> None:
        normalized = validate_deer_code(deer_code)
        row = self.session.scalar(
            select(DeerProfileRow).where(DeerProfileRow.deer_code == normalized)
        )
        if row is not None:
            row.is_active = False
            self.session.flush()
