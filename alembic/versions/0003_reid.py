"""add reid_subjects table

Revision ID: 0003_reid
Revises: 0002_watchlist
Create Date: 2026-05-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_reid"
down_revision = "0002_watchlist"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reid_subjects",
        sa.Column("global_id", sa.String(36), primary_key=True),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("camera_ids", sa.JSON(), nullable=False),
        sa.Column("plate_text", sa.String(32), nullable=True),
        sa.Column("vehicle_color", sa.String(32), nullable=True),
        sa.Column("vehicle_type", sa.String(32), nullable=True),
        sa.Column("match_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_watchlisted", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_reid_subjects_plate", "reid_subjects", ["plate_text"])
    op.create_index("ix_reid_subjects_last_seen", "reid_subjects", ["last_seen_at"])


def downgrade() -> None:
    op.drop_table("reid_subjects")
