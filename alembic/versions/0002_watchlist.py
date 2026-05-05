"""add watchlist table

Revision ID: 0002_watchlist
Revises: 0001_initial
Create Date: 2026-05-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_watchlist"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "watchlist",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("plate_pattern", sa.String(32), nullable=False, unique=True),
        sa.Column("reason", sa.String(512)),
        sa.Column("added_by_id", sa.Integer()),
        sa.Column("added_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index("ix_watchlist_plate_pattern", "watchlist", ["plate_pattern"], unique=True)


def downgrade() -> None:
    op.drop_table("watchlist")
