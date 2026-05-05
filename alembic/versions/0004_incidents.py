"""add incidents and crowd_snapshots tables

Revision ID: 0004_incidents
Revises: 0003_reid
Create Date: 2026-05-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_incidents"
down_revision = "0003_reid"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("itype", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="suspected"),
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("flags", sa.JSON(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("zone_id", sa.String(64), nullable=True),
        sa.Column("extras", sa.JSON(), nullable=True),
        sa.Column("is_false_positive", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("resolved_by_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_incidents_camera_id", "incidents", ["camera_id"])
    op.create_index("ix_incidents_itype", "incidents", ["itype"])
    op.create_index("ix_incidents_started_at", "incidents", ["started_at"])
    op.create_index("ix_incidents_status", "incidents", ["status"])
    op.create_index("ix_incidents_camera_status", "incidents", ["camera_id", "status"])
    op.create_index("ix_incidents_type_started", "incidents", ["itype", "started_at"])

    op.create_table(
        "crowd_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("total_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_density", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("stampede_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("stampede_level", sa.String(16), nullable=False, server_default="ok"),
        sa.Column("stampede_flags", sa.JSON(), nullable=True),
        sa.Column("zones_json", sa.JSON(), nullable=True),
    )
    op.create_index("ix_crowd_snapshots_camera_id", "crowd_snapshots", ["camera_id"])
    op.create_index("ix_crowd_snapshots_timestamp", "crowd_snapshots", ["timestamp"])
    op.create_index("ix_crowd_snapshots_camera_ts", "crowd_snapshots", ["camera_id", "timestamp"])


def downgrade() -> None:
    op.drop_table("crowd_snapshots")
    op.drop_table("incidents")
