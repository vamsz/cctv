"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("full_name", sa.String(255)),
        sa.Column("badge_number", sa.String(64)),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="operator"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_login_at", sa.DateTime()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "violations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("frame_idx", sa.Integer(), nullable=False),
        sa.Column("plate_text", sa.String(32)),
        sa.Column("plate_ocr_confidence", sa.Float()),
        sa.Column("rule_confidence", sa.Float(), nullable=False),
        sa.Column("frame_path", sa.String(512), nullable=False),
        sa.Column("annotated_path", sa.String(512)),
        sa.Column("plate_crop_path", sa.String(512)),
        sa.Column("bbox", sa.JSON()),
        sa.Column("extras", sa.JSON()),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("signature", sa.LargeBinary(), nullable=False),
        sa.Column("prev_id", sa.Integer()),
        sa.Column("chain_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("reviewer_id", sa.Integer()),
        sa.Column("reviewed_at", sa.DateTime()),
        sa.Column("review_notes", sa.Text()),
        sa.Column("pushed_to_chalana", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("pushed_at", sa.DateTime()),
        sa.Column("chalana_reference", sa.String(128)),
    )
    op.create_index("ix_violations_code", "violations", ["code"])
    op.create_index("ix_violations_camera_id", "violations", ["camera_id"])
    op.create_index("ix_violations_timestamp", "violations", ["timestamp"])
    op.create_index("ix_violations_plate_text", "violations", ["plate_text"])
    op.create_index("ix_violations_status", "violations", ["status"])
    op.create_index("ix_violations_status_timestamp", "violations", ["status", "timestamp"])
    op.create_index("ix_violations_camera_timestamp", "violations", ["camera_id", "timestamp"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("actor_id", sa.Integer()),
        sa.Column("actor_label", sa.String(128)),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(64)),
        sa.Column("target_id", sa.String(64)),
        sa.Column("detail", sa.JSON()),
        sa.Column("ip_address", sa.String(64)),
        sa.Column("user_agent", sa.String(512)),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])

    op.create_table(
        "camera_health",
        sa.Column("camera_id", sa.String(64), primary_key=True),
        sa.Column("last_frame_at", sa.DateTime()),
        sa.Column("last_violation_at", sa.DateTime()),
        sa.Column("fps_observed", sa.Float()),
        sa.Column("is_up", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_error", sa.Text()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("camera_health")
    op.drop_table("audit_log")
    op.drop_table("violations")
    op.drop_table("users")
