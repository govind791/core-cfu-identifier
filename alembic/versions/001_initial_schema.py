"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'plate_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('client_id', sa.String(100), nullable=False),
        sa.Column('sample_id', sa.String(100), nullable=False),
        sa.Column('status', sa.Enum('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', name='jobstatus'), nullable=False),
        sa.Column('progress', sa.Float(), nullable=True, default=0.0),
        sa.Column('error_message', sa.String(1000), nullable=True),
        sa.Column('plate_type', sa.String(50), nullable=False),
        sa.Column('capture_method', sa.String(50), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('operator_id', sa.String(100), nullable=True),
        sa.Column('facility_id', sa.String(100), nullable=True),
        sa.Column('dilution', sa.String(50), nullable=True),
        sa.Column('incubation_hours', sa.Float(), nullable=True),
        sa.Column('lighting_type', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_plate_jobs_client_id', 'plate_jobs', ['client_id'])
    op.create_index('ix_plate_jobs_sample_id', 'plate_jobs', ['sample_id'])

    op.create_table(
        'plate_images',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('original_filename', sa.String(500), nullable=False),
        sa.Column('storage_path', sa.String(1000), nullable=False),
        sa.Column('content_type', sa.String(100), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['plate_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id'),
    )

    op.create_table(
        'plate_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('cfu_count_total', sa.Integer(), nullable=True),
        sa.Column('detections', postgresql.JSONB(), nullable=True),
        sa.Column('quality', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', postgresql.JSONB(), nullable=True),
        sa.Column('artifacts', postgresql.JSONB(), nullable=True),
        sa.Column('model_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['plate_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id'),
    )

    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('actor', sa.String(200), nullable=False),
        sa.Column('actor_type', sa.String(50), nullable=False),
        sa.Column('details', postgresql.JSONB(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['plate_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_audit_logs_job_id', 'audit_logs', ['job_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'])


def downgrade() -> None:
    op.drop_table('audit_logs')
    op.drop_table('plate_results')
    op.drop_table('plate_images')
    op.drop_table('plate_jobs')
    op.execute('DROP TYPE IF EXISTS jobstatus')
