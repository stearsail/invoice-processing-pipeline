"""split extraction from ledgerentry

Revision ID: 03dbd3cc4a4a
Revises: 3d75385aa245
Create Date: 2026-11-26 15:19:37.668058

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '03dbd3cc4a4a'
down_revision: Union[str, Sequence[str], None] = '3d75385aa245'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('extraction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('run', sa.Integer(), nullable=False),
    sa.Column('extractor', sa.String(), nullable=False),
    sa.Column('model', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('invoice_data', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('parse_error', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('raw_output', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('issues', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('is_final', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('trigger', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['job_id'], ['job.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_extraction_job_id'), 'extraction', ['job_id'], unique=False)
    op.create_index('uq_extraction_one_final_per_job', 'extraction', ['job_id'], unique=True, postgresql_where=sa.text('is_final'), sqlite_where=sa.text('is_final'))
    op.add_column('ledgerentry', sa.Column('source_extraction_id', sa.Integer(), nullable=True))
    op.add_column('ledgerentry', sa.Column('corrected', sa.Boolean(), nullable=True))
    op.add_column('ledgerentry', sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=False))
    op.alter_column('ledgerentry', 'invoice_data',
               existing_type=postgresql.JSON(astext_type=sa.Text()),
               type_=sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'),
               nullable=False)
    op.drop_index(op.f('ix_ledgerentry_invoice_number'), table_name='ledgerentry')
    op.drop_index(op.f('ix_ledgerentry_needs_review'), table_name='ledgerentry')
    op.drop_index(op.f('ix_ledgerentry_vendor_name'), table_name='ledgerentry')
    op.create_index('ix_ledgerentry_vendor_invoice', 'ledgerentry', ['vendor_name', 'invoice_number'], unique=False)
    op.drop_constraint(op.f('ledgerentry_job_id_fkey'), 'ledgerentry', type_='foreignkey')
    op.create_foreign_key('ledgerentry_job_id_fkey', 'ledgerentry', 'job', ['job_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('ledgerentry_source_extraction_id_fkey', 'ledgerentry', 'extraction', ['source_extraction_id'], ['id'])
    op.drop_column('ledgerentry', 'extracted_by')
    op.drop_column('ledgerentry', 'reviewed_with_changes')
    op.drop_column('ledgerentry', 'needs_review')
    op.drop_column('ledgerentry', 'created_at')
    op.drop_column('ledgerentry', 'review_reason')


def downgrade() -> None:
    """Downgrade schema."""
    # approved entries can't be mapped back to the old draft shape: downgrade drops ledger data
    op.execute("DELETE FROM ledgerentry")
    op.add_column('ledgerentry', sa.Column('review_reason', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('ledgerentry', sa.Column('created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=False))
    op.add_column('ledgerentry', sa.Column('needs_review', sa.BOOLEAN(), autoincrement=False, nullable=False))
    op.add_column('ledgerentry', sa.Column('reviewed_with_changes', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('ledgerentry', sa.Column('extracted_by', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.drop_constraint('ledgerentry_source_extraction_id_fkey', 'ledgerentry', type_='foreignkey')
    op.drop_constraint('ledgerentry_job_id_fkey', 'ledgerentry', type_='foreignkey')
    op.create_foreign_key(op.f('ledgerentry_job_id_fkey'), 'ledgerentry', 'job', ['job_id'], ['id'])
    op.drop_index('ix_ledgerentry_vendor_invoice', table_name='ledgerentry')
    op.create_index(op.f('ix_ledgerentry_vendor_name'), 'ledgerentry', ['vendor_name'], unique=False)
    op.create_index(op.f('ix_ledgerentry_needs_review'), 'ledgerentry', ['needs_review'], unique=False)
    op.create_index(op.f('ix_ledgerentry_invoice_number'), 'ledgerentry', ['invoice_number'], unique=False)
    op.alter_column('ledgerentry', 'invoice_data',
               existing_type=sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'),
               type_=postgresql.JSON(astext_type=sa.Text()),
               nullable=True)
    op.drop_column('ledgerentry', 'reviewed_at')
    op.drop_column('ledgerentry', 'corrected')
    op.drop_column('ledgerentry', 'source_extraction_id')
    op.drop_index('uq_extraction_one_final_per_job', table_name='extraction', postgresql_where=sa.text('is_final'), sqlite_where=sa.text('is_final'))
    op.drop_index(op.f('ix_extraction_job_id'), table_name='extraction')
    op.drop_table('extraction')
