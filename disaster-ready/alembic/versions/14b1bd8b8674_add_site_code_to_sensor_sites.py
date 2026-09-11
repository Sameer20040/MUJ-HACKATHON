"""Add site_code to sensor_sites

Revision ID: 14b1bd8b8674
Revises: da1a3c15b39d
Create Date: 2026-09-11 21:22:17.544157

"""
from __future__ import with_statement

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '14b1bd8b8674'
down_revision = 'da1a3c15b39d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('sensor_sites', sa.Column('site_code', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('sensor_sites', 'site_code')