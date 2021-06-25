"""create table ohlc

Revision ID: c6bb586b1cae
Revises: 
Create Date: 2021-06-24 10:31:01.748655

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c6bb586b1cae'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ohlc",
        sa.Column("id", sa.INTEGER(),autoincrement=True),
        sa.Column("Symbol",sa.String(255)),
        sa.Column("Open",sa.INTEGER()),
        sa.Column("High",sa.INTEGER()),
        sa.Column("Low",sa.INTEGER()),
        sa.Column("Close",sa.INTEGER()),
        sa.Column("Vol",sa.INTEGER()),
        sa.Column("Previous_close",sa.INTEGER()),
        sa.Column("Turnover",sa.Float()),
        sa.Column("trns",sa.INTEGER()),
        sa.Column("Date",sa.Date()),
        sa.PrimaryKeyConstraint("id")
    )


def downgrade():
    op.drop_table("ohlc")
