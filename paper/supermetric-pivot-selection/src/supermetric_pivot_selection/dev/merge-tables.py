#!/usr/bin/env python3
import click
import duckdb
import glob
import sys


@click.command()
@click.argument("pattern")
@click.argument("output")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be merged without doing it"
)
def merge(pattern, output, dry_run):
    """Merge multiple DuckDB files into one.

    PATTERN: glob pattern for input files (e.g., "*.duckdb" or "data_*.db")
    OUTPUT: output file path
    """
    files = glob.glob(pattern)

    if not files:
        click.echo(f"No files found matching: {pattern}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(files)} files:")
    for f in files:
        click.echo(f"  {f}")

    try:
        conn = duckdb.connect(output)

        # Attach all files
        for i, f in enumerate(files):
            conn.execute(f"ATTACH '{f}' AS db{i}")

        # Get table names
        tables = conn.execute("select distinct name from (show all tables)").fetchall()
        tables = [r[0] for r in tables]

        skiplist = []
        for i in range(len(files)):
            try:
                for tab in tables:
                    conn.execute(f"select * from db{i}.{tab} limit 1")
            except:
                skiplist.append(i)
        if skiplist:
            print(f"skipping {len(skiplist)} databases because of incomplete tables")

        if dry_run:
            click.echo(f"\nWould merge tables {tables} into: {output}")
            return

        click.echo(f"\nMerging {len(tables)} tables:")

        # Merge each table
        for table in tables:
            unions = [
                f"SELECT * FROM db{i}.{table}"
                for i in range(len(files))
                if i not in skiplist
            ]
            conn.execute(f"CREATE TABLE {table} AS {' UNION ALL '.join(unions)}")

            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            click.echo(f"  {table}: {count} rows")

        conn.close()
        click.echo(f"\nMerged to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    merge()
