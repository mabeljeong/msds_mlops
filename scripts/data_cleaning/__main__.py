"""CLI entry point: ``python -m data_cleaning`` (with ``scripts/`` on sys.path)."""

from __future__ import annotations

from . import build_clean_dataset


def main() -> None:
    final_df = build_clean_dataset()
    if final_df is not None:
        print(f"\nFinal panel shape: {final_df.shape}")
        print(f"Columns ({len(final_df.columns)}): {list(final_df.columns)}")
    else:
        print("\nFinal panel not built (see messages above). Listings/Census/Redfin may still have been written.")


if __name__ == "__main__":
    main()
