"""Legacy compatibility entrypoint.

The original notebook-style training code was replaced by a reusable package.
Use:
    token-nlp train --data dataset.csv --text-column text --label-column label
"""

from token_efficient_nlp.cli import main


if __name__ == "__main__":
    main()
