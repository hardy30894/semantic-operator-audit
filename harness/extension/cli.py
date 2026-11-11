"""semaudit's command line with the v2 additions: `python3 -m extension.cli <command> ...`.

Same commands and options as `python3 -m semaudit`. The one difference: roster entries of kind
"anthropic" log a cost (extension/priced.py), so --max-cost-factor caps them too.
"""

import sys

from extension.priced import install

install()

from semaudit.__main__ import main  # noqa: E402  (after install, so runs build priced backends)

if __name__ == "__main__":
    sys.exit(main())
