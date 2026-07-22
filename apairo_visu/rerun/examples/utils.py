import argparse
from pathlib import Path


_DEFAULT_ROOT = Path.home() / "data" / "rellis"


def get_generic_argparser_rellis():
    """Generate an ArgumentParser with usual argument and a path to ~/data/rellis dataset.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--sequence", default=None, help="Restrict to one sequence ID.")
    p.add_argument("--every",    type=int, default=1,  help="Log every Nth frame.")
    p.add_argument("--idx",      type=int, default=0,  help="First frame index.")

    return p
