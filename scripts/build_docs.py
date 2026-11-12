"""Build documentation, reporting stalled Python stacks before the RTD timeout."""

import faulthandler
import runpy


if __name__ == "__main__":
    # A clean build takes seconds. RTD only saves command output when it exits;
    # fail early with all thread stacks instead of losing diagnostics at 15 min.
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        runpy.run_module("mkdocs", run_name="__main__")
    finally:
        faulthandler.cancel_dump_traceback_later()
