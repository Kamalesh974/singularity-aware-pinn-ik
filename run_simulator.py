"""Double-click / `py run_simulator.py` launcher for the interactive 7R PINN IK simulator."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ik_simulator.main import main

if __name__ == "__main__":
    main()
