"""
Entry point.

    py -m ik_simulator                          # open the simulator with runs/final_model.pt
    py -m ik_simulator --checkpoint runs/x.pt   # use another saved PINN
    py -m ik_simulator --selftest OUTDIR        # scripted end-to-end check that saves screenshots
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactive 7R PINN inverse-kinematics simulator")
    ap.add_argument("--checkpoint", default=None, help="saved PINN checkpoint (default: runs/final_model.pt)")
    ap.add_argument("--selftest", metavar="OUTDIR", default=None, help="run a scripted check and save screenshots there")
    args = ap.parse_args(argv)

    import torch  # must be imported before Qt on Windows (DLL load order)
    torch.set_num_threads(2)
    from PySide6.QtWidgets import QApplication

    from .gui import MainWindow

    app = QApplication(sys.argv[:1])
    win = MainWindow(args.checkpoint)
    win.show()
    if args.selftest:
        from .selftest import SelfTest
        SelfTest(win, app, args.selftest).start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
