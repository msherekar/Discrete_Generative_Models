"""Run the complete finite-sb teaching example; accepts the common runner flags."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import main
if __name__ == '__main__':
    main(['--method','finite-sb',*sys.argv[1:]])
