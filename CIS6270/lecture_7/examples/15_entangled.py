"""Run the complete entangled teaching example; accepts the common runner flags."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import main
if __name__ == '__main__':
    main(['--method','entangled',*sys.argv[1:]])
