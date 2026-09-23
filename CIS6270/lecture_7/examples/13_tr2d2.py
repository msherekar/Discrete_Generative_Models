"""Run the complete tr2d2 teaching example; accepts the common runner flags."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import main
if __name__ == '__main__':
    main(['--method','tr2d2',*sys.argv[1:]])
