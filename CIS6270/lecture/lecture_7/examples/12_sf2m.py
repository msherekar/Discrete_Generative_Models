"""Run the complete sf2m teaching example; accepts the common runner flags."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import main
if __name__ == '__main__':
    main(['--method','sf2m',*sys.argv[1:]])
