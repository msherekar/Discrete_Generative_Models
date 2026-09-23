"""Complete fisher training and generation example."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main
if __name__ == '__main__':
    main(['--method', 'fisher', '--out', 'outputs/fisher'] + sys.argv[1:])
