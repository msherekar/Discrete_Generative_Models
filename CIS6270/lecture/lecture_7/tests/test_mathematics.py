"""Independent identities and diagnostics for the completed seeded runs."""
import json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Mathematics(unittest.TestCase):
    def test_identities_and_recorded_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            for file in (ROOT/'verified_examples/mathematics').glob('*.json'):
                shutil.copy(file,Path(tmp)/file.name)
            subprocess.run([sys.executable,str(ROOT/'check_examples.py'),'--results',tmp],check=True)
            checks=json.loads((Path(tmp)/'checks.json').read_text())
            self.assertEqual(len(checks),24)
            self.assertTrue(all(checks.values()))
if __name__ == '__main__':
    unittest.main()
