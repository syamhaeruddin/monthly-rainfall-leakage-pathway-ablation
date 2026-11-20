"""Edit a COPY of the companion project (never the original): the station list is read from the STATIONS environment variable.

Usage:  python patch_copy.py <path to the copied project>
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

p = root / "src" / "common.py"; s = p.read_text(encoding="utf-8")
a = "STATIONS = STATIONS_ALL"
assert s.count(a) == 1, "common.py: pattern not found"
s = s.replace(a, 'STATIONS = [s for s in STATIONS_ALL if s in os.environ.get("STATIONS", ",".join(STATIONS_ALL)).split(",")]'
                 '   # two-station copy: stations chosen with env STATIONS')
p.write_text(s, encoding="utf-8", newline="")

p = root / "src" / "run_all.py"; s = p.read_text(encoding="utf-8")
a = '"Woomera", "Birdsville", "Perth", "Adelaide", "Melbourne", "Hobart"]'
assert s.count(a) == 1, "run_all.py: station pattern not found"
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace(a, a + nl + 'STATIONS = [s for s in STATIONS if s in os.environ.get("STATIONS", ",".join(STATIONS)).split(",")]   # two-station copy'
              + nl + 'FOLDS_AKTIF = [i for i, f in enumerate([["Darwin", "Brisbane", "Woomera"], ["Perth", "Cairns", "Birdsville"], '
              '["Tully", "Alice_Springs", "Coffs_Harbour"], ["Hobart", "Katherine", "Adelaide"], ["Broome", "Innisfail", "Melbourne"]]) '
              'if set(f) & set(STATIONS)]   # transfer folds that contain a chosen station')
b = "for f in range(5)]"
assert s.count(b) == 1, "run_all.py: fold pattern not found"
s = s.replace(b, "for f in FOLDS_AKTIF]")
p.write_text(s, encoding="utf-8", newline="")
print("patch OK")
