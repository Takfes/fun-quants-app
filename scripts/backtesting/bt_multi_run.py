import subprocess

import config
from tqdm import tqdm

cryptolist = config.PREXTICKS

for c in tqdm(cryptolist[:2]):
    list_files = subprocess.run(
        ["python", "backtesting.py", "futures1", c, "dic", "10000", "0.02"]
    )
    print("The exit code was: %d" % list_files.returncode)

# python backtesting.py futures1 SOLUSDT dic 10000 0.02
