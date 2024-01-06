import os
from datetime import datetime

import pandas as pd

if __name__ == "__main__":
    timetag = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_path)
    csv_in_dir = [x for x in os.listdir() if x.endswith(".csv")]
    df = pd.concat(pd.read_csv(x) for x in csv_in_dir)
    df.to_csv(f"concat_results_{timetag}.csv", index=False)
