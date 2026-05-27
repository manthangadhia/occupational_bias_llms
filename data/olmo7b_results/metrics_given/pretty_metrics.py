import json
from rich.table import Table
from rich.console import Console
import pandas as pd

df = pd.read_json("given_metrics_regression_coeffs.json")

console = Console()
for metric, group in df.groupby("metric"):
    table = Table(title=metric)
    for col in ["term", "coef", "stderr", "pvalue", "conf_low", "conf_high"]:
        table.add_column(col)
    for _, row in group.iterrows():
        table.add_row(*[f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in ["term", "coef", "stderr", "pvalue", "conf_low", "conf_high"]])
    console.print(table)