from pathlib import Path
import pandas as pd
from external import gsheets

# kc claw gsheets
kc_claw_sheets = {

    }

gsheet_links = {

    }
# %%

sheet_id = "11Y6dRdDwo2oUEGKUD0NZN7AmFcT3YjwMxYLdjqFsH3A"
gid = 0

gsheet_link = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid:d}"

# df = pd.read_csv(gsheet_link)
# %%

url = "https://docs.google.com/spreadsheets/d/11Y6dRdDwo2oUEGKUD0NZN7AmFcT3YjwMxYLdjqFsH3A/edit#gid=51485774"
csv_link = gsheets.browser2csv(url)

# %%
df = pd.read_csv(csv_link)
print(df.head())


