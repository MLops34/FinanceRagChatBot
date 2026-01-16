import pandas as pd
import glob
import pathlib as p
#-------------READING DOCUMENTS-------#
#-------------------------------------#

import pandas as pd

file = r"C:\Users\Acer\Downloads\db566-scheme-portfolio-details-december-2025.xlsx"
df = pd.read_excel(file, sheet_name="YO17")

print("Loaded YO16 with shape:", df.shape)
print(df.head()) 




