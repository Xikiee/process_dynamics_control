from asammdf import MDF 
import numpy as np

#done for eperiment 6
input_file = r"Project\data\24_02_25\rec1_005.mf4"

output_file = 'rec1_005.mf4'

mdf = MDF(input_file)

mdf.export(fmt='csv', filename=output_file)

print(f"{input_file} has been converted to {output_file}")