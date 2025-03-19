from asammdf import MDF 
import numpy as np

#done for eperiment 6
input_file = r"Project\data\rec1_006.mf4"

output_file = 'experiment_6.csv'

mdf = MDF(input_file)

mdf.export(fmt='csv', filename=output_file)

print(f"{input_file} has been converted to {output_file}")