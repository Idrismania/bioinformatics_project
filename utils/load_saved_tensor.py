import torch
import matplotlib.pyplot as plt
tensor = 'd:/Users/Horlings/ii_hh/bioinformatics_project/output_model/output.pt'

output = torch.load(tensor)

image = output[1][0]

print(image.detach())
