import torch
import matplotlib.pyplot as plt
tensor = '../output_model/outputs.pt'

output = torch.load(tensor)

image = output[1][0]

plt.imshow(image.detach(), cmap='grey')
plt.show()
