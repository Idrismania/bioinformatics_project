import matplotlib.pyplot as plt

losses = []

with open(r"E:\Users\Horlings\ii_hh\bioinformatics_project\SMA_loss_values.csv") as f:
    next(f)
    for line in f.readlines():
        
        line = line.strip()

        losses.append(line)

losses = [float(loss) for loss in losses]
iterations = list(range(1, len(losses) + 1))

num_epochs = 10
it_per_epoch = 1136

plt.plot(iterations, losses, linewidth=1)
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()