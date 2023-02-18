from mvbin import mvbin
import numpy as np

joint_prob = np.array([[0.2, 0.05, 0.15],
                       [0.05, 0.5, 0.45],
                       [0.15, 0.45, 0.8]])
p = np.diag(joint_prob)

# Population correlation:
corr = np.array([[1., -0.25, -0.0625],
                 [-0.25, 1., 0.25],
                 [-0.0625, 0.250, 1.]])

sample = mvbin(p=np.diag(joint_prob),
               joint_prob=joint_prob,
               size=10000)

# Sample correlation:
print(np.corrcoef(sample, rowvar=False))

