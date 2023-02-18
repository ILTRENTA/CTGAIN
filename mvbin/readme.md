## Multivariate binary data in python

This script generates multivariate and correlated binary data
using the procedure outlined in 

```
On the Generation of Correlated Artificial Binary Data
Friedrich Leisch, Andreas Weingessel, Kurt Hornik (1998)
```

and implemented in the `R` package `bindata`.

---

Replicating Example 1 in the original paper:

```{python}
from mvbin import mvbin
import numpy as np

# The joint probability matrix:
joint_prob = np.array([[0.2, 0.05, 0.15],
                       [0.05, 0.5, 0.45],
                       [0.15, 0.45, 0.8]])
p = np.diag(joint_prob)

# Population correlation matrix::
corr = np.array([[1., -0.25, -0.0625],
                 [-0.25, 1., 0.25],
                 [-0.0625, 0.250, 1.]])

# Sample:
sample = mvbin(p=np.diag(joint_prob),
               joint_prob=joint_prob,
               size=10000)

# Sample correlation:
print(np.corrcoef(sample, rowvar=False))
```

Which gives us the following sample correlation:

```{python}
[[ 1.         -0.25164281 -0.06168207]
 [-0.25164281  1.          0.25074679]
 [-0.06168207  0.25074679  1.        ]]
```