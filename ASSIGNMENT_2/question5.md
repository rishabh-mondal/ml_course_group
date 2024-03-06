Q 5.4.

1. Effect of Low-Rank on Reconstruction Quality: As you increase the value of  r, you might observe improved reconstruction quality. This is because a higher rank allows the model to capture more complex patterns and details in the image.
Conversely, lower values of r may result in a loss of finer details and more noticeable artifacts in the reconstructed image.


2. Computational Efficiency: Higher values of r generally require more computation during both training and inference. Training with a higher rank may take longer, but it could lead to a better representation of the data.
Lower values of r may provide a faster training process but may sacrifice some level of accuracy in the reconstruction.


3. Overfitting: With a very high r, there's a risk of overfitting the model to the training data, capturing noise and outliers in addition to true patterns. This may result in poor generalization to new data.