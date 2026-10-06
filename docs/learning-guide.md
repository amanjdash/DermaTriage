# DermaTriage learning and interview guide

Use this alongside the implementation. The answers describe design intent; measured behavior must come from the actual run artifacts.

## Data and leakage

**What the table means:** `image_id` is the filename stem used to find an image, `dx` is the seven-class target, and `lesion_id` groups images of the same lesion. The standard metadata has no patient identifier. The dataset preparation code joins IDs to files, verifies decodability, reports missing/corrupt IDs, and saves class counts.

**Why group by lesion?** Two dermoscopic views of one lesion share distinctive visual information. If one goes into training and its sibling goes into test, the test result can reward recognition of that lesion rather than generalization to a new lesion. A stratified grouped split balances labels while ensuring each lesion belongs to just one split.

**What to inspect:** `split_report.json`, train-only class weights, split CSV hashes, and the three lesion-ID intersections. Do not tune model choices against the test split.

## Neural-network terms used here

- **Tensor:** a typed, shaped array. One preprocessed image is typically `[3, 300, 300]`; a batch of 16 is `[16, 3, 300, 300]`.
- **Forward pass and logits:** EfficientNet transforms the batch into seven unnormalized scores. These logits are not probabilities.
- **Softmax and cross entropy:** softmax maps scores to a distribution that sums to one. Cross entropy penalizes probability assigned away from the true class; training uses logits directly with PyTorch's cross-entropy loss for numerical stability.
- **Backpropagation and optimizer:** backpropagation calculates each trainable weight's gradient. AdamW updates weights using those gradients and decoupled weight decay.
- **Learning rate:** update-size control. A large rate can overshoot; fine-tuning uses a smaller rate because pretrained features should change more cautiously.
- **Epoch and batch:** an epoch visits the training rows once; batches make that work fit memory and allow efficient accelerator operations.
- **CNN:** convolutions learn local patterns such as edges and textures; pooling/strided layers reduce spatial size while deeper features combine simpler patterns into larger structures.
- **Transfer learning:** ImageNet pretrained features start from useful generic visual patterns. The new seven-class head is trained first, then the feature extractor is unfrozen for lower-rate fine-tuning.
- **Overfitting and regularization:** the model can memorize training-specific details. Augmentation, dropout, weight decay, early stopping, and lesion-grouped evaluation reduce or reveal that risk; none guarantees generalization.
- **Dropout and BatchNorm:** Dropout masks activations during training. BatchNorm normalizes activations using running statistics. MC Dropout explicitly enables Dropout while leaving BatchNorm in evaluation mode so the latter's statistics do not change during predictions.
- **Mixed precision:** CUDA training may use lower-precision arithmetic for speed and memory savings while gradient scaling helps avoid underflow. It is disabled on CPU and should be checked for numerical stability on the training runtime.

## Evaluation and uncertainty

- **Accuracy** is the fraction correct, but a dominant class can make it look good while rare classes fail.
- **Macro F1** computes F1 per class and averages equally; **weighted F1** weights by class support. Per-class precision/recall and the confusion matrix show which mistakes occur.
- **ROC-AUC** measures ranking separation across thresholds; it does not choose a safe clinical threshold. The report skips it when labels make it undefined.
- **Calibration** asks whether predictions made at a confidence level are correct at about that frequency. ECE and a reliability diagram are descriptive checks; temperature scaling would be fitted on validation data only if there is a useful calibration need.
- **Confidence is not uncertainty.** Confidence is the largest mean softmax probability. Predictive entropy is `-Σ p̄ log(p̄)` and summarizes the spread of the mean probabilities. MC sample variance describes class-wise variation across stochastic passes. Neither tells how severe a lesion is.
- **Epistemic uncertainty:** MC Dropout approximates variation in the model's prediction under dropout masks. It can be compared with errors, but association does not make it a medical risk score or detect every out-of-distribution image.
- **Bootstrap interval:** evaluation resamples examples within each true class so every bootstrap replicate retains class representation. This describes sampling variability for macro F1; it does not capture all dataset or clinical uncertainty.

## Software path

- **REST request:** the browser sends `FormData` with one `image` field to `/predict`. The API checks request size, declared media type, decoded image format, and dimensions before preprocessing.
- **FastAPI lifespan:** application startup loads a configured checkpoint once and keeps the predictor in app state. If it is absent, health/model-info still work and prediction returns 503.
- **Thread pool and model lock:** PyTorch CPU work is blocking, so the async route runs it in a worker thread. A lock serializes MC calls because enabling Dropout changes module modes on the shared model.
- **React state:** the selected file, object-URL preview, loading state, API result, and request error are separate state values. The object URL is revoked when the selection changes or the component unmounts.
- **Docker image/container:** the image is a build artifact; a container is a running instance. Compose gives the frontend a network route to the backend. The backend is non-root; weights are mounted read-only instead of embedded in the code image.
- **Deployment:** CPU inference avoids assuming an accelerator, but model size, RAM, cold start, request timeout, and concurrent work must be measured on the selected host. A successful image build or health check is not the same as a tested production deployment.

## Questions to practice

1. Why can random image splitting leak lesion information? How does the split code prove there is no overlap?
2. Why is accuracy insufficient on HAM10000? Explain macro versus weighted F1.
3. What is the difference between logits, softmax probabilities, and confidence?
4. Why train the head before unfreezing EfficientNet-B3? What changes during fine-tuning?
5. Why keep BatchNorm in evaluation mode during MC Dropout?
6. What does predictive entropy measure, and what does it not measure?
7. How do calibration and discrimination differ? When could temperature scaling help?
8. Why does FastAPI load the checkpoint during lifespan instead of on each request?
9. Why send CPU inference to a thread pool, and why serialize calls that mutate Dropout mode?
10. What happens if the checkpoint is absent, the request is too large, or the image bytes are corrupt?
11. What hardware, preprocessing, warmups, and timed-run counts belong in a latency claim?
12. Which HAM10000, domain-shift, demographic, and clinical-validation limits constrain your conclusions?

