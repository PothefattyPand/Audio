# CLAUDE.md — ML Research Agent Rules (CSE428-Based)

You are the implementation agent for a machine-learning research/thesis project. Follow this protocol before and during coding.

## 1. Governing pipeline

Problem -> Data -> EDA -> Cleaning -> Transformation/Feature Selection -> Partitioning -> Model/Hyperparameter Selection -> Training -> Validation/Tuning -> Final Test Evaluation -> Deployment/Monitoring.

Do not skip directly to model training.

## 2. First response to a new project

Before writing model code, inspect available files/data and create or update a problem specification in the project documentation with:

- Task T
- Performance metric P
- Experience/data E
- input X
- target t
- unit of analysis
- task type: regression / binary / multiclass / multilabel / unsupervised / RL
- primary metric
- secondary metrics
- baseline
- data source
- split strategy
- leakage risks
- constraints
- ethical/privacy risks

If target, unit of analysis, or split logic cannot be determined, stop and ask.

## 3. Never violate these rules

1. Test data are never used for preprocessing fit, feature selection, threshold choice, hyperparameter tuning, architecture selection, or early stopping.
2. Learned preprocessing is fit on training data only.
3. Build a simple baseline before a complex model.
4. Model, loss, optimizer, parameters, hyperparameters, and metrics must be named correctly.
5. Complexity must be justified by observed limitations of simpler models.
6. Do not silently invent scientific assumptions.
7. Do not remove outliers or samples only to improve metrics.
8. Keep raw data immutable.
9. Save deterministic split manifests.
10. Every experiment must be reproducible from config + code + seed.

## 4. Task router

- Continuous target -> regression.
- Two mutually exclusive classes -> binary classification.
- One of K mutually exclusive classes -> multiclass classification.
- Multiple simultaneous labels -> multilabel classification.
- No labels / structure discovery -> unsupervised learning.
- Sequential actions + rewards -> reinforcement learning.

## 5. Output/loss router

### Regression
- output: linear/unbounded unless task requires otherwise
- loss: MSE or MAE as justified
- metrics: RMSE/MAE/MSE/R2

### Binary classification
- score: `z = theta^T x`
- probability: `sigmoid(z)` for a single-logit output, or `softmax(z)` for a two-logit output
- loss: binary cross-entropy for a single-logit output, or cross-entropy for a two-logit output
- threshold: chosen on validation data

This project uses two logits, `CrossEntropyLoss`, and `softmax`; do not describe it as a single-sigmoid model.

### Multiclass classification
- K logits
- Softmax probabilities
- categorical/sparse cross-entropy as appropriate
- prediction = argmax

### Multilabel classification
- K independent sigmoid outputs
- binary cross-entropy per label
- do NOT use Softmax for independent labels

## 6. Data checks before training

Report:
- sample count
- feature types
- target distribution
- missingness
- duplicates
- outliers/anomalies
- class imbalance
- correlated features
- group/person/session/site structure
- temporal ordering
- image shape/channels/intensity range if image data

For this project, the EDA and preprocessing record is `docs/EDA_PREPROCESSING.md`.

## 7. Split selection

- IID tabular -> random split
- imbalanced classes -> stratified where possible
- repeated subject/site/object -> group-aware split
- time-series -> chronological split
- small data -> CV on development set where appropriate

For this project, use `data/splits.csv` and the `split_binary_classification` column as the frozen participant-level split manifest.

## 8. Preprocessing

Possible operations:
- missing-value handling
- duplicate removal
- valid outlier treatment
- standardization
- MinMax/MaxAbs scaling
- L1/L2 normalization
- label/one-hot encoding
- binning
- log transform
- polynomial/interaction features
- FT/FFT/STFT/Wavelet/DCT/MFCC when scientifically justified
- image resizing/normalization
- training-only augmentation

Fit stateful transformations on train only.

## 9. Baseline-first model selection

Use the simplest justified model first.

Examples:
- continuous tabular -> linear regression baseline
- binary tabular -> logistic regression baseline
- image -> simple CNN or transfer-learning baseline, optionally compare flattened linear/MLP sanity baseline

Do not select a deep model solely because it is more advanced.

## 10. Optimization

Gradient-descent rule:

`theta <- theta - learning_rate * gradient`

Diagnose:
- too-small LR -> slow convergence
- too-large LR -> oscillation/divergence
- spiky minibatch loss alone != overfitting

For neural nets use proper random/Xavier/He/pretrained initialization; do not zero-initialize all hidden weights.

Training loop order:
1. forward propagation
2. compute loss
3. zero gradients
4. backward propagation
5. optimizer step

Learning rate is a hyperparameter controlling the update step size. Tune it on validation behavior, not test performance.

## 11. Deep-learning knowledge and model router

Deep Learning is a subset of machine learning based on artificial neural networks with multiple processing layers. Deeper layers can learn progressively higher-level representations from data.

Why deep learning may be justified:
- large and complex datasets
- automatic feature extraction instead of fully manual feature engineering
- unstructured inputs such as images, video, audio, or text
- transfer learning from a pretrained network when labeled data or compute are limited

Do not assume deep learning is automatically better. It can require large data, high computation, careful regularization, and may be difficult to interpret.

Common deep-learning model families and their roles:
- Feedforward Neural Network (FNN/MLP): one-way flow from input to output; common for tabular classification/regression
- Convolutional Neural Network (CNN): spatial feature learning for images/videos and other grid-like data
- Recurrent Neural Network (RNN): sequential data with hidden-state dependence across steps
- Autoencoder: encoder-decoder reconstruction; dimensionality reduction/anomaly detection
- Generative Adversarial Network (GAN): generator and discriminator trained competitively to generate data
- Transformer: self-attention-based sequence modeling with parallel processing
- Deep Reinforcement Learning: sequential decision making with rewards

Before selecting a deep model, state why its inductive bias matches the input structure.

## 12. Neural networks and MLP rules

A neural network layer performs an affine transformation followed by an activation:

- `z[l] = W[l] a[l-1] + b[l]`
- `a[l] = g(z[l])`

For the first hidden layer, `a[0] = x`.

For an L-layer network, the model is a composition of layer functions:

`y = f[L] o f[L-1] o ... o f[2] o f[1](x)`

A generic MLP must be understood in terms of three parts:
- Model: forward propagation defines the input-to-output mapping
- Cost function: e.g., MSE for regression; cross-entropy for classification
- Optimization: backpropagation + gradient descent update the weights and biases

Nonlinear activation functions are required between affine layers for nonlinear representational power. A stack of purely linear layers collapses to an equivalent single linear transformation.

Do not call a network "deep" merely because it has many neurons; depth refers to multiple processing layers.

## 13. Activation-function router

Use activation functions according to role; do not use them interchangeably.

### Identity / Linear

`g(z) = z`

Use mainly for regression output when an unrestricted continuous value is required. It does not add non-linearity and is usually unsuitable for hidden layers.

### Sigmoid

`sigmoid(z) = 1 / (1 + exp(-z))`

Properties:
- maps real values to `(0,1)`
- natural probability output for binary classification
- may suffer from vanishing gradients when inputs are far from zero

Typical use:
- binary output layer
- independent outputs in multilabel classification

### Tanh

`tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))`

Properties:
- output range `(-1,1)`
- zero-centered
- can be useful when negative and positive activations are needed
- may still suffer from vanishing gradients for large-magnitude inputs

### ReLU

`ReLU(z) = max(0, z)`

Properties:
- zero for negative input, identity for positive input
- commonly used in hidden layers
- reduces saturation on the positive side and helps mitigate vanishing-gradient problems
- can suffer from "dying ReLU" when a neuron remains in the negative region and receives zero gradient

### Leaky ReLU

`LeakyReLU(z) = max(alpha*z, z)`, commonly with small `alpha` such as `0.01`.

Use as a ReLU alternative when dying ReLU is a concern because negative inputs retain a small non-zero gradient.

### Softmax

For class `i` among `K` classes:

`Softmax(z_i) = exp(z_i) / sum_j exp(z_j)`

Properties:
- converts logits into class probabilities
- each value is between 0 and 1
- probabilities sum to 1

Use for mutually exclusive multiclass output. Do not use Softmax for independent multilabel targets.

## 14. Forward propagation

Forward propagation passes input data through hidden layers to the output layer.

For layer `i`:

`h[i] = phi(W[i]^T h[i-1] + b[i])`

with `h[0] = x`.

Output rules:
- regression -> linear output
- binary classification -> sigmoid output
- multiclass classification -> Softmax output

For classification, convert probabilities/scores to labels only after the output layer:
- binary -> validation-selected threshold
- multiclass -> argmax

During debugging, log tensor shapes and verify every matrix multiplication and activation is dimensionally valid.

## 15. Backpropagation and gradient problems

Backpropagation is the mechanism that propagates prediction error backward through the network and uses the chain rule to compute gradients of weights and biases.

Conceptual training sequence:
1. forward propagate and compute activations/predictions
2. compute the cost/loss
3. compute gradients beginning from the output layer
4. propagate gradients backward using the chain rule
5. update weights/biases using gradient descent or another optimizer
6. repeat over batches/epochs

### Vanishing gradients

Symptoms:
- gradients become extremely small in earlier layers
- early layers update very slowly
- training may stall even when later layers change

Common causes:
- deep networks
- repeated multiplication by small derivatives
- saturated sigmoid/tanh activations

Possible fixes:
- ReLU or Leaky ReLU
- Batch Normalization
- Xavier/He initialization
- skip/residual connections

### Exploding gradients

Symptoms:
- excessively large gradients
- very large weight updates
- unstable or oscillating loss
- divergence or NaN values

Common causes:
- very large weights
- repeated multiplication of large values through deep networks

Possible fixes:
- gradient clipping
- L2/weight regularization
- Xavier/He/normalized initialization
- smaller learning rate

Do not label every unstable curve as overfitting. Check gradient norms, learning rate, scaling, initialization, optimizer, and batch size.

## 16. Dropout and Batch Normalization

### Dropout

Dropout is a neural-network regularization method used to reduce overfitting.

During training:
- randomly deactivate a fraction of neurons according to dropout probability `p`
- this prevents excessive dependence on individual neurons
- the model is forced to learn more robust distributed features

During evaluation/testing:
- dropout must be disabled
- use the framework's correct evaluation mode so train/test behavior is consistent

Treat dropout rate as a hyperparameter and tune it on validation data only.

### Batch Normalization

Batch Normalization stabilizes and often accelerates neural-network training by normalizing activations within each mini-batch.

For each mini-batch:
1. compute activation mean and variance
2. normalize the activations
3. apply learnable scale `gamma`
4. apply learnable shift `beta`

Benefits described in the course material:
- more stable activations/gradients
- faster convergence
- mitigation of vanishing/exploding-gradient behavior
- can support higher learning rates

Trainable BatchNorm parameters per feature channel:

`#trainable_params = 2 * C_out`  (gamma and beta)

After a dense layer:

`#trainable_params = 2 * #output_units`

Running mean and running variance are non-trainable state in common implementations.

When using BatchNorm, ensure train/eval mode is switched correctly during validation and testing.

## 17. Image-data and CNN entry checks

Images and videos are unstructured data represented as pixel grids/arrays rather than ordinary tabular rows and columns.

Before training an image model, inspect:
- image height, width, and channels
- grayscale vs RGB vs multi-channel data
- intensity range
- resolution consistency
- corrupt files
- class distribution
- duplicate/near-duplicate images
- subject/site/object grouping that could cause leakage

Important properties of images/videos:
- high dimensionality
- implicit rather than directly interpretable pixel features
- spatial dependency among nearby pixels
- temporal dependency across video frames

Why an MLP is often a poor image model:
- flattening destroys explicit 2D spatial organization
- fully connected layers create an explosive number of parameters on high-dimensional images

CNNs address this through local connectivity, weight sharing, convolutional filters, and hierarchical feature learning.

## 18. CNN convolution mechanics

Track tensors as `(B,C,H,W)` in PyTorch.

### Kernel / filter

A kernel is a small matrix that slides over an input region. At each location it performs element-wise multiplication followed by summation to produce a feature-map value.

In CNN implementation, learned filters are optimized during training rather than manually chosen edge/sharpening filters.

### Feature maps

A convolutional layer usually applies multiple learned filters. Each filter produces one output feature map representing a learned pattern such as an edge, texture, shape, or more complex feature.

Number of output channels = number of filters.

### Channels

- grayscale image -> 1 channel
- RGB image -> 3 channels
- specialized/medical/hyperspectral images may have more

A convolutional filter spans all input channels for a standard convolution.

### Padding

Padding adds values around the input border.

Use padding when:
- border information is important
- output spatial dimensions need to be preserved or deliberately controlled

For a 3x3 kernel with stride 1, padding 1 on each side preserves width/height.

### Stride

Stride is how many pixels the filter moves per convolution step.

Larger stride:
- produces smaller feature maps
- reduces computation
- performs downsampling

### Receptive field

The receptive field of a neuron is the region of the original input that can affect that neuron's value.

- first convolutional layer: receptive field roughly equals the kernel size
- deeper layers: receptive fields grow, allowing neurons to integrate larger context
- early layers capture local details such as edges/textures
- deeper layers can combine them into shapes, object parts, and whole-object representations

### Translation behavior

Because the same filter slides across locations, a CNN can detect the same learned feature at different positions more naturally than a fully connected MLP.

## 19. CNN building blocks and their purposes

A common end-to-end CNN contains:
- Convolutional Layer
- Activation Layer
- Pooling Layer
- optional Batch Normalization
- optional Dropout
- Flattening or Global Average Pooling
- Fully Connected/Dense Layer
- Output Layer

### Convolutional layer

Primary objective: feature extraction, not dimensionality reduction.

It applies multiple learned filters to produce multiple feature maps.

### Pooling layer

Primary objective: reduce spatial dimensions while retaining useful information.

Pooling operates independently over channels and has no learned kernel weights.

Common forms:
- Max Pooling -> keeps the maximum value in each region; emphasizes prominent features
- Average Pooling -> keeps the regional average; gives a smoother summary
- Global Average Pooling (GAP) -> reduces each feature map to one value and may replace Flatten + large Dense layers

Pooling can reduce computation/memory, improve translation tolerance, and help generalization.

### Flattening

Flattening reshapes a multidimensional feature tensor into a one-dimensional feature vector for dense layers.

It changes shape, not the underlying values, and has no learned parameters.

### Dense layer

Dense layers combine the extracted global features and produce decisions/predictions.

Output-layer role:
- regression -> continuous output
- binary/multilabel classification -> sigmoid as appropriate
- multiclass classification -> Softmax

## 20. CNN dimensions and parameter calculation

For convolution/pooling dimensions, use integer-valid output sizes. The course formula is:

`W_out = floor((W_in - F + 2P) / S) + 1`

`H_out = floor((H_in - F + 2P) / S) + 1`

For convolution:

`C_out = number_of_filters`

For pooling:

`C_out = C_in`

For batch size `N`:
- input -> `[N, C_in, H_in, W_in]` in PyTorch convention
- output -> `[N, C_out, H_out, W_out]`

Flatten output size per sample:

`N_flat = C * H * W`

Dense output size:

`#output_units`

### Convolutional parameters

`#params = (k_h * k_w * C_in + 1) * C_out`

The `+1` is one bias per output filter.

### Dense parameters

`#params = (N_in + 1) * N_out`

The `+1` contributes one bias per output neuron.

### Pooling / flattening

- Pooling trainable parameters = 0
- Flatten trainable parameters = 0

### BatchNorm

For the trainable gamma/beta pair:

`#trainable_params = 2 * C_out`

or after a Dense layer:

`#trainable_params = 2 * N_out`

When solving architecture questions, make a table with:
- layer name
- input dimensions
- kernel/pool size
- stride
- padding
- filters/neurons
- output dimensions
- trainable parameter count

Never guess the flatten size. Compute it from the preceding feature-map dimensions.

## 21. CNN architecture-design rules

General architecture pattern:

`Input -> [Conv -> ReLU -> optional BatchNorm -> Pool] x several blocks -> Flatten or GAP -> Dense -> optional Dropout -> Output`

Course design intuitions:
- 3x3 filters are a common default for local feature extraction
- filter count often increases with depth, e.g., `32 -> 64 -> 128`
- deeper networks can learn more abstract features but are harder to train
- use padding deliberately: `same` preserves size; `valid` shrinks
- stride controls downsampling speed and can sometimes replace pooling
- max pooling emphasizes strong/sharp responses; average pooling produces smoother summaries
- ReLU is the default hidden activation in many CNNs; Leaky ReLU/ELU are alternatives
- BatchNorm can stabilize training
- Dropout can reduce overfitting, especially around dense layers
- Global Average Pooling can reduce the parameter count and overfitting relative to large Flatten + Dense heads

A sample pattern from the lecture is:

`224x224x3 -> Conv(3x3,32)+ReLU -> MaxPool -> Conv(3x3,64)+ReLU -> MaxPool -> Conv(3x3,128)+ReLU -> GAP -> Dense(128)+ReLU+Dropout -> Dense(10)+Softmax`

Do not copy a sample architecture blindly. Adapt capacity, input size, output classes, and regularization to the actual task/data.

## 22. Modern CNN architecture knowledge

Use these as established architectural references, not automatically as the proposed method.

### LeNet-5

Key idea:
- early foundational CNN for handwritten-digit recognition
- two convolution + average-pooling blocks followed by flattening and three dense layers (including output)
- demonstrated that CNNs can learn spatial hierarchies for image tasks

### AlexNet

Key ideas:
- 5 convolutional layers followed by flattening and 3 dense layers
- used max pooling
- popularized ReLU for faster training
- used dropout to reduce overfitting in fully connected layers
- used data augmentation and GPU-based training
- demonstrated large-scale CNN success on ImageNet

### VGG-16 / VGG-19

Key ideas:
- simple, repeated 3x3 convolutions
- stride 1, padding 1
- multiple convolutions before 2x2 max-pooling with stride 2
- increased depth while keeping the design simple

Limitations:
- very large parameter counts (e.g., VGG-16 about 138M)
- memory/computation heavy
- no skip connections; difficult to train at great depth without proper regularization

### GoogLeNet / Inception

Key ideas:
- parallel 1x1, 3x3, 5x5 convolution and pooling branches
- 1x1 convolutions used as bottlenecks to reduce dimensionality before expensive operations
- outputs from parallel branches are concatenated along channel depth
- global average pooling reduces reliance on large fully connected layers
- auxiliary classifiers were used during training to aid gradient flow

Use when multi-scale feature extraction and parameter efficiency are relevant.

### ResNet

Core residual idea:

`H(x) = F(x) + x`

with a skip/shortcut connection carrying `x` across layers.

Why it matters:
- improves gradient flow
- mitigates vanishing-gradient difficulty
- enables much deeper networks

Common variants include ResNet-18, 34, 50, 101, and 152.

### DenseNet

Core idea:
- each layer receives concatenated feature maps from previous layers
- promotes feature reuse and short gradient paths

Dense Blocks concatenate features; Transition Layers use 1x1 convolution and pooling to control channel/spatial growth.

Distinguish:
- ResNet -> additive skip connections
- DenseNet -> concatenative connectivity

DenseNet can be parameter-efficient but memory intensive.

### MobileNet

Designed for mobile/edge efficiency.

Core idea: depthwise separable convolution:
1. depthwise convolution -> one spatial filter per input channel
2. pointwise convolution -> 1x1 convolution combines channels

Also uses:
- width multiplier `alpha` to control channel count
- resolution multiplier `rho` to control input resolution

Use when model size, latency, or edge deployment is a major constraint.

### EfficientNet

Core idea: compound scaling of:
- depth
- width
- input resolution

Rather than scaling only one dimension, EfficientNet grows them together using coordinated scaling factors.

Use as a candidate when balancing accuracy, model size, and compute efficiency is important.

## 23. Image-classification protocol

Image classification predicts a class label from visual content.

CNN classification flow:
1. raw pixels enter the network
2. convolution/pooling layers learn hierarchical features
3. early features: edges/textures
4. middle features: shapes/patterns/object parts
5. deeper features: higher-level object representation
6. classifier maps learned representation to class scores/probabilities

Before training:
- define exact classes and class mapping
- verify folder/label mapping
- inspect class imbalance
- choose image resize/normalization from train protocol
- use augmentation on training data only
- ensure subject/patient/object duplicates do not cross splits

For multiclass image classification, output dimension must equal number of classes.

## 24. Transfer learning protocol

Transfer learning reuses a model pretrained on a large dataset such as ImageNet for a related new task.

Why it works:
- early CNN layers often learn generic visual features such as edges, blobs, and textures
- final/deeper layers are more task-specific
- pretrained weights can reduce data and compute requirements

Two primary strategies:

### Feature extraction

- load pretrained CNN
- remove/replace the original final classification layer
- freeze the pretrained feature extractor
- train the new task-specific classifier head

### Fine-tuning

- begin from pretrained weights
- replace the output head
- unfreeze some higher/deeper pretrained layers
- continue training on the new dataset

Candidate pretrained families from the lecture include:
- VGG
- Inception
- ResNet
- EfficientNet

Research protocol:
1. establish a simple baseline
2. establish a frozen-feature transfer baseline
3. fine-tune only if validation evidence justifies it
4. use a smaller learning rate for pretrained layers than for a randomly initialized head when implementation supports parameter groups
5. never choose the freeze/unfreeze boundary using the test set
6. document pretrained dataset, checkpoint/version, frozen layers, learning rates, and augmentation

Do not assume transfer learning is valid when the source and target domains are extremely mismatched; report the domain difference as a limitation.

## 25. Overfitting diagnosis

- train high + val high -> underfitting/optimization/feature issue
- train low + val high/rising -> overfitting
- oscillating train loss -> inspect LR/scaling/optimizer/batch size

Fix only the diagnosed cause.

Possible anti-overfitting tools:
- L1/L2
- dropout
- augmentation
- early stopping
- more data
- reduced complexity
- feature selection
- transfer learning when appropriate
- Global Average Pooling instead of a very large dense head when suitable

For neural networks, remember that high capacity can memorize training data and noise. Generalization, not training loss alone, determines success.

## 26. Hyperparameter tuning

Use validation/CV only.

Possible neural-network hyperparameters include:
- learning rate
- optimizer
- batch size
- number of layers
- hidden units / filter counts
- kernel size
- stride/padding design
- dropout rate
- weight decay
- augmentation strength
- BatchNorm placement
- pretrained freeze/unfreeze depth

Log every trial:
- config
- seed
- validation metric
- best epoch
- checkpoint

Do not run final test after every trial.

## 27. Final test protocol

Only after model/preprocessing/hyperparameters/threshold are locked:

- load locked checkpoint
- use evaluation mode (`model.eval()` in PyTorch)
- disable gradient computation when appropriate
- run untouched test set once
- compute predefined metrics
- save predictions
- save confusion matrix or residual plots
- perform error analysis

For image classification also inspect:
- per-class performance
- confusion pairs
- representative false positives/false negatives
- whether errors correlate with image quality, viewpoint, lighting, site, subject, or class imbalance

## 28. Research requirements

A thesis result needs:
- naive/simple baseline
- relevant established baseline when feasible
- proposed method
- same split/evaluation protocol for comparison
- ablations for claimed components
- limitations
- ethics/privacy discussion
- reproducibility metadata

For a CNN/deep-learning thesis, ablations may include:
- removing/adding BatchNorm
- removing/adding dropout
- augmentation vs no augmentation
- GAP vs Flatten + Dense
- pretrained vs scratch
- frozen feature extractor vs fine-tuning
- proposed block/component removed

Do not claim a component is responsible for an improvement unless an ablation or controlled comparison supports the claim.

## 29. Required project layout

Prefer:

```text
configs/
data/raw/
data/interim/
data/processed/
metadata/
docs/
notebooks/
src/data.py
src/preprocess.py
src/features.py
src/models.py
src/train.py
src/evaluate.py
src/metrics.py
scripts/
tests/
checkpoints/
outputs/
logs/
```

For image/deep-learning projects also consider:

```text
src/augmentations.py
src/datasets.py
src/transforms.py
metadata/class_mapping.json
outputs/confusion_matrix.png
outputs/error_analysis/
```

## 30. Before declaring success

Verify:
- no leakage
- target/output/loss alignment
- shape correctness
- activation/output correctness
- baseline comparison
- validation-based tuning
- untouched test
- failure analysis
- experiment logging
- reproducibility
- correct train/eval mode for Dropout and BatchNorm
- correct CNN output dimensions and parameter counts
- transfer-learning freeze/unfreeze state documented when used
- model complexity justified by the data/problem

If any item is missing, report it as incomplete rather than claiming the project is finished.
