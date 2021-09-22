# MLPMixer-jax2tf

This repository hosts code for converting the original MLP-Mixer models [1] (JAX) to TensorFlow. The converted models are hosted
on TensorFlow Hub and can be found here: [TODO].

Several model variants are available:

### **SAM [2] pre-trained** (these models were pre-trained on ImageNet-1k):

* B/16 (classification, feature-extractor)
* B/32 (classification, feature-extractor)

### **ImageNet-1k fine-tuned**:

* B/16 (classification, feature-extractor)
* L/16 (classification, feature-extractor)

### **ImageNet-21k pre-trained**:

* B/16 (classification, feature-extractor)
* L/16 (classification, feature-extractor)

For more details on the training protocols, please follow [1, 3].

The original model classes and weights [4] using the `jax2tf` tool [5]. For details on the conversion process,
please refer to the [`conversion.ipynb`](https://colab.research.google.com/github/sayakpaul/MLPMixer-jax2tf/blob/main/conversion.ipynb) notebook.

## Other notebooks

* [`classification.ipynb`](https://colab.research.google.com/github/sayakpaul/MLPMixer-jax2tf/blob/main/classification.ipynb): Shows how to load a Vision Transformer model from TensorFlow Hub and run image classification.
* [`fine-tune.ipynb`](https://colab.research.google.com/github/sayakpaul/MLPMixer-jax2tf/blob/main/fine-tune.ipynb): Shows how to
  fine-tune a Vision Transformer model from TensorFlow Hub on the `tf_flowers` dataset.

## References

[1] [MLP-Mixer: An all-MLP Architecture for Vision by Tolstikhin et al.](https://arxiv.org/abs/2105.01601)

[2] [Sharpness-Aware Minimization for Efficiently Improving Generalization by Foret et al.](https://arxiv.org/abs/2010.01412)

[3] [When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations by Chen et al.](https://arxiv.org/abs/2106.01548)

[4] [Vision Transformer GitHub](https://github.com/google-research/vision_transformer)

[5] [jax2tf tool](https://github.com/google/jax/tree/main/jax/experimental/jax2tf/)
