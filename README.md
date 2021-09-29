# MLPMixer-jax2tf

<p align="center">
  <img src="https://i.ibb.co/tmGYFTv/mixer.png" width=650/><br>
  <sup>Example usage.</sup>
</p>

This repository hosts code for converting the original MLP-Mixer models [1] (JAX) to TensorFlow. The converted models are hosted
on TensorFlow Hub and can be found here: https://tfhub.dev/sayakpaul/collections/mlp-mixer/1.

**Note that it's a requirement to use TensorFlow 2.6 or greater to use the converted models.**

Several model variants are available:

### **SAM [2] pre-trained** (these models were pre-trained on ImageNet-1k):

* B/16 ([classification](https://tfhub.dev/sayakpaul/mixer_b16_sam_classification/1), [feature-extractor](https://tfhub.dev/sayakpaul/mixer_b16_sam_fe/1))
* B/32 ([classification](https://tfhub.dev/sayakpaul/mixer_b32_sam_classification/1), [feature-extractor](https://tfhub.dev/sayakpaul/mixer_b32_sam_fe/1))

### **ImageNet-1k fine-tuned**:

* B/16 ([classification](https://tfhub.dev/sayakpaul/mixer_b16_i1k_classification/1), [feature-extractor](https://tfhub.dev/sayakpaul/mixer_b16_i1k_fe/1))
* L/16 ([classification](https://tfhub.dev/sayakpaul/mixer_l16_i1k_classification/1), [feature-extractor](https://tfhub.dev/sayakpaul/mixer_l16_i1k_fe/1))

### **ImageNet-21k pre-trained**:

* B/16 ([classification](https://tfhub.dev/sayakpaul/mixer_b16_i21k_classification/1), [feature-extractor](https://tfhub.dev/sayakpaul/mixer_b16_i21k_fe/1))
* L/16 ([classification](https://tfhub.dev/sayakpaul/mixer_l16_i21k_classification/1), [feature-extractor](https://tfhub.dev/sayakpaul/mixer_l16_i21k_fe/1))

For more details on the training protocols, please follow [1, 3].

The original model classes and weights [4] were converted using the `jax2tf` tool [5]. For details on the conversion process,
please refer to the [`conversion.ipynb`](https://colab.research.google.com/github/sayakpaul/MLPMixer-jax2tf/blob/main/conversion.ipynb) notebook.

I independently validated two models on the ImageNet-1k validation set. The table
below reports the top-1 accuracies along with their respective logs from tensorboard.dev.

| **Model** | **Top-1 Accuracy** | **tb.dev link** |
|:---:|:---:|:---:|
| [B-16 fine-tuned on<br> ImageNet-1k](https://tfhub.dev/sayakpaul/mixer_b16_i1k_classification/1) | 75.31% | [Link](https://tensorboard.dev/experiment/trMCPE2SQYG51FYqyjgh3Q) |
| [B-16 pre-trained on<br> ImageNet-1k using SAM](https://tfhub.dev/sayakpaul/mixer_b16_i1k_classification/1) | 75.58% | [Link](https://tensorboard.dev/experiment/52LkVYfnQDykgyDHmWjzBA/) |

[Here is a tensorboard.dev run](https://tensorboard.dev/experiment/3hbqCglPSNC5OZPnYbeHew/) that logs fine-tuning results (using [this model](https://tfhub.dev/sayakpaul/mixer_b16_i1k_fe/1))
for the [Flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers).

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

## Acknowledgements

Thanks to the [ML-GDE program](https://developers.google.com/community/experts) for providing GCP Credit support that helped me execute the experiments for this project.
