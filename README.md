# Official Implementation of ‘Learning Transferable Negative Prompts for Out-of-Distribution Detection’

Existing prompt learning methods have demonstrated certain capabilities in Out-of-Distribution (OOD) detection, but their lack of perception of OOD images in the target dataset can lead to mismatches between OOD images and In-Distribution (ID) categories, leading to a high false positive rate. To address this issue, we introduce a novel OOD detection method, named `NegPrompt', which is designed to learn a set of negative prompts, each representing a negative connotation of a given class label, to delineate the boundaries between ID and OOD images. It learns such negative prompts with ID data only, eliminating its reliance on external data. Further, current methods assume the availability of samples of all ID classes, rendering them ineffective in open-vocabulary learning scenarios where the inference stage can contain novel ID classes not present in the training data. In contrast, our learned negative prompts are transferable to novel class labels. Experiments on various ImageNet-based  benchmarks demonstrate that NegPrompt surpasses state-of-the-art prompt-learning-based OOD detection methods and maintains a consistent lead in hard OOD detection in closed- and open-vocabulary classification scenarios.

## Overall Architecture

![method](https://github.com/mala-lab/NegPrompt/blob/main/img/method.png)

## Installation

The environments for NegPropmt need to be prepared:

```bash
conda create -n NegPrompt python=3.8
conda activate NegPrompt
pip install -r requirements.txt
```



## Data preparation

First, make a new directory ./data to store the dataset.

Then download the dataset like https://github.com/AtsuMiyai/LoCoOp did.

Arrange the data directory like this:

```
NegPrompt
|-- data/
    |-- ImageNet1k/
	|-- ILSVRC/
		|-- Data/
			|-- CLS-LOC/
				|-- train/
				|-- val/
		｜protocols/
    |-- iNaturalist/
	|--images/
    |-- SUN/
	|--images/
    |-- Places/
	|--images/
    |-- dtd/
	|--images/
    ...
```



## Train and validation

To train the NegPrompt for convetional OOD：

```bash
conda activate NegPrompt
python ./scripts/train_test_ood.py
```

To train the NegPrompt for hard OOD:

```bash
conda activate NegPrompt
python ./scripts/train_test_openset.py
```

