# Project Title
Preserving logical and functional dependencies in synthetic tabular data

## Description
Dependencies among attributes are a common aspect of tabular data. However, whether existing synthetic tabular data generation algorithms preserve these dependencies 
while generating synthetic data is yet to be explored. Moreover, no well-established measures can quantify logical dependencies among attributes. In this article, 
we provide a measure to quantify logical dependencies among attributes in tabular data. Utilizing this measure, we compare several state-of-the-art synthetic data 
generation algorithms and test their capability to preserve logical and functional dependencies on several publicly available datasets. We demonstrate that currently 
available synthetic tabular data generation algorithms do not fully preserve functional dependencies when they generate synthetic datasets. In addition, we also showed
that some tabular synthetic data generation models can preserve inter-attribute logical dependencies. Our review and comparison of the state-of-the-art reveal research
needs and opportunities to develop task-specific synthetic tabular data generation models.

## Installation
#### Clone the Repository
Clone the repository to your local machine using the following command:
bash
git clone https://github.com/Chaithra-U/Dependency_preservation

#### Explore the Repository
After cloning, you'll find two main folders: functional and logical_dependencies.

Choose the experiment you want to run based on the type of dependencies you're interested in.

##### Logical Dependencies Experiment
Inside the logical_dependencies folder, you’ll find both real and synthetic datasets generated by seven state-of-the-art tabular synthetic data generation algorithms.

If you need to generate synthetic data for a specific dataset, use the provided links

CTGAN: https://github.com/sdv-dev/CTGAN

CTABGAN: https://github.com/Team-TUD/CTAB-GAN

CTABGAN+: https://github.com/Team-TUD/CTAB-GAN-Plus-DP

NextConvGeN: https://github.com/manjunath-mahendra/NextConvGeN

TabDDPM: https://github.com/yandex-research/tab-ddpm

TabuLa: https://github.com/zhao-zilong/Tabula

After generating the synthetic data, you can compare the inter-attribute logical dependencies with the real data by running one of the five provided Jupyter notebooks.

To perform this experiment on a different dataset, follow the same steps outlined in the notebook, but use your data instead.
##### Functional Dependencies Experiment
For functional dependencies, the FDTool algorithm has been used, which is publicly available for extracting functional dependencies from any dataset.

In the functional folder, scripts are provided for two datasets: Airbnb and Migraine.

To find common functional dependencies for your own data, repeat the steps in the provided scripts with your dataset.


