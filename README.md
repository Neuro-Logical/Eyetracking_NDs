# Automating the analysis of eye movement for different neurodegenerative disorders

Code for paper [Automating the analysis of eye movement for different neurodegenerative disorders](https://www.medrxiv.org/content/10.1101/2023.05.30.23290745v1).

An automatic approach to identify eye motion states from eye tracking data and extract interpretable features, followed by calculating p values using statistical tests. 

## Setup

``` sh
# Create environment
conda create -n eye
conda activate eye

# Install packages
pip install -r requirements.txt
```

## Usage

General eye movement features:

``` sh
python run_processing.py
```

Task-specific features:

``` sh
python run_saccade.py
```

Plot the boxplot and calculate statistics:
``` sh
python run_summary.py
```


## Citation

Please cite our work if you find this repository helpful to your project.

```sh
@article{li2023automating,
  title={Automating analysis of eye movement and feature extraction for different neurodegenerative disorders},
  author={Li, Deming and Butala, Ankur A and Meyer, Trevor and Oh, Esther S and Motley, Chelsey and Moro-Velazquez, Laureano and Dehak, Najim},
  journal={medRxiv},
  pages={2023--05},
  year={2023},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

## Acknowledgement

* Visualizations of statistical significance are adapted from [statannotations](https://github.com/trevismd/statannotations.git).
