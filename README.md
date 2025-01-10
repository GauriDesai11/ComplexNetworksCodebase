# ComplexNetworksCodebase

This repository contains:
1) 2 Python scripts:
    - `extract_data.py` which collects the data using the Perigon API .
    - `network_analysis.py` which consolidates various network analysis functions.
2) A CSV files containing the communities found using the Louvain method for   the full graph originally provided in this repository.
3) Folders containing the results after running the scripts.

## Requirements

- Python 3.x
- [NetworkX](https://networkx.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/) (for ARI and NMI in community comparisons)
- [community](https://pypi.org/project/python-louvain/) (for Louvain detection)

You can install them via:
```bash
pip install networkx matplotlib numpy scikit-learn python-louvain
```

## Usage

### Extracting data

1. Sign up with Perigon (https://www.perigon.io)
2. Get your unique API key
3. Edit line 8 with your API key
4. Edit line 72 to select the year for which you wish to extract data

``` python3 extract_data.py ```

- Articles are extract for each 5 day interval across the chosen year and results are saved in folder `articles_by_year`.

### Running network analysis

``` python3 network_analysis.py ```

- Results are saved inside folder called `analysed_topic_data` and the graphs are saved in `topic_topic_graph`.
- The exact Louvain communities are not printed out.