# FLEE-GNN

## Environment Configuration

### No plot

```
conda create -n resilience python=3.9

python -m pip install pandas numpy tqdm networkx matplotlib

```

### Plot (require geopandas support)

- Check `gdal-config`

```
brew install gdal
gdal-config --version
```

- Install python packages

```
python -m pip install gdal fiona pysal geopandas
```

## Scripts

### Generate Data

Generate <size> of graphs with 1000-2000 edges

- Random Generator

```
python ./scripts/generate_data_random.py --save-dir ./training/random_data/ --size 100
```

- Noise Generator

```
python ./scripts/generate_data_from_real_data.py --data-path ./data/2012_data.csv --save-dir ./training/noise_data_0.2/ --size 100 --noise-rate 0.2
```

### Generate Labels

Generate entropy-based resilience for all th data saved in <data-dir>. Some graphs is poor constructed, therefore, the computation will fail, but the error will be well-handled.

```
python ./scripts/generate_label.py --info-dir ./data/info --data-dir ./training/noise_data_0.2/ --label-dir ./training/noise_label_0.2/ --size 100
```

### Train Model

Save the best model and running loss to the <output-dir>

```
python ./scripts/train.py --info-dir ./data/info/ --data-dir ./training/noise_data_0.2/ --label-dir ./training/random_output/ --output-dir ./training/random_output/ --size 100 --epochs 100
```

### Evaluate Model

Use the pre-trained model predict the resilience of a given graph, the predication value will will be saved in <output-dir>/`predict.csv`

```
python ./scripts/evaluation.py --info-dir ./data/info/ --data-path ./data/2012_data.csv --label-path ./data/2012_label.csv --model-path ./training/random_output/best_model.pth --output-dir ./training/random_output/
```

### Plot Resilience Map

Plot the resilience heatmap for ground true label saved in <true-label-path>, predict value saved in <predict-label-path>, and the difference between these two.

```
python ./scripts/plot.resilience.py --true-label-path ./data/2012_label.csv --predict-label-path ./training/random_output/predict.csv --info-dir ./data/info/ --output-dir ./training/random_output/
```
