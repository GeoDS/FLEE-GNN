# FLEE-GNN

## Environment Configuration

```
conda create -n resilience python=3.9

python -m pip install pandas numpy tqdm networkx matplotlib

```

## Scripts

### Generate Data

- Random Generator

```
python ./scripts/generate_data_random.py --save-dir ./training/random_data/ --size 100
```

- Noise Generator

```
python ./scripts/generate_data_from_real_data.py --data-path ./data/2012_data.csv --save-dir ./training/noise_data_0.2/ --size 100 --noise-rate 0.2
```

### Generate Labels

```
python ./scripts/generate_label.py --info-dir ./data/info --data-dir ./training/noise_data_0.2/ --label-dir ./training/noise_label_0.2/ --size 100
```

### Train Model

```
python ./scripts/train.py --info-dir ./data/info/ --data-dir ./training/noise_data_0.2/ --label-dir ./training/random_output/ --output-dir ./training/random_output/ --size 100 --epochs 100
```

### Evaluate Model

```
python ./scripts/evaluation.py --info-dir ./data/info/ --data-path ./data/2012_data.csv --label-path ./data/2012_label.csv --model-path ./training/random_output/best_model.pth --output-dir ./training/random_output/
```

### Plot Resilience Map

```
python ./scripts/plot.resilience.py --true-label-path ./data/2012_label.csv --predict-label-path ./training/random_output/predict.csv --info-dir ./data/info/ --output-dir ./training/random_output/
```
