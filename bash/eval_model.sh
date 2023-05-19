noises=(0.1 0.2 0.3 0.4 0.5 0.6)
for noise in ${noises[@]}
do
    python ./scripts/evaluation.py --info-dir ./data/info/ --data-path ./data/2012_data.csv --label-path ./data/2012_label.csv --model-path ./training/noise_output_${noise}/best_model.pth --output-dir ./training/noise_output_${noise}/
done
