noises=(0.1 0.2 0.3 0.4 0.5 0.6)
for noise in ${noises[@]}
do
    echo "noise: ${noise}"
    python ./scripts/calculate_resilience.py --true-label-path ./data/2012_label.csv --predict-label-path ./training/noise_output_${noise}/predict.csv
done

python ./scripts/calculate_resilience.py --true-label-path ./data/2012_label.csv --predict-label-path ./training/random_output/predict.csv