noises=(0.3 0.4 0.5 0.6)

for noise in ${noises[@]}
do
    python ./scripts/generate_data_from_real_data.py --data-path ./data/2012_data.csv --save-dir ./training/noise_data_${noise}/ --size 500 --noise-rate ${noise}
done