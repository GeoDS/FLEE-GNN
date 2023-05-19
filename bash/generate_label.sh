noises=(0.1 0.2 0.3 0.4 0.5 0.6)

for noise in ${noises[@]}
do
    python ./scripts/generate_label.py --info-dir ./data/info --data-dir ./training/noise_data_${noise}/ --label-dir ./training/noise_label_${noise}/ --size 500
done