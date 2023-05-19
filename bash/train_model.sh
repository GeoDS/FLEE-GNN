noises=(0.1 0.2 0.3 0.4 0.5 0.6)
for noise in ${noises[@]}
do
    python ./scripts/train.py --info-dir ./data/info/ --data-dir ./training/noise_data_${noise}/ --label-dir ./training/noise_label_${noise} --output-dir ./training/noise_output_${noise} --size 500 --epochs 100
done
