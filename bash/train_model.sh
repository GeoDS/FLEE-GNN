noises=(0.1 0.2 0.3 0.4 0.5 0.6)
for noise in ${noises[@]}
do
    python ./scripts/train.py --info-dir ./data/info/ --data-dir ./training/noise_data_${noise}/ --label-dir ./training/noise_label_${noise} --output-dir ./training/noise_output_${noise} --size 500 --epochs 100
done

 python ./scripts/train.py --info-dir ./data/info/ --data-dir ./training/noise_data_0.3/ --label-dir ./training/noise_label_0.3 --output-dir ./training/noise_output_0.3_drop_columns --size 500 --epochs 100


python ./scripts/train.py --info-dir ./data/info/ --data-dir ./training/2012_1-6_noise_0.3/ --label-dir ./training/2012_1-6_noise_0.3_label --output-dir ./training/2012_1-6_noise_0.3_output --size 500 --epochs 100