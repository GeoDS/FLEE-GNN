noises=(0.1 0.2 0.3 0.4 0.5 0.6)
for noise in ${noises[@]}
do
    python ./scripts/evaluation.py --info-dir ./data/info/ --data-path ./data/2012_data.csv --label-path ./data/2012_label.csv --model-path ./training/noise_output_${noise}/best_model.pth --output-dir ./training/noise_output_${noise}/
done

python ./scripts/evaluation.py --info-dir ./data/info/ --data-path ./data/2012_data.csv --label-path ./data/2012_label.csv --model-path ./training/noise_output_0.3_dropAVGMILE/best_model.pth --output-dir ./training/noise_output_0.3_dropAVGMILE/



python ./scripts/evaluation.py --info-dir ./data/info/ --data-path ./data/partial/2017_data_1-6.csv --label-path ./data/partial/2017_label_1-6.csv --model-path ./training/2012_1-6_noise_0.3_output/best_model.pth --output-dir ./training/2012_1-6_noise_0.3_output/