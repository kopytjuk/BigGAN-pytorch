$datafolder = "C:\Users\kopyt\Documents\DATA\cub-200-cropped"
$experiments_path = "./experiments/1"

python main.py --batch_size 64  --dataset off --adv_loss hinge --version alpha_v1 --image_path "$datafolder" --experiment_path "$experiments_path"
