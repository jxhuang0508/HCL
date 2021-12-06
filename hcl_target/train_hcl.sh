## Gnerate pseudo labels
# python generate_plabel_cityscapes_advent.py  --restore-from /home/jiaxinghuang/research/HCL_model_ada/Seg-Uncertainty/snapshots/HCL_source_only/GTA5_25000.pth

## Re-train networks, HCL, historical models + Aug + HCID
python train_ft_advent_hcl.py --snapshot-dir ./snapshots/HCL_historical_momentum_scaleaug_HCID \
--restore-from /home/jiaxinghuang/research/HCL_model_ada/Seg-Uncertainty/snapshots/HCL_source_only/GTA5_25000.pth \
--drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 \
--lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 \
--max-value 7 --gpu-ids 0,1,2 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False --save-pred-every 300


### Re-train networks, threshold base
#python train_ft_base.py --snapshot-dir ./snapshots/base \
#--restore-from /home/jiaxinghuang/research/HCL_model_ada/Seg-Uncertainty/snapshots/HCL_source_only/GTA5_25000.pth \
#--drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 \
#--lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 \
#--max-value 7 --gpu-ids 0,1,2 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False --save-pred-every 300
