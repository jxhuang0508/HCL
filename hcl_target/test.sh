for i in {300..10000000..300}
do
	echo "TEST $i MODEL"
	python evaluate_cityscapes_advent_best.py --restore-from ./snapshots/HCL_historical_momentum_scaleaug_HCID/GTA5_$i.pth
done

