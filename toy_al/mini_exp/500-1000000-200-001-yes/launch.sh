#!/bin/bash

echo "Start first sampling"
python ../sample_sim.py \
	--seed 42 \
	--phase 0 \
	--num_train_sample 500 \
	--num_test_sample 1000000 \
	--num_global_test_sample 200 \
	--sigma 0.01 \
	--sample_input_filename test_output.hdf5 \
	--global_test_filename global_test.hdf5

echo "Start first training"
python ../training.py \
	--batch_size 128 \
       	--epoch 1000 \
	--lr 0.001 \
	--seed 43 \
	--device gpu \
	--cont_train no \
	--num_workers 1 \
	--phase 0 \
	--ckpt_dir ./ \
	--output_filename test_output.hdf5 \
 	--global_test_filename global_test.hdf5

echo "Start second sampling"
python ../sample_sim.py \
	--seed 44 \
	--phase 1 \
	--num_train_sample 500 \
	--num_test_sample 1000000 \
	--num_global_test_sample 200 \
	--sigma 0.01 \
	--sample_input_filename test_output.hdf5 \
	--global_test_filename global_test.hdf5

echo "Start second trainig"
python ../training.py \
	--batch_size 128 \
       	--epoch 1000 \
	--lr 0.001 \
	--seed 52 \
	--device gpu \
	--cont_train yes \
	--num_workers 1 \
	--phase 1 \
	--ckpt_dir ./ \
	--output_filename test_output.hdf5 \
 	--global_test_filename global_test.hdf5

#python sample_sim.py \
#	--seed 44 \
#	--phase 1 \
#	--num_train_sample 240 \
#	--num_test_sample 120 \
#	--num_global_test_sample 50 \
#	--sigma 0.02 \
#	--sample_input_filename test_output.hdf5 \
#	--global_test_filename global_test.hdf5

#python sample_sim.py 40 1
#python training.py --batch_size 128 --epoch 200 --lr 0.001 --seed 42 --device gpu --num_workers 8 --data_root_dir ./ --phase 0 --ckpt_dir ./ --input_filename input_for_training_phase_1.hdf5 
