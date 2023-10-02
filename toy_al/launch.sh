python sample_sim.py 10 0
python training.py --batch_size 128 --epoch 200 --lr 0.001 --seed 42 --device gpu --num_workers 8 --data_root_dir ./ --phase 0 --ckpt_dir ./ --input_filename input_for_training_phase_0.hdf5 --output_filename_prefix test_output 
python sample_sim.py 40 1
python training.py --batch_size 128 --epoch 200 --lr 0.001 --seed 42 --device gpu --num_workers 8 --data_root_dir ./ --phase 0 --ckpt_dir ./ --input_filename input_for_training_phase_1.hdf5 
