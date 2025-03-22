python process_dataset/extract_umi_data.py "mouse_arrangement_0,mouse_arrangement_1,towel_folding_0,cup_arrangement_0,cup_arrangement_1" --data_dir=data/umi_data/lz4 --output_dir=/dev/shm/uva/umi_data/zarr

accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=uva_umi_multi.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.different_history_freq=True \
    model.policy.optimizer.learning_rate=1e-4 \
    task.dataset.dataset_root_dir=/dev/shm/uva/umi_data/zarr \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_umi_multitask_video"