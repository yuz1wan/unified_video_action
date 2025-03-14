CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 train.py \
    --config-dir=. \
    --config-name=uva_pusht.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_pusht_video_model"