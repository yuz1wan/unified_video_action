CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 train.py \
    --config-dir=. \
    --config-name=uva_human_pp.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_human_pp_video_model"