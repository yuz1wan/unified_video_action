# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 train.py \
#     --config-dir=. \
#     --config-name=uva_pusht.yaml \
#     model.policy.action_model_params.predict_action=False \
#     model.policy.selected_training_mode=video_model \
#     model.policy.optimizer.learning_rate=1e-4 \
#     logging.project=uva \
#     hydra.run.dir="checkpoints/uva_pusht_video_model" \
#     training.debug=True \

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 train.py \
    --config-dir=. \
    --config-name=uva_human_pp.yaml \
    model.policy.autoregressive_model_params.pretrained_model_path=checkpoints/uva_human_pp_video_model/checkpoints/latest.ckpt \
    model.policy.action_model_params.predict_action=True
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva \
    hydra.run.dir="uva_human_pp_video_act_model" \
    training.debug=True \