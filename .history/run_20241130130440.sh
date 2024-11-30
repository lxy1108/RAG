# export CUDA_VISIBLE_DEVICES=2,3
python -u run.py \
    --dense_retriever_lists m3e \
    --recursive_answer \
    --rephrase_context \
    --rephrase_before_retrieve \
    --answer_before_retrieve \


