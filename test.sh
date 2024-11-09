# 1 window, 10 shots
# python run_evaluation.py \
# --dataset banking77 \
# --model togethercomputer/LLaMA-2-7B-32K \
# --subsample-test-set 250 \
# --n-runs 1 \
# --n-shots-per-window 838 \
# --n-windows 1 \
# --fp16 \
# --output-dir ./test
# fp32 max 400 nspw for banking 77 32k on NVIDIA L40S

# block attention
# examples_stride=50
python run_evaluation.py \
--dataset banking77 \
--model togethercomputer/LLaMA-2-7B-32K \
--subsample-test-set 250 \
--n-runs 1 \
--n-shots-per-window 838 \
--examples-stride 50 \
--n-windows 1 \
--fp16 \
--output-dir ./test
