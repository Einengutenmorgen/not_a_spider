cd ~/not_a_spider && source venvSpider/bin/activate
export H=/tmp/hau G=/GPUdata/hau HF_HOME=/GPUdata/hau/hf
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=5                    # a GPU with at least 25 GB free for the 7B
echo "H=$H G=$G HF_HOME=$HF_HOME"
ls $H/olmo7_ref/positions.jsonl $H/olmo7_idx/index_0.jsonl
