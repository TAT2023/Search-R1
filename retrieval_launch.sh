
file_path=/root/shared-nvme/wiki
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/root/shared-nvme/models/e5-base-v2
LOG_FILE=retrieval_server.log

# Optional GPU pinning for retrieval service.
# Example: RETRIEVAL_CUDA_DEVICES=6,7 bash retrieval_launch.sh
if [ -n "$RETRIEVAL_CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$RETRIEVAL_CUDA_DEVICES"
else
    export CUDA_VISIBLE_DEVICES=6,7
fi

nohup python search_r1/search/retrieval_server.py --index_path $index_file \
                                                   --corpus_path $corpus_file \
                                                   --topk 3 \
                                                   --retriever_name $retriever_name \
                                                   --retriever_model $retriever_path \
                                                   --faiss_gpu \
                                                   > $LOG_FILE 2>&1 &

RETRIEVAL_PID=$!
echo "Retrieval server started in background. PID=$RETRIEVAL_PID, log=$LOG_FILE"
