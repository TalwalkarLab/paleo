#! /bin/sh

OUT_FILE=results/case_mxnet

PPP_COMP=0.49
PPP_COMM=0.72

echo "MXNET experiment\n$(date)\n" > $OUT_FILE

./paleo.sh simulate nets/inception_v3.json \
    --batch_size=32 \
    --network_name=ethernet20 \
    --device_name=K80 \
    --num_workers=1,2,4,8,16,32,64,128 \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=weak >> $OUT_FILE

echo "\n\nAlexNet 512 \n\n" >> $OUT_FILE

./paleo.sh simulate nets/alex_v2.json \
    --batch_size=512 \
    --network_name=ethernet20 \
    --device_name=K80 \
    --num_workers=1,2,4,8,16,32,64,128 \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=weak >> $OUT_FILE

