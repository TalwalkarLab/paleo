#! /bin/sh

NET_FILE=nets/alex_v2.json
OUT_FILE=results/case_alexnet_scalability

PPP_COMP=0.62
PPP_COMM=1.0

echo "AlexNet Scalability experiment\n$(date)\n" > $OUT_FILE

# Scalability simulation.

./paleo.sh simulate $NET_FILE \
    --batch_size=2048 \
    --network_name=ethernet20 \
    --device_name=K80 \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=strong,weak >> $OUT_FILE

./paleo.sh simulate $NET_FILE \
    --batch_size=256 \
    --network_name=ethernet20 \
    --device_name=K80 \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=strong,weak >> $OUT_FILE
