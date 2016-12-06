#! /bin/sh

NET_FILE=nets/inception_v3.json
OUT_FILE=results/case_inception

PPP_COMP=0.62
PPP_COMM=0.72

echo "Inception v3 experiment\n$(date)\n" > $OUT_FILE

# Summary, too long.
# ./paleo.sh summary $NET_FILE

./paleo.sh simulate $NET_FILE \
    --batch_size=256 \
    --network_name=ethernet\
    --device_name=K40 \
    --num_workers=1,2,4,8,16,50,100 \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=weak >> $OUT_FILE



