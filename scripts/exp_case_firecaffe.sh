#! /bin/sh

NET_FILE=nets/nin.json
OUT_FILE=results/case_firecaffe

PPP_COMP=0.21
PPP_COMM=0.56

echo "FireCaffe NiN experiment\n$(date)\n" > $OUT_FILE

# Summary
./paleo.sh summary $NET_FILE >> $OUT_FILE

echo '\n\n' >> $OUT_FILE

# Scalability simulation.
./paleo.sh simulate $NET_FILE \
    --batch_size=256 \
    --network_name=infiniband \
    --device_name=K20X \
    --use_only_gemm \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=strong >> $OUT_FILE

echo '\n' >> $OUT_FILE

./paleo.sh simulate $NET_FILE \
    --batch_size=1024 \
    --network_name=infiniband \
    --device_name=K20X \
    --use_only_gemm \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --scaling=strong >> $OUT_FILE

