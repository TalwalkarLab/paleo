#! /bin/sh

OUT_FILE=results/case_alexnet_hybrid

PPP_COMP=0.46
PPP_COMM=0.75

echo "AlexNet experiment\n$(date)\n" > $OUT_FILE

# Scalability simulation.

./paleo.sh simulate nets/alex_v2_1gpu.json \
    --hybrid_workers=1 \
    --batch_size=128   \
    --parallel=hybrid  \
    --device_name=K20  \
    --network_name=pcie2 \
    --use_only_gemm    \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    >> $OUT_FILE

echo '\n\n' >> $OUT_FILE

./paleo.sh simulate nets/alex_v2_2gpu.json \
    --hybrid_workers=2 \
    --batch_size=128   \
    --parallel=hybrid  \
    --device_name=K20  \
    --network_name=pcie2 \
    --use_only_gemm    \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    >> $OUT_FILE

echo '\n\n' >> $OUT_FILE

./paleo.sh simulate nets/alex_v2_4gpu.json \
    --hybrid_workers=4 \
    --batch_size=128   \
    --parallel=hybrid  \
    --device_name=K20  \
    --network_name=pcie2 \
    --use_only_gemm    \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    >> $OUT_FILE

echo '\n\n' >> $OUT_FILE

./paleo.sh simulate nets/alex_v2_8gpu.json \
    --hybrid_workers=8 \
    --batch_size=128   \
    --parallel=hybrid  \
    --device_name=K20  \
    --network_name=pcie2 \
    --use_only_gemm    \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    >> $OUT_FILE

echo '\n\nComparison: weak scaling.\n' >> $OUT_FILE

./paleo.sh simulate nets/alex_v2_1gpu.json \
    --batch_size=128 \
    --network_name=pcie2\
    --device_name=K20 \
    --num_workers=1,2,4,8 \
    --ppp_comp=$PPP_COMP \
    --ppp_comm=$PPP_COMM \
    --use_only_gemm \
    --scaling=weak >> $OUT_FILE
