#! /bin/sh

NET_FILE=$1
OUT_FILE=$2

PPP_COMP=0.62

# Summary
./paleo.sh summary $NET_FILE > $OUT_FILE

# Fullpass Tensorflow execution
echo '\n\n' >> $OUT_FILE
./paleo.sh fullpass $NET_FILE >> $OUT_FILE

# Layerwise profiling
echo '\n\n' >> $OUT_FILE
./paleo.sh profile $NET_FILE \
    --direction=forward \
    --executor=cudnn \
    --ppp_comp=$PPP_COMP \
    >> $OUT_FILE
./paleo.sh profile $NET_FILE \
    --direction=forward \
    --executor=tensorflow \
    >> $OUT_FILE

echo '\n\n' >> $OUT_FILE
./paleo.sh profile $NET_FILE \
    --direction=backward \
    --gradient_wrt=data \
    --executor=cudnn \
    --ppp_comp=$PPP_COMP \
    >> $OUT_FILE
./paleo.sh profile $NET_FILE \
    --direction=backward \
    --gradient_wrt=data \
    --executor=tensorflow \
    >> $OUT_FILE

echo '\n\n' >> $OUT_FILE
./paleo.sh profile $NET_FILE \
    --direction=backward \
    --gradient_wrt=filter \
    --executor=cudnn \
    --ppp_comp=$PPP_COMP \
    >> $OUT_FILE
./paleo.sh profile $NET_FILE \
    --direction=backward \
    --gradient_wrt=filter \
    --executor=tensorflow \
    >> $OUT_FILE


