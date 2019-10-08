#!/bin/bash

i="0"

while [ $i -lt 4 ]
do
    #echo "aaaa"
    for ff in $(ls experiments); do
        # echo $ff
        RID=$(echo $ff | cut -d'_' -f 1)
        cp experiments/$ff/loss.png ~/Dropbox/lctmp/${RID}_loss.png
        cp experiments/$ff/log.log ~/Dropbox/lctmp/${RID}_log.log
    done

    sleep 600
    
done

