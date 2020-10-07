#!/bin/bash
test=`echo \$HOSTNAME`
for i in {1..3}
do
	for FILE in $(ls assets)
		do     
			if [[ $FILE  = '21.jpg' ]]; then
				./darknet detect cfg/yolov3.cfg yolov3.weights  /darknet/assets/$FILE -out /tmp/output/$test
			else
				./darknet detect cfg/yolov3.cfg yolov3.weights  /darknet/assets/$FILE
			fi
	done
done

