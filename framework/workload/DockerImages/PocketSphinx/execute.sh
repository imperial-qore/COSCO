#!/bin/bash
test=`echo \$HOSTNAME`
for i in {1..10}
	do
	for FILE in $(ls assets)
		do  pocketsphinx_continuous -infile ~/assets/$FILE -logfn /dev/null
	done 
done


