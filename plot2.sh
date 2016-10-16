#/bin/bash

grep ^\ 0\  stdout.log | awk '{print $2}'

