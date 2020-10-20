#!/bin/bash
while IFS= read -r line; do
    rsync -P -u -h hep03:"$line" ./i3/
done < files.txt

line=$(head -n 1 gcd.txt)
rsync -P -u -h hep03:"$line" ./gcd/
