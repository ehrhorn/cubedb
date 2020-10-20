#!/bin/bash
for i in ./i3/*.zst ; do mv "$i" "${i%.zst}.i3.zst" ; done