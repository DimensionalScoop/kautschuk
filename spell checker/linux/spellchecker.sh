#!/bin/sh
for f in *.tex; do aspell -t --lang DE_de -c $f; done