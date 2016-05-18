for f in *.tex; do aspell -t --lang DE_de -c $f; done

#mkdir -p uncorrected-versions
#mv *.tex.bak uncorrected-versions/ 2>/dev/null
