This directory contains the files for the current best model trained as a combined part-of-speech tagger and supertagger.

`superpos.py -m <ModelFile> -b <SuperBeta> -i <InputFile> -o <Outputfile>`


Results on the training and development data are the following.

| Model | Pos1 train | Pos1 dev | Pos2 train | Pos2 dev | Super train | Super dev |
|:---------------|:------|:------|:------|:------|:------|:------|
superposmodel.h5 | 98.65 | 98.70 | 98.26 | 98.32 | 93.54 | 92.41 |
