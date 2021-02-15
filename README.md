# DeepGrail

This repository contains an experimental set of Jupyter notebooks and Python files replacing my older part-of-speech taggers and supertaggers with modern deep learning versions. A summary of my deep learning experiments using [TLGbank data](https://richardmoot.github.io/TLGbank/) to develop such part-of-speech taggers and supertaggers can be found below.

This code has been designed to work together with the [GrailLight parser](https://github.com/RichardMoot/GrailLight) to provide a wide-coverage syntactic and semantic parser for French. 

### Supertagger results and comparison with previous results

The current best LSTM supertagger assigns 93.2 percent of words their correct formula. For comparison, the [best maximum entropy models](https://github.com/RichardMoot/models) for the same data assign 90.41 percent correct, with the gold part-of-speech tag given. The correct part-of-speech tags helps the maximum entropy supertagger quite a bit: a more honest comparison, without the gold part-of-speech tags, has only 88.86 percent of the formulas correct. This improvement is maintained in the more realistic case where multiple formulas are assigned to each word: for an average of 2.4 formulas per word, the maximum entropy supertagger has 97.57 formulas correct whereas the LSTM supertagger has 99.0 correct. The LSTM supertagger therefore represents a significant improvement over the earlier results.

#### Detailed results

The table below lists the percentage of correct formulas (under `Correct`) and the average number of formulas assigned to each word (under `Formulas`).

| Beta | Correct | Formulas|
|:-----|--------:|--------:|
1.0   | 92.22 | 1.00 |
0.1   | 95.81 | 1.15 |
0.05  | 96.54 | 1.22 |
0.01  | 97.86 | 1.49 |
0.005 | 98.29 | 1.67 |
0.001 | 98.97 | 2.40 |
0.0005 | 99.17 | 2.91 |
0.0001 | 99.48 | 4.73 |

### Comparison

The table and image below compare the maximum entropy part-of-speech/supertagger  with the LSTM version and the LSTM version using ELMo vector embeddings (on the [same data](https://richardmoot.github.io/TLGbank/)). The trade-off here is to get an error percentage which is as small as possible (to ensure the correct formula is among those assigned) while having keeping the average number of possible words per formula low (for parsing efficiency).

| Model | POS | Super | 0.1 | 0.01 | 0.001 |
|-------|-----|-------|-----|------|-------|
| MaxEnt | 97.8 | 90.6 | 96.4 (1.4) | 98.4 (2.3) | 98.8 (4.7) |
| LSTM | 98.4 | 92.2 | 95.8 (1.2) | 97.9 (1.5) | 99.0 (2.4) |
| LSTM+ELMo | 99.1 | 93.2 | 97.6 (1.1) | 98.6 (1.5) | 99.3 (3.0) |

![visual map of the average number of formulas/word versus the error percentage for the different models](https://github.com/RichardMoot/Slides/blob/master/eval_deep.png)

## Access

The current repository is private until the code has stabilised and a paper describing the results has been published. Until then, contact me for access. 

### Dependencies

These scripts and notebooks require some package for producing distributional vector representations for words. This can be either `fastText` (using https://fasttext.cc/docs/en/crawl-vectors.html) or `ELMo` (recommended, using https://github.com/HIT-SCIR/ELMoForManyLangs).

## Using the taggers

The model files are too big to be included in the repository. Contact me if you want the latest models.

There are two main tagger scripts `super.py` and `elmo_super.py`. The main difference between these two scripts is that the first uses the averaged ELMo vectors as input (one 1024-float vector per word) whereas the second uses all ELMo vectors as input (three 1024-float vectors per word). In practice, there doesn't appear to be a big difference in performance between the two, so for most people the `super.py` script is preferred. 

The script is invoked as follows.
```
super.py --input input.txt --output super.txt --model=modelfile --beta 0.01 
```
The input and output files can be specified using the `--input` and `--output` options (they default to `input.txt` and `super.txt` if not explicitly specified). The model file contains the filename of the tagger model to be loaded. Finally, the `beta` parameter specifies the number of formulas to output for each word as a function of the probability assigned to the most likely formula. For example, a beta value of 0.01 and most likely assignment of 0.9 means all formula with probability over 0.009 will be output. 


## Training your own models

To train your own models, you need the training data stored in the `TLGbank` directory. Each file `sentXXXXXX.npz` contains sentence number XXXXXX, and sentence should be numbered from 0 to the value of `treebank_sentences` minus 1.

Traning is done using the following command.
```
avg_sequence_script.py
```
There are a number of variant training scripts (`all_sequence_script.py` concatenates the different ELMo vectors, whereas `w_avg_sequence_script.py` estimates a weighted average as part of its model parameters).

In each case, the model is fairly simple: the ELMo embedding is fed to two bidirectional LSTM layers and a final dense layer computes the tags. Regularisation parameters and the optimiser can easily be adapted, as can the number of epochs (the default is 100).

Intermediate data is stored in the files `current_gen_elmo_superpos.h5` (model for the last completed epoch) and `best_gen_elmo_superpos.h5` (the current best performing model on the validation data). Training statistics are ouput in `elmo_training_log.csv`.

If desired, training can be continued after the last epoch (or after execution has been aborted) by using `avg_sequence_script_continue.py` (or `all_sequence_script_continue.py` for any of the scripts requiring all ELMo vectors as input). The resumes training from `current_gen_elmo_superpos.h5` and continues for 100 more epochs.

## Presentations of this material

This material has been [presented](https://richardmoot.github.io/Slides/) on a number of occasions, notably at:
* [WoLLIC2019](https://richardmoot.github.io/Slides/WoLLIC2019.pdf) (Utrecht), where I talk about vector representations, wide-coverage semantics, and reconciling vector-based semantics with formal semantics,
* [Düsseldorf](https://richardmoot.github.io/Slides/WCS_Dusseldorf.pdf), one of the first presentations of Grail Light with the deep learning component (DeepGrail); also has some discussion about the difference between type-logical grammars and combinatory categorial grammars.

## References

Moot, R. (2015) _A Type-Logical Treebank for French_, Journal of
Language Modelling **3(1)**, pp. 229-265 ([Github](https://richardmoot.github.io/TLGbank/), [HAL](https://hal.archives-ouvertes.fr/hal-02102867v1)).

Moot, R. (2019) _Grail Light_ ([Github](https://github.com/RichardMoot/GrailLight), [HAL](
https://hal.archives-ouvertes.fr/hal-02101396/))

Kogkalidis, K., Moortgat, M., Moot, R. and Tziafas, G. (2019) _Deductive Parsing with an Unbounded Type Lexicon_, Proceedings SEMSPACE 2019 ([HAL](https://hal-lirmm.ccsd.cnrs.fr/lirmm-02313572/))
