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

## Presentations of this material

This material has been [presented](https://richardmoot.github.io/Slides/) on a number of occasions, notably at:
* [WoLLIC2019](https://richardmoot.github.io/Slides/WoLLIC2019.pdf) (Utrecht), where I talk about vector representations, wide-coverage semantics, and reconciling vector-based semantics with formal semantics,
* [Düsseldorf](https://richardmoot.github.io/Slides/WCS_Dusseldorf.pdf), one of the first presentations of Grail Light with the deep learning component (DeepGrail); also has some discussion about the difference between type-logical grammars and combinatory categorial grammars.

## References

Moot, R. (2015) _A Type-Logical Treebank for French_, Journal of
Language Modelling **3(1)**, pp. 229-265 ([Github](https://richardmoot.github.io/TLGbank/), [HAL](https://hal.archives-ouvertes.fr/hal-02102867v1)).

Moot, R. (2019) _Grail Light_ ([Github](https://github.com/RichardMoot/GrailLight), [HAL](
https://hal.archives-ouvertes.fr/hal-02101396/))
