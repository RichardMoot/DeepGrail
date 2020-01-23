# DeepGrail

This repository contains an experimental set of Jupyter notebooks and Python files replacing my older part-of-speech taggers and supertaggers with modern deep learning versions. A summary of my deep learning experiments using [TLGbank data](https://richardmoot.github.io/TLGbank/) to develop such part-of-speech taggers and supertaggers can be found below.

This code has been designed to work together with the [GrailLight parser](https://github.com/RichardMoot/GrailLight) to provide a wide-coverage syntactic and semantic parser for French. 

### Supertagger results and comparison with previous results

The current best LSTM supertagger assigns 92.218 percent of words their correct formula. For comparison, the [best maximum entropy models](https://github.com/RichardMoot/models) for the same data assign 90.41 percent correct, with the gold part-of-speech tag given. The correct part-of-speech tags helps the maximum entropy supertagger quite a bit: a more honest comparison, without the gold part-of-speech tags, has only 88.86 percent of the formulas correct. This improvement is maintained in the more realistic case where multiple formulas are assigned to each word: for an average of 2.4 formulas per word, the maximum entropy supertagger has 97.57 formulas correct whereas the LSTM supertags has 98.97 correct. The LSTM supertagger therefore represents a significant improvement over the earlier results.

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

## Access

The current repository is private until the code has stabilised and a paper describing the results has been published. Until then, contact me for access. 

### Dependencies

These scripts and notebooks require some package for producing distributional vector representations for words. This can be either `fastText` (using https://fasttext.cc/docs/en/crawl-vectors.html) or `ELMo` (recommended, using https://github.com/HIT-SCIR/ELMoForManyLangs).

## Presentations of this material

This material has been presented at several [presentations](https://richardmoot.github.io/Slides/), notable at [WoLLIC2019](https://richardmoot.github.io/Slides/WoLLIC2019.pdf) (Utrecht) and in [Düsseldorf](https://richardmoot.github.io/Slides/WCS_Dusseldorf.pdf).

## References

Moot, R. (2015) _A Type-Logical Treebank for French_, Journal of
Language Modelling **3(1)**, pp. 229-265 ([Github](https://richardmoot.github.io/TLGbank/))

Moot, R. (2019) _Grail Light_ ([Github](https://github.com/RichardMoot/GrailLight), [Hal](
https://hal.archives-ouvertes.fr/hal-02101396/))
