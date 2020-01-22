# DeepGrail

This repository contains an experimental set of Jupyter notebooks and Python files replacing my older part-of-speech taggers and supertaggers with modern deep learning versions. A summary of my deep learning experiments using TLGbank data to develop such part-of-speech taggers and supertaggers can be found below.


### Dependencies

These scripts and notebooks require some package for producing distributional vector representations for words. This can be either `fastText` (using https://fasttext.cc/docs/en/crawl-vectors.html) or `ELMo` (recommended, using https://github.com/HIT-SCIR/ELMoForManyLangs).

### Supertagger results

Percentage of development data correctly tagged: 92.218 (for comparison: best maximum entropy results for the same data are 90.4)

#### Detailed results

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

## References

Moot, R. (2015) _A Type-Logical Treebank for French_, Journal of
Language Modelling **3(1)**, pp. 229-265.

Moot, R. (2019) _Grail Light_ (Github)[https://github.com/RichardMoot/GrailLight] (Hal)[
https://hal.archives-ouvertes.fr/hal-02101396/]
