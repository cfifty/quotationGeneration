# quotationGeneration
Given a corpus of quotations, generate a new quotation using Language Model techniques

## Report
1. Final Report Link: https://docs.google.com/document/d/18K8nKzGEptEv4apl8lS2JtXQjp7exSSntbZDmsjV7wg/edit?usp=sharing
2. Presentation Link: https://docs.google.com/presentation/d/1-x0Ac8Pd5EQoOYOBhsiV3Dr5gEi1hXCUKomJldLNXJQ/edit?usp=sharing
3. Proposal Link: https://docs.google.com/document/d/1sa6AfUZY2qDRpmPiJ2wMs5Fvc-Cz3BsqqB8GCFzeKQ4/edit?usp=sharing

## Neural Net Notes:
Tutorial: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
CUDA GPU on AWS instructions: https://github.com/dennybritz/rnn-tutorial-rnnlm 

### Virtual Environment Setup
- Python, pip v 2.7
- [virtualenv](https://virtualenv.pypa.io/en/latest/)

```bash
# Create a new virtual environment 
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# If Theano is giving you trouble (upgraded Mac Systems)
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

## Language Model Notes
* Use simplified Kneser-Ney smoothing for bigram and trigrams (literature suggests it is superior to +1 laplace smoothing)
  * Subtract 0.75 from all n-grams which appear 2 or more times
  * Subtract 0.5 from n-grams which appear once 
  * Add 0.000027 to unseen n-grams
  * https://www.researchgate.net/profile/Frankie_James2/publication/255479295_Modified_Kneser-Ney_Smoothing_of_n-gram_Models/links/54d156750cf28959aa7adc08.pdf
  * http://ieeexplore.ieee.org/abstract/document/4244538/?part=1

## Evaluation
### Language Models
* Donald Trump: https://goo.gl/forms/yTv29gNEOguivH0B3
* Cicero: https://goo.gl/forms/eYgZmQsJMMdfATjw1

### Recurrent Neural Net
*
* 

### LSTM Neural Net
*
*
