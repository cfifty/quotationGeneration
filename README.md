# quotationGeneration
Given a corpus of quotations, generate a new quotation using Language Model techniques

## Language Model Notes
* Use simplified Kneser-Ney smoothing for bigram and trigrams (literature suggests it is superior to +1 laplace smoothing)
  * Subtract 0.75 from all n-grams which appear 2 or more times
  * Subtract 0.5 from n-grams which appear once 
  * Add 0.000027 to unseen n-grams
  * https://www.researchgate.net/profile/Frankie_James2/publication/255479295_Modified_Kneser-Ney_Smoothing_of_n-gram_Models/links/54d156750cf28959aa7adc08.pdf
  * http://ieeexplore.ieee.org/abstract/document/4244538/?part=1
* 
