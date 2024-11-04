# LLM-Intro

### Data set taken from: https://github.com/karpathy/makemore/blob/master/names.txt 

# Linear Network
## Approach 1:
### Implement Bigram Model 
- Take each name and arrange into pairs... ex. emma --> .e, em, mm, ma, m.
- Count appearance of each pair
- Take counts and create probability for how likley the second letter is to come out the first
- Feed it through tool that samples based off of probability and concatenate to form names
- Calculate loss by taking the mean of the negative of the log of the probability 

## Approach 2:
### Implement Nerual Net
- First form data set of bigrams (first letter is an input and the second letter is the output)
- Encode x using F.one_hot()...this creates vector that has a one at xth index  --> ex. "a" = 1 so [0, 1, 0, 0...]
- Create weight matrix using torch.randn 
- Multiply input vector with weight matrix to get ouput
- Expenentiate values then sum along the rows 
- Calculate loss with this by taking the negative log and then mean of the probabilities


## On My Own:
### Trigram Model:
#### Approach One 
-  Built trigram model using approach 1
-  Take name and make into groups of three... ex. emma --> ..e, .em, emm, mma, ma., a..
-  Count appearance of each three
-  Normalize rows to find probability of each group
-  Use torch.multinomial() to take sample based off of probability
-  Take sample and attatch it to name but then make the new sample by feeding previous 2 words to get third word
-  Calculate loss by taking the average of the log likelihood (log_liklihood += torch.log(probabilities))
-  Take negative for NLLLoss
-  Loss is at 1.9166

#### Approach Two
- Something went wrong because I am not getting the same loss as my counting model and this model isn't performing well
- First form data set of trigrams (first two letters is an input and the third letter is the output)
- Encode x using multi_hot(tuple, num_classes)...this creates vector that has a one at x[i] and x[j] index  --> ex. ("a", "b") = (1, 2) so [0, 1, 2, 0...]
- Create weight matrix using torch.randn((r, c)) 
- Multiply input vector with weight matrix to get ouput
- Expenentiate values then sum along the rows 
- Calculate loss with this by taking the negative log and then mean of the probabilities

# Multi-Layer Perceptron (https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
## Draft One
- Initialize weights and biases
- Take input and embed it using the look up table 
- Backwards pass to optimize model
- Use learning rate decay to minimize loss
- Sample using torch multinomial

## Draft Two:
#### Context Length Tuning
- Loss when context = 3 --> 2.16 (11,897)
- Loss when context = 4 --> 2.12 (13,897)
- Loss when context = 7 --> 2.21 (19,897)
- Loss when context = 5 --> 2.16 (15,897)
- Best were context length of 4 and 5
- Name comparison: 
    - CL = 4 --> [arcee, athik, raylynn, svin, evia, willan, araiyannony, blarianna, rohson, reinder]
    - CL = 5 --> [tethea, beel, amaryia, tasia, maylee, deyssa, syove, maleesen, guilla, remton]
- Names when context length = 5 are much better 
#### Hidden Layer Tuning
- Hidden Layer with 400 neurons (31,497)
    - Loss: 2.15 
    - Names are worse
- Hidden Layer with 300 Neurons (23,697)
    - Loss: 2.15
    - Names are a bit better but still worse than 200 neurons
- Hidden Layer with 100 Neurons (8,097)
    - Loss: 2.16
    - Names were much better than the 300 & 400 Neuron 
- Hidden Layer with 50 Neurons (4,197)
    - Loss 2.20
    - Names are much worse
- Hidden Layer with 800 Neurons (62,697)
    - Loss: 2.26
    - Names aren't very good
- TODO:
    - Implement something else from paper


# Activations & Gradients, Batch Norm
- 
