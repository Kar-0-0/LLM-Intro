# LLM-Intro
# Steps

### Data set taken from: https://github.com/karpathy/makemore/blob/master/names.txt 

# Approach 1:
## Implement Bigram Model 
- Take each name and arrange into pairs... ex. emma --> .e, em, mm, ma, m.
- Count appearance of each pair
- Take counts and create probability for how likley the second letter is to come out the first
- Feed it through tool that samples based off of probability and concatenate to form names
- Calculate loss by taking the mean of the negative of the log of the probability 

# Approach 2:
-


# On My Own:
## Trigram Model:
- - Built trigram model using approach 1
- - Take name and make into groups of three... ex. emma --> ..e, .em, emm, mma, ma., a..
- - Count appearance of each three
- - Normalize rows to find probability of each group
- - Use torch.multinomial() to take sample based off of probability
- - Take sample and attatch it to name but then make the new sample by feeding previous 2 words to get third word
- - Calculate loss by taking the average of the log likelihood (log_liklihood += torch.log(probabilities))
- - Take negative for NLLLoss
- - Loss is at 1.9166
