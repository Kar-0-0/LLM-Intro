# LLM-Intro
# Steps

### Data set taken from: https://github.com/karpathy/makemore/blob/master/names.txt 

# Approach 1:
### Implement Bigram Model 
- Take each name and arrange into pairs... ex. emma --> .e, em, mm, ma, m.
- Count appearance of each pair
- Take counts and create probability for how likley the second letter is to come out the first
- Feed it through tool that samples based off of probability and concatenate to form names
- Calculate loss by taking the mean of the negative of the log of the probability 

# Approach 2:
-
