# def custom_prompt_sentiment(text):
#     prompt=f''''
# you are an NLP expert.custom_prompt_sentiment
# Analyze the sentiment of the following sentence and classify it as
# Positive,Negative,or Neutral.
# Sentence:"{text}"
# Answer format:
# Sentiment:<positive/Negative/Neutral>
# Explanation:<short reason>
# return prompt
# sentence= "The new AI tool is extremely useful and easy to use."
# prompt_output = custom_prompt_sentiment(sentence)
# print("Generated custom prompt:/n")
# print(prompt_output) 


import nltk
from nltk.sentiment.vadar