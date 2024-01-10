from transformers import MT5ForConditionalGeneration, MT5Tokenizer

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-xxl')
model = MT5ForConditionalGeneration.from_pretrained('/network/scratch/p/poradaia/mt5_coref')
model = model.cuda()

text = "Speaker-A [1 I ] still have n’t gone to that fresh French restaurant by your house # Speaker-A [1 I ] ’m like dying to go there | # Speaker-B You mean the one right next to the apartment **"

text = 'coref: w | _ The Eiffel Tower ( French : tour Eiffel ) is a wrought - iron lattice tower on the Champ de Mars in Paris , France . It is named after the engineer Gustave Eiffel , whose company designed and built the tower . ** _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World \' s Fair . _ Although initially criticised by some of France \' s leading artists and intellectuals for its design , it has since become a global cultural icon of France and one of the most recognisable structures in the world . _ The Eiffel Tower is the most visited monument with an entrance fee in the world : 6 . 91 million people ascended it in 2015 . _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 .'

inputs = tokenizer(text, return_tensors="pt").input_ids.cuda()
outputs = model.generate(inputs, max_new_tokens=100)
decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_outputs)
