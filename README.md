# PROBLEM:
Define and Develop applications using 2 of the following 4 API (in hugging face): Stable Diffusion, Bert base model, CLIP or Fine-tuned XLSR-53

# DEFINITION:
Using Stable Diffusion and CLIP API to find perfect playlists of songs for any moods.  
It can be a useful tool for people looking to find the perfect playlist for any mood, whether it be for working out, studying, 
relaxing or partying.

# DESCRIPTION AND USAGE:
You enter your mood (any mood), and Stable Diffusion will transform it into an output image that combines your input 
"mood" with face. After that, Clip API will use this output as an input and categorize based on the four fundamental 
emotions: "happy, anger, sad, and love". Then, the different probabilities for the aforementioned categories are output. In the 
code, I added a YouTube playlist for each of the four emotional categories. The output with the highest likelihood will then be 
printed along with its associated playlist. 
Output Example:( The mood+face image is output before the probability output line; however, it is impossible to depict it 
here in Word/PDF) 
Hello, I am a ML DJ which gives playlist depending on your current mood, I would like to help you give the best playlist 
************************************ 
Please enter your mood in one word: joy 
tensor([[0.8247, 0.0857, 0.0704, 0.0193]], grad_fn=<SoftmaxBackward0>) 
The category with the highest probability is: happy 
We recommend you this playlist, please click the link:  
English: https://www.youtube.com/watch?v=ru0K8uYEZWw&list=PLW9z2i0xwq0F38LieqflLLWLWZQgvhEX  
Hindi: https://www.youtube.com/watch?v=Cc_cNEjAh_Y&list=PL8U7gDbfLksNOQIbN_jfC9DVQYt4xX
 To&ab_channel=SonyMusicIndiaVEVO 
Hope you enjoy the music 

# GOOD EXAMPLES:
VALID FOR ALL KINDS OF “MOOD” INPUT:  
You can enter moods that are not included in the categories, and the program will nevertheless produce a playlist that is 
closely related to that mood. The output for joy in the aforementioned example is likewise "happy," which has a comparable 
meaning. 
USEFUL IN OTHER FIELDS: 
We learn about the moods connected with the highest output likelihood along with it, which is useful information for 
philosophical research. However, this may be a bad example for playlist accuracy.

# BAD EXAMPLES:
ACCURACY: 
Accuracy highly depends on Stable Diffusion Output image and Clip API classification. Likewise, various moods (or 
emotions) have comparable facial expressions. E.g., emotions like happy and love have similar face expressions. However, 
this can be a good proof for philosophers that face does not provide accurate emotions 
LIMITATIONS: 
For this model, we have use CLIPAPI for only four basic human emotions, so every mood has to be categorized into these 
four emotions/moods which means it have to choose playlist amongst the four emotions/moods.
