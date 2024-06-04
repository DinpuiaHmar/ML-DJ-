print("Hello, I am a ML DJ which gives playlist depending on your current mood, I would like to help you give the best playlist")
print("************************************")
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model_id = "stabilityai/stable-diffusion-2-1"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def text_to_image(prompt):
  image = pipe(prompt, height=768, width=768).images[0]
  return image

MOOD = input("Please enter your mood in one word: ")
MOOD = "A person with" + MOOD + "face"
image = text_to_image(MOOD)
image.show()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=["happy", "anger", "love", "sad"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)


playlist_links = {
    "happy": ["English: https://www.youtube.com/watch?v=ru0K8uYEZWw&list=PLW9z2i0xwq0F3-8LieqflLLWLWZQgvhEX", "Hindi: https://www.youtube.com/watch?v=Cc_cNEjAh_Y&list=PL8U7gDbfLksNOQ-IbN_jfC9DVQYt4xXTo&ab_channel=SonyMusicIndiaVEVO"],
    "anger": ["English: https://www.youtube.com/watch?v=Vrr3lRLjZ1Y&list=PLknqyEOvGo1YgL11BN1m-YOxaFHl29elY", "Hindi: https://www.youtube.com/watch?v=Gh5wHtqW9Ek&list=PL9bw4S5ePsEG1BSA7I5EtqskLWRaQojwR&ab_channel=T-Series"],
    "love": ["English: https://www.youtube.com/watch?v=lp-EO5I60KA&list=PL64G6j8ePNureM8YCKy5nRFyzYf8I2noy&ab_channel=EdSheeran", "Hindi: https://www.youtube.com/watch?v=atVof3pjT-I&list=PL9bw4S5ePsEGpT9PdWJYN8joMa2eWAxJf&ab_channel=T-Series  "],
    "sad": ["English: https://www.youtube.com/watch?v=CkJ6w-V54EA", "Hindi: https://www.youtube.com/watch?v=SBWYGGDYmhg&list=PLHuHXHyLu7BGi-vR7X6j_xh_Tt9wy7pNA&ab_channel=SonyMusicIndia"]
}

# Get the highest probability mood category
max_prob, max_prob_idx = torch.max(probs, dim=1)
max_prob_idx = max_prob_idx.item()
max_prob_mood = list(playlist_links.keys())[list(playlist_links.values()).index(playlist_links[list(playlist_links.keys())[max_prob_idx]])]

# Get the playlist link for the highest probability mood category
playlist_link = playlist_links[max_prob_mood]


print("The category with the highest probability is:", max_prob_mood)

print("We recommend you this playlist, please click the link: \n" ,  playlist_link[0], "\n", playlist_link[1])

print("Hope you enjoy the music")
