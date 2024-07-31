"""
YouTube vid summarizer

Uses captions of YouTube video to generate summary and details about the video.
The summary is made with GPT-3.5 model

Tools

- youtube-transcript-api
- transformers
- youtube-transcript-api
- youtube-transcript-api
- youtube-transcript-api
- PyTorch


"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse




# define pipe
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

url = input('Enter YouTube video URL: ')

video_id = url.replace('https://www.youtube.com/watch?v=', '')

transcript = YouTubeTranscriptApi.get_transcript(video_id)

# format text
formatter = TextFormatter()
transcript = formatter.format_transcript(transcript)

# print
print(transcript)

# LLM model

input_text = "Summarize: \n" + transcript + "\nPlace the tools, people mentioned, sources and website in bullet points each in a new line. And if applicable step-by-step actions.\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
final_output = tokenizer.decode(outputs[0])
print(tokenizer.decode(outputs[0]))


# save to file
with open('summary.txt', 'w') as f:
    f.write(final_output)