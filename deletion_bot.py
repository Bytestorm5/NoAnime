import discord
from dotenv import load_dotenv
import os
import json
import urllib
import keras
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from PIL import Image

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

model = keras.models.load_model('final_model')
THRESHOLD = 0.5325325325325325325

channel_targets = json.load(open("target_channels.json"))

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

def predict(img):
    imgr = cv2.resize(img, (256, 256))
    imgr=np.expand_dims(imgr,axis=0)
    pred = model.predict(imgr)
    return pred[0][0]

def handle_video(url):
    clip = VideoFileClip(url) 
    clip.iter_frames()

    preds = []

    i = 0
    for frame in clip.iter_frames():
        if i > 100: #break in case things get goofy, we don't need this much from one gif
            break
        if i % 5 == 0:
            preds.append(predict(cv2.resize(frame, (256, 256))))
        i += 1

    clip.close()

    os.remove(url)
    print(np.average(preds))
    return np.average(preds) > THRESHOLD ##if more than half of the frames are considered bad, mark
def handle_image(url):
    image = cv2.imread(url)
    pred = predict(image)  
    os.remove(url)  
    return pred > THRESHOLD

async def react_to_result(result, message):
    if result:
        await message.add_reaction('🤡')
    else:
        await message.add_reaction('👍')
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('cheems!salvation'):
        channel_targets['channels'].append(message.channel.id)
        await message.channel.send('Salvation has arrived.')
    
    if message.channel.id in channel_targets['channels']:
        file_index = 0
        #check if embed is anime
        for embed in message.embeds:
            match (embed.type):
                case 'gifv' | 'video':
                    print(embed.video.url)
                    path = os.path.join("temp", f"{message.id}_{file_index}{embed.video.url[-4:0]}")
                    urllib.request.urlretrieve(embed.video.url, path)

                    react_to_result(handle_video(path))
                case 'image':
                    print(embed.url)
                    path = os.path.join("temp", f"{message.id}_{file_index}{embed.url[-4:0]}")
                    urllib.request.urlretrieve(embed.url, path) #Currently gives 403 error
                    
                    react_to_result(handle_image(path))
            file_index += 1

        #check if attachment is anime
        for attach in message.attachments:
            if attach.content_type.startswith('image'):
                print(attach.url)
                path = os.path.join("temp", f"{message.id}_{file_index}_{attach.filename}")
                await attach.save(path)
                
                react_to_result(handle_image(path))
            elif attach.content_type.startswith('video'):
                print(attach.url)
                path = os.path.join("temp", attach.filename)
                await attach.save(path)
                react_to_result(handle_video(path))
            file_index += 1
        



client.run(TOKEN)