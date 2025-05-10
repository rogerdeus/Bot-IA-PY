import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
discord_token = os.getenv("discord_token")
chave_ia = os.getenv("chave_ia")


if not discord_token or not chave_ia:
    raise ValueError("Erro: DISCORD_TOKEN ou CHAVE_IA não encontrados no .env")


genai.configure(api_key=chave_ia)
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    }
)


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user} (ID: {bot.user.id})")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    if bot.user in message.mentions:
        user_message = message.content.replace(f"<@!{bot.user.id}>", "").strip()
        if not user_message:
            await message.reply("Por favor, envie uma mensagem válida!")
            return
        await message.channel.typing()
        try:
            response = model.generate_content(user_message)
            reply = response.text[:2000]
            await message.reply(reply)
        except Exception as e:
            await message.reply(f"Erro: Processamento de mensagem falhou: {str(e)}")

    await bot.process_commands(message)

bot.run(discord_token)