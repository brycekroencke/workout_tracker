"""
A discord bot for posting users workouts from the json database to a discord channel
"""

import os
import discord
from discord.ext import commands
import random
from dotenv import load_dotenv
import difflib
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import re
import time
import sys
import pyttsx3
import json
import calendar
from datetime import date
from collections import Counter

#connect to discord
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
# client = discord.Client()
client = commands.Bot(command_prefix = '$')


def gen_header(date, day, user):
    header = ("**%s - %s**" % (day, date))
    header_centered = header.center(38, ' ')
    user = ("**%s**" % user)
    user_centered = user.center(38, ' ')
    return ("█" * 18) + "\n" + user_centered + "\n" + header_centered + "\n" + ("▄" * 18)

def gen_date_header(date, day):
    header = ("**%s - %s**" % (day, date))
    header_centered = header.center(38, ' ')
    return ("▄" * 18) + "\n"  + header_centered + "\n" + ("▄" * 18)

def gen_user_header(user):
    user = ("**%s**" % user)
    user_centered = user.center(  38, ' ')
    return ("█" * 18) + "\n"  + user_centered + "\n"

def load_json(filepath):
    with open(filepath) as f:
        if os.stat(filepath).st_size == 0:
            print('File is empty')
            return
        else:
            print('Loading data')
            data = json.load(f)
            return data

def check_date_format(date):
    r = re.compile('.{2}/.{2}/.{4}')
    if r.match(date):
        return True
    else:
        print("invalid date format")
        return False


@client.event
async def on_message(message):
    print(message.channel)
    print(message.author)
    if(message.author == client.user):
        return
    elif message.content[0] == '$':
        await client.process_commands(message)


@client.command()
async def post(ctx, *args):
    date = " ".join(args)
    if ctx.message.author == client.user:
        return

    if not check_date_format(date):
        await ctx.message.channel.send("the date: %s is not in the proper formate of mm/dd/yyyy" % (date))
        return


    data = load_json("workout_overview/workout_log.json")
    user = str(ctx.message.author).split('#')[0]
    if date in data.keys():
        day_str = list(calendar.day_name)[data[date]["day"]]
        lifts = data[date]["lifts"]

        out = gen_header(date, day_str, user)
        for lift in lifts:
            out += "\n__%sS__\n" % lift.upper()
            for weight in lifts[lift]:
                rep_list = []
                for reps in lifts[lift][weight]:
                    rep_list.append(reps["reps"])
                set_rep_count = Counter(rep_list)
                for key in set_rep_count:
                    out += ("%s by %s at %s lbs.\n" % (set_rep_count[key],key, weight))

        await ctx.message.channel.send(out)
    else:
        await ctx.message.channel.send("%s had not data logged on %s" % (user, date))

@client.command()
async def post_all(ctx, *args):
        data = load_json("workout_overview/workout_log.json")
        user = str(ctx.message.author).split('#')[0]
        out = gen_user_header(user)
        first = True
        for date in data:
            day_str = list(calendar.day_name)[data[date]["day"]]
            lifts = data[date]["lifts"]
            header = gen_date_header(date, day_str)
            if first:
                out += header + "\n"
                first = False
            else:
                out = header + "\n"

            for lift in lifts:
                out += "\n__%s__\n" % lift.upper()
                for weight in lifts[lift]:
                    rep_list = []
                    for reps in lifts[lift][weight]:
                        rep_list.append(reps["reps"])
                    set_rep_count = Counter(rep_list)
                    for key in set_rep_count:
                        out += ("%s by %s at %s lbs.\n" % (set_rep_count[key],key, weight))

            await ctx.message.channel.send(out)


@client.command()
async def post_today(ctx, *args):
    text = " ".join(args)
    if ctx.message.author == client.user:
        return
    data = load_json("workout_overview/workout_log.json")
    today = date.today()
    date_str = today.strftime("%m/%d/%Y")


    day_str = list(calendar.day_name)[data[date_str]["day"]]
    lifts = data[date_str]["lifts"]
    user = str(ctx.message.author).split('#')[0]
    out = gen_header(date_str, day_str, user)
    for lift in lifts:
        out += "\n__%s__\n" % lift.upper()
        for weight in lifts[lift]:
            rep_list = []
            for reps in lifts[lift][weight]:
                rep_list.append(reps["reps"])
            set_rep_count = Counter(rep_list)
            for key in set_rep_count:
                out += ("%s by %s at %s lbs.\n" % (set_rep_count[key],key, weight))

    await ctx.message.channel.send(out)


@client.command()
async def kill(ctx):
    sys.exit()



client.run(TOKEN)
