"""
File: test.py
Description: Test file for OpenAI API functionality.
"""
from openai import OpenAI
from utils import openai_api_key

client = OpenAI(api_key=openai_api_key)

def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    completion = client.chat.completions.create(model="gpt-4o-mini", 
    messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"

prompt = """
---
Character 1: agent_js is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: agent_av is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, agent_js and agent_av were already conversing about conversing about agent_js's research paper mentioned by agent_av This context takes place after that conversation.

Current Context: agent_js was attending her Physics class (preparing for the next lecture) when agent_js saw agent_av in the middle of working on his research paper at the library (writing the introduction).
agent_js is thinking of initating a conversation with agent_av.
Current Location: library in X College

(This is what is in agent_js's head: agent_js should remember to follow up with agent_av about his thoughts on her research paper. Beyond this, agent_js doesn't necessarily know anything more about agent_av) 

(This is what is in agent_av's head: agent_av should remember to ask agent_js about her research paper, as she found it interesting that he mentioned it. Beyond this, agent_av doesn't necessarily know anything more about agent_js) 

Here is their conversation. 

agent_js: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""


print (ChatGPT_request(prompt))












