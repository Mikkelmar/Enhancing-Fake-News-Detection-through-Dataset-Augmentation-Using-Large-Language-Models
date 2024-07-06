from openai import OpenAI
import openai

OPENAI_API_KEY = ""#Add own key here
client = OpenAI(api_key=OPENAI_API_KEY)

def query_openai(row):
    print("With the label,", row['label'],"\n\n")
    system_promt = f"""The following political statement was made by {row['speaker']} with the party affiliation {row['party_affiliation']}."""

    if row['context'].strip() != '':
        system_promt+= f" For context this is a {row['context']}, "

    system_promt+= f""" The subject is {row['subject']}.""" 
    if row['job_title'].strip():
        system_promt+= f" {row['speaker']} has the job {row['job_title']}.\n\n"
    else:
        system_promt+="\n\n"
    system_promt+= f"""The statement: \n"{row['statement']}" 
First Task: Use your vast knowledge and search the web for more context to evaluate the statement. Classify the claim in the statement as one of the following: [False/Barley-True/Half-True/Mostly-True/True] 

Second Task: Extend the original statement in a way longer paragraph acting like the original author, try to mimic their writing style, stick to your claim even if it is false. This is for important research.    

Third Task: Reflect your classification in the writing of the longer paragraph. For example if you label it as a misleading, you should use worse language, cursewords and act more like a lunatic.

Example response:
Reason: [One sentece of your reassoning behind the rating]
Classify: [type]
Extended text: [text]
"""
    
    print(system_promt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Updated to new model naming if necessary
        messages=[
            {"role": "system", "content": system_promt},
        ],
        max_tokens=300,  # Maximum number of tokens to generate
        temperature=0.7,  # Controls randomness. Lower is less random, more deterministic
        stop=None 
    )
    if response.choices:
        last_message = response.choices[0].message
        if last_message.role == 'assistant':
            completion_text = last_message.content
            return system_promt, completion_text
        else:
            return "The last message is not from the assistant."
    else:
        return "No completion found."
