from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline
from flask import Flask
from flask import request, jsonify

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)

app = Flask(__name__)

conversation = Conversation()

@app.route('/add_input', methods = ['GET', 'POST'])
def add_input():
     text = request.json['text']
     conversation.add_user_input(text)
     result = nlp([conversation], do_sample=False, max_length=1000)
     messages = []
     for is_user, text in result.iter_texts():
          messages.append({
               'is_user': is_user,
               'text': text
          })
     return jsonify({
          'uuid': result.uuid,
          'messages': messages
     })

print(add_input())
FLASK_APP=main, FLASK_ENV=development, flask=run

# curl -X POST http://localhost:5000/add_input -H "Content-Type: application/json" - data '{"text": "hello"}'

# {
#   "messages": [
#     {
#       "is_user": True, 
#       "text": "hello"
#     }, 
#     {
#       "is_user": False, 
#       "text": " Hello! How are you doing today? I just got back from a walk with my dog."
#     }
#   ], 
#   "uuid": "ef7fe782-e57e-44b0-9001-d2e31aa35de4"
# }


