# OpenAI API

## Making Request to the OpenAI Api

```Python
from openai import OpenAI
client = OpenAI(api_key="your_api_key_here")
prompt = """some prompt here
            """

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How can I use the OpenAI API?"},
        {"role": "user", "content":prompt}
    ],
    temperature=0.7,# Controls amount of randomness 1 is the default 0 deterministic 2 highly random
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    max_completion_tokens=5 #control response length
    
    
)

print(response.choices[0].message.content)

```

## Summarizing and Editing Text

- **Tokens**: units of text that help the AI understand and interpret text
- Cost: based on model and number of tokens, input and output tokens may have different costs
- Increasing `max_completion_tokens` increases cost

## Calculating Cost

```Python
# Define Price per Token
max_completion_tokens = 500
input_token_price = 0.15 / 1_000_000  # $0.03 per 1,000,000 input tokens
output_token_price = 0.6 / 1_000_000  # $0.06 per 1,000,000 output tokens
# Extract Token Usage from Response
input_tokens = response.usage.prompt_tokens
#output_tokens = response.usage.completion_tokens
output_tokens = max_completion_tokens

#Calculate Cost
cost = (input_tokens * input_token_price + output_tokens * output_token_price)
print(f"Total Cost: ${cost:.10f}")

```

## Shot Prompting

- Providing examples to the prompt to get a better response
  - **Zero-Shot**: No examples, just instructions
  - **One-Shot**: One example guides the response
  - **Few-Shot**: multiple examples provide more context

## Chat Roles and System Messages

- Roles:
  - System - control's assistant's behavior (e.g. you are a python tutor who speaks concisely)
  - User - used to provide an instruction to the assistant
  - Assistant - response to user instruction

- system messages can also include guardrails, including what the model is not allowed to do


Assistant Role:

```Python

```

## Multi-Turn Conversations

```Python
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

messages = [
    {"role": "system", "content": "You are a helpful math tutor that speaks concisely."},
    {"role": "user", "content": "Explain what pi is."}
]

# Send the chat messages to the model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_completion_tokens=100
)

# Extract the assistant message from the response
assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}

# Add assistant_dict to the messages dictionary
messages.append(assistant_dict)
print(messages)
```

***

## Formatting

"""some text here then insert an external text ```{text}```"""

```Python
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create a request to complete the story
prompt = f"""complete the given story with only two paragraphs in the style of Shakespeare, the story is delimited with triple backticks ```{story}```"""

# Get the generated response
response = get_response(prompt)

print("\n Original story: \n", story)
print("\n Generated story: \n", response)
```

***

## Chain of Thought and self-consistency prompting

- Requires LLMs to provide reasoning steps (thoughts) before giving answer
- used for complex reasoning tasks
- help reduce model errors

- Self consistency prompting generates multiple chain of thoughts by prompting the model several times

***

## Chatbot development

- system_prompt: guidelines on what the bot's role is
- user_prompt: question user asks
- specify audience, tone, length, structure

***

# Hugging Face

## The Pipeline

```Python
from transformers import pipeline

gpt2_pipeline = pipeline(task='text-generation', model='openai-community/gpt2')

results = gpt2_pipeline("Once upon a time", max_new_tokens=10 , max_length=50, num_return_sequences=2)

#loop because of the number of returned sequences is more than 1
for result in results:
    print(result['generated_text'])

```

## Using inference providers

```Python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="together",
    api_key=os.environ("HF_TOKEN")
)

completion = client.chat.completions.create(
    model="gemini-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How can I use the Hugging Face Inference API?"}
    ],
    max_new_tokens=100
)
print(completion.choices[0].message.content)
```
## Hugging Face Datasets
```Bash
pip install datasets
```
```Python
from datasets import load_dataset
data = load_dataset("ag_news", split="train")

#uses arrow datasets so different patterns
filtered = data.filter(lambda example: example['label'] == 0)
print(filtered[:5])

#selecting rows from an index in arrow
sliced = filtered.select(range(10)) #first 10 rows
print(sliced[0]['text']) <-index then column

```

***

## Text Classification

```Python
from transformers import pipeline

my_pipeline = pipeline(task='text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
print(my_pipeline("I love using transformers library!"))

#Category Assignment

classifier = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')
text = "I had a wonderful experience at the restaurant. The food was delicious and the service was excellent."
categories = ["food", "service", "ambiance", "price"]
result = classifier(text, candidate_labels=categories)
print(f"Top Label: {result['labels'][0] with score: {result['scores'][0]}")
```
- challenges are ambiguity, sarcasm and irony, multiligual complexities, often require more pre-processing and more robust models

***

## Text Summarization

```Python
from transformers import pipeline

summarizer = pipeline(task='summarization', model='facebook/bart-large-cnn')
text = """The Transformers library by Hugging Face provides a wide range of pre-trained models for natural language processing tasks. It is widely used for tasks such as text classification, named entity recognition, question answering, and text generation. The library is built on top of PyTorch and TensorFlow, making it easy to integrate into existing machine learning workflows."""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])

# Repeat for a summary between 50 and 150 tokens
long_summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", min_new_tokens=50, max_new_tokens=150)

long_summary_text = long_summarizer(original_text)

print(long_summary_text[0]["summary_text"])


```

***

## Auto Models and Tokenizers

- Auto Classes - flexible way to access models and tokenizers
- more control - over model behavior and outputs
- perfect for advanced tasks
- Pipelines = quick ; Auto Classes = flexible

```Python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)

#Tokenizers prepare text input data
# Recommend to use tokenizer paired with the model
# Clean and input and split data into tokens
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokens = tokenizer.tokenize("I love using transformers library!")
print(tokens)

```
```Python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

my_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
my_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

my_pipeline = pipeline(task='text-classification', model=my_model, tokenizer=my_tokenizer)

```

***

## Document Q&A

```Python
from pypdf import PdfReader
reader = PdfReader("sample.pdf")

document_text = ""
for page in reader.pages:
    document_text += page.extract_text() + "\n"

qa_pipeline = pipeline(task='question-answering', model='distilbert-base-uncased-distilled-squad')
result = qa_pipeline(question="What is the main topic of the document?", context=document_text)
print(f"Answer: {result['answer']}")


```

***

## Structuring an API call

```Python
from openai import OpenAI

client = OpenAI(api_key="your_api_key_here")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How can I use the OpenAI API?"}
    ],
    temperature=0.7,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    max_completion_tokens=5,
    response_format={
        "type": "json_object",
        "properties": {
            "summary": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["summary", "keywords"]
    }
)


```

## Handling Exceptions

```Python
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How can I use the OpenAI API?"}
        ],
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        max_completion_tokens=5
    )
    print(response.choices[0].message.content)
except openAI.AuthenticationError as e:
    print(f"An error occurred: {e}")
    pass
except openAI.RateLimitError as e:
    print(f"An error occurred: {e}")
    pass
except openAI.OpenAIError as e:
    print(f"An error occurred: {e}")
    pass
```

## Retrying

```Python
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_random_exponential
)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))

def get_response(model, message):
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        max_completion_tokens=5,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content
```

## Batching

```Python
countries = ["France", "Germany", "Italy", "Spain", "Portugal"]
message=[{"role": "system", "content": "You are a helpful assistant. you are given a series of countries and you need to provide the capital of each country. Provide each of the questions with an answer in the response as a separate content"}]
[message.append({"role":"user","content":i}) for i in countries]
```

## Reducing tokens

```Python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")
prompt = "some long prompt here"
num_tokens = len(encoding.encode(prompt))

client = OpenAI(api_key="<OPENAI_API_TOKEN>")
input_message = {"role": "user", "content": "I'd like to buy a shirt and a jacket. Can you suggest two color pairings for these items?"}

# Use tiktoken to create the encoding for your model
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
# Check for the number of tokens
num_tokens = len(encoding.encode(input_message["content"]))

# Run the chat completions function and print the response
if num_tokens <= 100:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[input_message])
    print(response.choices[0].message.content)
else:
    print("Message exceeds token limit")

```

***

## Function Calling

- OpenAI's tools
- return more specific information

```Python
from openai import OpenAI

client = OpenAI(api_key="your_api_key_here")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=function_definition
)

print(response.choices[0].message.tool_calls[0].function_arguments)

function_definition = [
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                }
            },
            'required': ['location']
        }
    }
]

#Example
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

response= client.chat.completions.create(
    model="gpt-4o-mini",
    # Add the message
    messages= message_listing,
    # Add your function definition
    tools=function_definition
)

# Print the response
print(response.choices[0].message.tool_calls[0].function.arguments)

#example of tool call
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

response= client.chat.completions.create(
    model=model,
    messages=messages,
    # Add the function definition
    tools=function_definition,
    # Specify the function to be called for the response
    tool_choice={"type": "function", "function": {"name": "extract_review_info"}}
)

# Print the response
print(response)


```

***

## External APIs

```Python

function_definition=[{"type": "function",
    "function" : {
        "name": "get_artwork",
        "description": "This function calls the Art Institute of Chicago API to find artwork that matches a keyword",
        "parameters": 
            {  "type": "object",
               "properties": {
                   "artwork keyword": {
                       "type": "string",
                       "description": "The keyword to be passed to the get_artwork function."}}},
            "result": {"type": "string"}            
    }} ] )



import json

if response.choices[0].finish_reason=='tool_calls':
    function_call = response.choices[0].message.tool_calls[0].function
    if function_call.name=="get_artwork":
        artwork_keyword = json.loads(function_call.arguments)["artwork keyword"]
        artwork = get_artwork(artwork_keyword)
        if artwork:
            print(f"Here are some recommendations:
                {[item['title'] for item in json.loads(artwork)['data']]}")
        else:
            print("Apologies, I couldn't find any artwork matching that keyword.")
    else:
        print("Apologies, I cannot handle that function call.")
else:
    print("I am sorry, I couldn't process your request.")
```

## Combining Features with F Strings

```Python
articles = [..., {"headline": "1.5 Billion Tune-in to the World Cup ",
    "topic": "Sport",
    "keywords": ["soccer", "world cup", "tv"]}]

def create_article_text(article):
    return f"""Headline: {article['headline']}
    Topic: {article['topic']}
    Keywords: {', '.join(article['keywords'])}"""

print(create_article_text(articles[-1]))
```

## Creating Enriched Embeddings
```python
article_texts = [create_article_text(article) for article in articles]
article_embeddings = create_embeddings(article_texts)
print(article_embeddings)
```

## Computing Distances
```python
from scipy.spatial import distance
def find_n_closest(query_vector, embeddings, n=3):  
    distances = []
    for index, embedding inenumerate(embeddings):    
        dist = distance.cosine(query_vector, embedding)    
        distances.append({"distance": dist, "index": index})  
        distances_sorted = sorted(distances, key=lambda x: x["distance"])
        return distances_sorted[0:n]
```
## Returning Search Results

```python
query_text = "AI"
query_vector = create_embeddings(query_text)[0]
hits = find_n_closest(query_vector, article_embeddings, n=5)

for hit in hits:  
    article = articles[hit['index']]
    print(article['headline'])
```

## Recommendation Systems
 - Similar to Semantic Search
 - Embed the potential recommendation
 - Calculate Distances
 - Recommend Closest Items

 ```Python
def create_article_text(article):
   return f"""Headline: {article['headline]}
   Topic: {article['topic]}
   Keywords: {', '.join(article['keywords])}"""
  
history_texts = [create_article_text(article) for article in user_history]
history_embeddings = create_embeddings(history_texts)
mean_history_embeddings = np.mean(history_embeddings, axis=0)

articles_filtered = [article for article in articles if article not in user_history]

hist = find_n_closest(mean_history_embeddings, article_embeddings, n=5)

for hit in hits:
    article = articles_filtered[hit['index']]
    print(article['headline'])
 ```
## Embedding for classification tasks

 - Assign Labels to items
 - categorization
 - sentiment analysis
 - embeddings capture semantic meaning

 ```Python
 topics = [
     {'label': ;'Tech'},
     {'label': ;'Science'},
     {'label': ;'Sports'},
     {'label': ;'Business'},
 ]
 class_descriptions = [topic['label'] for topic in topics]
 class_embeddings = create_embeddings(class_descriptions)
 
 article_text = create_article_text(article)
 article_embeddings = create_embeddings(article_text)[0]
 
 def find_closest(query_vector, embeddings):
     distances = []
     for index, embedding in enumerate(embeddings):
         dist = distance.cosine(query_vector, embedding)
         distances.append({"distance": dist, "index": index})
     return min(distances, key=lambda x: x["distance"])
     
label = topics[closest["index"]]['label']
print(label)

```

***

# Vector Databases for Embedding Systems

- Embeddings
- Source Texts
- Metadata
  - Ids
  - references
  - Additional data useful for filtering results

## Creating Vector databases with ChromaDB

- simple yet powerful vector db
- two flavors
  - local: great for dev and proto
  - client/server: made for production

```Python
import chromadb

client = chromadb.PersistentClient(path="path/to/your/database or to save to")

# Create collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
collection = client.create_collection(
    name = "my_collection",
    embedding_function = OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key="..."
    )
)

client.list_collections()

#add embeddings
collection.add(ids=['My-doc'], documents=["this is the source text"])

# multiple documents

collection.add(
    ids=["my-doc-1","my-doc-2"],
    documents=["this is documents 1 ", "this is documents 2 "]
)


collection.count()
collection.peek() #return the first 10 items
collection.get(ids=["my-doc-1","my-doc-2"])


import tiktoken

enc = tiktoken.encoding_for_model("text-embedding-3-small")
total_tokens = sum(len(enc.encode(text)) for text in documents)
cost_per_1k_tokens = 0.00002
print('Total Tokens:', total_tokens)
print('Cost:', total_tokens * cost_per_1k_tokens / 1000)


```

## Querying and Updating the Database


```Python
collection = client.get_collection(
    name = 'netflix_titles',
    embedding_function = OpenAIEmbeddingFunction(api_key="...")
)

result = collection.query(
    query_texts=['movies where people sing a lot'],
    n_results=3
)
print(result)

```
- query() returns a dict with multiple keys:
  - ids: the ides of the returned items
  - embeddings: the embeddings of the returned items
  - documents: the source texts of the returned items
  - metadatas: the metadatas of the returned items
  - distances: the distances of the returned items for the query text

## Updating a Collection

```Python
collection.update(
    ids=['id-1','id-2'],
    documents=['New Document 1', 'New Document 2']
)
```

## Deleting
```Python
collection.delete(ids=['id-1','id-2'])

#delete all collections and items

client.reset()
```

## Multiple Queries and Filtering

```Python
reference_ids = ['s8170','s8103']
reference_texts = collection.get(ids=reference_ids)['documents']
result = collection.query(
    query_texts = reference_texts,
    n_results=3,
    where ={
        "type":"Movie"
    }
)

# Same as 
# 
result = collection.query(
    query_texts = reference_texts,
    n_results=3,
    where ={
        "type":{
            "$eq":"Movie"
    }
}
)

# Multiple where filters with and

result = collection.query(
    query_texts = reference_texts,
    n_results=3,
    where ={
        "$and":[
            {"type":
                {"$eq":"Movie"}
            },
            {"release_year":
                {"$gt":2020}
            }
            
        ]
    }
)


```

# Vector DB Embeddings with Pinecone

- **Indexes:**
  - Store vectors
  - Serve queries and other vector manupulations
  - Index contains records for each vector, including metadata
  - can create multiple indexes
- **Two types:**
  - **Serverless:**
    - No resource management
    - indexes scale automatically
    - run on cloud and store in blob
    - easier to use and often lower cost
  - **Pod Based**
    - choose hardware to create index
    - pod type, determines storage, query, latency, query throughput

```Python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your_api_key")

pc.create_index(
    name='datacamp-index',
    dimension = 1536,
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

pc.list_indexes()

index = pc.Index('datacamp-index')
index.describe_index_stats()

pc.delete_index('datacamp-index')
```

- Namespaces:
  - containers for partioning indexes
    - Separate datasets
    - Data Versioning
    - Separate groups

- Organizations:
  - Can contain multiple projects, which can contain multiple indexes, and namespaces each

## Vector Ingestion

```Python
vector_dims = [len(vector['values']) == 1536 for vector in vectors]
all(vector_dims)

index.upsert(
    vectors=vectors
)

index.describe_index_stats()
```

- Metadata useful for filtering

## Retrieving Vectors

- 2 Main methods:
  - Fetching:
    - Retrieve vectors based on their IDs
  - Querying
    - Retrieve similar vectors based on an input vector

```Python
index.fetch(
    ids=['id1','id2']
    namespace='namespace1'
)

# Read Units are resources consumed during read operations

# Extract the metadata from each result in fetched_vectors
metadatas = [fetched_vectors['vectors'][id]['metadata'] for id in ids]
print(metadatas)

#Querying

index.query(
    vector=[-0.250919762305275,...],
    top_k=3,
    include_values=True
)

```
- **Distance Metrics**:
  - Cosine Similarity
  - Euclidean Distance
  - Dot Product

```Python
pc.create_index(
    name='datacamp-index',
    dimension=1536,
    metric='dotproduct',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)
```

## Metadata Filtering

```Python
index.query(
    vector=[-0.250919762305275,...],
    filter={
        "genre":{"$eq":"documentary"},
        "year":2019
    },
    top_k=3,
    include_values=True,
    include_metadatas=True
)
```

## Updating and Deleting Vectors

```Python
index.fetch(ids=['1'])

index.update(
    id='1',
    values=[0.370695321,...],
    set_metadata={"genre":"comedy","rating":5}
)

index.delete(ids=['1','2'])

index.delete(
    filter={
        "genre":{"$eq":"action"},
    }
)

index.delete(delete_all=True, namespace='namespace1')

```

# Performance Tuning and AI Applications

```Python
#Chunking Function
def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk=tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it,batch_size))
        
#Batching
pc.Pinecone(api_key='your_api_key')
index = pc.Index('datacamp-index')

for chunk in chunks(vectors):
    index.upsert(vectors=chunk)
    
#Parallel batching

pc = Pinecone(api_key='your_api_key', pool_threads=30)

with pc.Index('datacamp-index', pool_threads=30) as index:
    async_results = [index.upsert(vectors=chunk, async_req=True)
        for chunk in chunks(vectors, batch_size=100)]
    
    [async_result.get() for async_result in async_results]

```

## Multi-Tenancy and Namespaces

- Serve multiple tenants in isolation
- Separate different customer's data
  - security and privacy
- Reduce query latency

- Namespaces
  - Advantages: Reduces the need for additional indexes
  - Disadvantages: Tenants share resources, complex data
- Metadata Filtering:
  - Advantages: Allows querying across multiple tenants
  - Disadvantages: Shared resources, challenging cost tracking
- Separate Indexes:
  - Advantages: Physically separates tenants, allocates individual resources
  - Disadvantages: requires more effort and cost

## Semantic Search with Pinecone

```Python
batch_limit = 100

for batch in np.array_split(df, len(df) / batch_limit):    
    metadatas = [{"text_id": row['id'], "text": row['text'], "title": row['title']} 
        for _, row in batch.iterrows()]    
    texts = batch['text'].tolist()    
    ids = [str(uuid4()) for _ inrange(len(texts))]    
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")    
    embeds = [np.array(x.embedding) for x in response.data]    
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace="squad_dataset")

index.describe_index_stats()


query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
query_response = client.embeddings.create(
    input=query,    
    model="text-embedding-3-small")
query_emb = query_response.data[0].embedding
retrieved_docs = index.query(vector=query_emb,                             
                             top_k=3,                              
                             namespace=namespace,                             
                             include_metadata=True)


for result in retrieved_docs['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

```
