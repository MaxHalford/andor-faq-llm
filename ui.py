import os
import textwrap
import streamlit as st
import numpy as np
import pandas as pd
import requests

st.image('http://andor.iello.fr/wp-content/uploads/2014/11/header-image.png')

st.info("""Cet outil est une preuve de concept. C'est un système de question/réponse basé sur le jeu de société [Andor](https://andor.iello.fr/). Ce dernier est un jeu de rôle de plateau avec des règles relativement complexes. Le système est un modèle de langage qui a indexé le livre de règles français du jeu. Cliquez [ici](https://github.com/MaxHalford/andor-faq-llm) pour en savoir plus.

Le système va suggérer des parties du livret de règles qui correspondent à la question posée. Il va ensuite suggérer une réponse synthétique en fusionant ce qu'il sait déjà du jeu, et les parties suggérées. Il faut scroller en bas de la page pour voir la réponse.
""")

embeddings_df = pd.read_csv('embeddings.csv', index_col=0)

question = st.text_input(label='Posez une question', placeholder='Ex: Comment se déroule un combat?')
with st.sidebar:
    k = st.slider(label='Nombre de suggestions', min_value=1, max_value=30, value=10)

if not question:
    st.stop()

# Embed the question
with st.spinner('Waiting on OpenAI...'):
    api_token = os.environ['OPENAI_API_TOKEN']
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json={"input": question, "model": "text-embedding-ada-002"}
    )
    embedding = response.json()['data'][0]['embedding']

# Compute the cosine similarity between the question and the rule book
similarities = np.dot(embeddings_df, embedding)
top_k = similarities.argsort()[-k:][::-1]

# Display the top k suggested crops
st.markdown('## Suggestions')
suggested_crops = []
for idx in top_k:
    with (
        open(f'crops/{embeddings_df.index[idx]}/text_clean.txt') as f
    ):
        st.image(f'crops/{embeddings_df.index[idx]}/image.png')
        suggested_crops.append(f.read().strip())

st.markdown('## Synthèse')

# Now let's suggest an answer
suggested_crops_cat = '\n'.join(f'{i}.\n\n{sc}' for i, sc in enumerate(suggested_crops, start=1))
prompt = f"""Nous jouons au jeu de plateau Andor. Nous souhaitons répondre à la question suivante:

{question}

Voici des éléments de réponse que nous avons trouvé dans la documentation du jeu:

{suggested_crops_cat}

Je veux que tu répondes à la question posée avec ce que tu sais déjà des régles de jeu d'Andor, et optionnellement avec les éléments de réponse que je t'ai donné.

"""

with st.spinner('Waiting on OpenAI...'):
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={"messages": [{"role": "user", "content": prompt}], "model": "gpt-3.5-turbo"}
    )
    answer = response.json()['choices'][0]['message']['content']
    st.text(textwrap.fill(answer, width=80, break_long_words=False, replace_whitespace=False))
