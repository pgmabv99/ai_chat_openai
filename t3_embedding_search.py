# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search




class test:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    GPT_MODEL = "gpt-3.5-turbo"
    def __init__(self) -> None:
        self.client = OpenAI()
        pass

    def ask_no_prompt(self, query) -> str:

        """ ask without prompt"""
        response = self.client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
                {'role': 'user', 'content': query},
            ],
            model=test.GPT_MODEL,
            temperature=0,
        )

        return(response.choices[0].message.content)
        pass

    def get_embed(self):
        # download pre-chunked text and pre-computed embeddings
        # this file is ~200 MB, so may take a minute depending on your connection speed
        """get list of embedding for wiki from external site"""
        embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"

        # self.df = pd.read_csv(embeddings_path, nrows=3)
        self.df = pd.read_csv(embeddings_path)
        # convert embeddings from CSV str type back to list type
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)
        print(self.df.head(5))
        pass

    # search function
    def strings_ranked_by_relatedness(self,
        query: str,
        top_n: int = 5
    ) -> tuple[list[str], list[float]]:

        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = self.client.embeddings.create(
            model=test.EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y)
        strings_and_relatednesses = []
        for i, row in self.df.iterrows():
            dist=1-relatedness_fn(query_embedding, row["embedding"])
            strings_and_relatednesses.append((row["text"],dist))

        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        # list of pairs to pair of lists
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def get_num_tokens(self,
                   text: str
                   ) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(test.GPT_MODEL)
        return len(encoding.encode(text))


    def query_prompt(self,
        query: str,
        token_budget: int
    ) -> str:
        """Return a prompt for GPT, with relevant source texts pulled from a dataframe."""
        prompt=""
        # add introduction
        introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        introduction = ''
        prompt += introduction

        # add top N closes articles
        question = f"\n\nQuestion: {query}"
        strings, _= self.strings_ranked_by_relatedness(query)
        for string in strings:
            next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
            num_tokens=self.get_num_tokens(prompt+ question + next_article)
            if (num_tokens> token_budget):
                print("budget exceeded :", num_tokens)
                break
            else:
                prompt += next_article

        # add user question

        prompt+=question
        return prompt


    def ask_with_prompts(self,
            query: str,
            token_budget: int = 4096 - 500,
            print_message: bool = False,
            ) -> str:

        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        prompt = self.query_prompt(query,  token_budget=token_budget)
        if print_message:
            print(prompt)

        messages = [
            {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=test.GPT_MODEL,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

t1=test()
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'
answer=t1.ask_no_prompt(query)
print("NO  prompt answer=====", answer)
# exit()
t1.get_embed()
# examples
# strings, relatednesses = t1.strings_ranked_by_relatedness("curling gold medal",  top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}")
#     print(string[:50])
answer=t1.ask_with_prompts(query)
print("with prompt answer======", answer)
