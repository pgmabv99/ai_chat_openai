# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
from transformers import GPT2Tokenizer
import difflib
import time





class test:
    # EMBEDDING_MODEL = "text-embedding-ada-002"
    #  operands could not be broadcast together with shapes (3072,) (1536,)
    # EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_MODEL = "text-embedding-3-small"
    # GPT_MODEL = "gpt-3.5-turbo"
    # GPT_MODEL = "gpt-4"
    GPT_MODEL = "gpt-4-turbo-preview"
    def __init__(self) -> None:
        self.client = OpenAI()
        self.token_budget = 4096 - 500
        # self.token_budget =  10**5
        self.max_prompts=1
        self.df_file="output/df.file"
        pass



    def embedding_get(self):
        # download pre-chunked text and pre-computed embeddings
        # this file is ~200 MB, so may take a minute depending on your connection speed
        """get list of embedding for wiki from external site"""
        embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"

        self.df = pd.read_csv(embeddings_path)
        # convert embeddings from CSV str type back to list type
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)
        # print(self.df.head(5))
        print("len of embedding df ", len(self.df))
        # self.df.to_csv(self.df_csv, index=False)
        self.df.to_pickle(self.df_file)
        pass
    def embedding_load(self):
        self.df=pd.read_pickle(self.df_file)
        pass

    # search function
    def strings_ranked_by_relatedness(self,
        query: str,
        n_prompts :int,
    ) -> tuple[list[str], list[float]]:

        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = self.client.embeddings.create(
            model=test.EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        # relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y)
        relatedness_fn=lambda x, y: spatial.distance.cosine(x, y)
        strings_and_relatednesses = []
        for i, row in self.df.iterrows():
            dist=1-relatedness_fn(query_embedding, row["embedding"])
            strings_and_relatednesses.append((row["text"],dist))

        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        # list of pairs to pair of lists
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:n_prompts], relatednesses[:n_prompts]

    def get_num_tokens(self,
                   text: str
                   ) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(test.GPT_MODEL)
        tokens=encoding.encode(text)
        num_tokens=len(tokens)
        # print("====num_tokens", num_tokens)
        # print(tokens)
        return num_tokens

    def get_real_tokens_gpt2(self,text):
        # Load the GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.tokenize(text)
        print(tokens)


    def prompt_build(self, query, n_prompts):
        """Return a prompt for GPT, with relevant source texts pulled from a dataframe."""
        prompt=""
        # add introduction
        introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        introduction = ''
        prompt += introduction

        # add top N closes articles
        question = f"\n\nQuestion: {query}"
        strings, relatedeness= self.strings_ranked_by_relatedness(query,n_prompts)
        # for i in range(len(strings)):
        #     print(relatedeness[i],strings[i][:50])
        for string in strings:
            next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
            num_tokens=self.get_num_tokens(prompt+ question + next_article)
            if (num_tokens> self.token_budget):
                print("budget exceeded :", num_tokens)
                break
            else:
                prompt += next_article

        # add user question
        prompt+=question
        return prompt

    def ask(self,query,n_prompts) :
        """Answers a query using GPT with/without prompts of additonal texts."""
        if n_prompts >0 :

            # self.embedding_get()
            prompt = self.prompt_build(query,n_prompts)
        else:
            prompt = query


        messages = [
            {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=test.GPT_MODEL,
            messages=messages,
            temperature=0,
            seed=1,
            # temperature=1
        )
        return response.choices[0].message.content

    def comp(self):
        differ = difflib.Differ()
        query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'

        answers=[]
        for n_prompts in range(self.max_prompts+1):
            answer=t1.ask(query,n_prompts)
            print("===== n_prompt = ", n_prompts, "\n",answer)

            with open("output/answer_"+str(n_prompts)+ ".txt" , "w" ) as file:
                file.write(answer)
            answers.append(answer)

            if n_prompts:
                print("\n \n difference ============ with previous\n ", n_prompts  )
                diff = list(differ.compare(answers[n_prompts].splitlines(),answers[n_prompts-1].splitlines()))
                with open("output/diff_"+str(n_prompts)+ ".txt" , "w" ) as file:
                    for line in  diff:
                        print(line)
                        file.write(line)

        return


    def test(self):
        print("in test")
        time.sleep(5)

t1=test()
# t1.embedding_get()
t1.embedding_load()
t1.comp()




# # t1.get_num_tokens("dear ,come here dear")
# # t1.get_real_tokens_gpt2("dear ,come here dear")
