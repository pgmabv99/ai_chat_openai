""" test OpenAI building of prompts using proximity search on related documents """
# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
from transformers import GPT2Tokenizer
import difflib
# import time
from datetime import  datetime
import os



class test:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    # EMBEDDING_MODEL = "text-embedding-3-large"  # has too many dimensions compared to ada-002
    # EMBEDDING_MODEL = "text-embedding-3-small"  # same dimansions but distance reversed  compared to ada-002

    # GPT_MODEL = "gpt-3.5-turbo"
    # GPT_MODEL = "gpt-4"
    GPT_MODEL = "gpt-4-turbo-preview"

    def __init__(self) -> None:
        self.client = OpenAI()
        self.token_budget = 4096 - 500
        # self.token_budget =  10**5
        self.max_docs=5
        self.df_file="catalog/df.file"
        pass



    def embedding_get(self):
        # download pre-chunked text and pre-computed embeddings
        # this file is ~200 MB, so may take a minute depending on your connection speed
        """get list of embedding for wiki from external site and save it"""
        print(datetime.now(),"starting to read remote csv")
        embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
        self.df = pd.read_csv(embeddings_path)

        print(datetime.now(),"starting to convert  csv")
        # convert embeddings from CSV str type back to list type
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)
        print("len of embedding df ", len(self.df))

        # print(datetime.now(),"starting save to file csv")
        # self.df.to_csv(self.df_file, index=False)
        print(datetime.now(),"starting  save to file pickle")
        self.df.to_pickle(self.df_file)
        print(datetime.now(),"finished all ")
        pass
    def embedding_load(self):
        self.df=pd.read_pickle(self.df_file)
        pass

    # search function

    def get_top_docs(self,
        n_prompts :int,
        query_embedding,
    ):
        """create df with top N docs."""
        print(datetime.now(),"start  get_top_docs")

        self.df1=self.df
        # add column
        self.df1["dist"]=self.df1.embedding.apply(lambda x :1 - spatial.distance.cosine(x, query_embedding))
        # sort and take highest
        self.df1=self.df1.sort_values("dist",ascending=False).head(n_prompts)

        # save docs to disk
        import mwparserfromhell
        docs=self.df1["text"].tolist()
        dists=self.df1["dist"].tolist()
        for i  in range(len(docs)):
            string_wiki=mwparserfromhell.parse(docs[i])
            string_text=string_wiki.strip_code()
            string_list=string_text.split(".")
            string_text="\n".join(string_list)
            dist=str(round(dists[i],3))
            with open("output/doc{}_{}".format(i, dist),"w")as f1:
                f1.write(string_text)
        print(datetime.now(),"end  get_top_docs")
        return

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


    def prompt_build_from_docs(self,
                     query,
                     ):
        """Return a prompt for GPT, with relevant docs pulled from a dataframe."""
        prompt=""
        # add introduction
        introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        introduction = ''
        prompt += introduction

        # add top N closest docs
        question = f"\n\nQuestion: {query}"
        docs=self.df1["text"]
        for doc in docs:
            next_article = f'\n\nWikipedia article section:\n"""\n{doc}\n"""'
            num_tokens=self.get_num_tokens(prompt+ question + next_article)
            if (num_tokens> self.token_budget):
                print("budget exceeded :", num_tokens)
                break
            else:
                prompt += next_article

        # add user question
        prompt+=question
        return prompt

    def ask(self,query,n_docs) :
        """Answers a query using GPT n_docs of    additonal docs."""
        print(datetime.now(),"starting ask")
        if n_docs >0 :
            query_embedding_response = self.client.embeddings.create(
                model=test.EMBEDDING_MODEL,
                input=query,
            )
            query_embedding = query_embedding_response.data[0].embedding
            self.get_top_docs(n_docs, query_embedding)
            prompt = self.prompt_build_from_docs(query)
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
        answer=response.choices[0].message.content
        print(datetime.now(),"end  ask")
        return answer

    def comp(self,query):
        """ compare experiments  with variable number of docs"""
        os.system("rm output/*")
        for n_prompts in range(0,self.max_docs+1):
            answer=t1.ask(query,n_prompts)
            with open("output/answer_"+str(n_prompts) , "w" ) as file:
                file.write(answer+"\n")
            if n_prompts >0 :
                command = ("diff" + " output/answer_"+ str(n_prompts) +
                                    " output/answer_"+ str(n_prompts-1) +">"
                                    + " output/diff" + str(n_prompts))

                os.system(command)
        return



t1=test()
# t1.embedding_get()
t1.embedding_load()
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'
t1.comp(query)
# print(t1.ask(query,n_docs=1))





# # t1.get_num_tokens("dear ,come here dear")
# # t1.get_real_tokens_gpt2("dear ,come here dear")
