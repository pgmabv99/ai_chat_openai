from openai import OpenAI

class test:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.m_list=[]
        self.m_list.append({"role": "system", "content": "You are a repair person skilled in appliance repair."})
        pass

    def run(self, question):
        self.m_list.append({"role": "user", "content": question})


        self.completion = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=self.m_list
        )
        self.m_list.append({"role": "assistant", "content": self.completion.choices[0].message.content})

        # print(completion.choices[0].message)
        print("=======model",self.completion.model , "usage",  self.completion.usage)
        tok_list=self.completion.choices[0].message.content.split("\n")
        for tok in tok_list:
            print(tok)

t1=test()
t1.run("2 suggestion  how to fix bosh dishwasher leak.")
t1.run(" more ")
t1.run(" more ")
t1.run(" more ")
