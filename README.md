# Complex-QA-with-KG-and-GPT
langchain hackathon project 

Built with [OpenAI(text-davinci-003)](https://platform.openai.com/docs/models) and [Langchain](https://github.com/hwchase17/langchain). 


Step 1: Build a (2-hop) knowledge graph on a topic. The konwleddge graph is extracted from the Wikipedia pages of the topic and its neighbor entities.
```
python build_knolwedge_graph.py --topic Taylor Alison Swift
```

Step 2: Ask it a question agains a QA
```
python run_qa.py --question "What is the city where Taylor Alison Swift went to known for?"
python run_qa.py --question "What album did Taylor Alison Swift release that is nominated for Best Country award?"
```

This is my 3-hr Hackathon project so there is much to improve. Please comment in issues. 