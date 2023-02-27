from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from langchain.chains import GraphQAChain
from langchain.graphs.networkx_graph import NetworkxEntityGraph

from langchain import PromptTemplate
import networkx as nx
import argparse
import pickle



parser = argparse.ArgumentParser(description='Run QA against the knowledge graphs')

parser.add_argument('--file', dest='file',default='knowledge_graph.pkl',
                    help='the knowledge graph file to QA on')

parser.add_argument('--question', dest='question',default='What album did Taylor Alison Swift release that is nominated for Best Country award?',
                    help='the question to ask')


args = parser.parse_args()


def answer_complex_question(index_creator:GraphIndexCreator, question: str, prompt:str, knowlege_graph:NetworkxEntityGraph):
  print(f"""question: {question}""")
  graph_question = index_creator.from_text(question)
  kg_queries = graph_question.get_triples()
  print(f"Questions are divided into {len(kg_queries)} KG queries: {kg_queries}\n")
  question_asking_llm = OpenAI()
  sub_questions = []
  last_answer = ""
  qa_graph_chain = GraphQAChain.from_llm(OpenAI(), graph=knowlege_graph, verbose=True)
  for i, triplet in enumerate(kg_queries):
    sub_question = question_asking_llm(prompt.format(triple=str(triplet)))
    if last_answer != "": 
      sub_question = f"""{result["result"]}, {sub_question}"""
    print(f"\n\n\n Step #{i+1}: query {kg_queries[i]} turned into sub_question: {sub_question}")
    result = qa_graph_chain(sub_question)
    print( f"""Question: {result["query"]} \nAnswer:  {result["result"]}""")  
    last_answer = result["result"]
    if "I don't know" in last_answer:
      last_answer = ""
  
  print("/n Final answer: ", last_answer)

if __name__ == '__main__':
    template = """
    I want you to turn a knowledge graph triple that describes (subjecet, object, predicate) into a question to ask. Each question start with what.
    The question should be clear and succint, and it should use contain all the verbs and nouns from the triple. It should not have information from outside of the triple

    Here are some examples:

    - Triple: ('Adam', 'city', 'lives in') - what city does Adam live in?
    - Triple: ('city', something, 'is known for') - what is something the city is known for?
    - Triple: ('song', 'album A', 'was released in') - what song was released in album A?

    Triple: {triple} -

    """
    prompt = PromptTemplate(
        input_variables=["triple"],
        template=template,
    )
    index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
    question = args.question
    with open(args.file, 'rb') as f:
        indexed_kg = pickle.load(f)
    answer_complex_question(index_creator, question, prompt, indexed_kg)

