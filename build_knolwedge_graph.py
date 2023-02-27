from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
import wikipedia
import networkx as nx
from langchain.graphs.networkx_graph import NetworkxEntityGraph, KnowledgeTriple
import argparse
import pickle


parser = argparse.ArgumentParser(
    description='Build a knowledge graph (2-hops) given a topic.')

parser.add_argument('--topic', dest='topic', default='Taylor Alison Swift',
                    help='the topic you want to build a knowledge graph based on its wikipedia page')

parser.add_argument('--dest', dest='dest', default='knowledge_graph.pkl',
                    help='where the knowledge graph is stored')

args = parser.parse_args()


def index_topic(index_creator: GraphIndexCreator, topic: str = "Taylor Alison Swift"):
    indexed_kg = NetworkxEntityGraph()
    first_entity_graph = index_creator.from_text(wikipedia.summary(topic))
    triples = first_entity_graph.get_triples()
    neighbor_names = [triple[1] for triple in triples]
    for triplet in triples:
        indexed_kg.add_triple(KnowledgeTriple(
            triplet[0], triplet[2], triplet[1]))
    for neighbor in neighbor_names:
        try:
            context = wikipedia.summary(neighbor)
            graph = index_creator.from_text(context)
            for triplet in graph.get_triples():
                indexed_kg.add_triple(KnowledgeTriple(
                    triplet[0], triplet[2], triplet[1]))
        except:
            pass
    return indexed_kg


def print_knowledge_graph(graph: NetworkxEntityGraph):
    print(len(graph.get_triples()))
    for k in graph.get_triples():
        print(k)


if __name__ == '__main__':
    index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
    print("Building a 2-hop knowledge graph for ", args.topic)
    indexed_kg = index_topic(index_creator, args.topic)
    with open(args.dest, 'wb') as f:
        pickle.dump(indexed_kg, f)
    print("Done! It can be accessed by ", args.dest)
