from annoy import AnnoyIndex
import numpy as np
import os
import torch
import arxiv_lib.load_mathml
import ml.graph_cnn
import ml.heatmaps
def build_index():
	vectors = np.load("train_embeddings.npy")
	ids = open("train_keys.csv", "r").read().splitlines()
	index = AnnoyIndex(64, 'angular')
	n_trees = 200
	for i in range(len(vectors)):
		index.add_item(i, vectors[i])
	index.build(n_trees)
	index.save("index.ann")
	return index, ids

def load_index(path="index.ann"):
	ids = open("train_keys.csv", "r").read().splitlines()
	index = AnnoyIndex(64, 'angular')
	index.load(path)
	return index, ids

class MathMl():
	def __init__(self, x):
		self.raw = x
	def __str__(self):
		return "\n{}\n\n".format(self.raw.replace("span", "div"))

def get_mathml(path, test=False):
	if test:
		raw = open(os.path.join("/data/pfahler/arxiv_processed/subset_ml/test/mathml", path[1:]), "r").read()
	else:
		raw = open(os.path.join("/data/pfahler/arxiv_processed/subset_ml/train/mathml", path[1:]), "r").read()
	# print("\n"+raw.replace("\n","")+"\n")
	return "\n"+raw.replace("\n", "").replace("span", "div").replace("<mo>⁡</mo>", "") + "\n"

import streamlit as st

def main(query=None):

    p = pprint.PrettyPrinter()

    st.write("# Index Test")
    st.write("<style type='text/css'>div.katex{text-align:center;}</style>", unsafe_allow_html=True)
    if query is None:
        query = np.random.randint(0, len(keys))
    st.write("## Query ({})".format(query))
    st.write(get_mathml(keys[query]), unsafe_allow_html=True)

    embedding = torch.tensor(index.get_item_vector(query))
    model = ml.graph_cnn.GraphCNN()
    model.load_state_dict(torch.load('/home/richter2/DataSaliency/models/178/equation_encoder_graph_cnn_checkpoint19.pt', map_location='cpu'))
    heatmap = ml.heatmaps.Heatmap(model)

    st.write("## Results")
    nn, dists = index.get_nns_by_item(query, 10, search_k=5*10*100, include_distances=True)
    for i, x in enumerate(nn):
        path = '/home/richter2/DataSaliency/mathml' + keys[nn[i]]
        formular = heatmap.saliency_map([path, path], [embedding, embedding], parent_sum=True, color_mode='lin', value_mode='sim')
        formular = formular[0]
        st.write("Result {}".format(formular), unsafe_allow_html=True)


        formular = heatmap.saliency_map([path], [embedding], parent_sum=False, color_mode='lin', value_mode='sim')
        formular = formular[0]
        st.write("Result {}".format(formular), unsafe_allow_html=True)


        formular = heatmap.saliency_map([path], [embedding], parent_sum=True, color_mode='log', value_mode='sim')
        formular = formular[0]
        st.write("Result {}".format(formular), unsafe_allow_html=True)


        formular = heatmap.saliency_map([path], [embedding], parent_sum=False, color_mode='lin', value_mode='grad')
        formular = formular[0]
        st.write("Result {}".format(formular), unsafe_allow_html=True)


        formular = heatmap.saliency_map([path], [embedding], parent_sum=True, color_mode='log', value_mode='grad')
        formular = formular[0]
        st.write("Result {}".format(formular), unsafe_allow_html=True)

build_index()
model = ml.graph_cnn.GraphCNN()
model.load_state_dict_from_path("model.pt")
colors = ml.heatmaps.Heatmap(model,False)
index, keys = load_index()
query = 0
print("test")
query = st.number_input('Insert a number', value=query)
if st.button("Feeling Lucky!"):
	query = None
main(query)
