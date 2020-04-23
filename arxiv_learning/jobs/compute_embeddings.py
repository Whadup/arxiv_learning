import zipfile
import argparse
import os
import pickle
import torch
import numpy as np
from torch_geometric.data import DataLoader
import arxiv_lib
import arxiv_lib.load_mathml
import ml.graph_cnn
from multiprocessing import Pool
from functools import partial

def load(x, zip=None, alphabet=None):
	try:
		xml = zip.open(x, "r").read()
		return arxiv_lib.load_mathml.load_pytorch(None, alphabet, string=xml)
	except:
		return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Extract embeddings from model')
	parser.add_argument('--new', action="store_true")
	parser.add_argument('--checkpoint', type=int, default=-1)
	args = parser.parse_args()

	model = ml.graph_cnn.GraphCNN()
	if args.checkpoint >= 0:
		model.load_checkpoint(args.checkpoint)
	else:
		model.load()

	model = model.cuda().eval()

	path = "/rdata/pfahler/arxiv_processed/mathml"
	test_path = "/data/s1/pfahler/arxiv_processed/subset_ml/test/mathml"

	all_mathml = zipfile.ZipFile("/rdata/pfahler/arxiv_processed/mathml.zip", "r")
	print("Zip loaded")
	# data = sum([os.listdir(os.path.join(path, x)) for x in os.listdir(path)], [])
	if args.new:
		data = all_mathml.namelist()
		print("data identified",len(data))
		#data = sum([[os.path.join(path, x, y) for y in os.listdir(os.path.join(path, x))] for x in os.listdir(path)], [])
		# shuffle(data)
		alphabet = arxiv_lib.load_mathml.load_alphabet(
			os.path.join(os.path.split(path)[0], "alphabet.pickle"))
		#with Pool(1) as p:
		#	X = p.imap(partial(load,zip=all_mathml,alphabet=alphabet), data, chunksize=10000)
		X = [load(x,zip=all_mathml, alphabet=alphabet) for x in data]
		print("torch tensors ready")
		#DO NOT CHANGE ORDER!
		keys = [x for i, x in enumerate(data) if X[i] is not None]
		X = [x for x in X if x is not None]
		print("filtered")
		with open("/rdata/pfahler/arxiv_processed/train_keys.csv", "w") as f:
			# f.write("\n".join(keys).replace(path, ""))
			f.write("\n".join(keys))
		print("stored keys")
		pickle.dump(X, open("/rdata/pfahler/arxiv_processed/train_mathmls.pickle", "wb"))
		print("stored pickle")
		# test_data = sum([[os.path.join(test_path, x, y) for y in os.listdir(os.path.join(test_path, x))] for x in os.listdir(test_path)], [])
		# # shuffle(data)

		# test_X = [load(x, alphabet) for x in test_data]
		# #DO NOT CHANGE ORDER!
		# test_keys = [x for i, x in enumerate(test_data) if test_X[i] is not None]
		# test_X = [x for x in test_X if x is not None]

		# with open("test_keys.csv", "w") as f:
		# 	f.write("\n".join(test_keys).replace(test_path, ""))
		# pickle.dump(test_X, open("test_mathmls.pickle", "wb"))
	else:
		X = pickle.load(open("train_mathmls.pickle", "rb"))
		test_X = pickle.load(open("test_mathmls.pickle", "rb"))


	# loader = DataLoader(X, batch_size=128)
	# emb = []
	# with torch.no_grad():
	# 	for d in loader:
	# 		d = d.to("cuda")
	# 		ee = model(d).detach().cpu().numpy()
	# 		for e in ee:
	# 			emb.append(e)


	# np.save("train_embeddings.npy", np.array(emb).reshape(-1, 64))

	# loader = DataLoader(test_X, batch_size=128)
	# emb = []
	# with torch.no_grad():
	# 	for d in loader:
	# 		d = d.to("cuda")
	# 		ee = model(d).detach().cpu().numpy()
	# 		for e in ee:
	# 			emb.append(e)


	# np.save("test_embeddings.npy", np.array(emb).reshape(-1, 64))
