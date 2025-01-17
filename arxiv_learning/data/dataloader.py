"""Load pickled triple data"""
import pickle
import gzip
import ray
import arxiv_learning.data.load_mathml
SAME_SECTION = 1
SAME_PAPER = 2
ALONG_CITATION = 3
DIFFERENT_PAPER = 4

# def load_xml(archive, alphabet, file):
#     xml = archive.open(file, "r").read()
#     if not xml:
#         raise FileNotFoundError(file)
#     return arxiv_learning.data.load_mathml.load_pytorch(None, alphabet, string=xml)


class RayManager():

    def __init__(self, custom_heuristics=None, data_augmentation=True, total=100, batch_size=128, blowout=1, test=False):
        # ray.init(address="129.217.30.174:6379")
        ray.init(address="auto", ignore_reinit_error=True)
        # ray.init(address='auto', redis_password='5241590000000000')
        # ray.init()
        self.total = total
        self.blowout = blowout
        self.batch_size = batch_size
        self.test = test
        print("DATA_AUGMENTATION", data_augmentation)
        if custom_heuristics is not None:
            self.heuristics = custom_heuristics
        else:
            import arxiv_learning.data.heuristics.equations
            import arxiv_learning.data.heuristics.context
            self.heuristics = {
                "same_paper_heuristic": {
                    "data_set" : arxiv_learning.data.heuristics.context.SamePaper,
                    "head": None
                },
                "same_section_heuristic": {
                    "data_set" : arxiv_learning.data.heuristics.context.SameSection,
                    "head": None
                },
                # "equalities_heuristic": {
                # 	"data_set" : arxiv_learning.data.heuristics.equations.EqualityHeuristic,
                # 	"head": None
                # },
            }
        self.heuristics = {
            "{}/{}".format(k, i) : {
                "data_set" : v["data_set"].remote(test=self.test, batch_size=self.batch_size, data_augmentation=data_augmentation),
                "head" : v["head"]
            }
            for k, v in self.heuristics.items()
            for i in range(v.get("blowout", self.blowout))
        }
        self.round = []
        self.robin = []

    def __del__(self):
        ray.shutdown()

    def __iter__(self):
        from arxiv_learning.data.heuristics.heuristic import GZIP
        counter = 0
        for i, v in enumerate(self.heuristics.values()):
            v["data_set"].seed.remote(i)
        self.round = []
        self.robin = {}
        for k,v in self.heuristics.items():
            item = v["data_set"].generate.remote()
            self.round.append(item)
            self.robin[item] = k
        while counter < self.total and len(self.round) > 0:
            data, queue = ray.wait(self.round)
            data = data[0]
            dataset = self.robin[data]
            if GZIP:
                io_buffer = ray.get(data)
                if io_buffer is not None:
                    with gzip.open(io_buffer, "rb") as gzip_obj:
                        buffer = pickle.load(gzip_obj)
                        for x in buffer:
                            counter += 1
                            yield dataset.split("/")[0], x
                            if counter == self.total:
                                break
                else:
                    x = None # fuck you for this hack!
            else:
                x = ray.get(data)
                if x is not None:
                    counter += 1
                    yield dataset.split("/")[0], x
            del self.robin[data]
            if x is not None:
                new_item = self.heuristics[dataset]["data_set"].generate.remote()
                self.round = queue + [new_item]
                self.robin[new_item] = dataset
            else:
                self.round = queue
        # empty everything
        if self.round:
            # for x in self.round:
            #     ray.cancel(x)
            ray.get(ray.wait(self.round, num_returns=len(self.round))[0])
