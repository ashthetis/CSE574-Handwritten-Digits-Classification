import pickle


pickle_obj = pickle.load(file=open('params.pickle', 'rb'))
print(pickle_obj)