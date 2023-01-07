from sklearn.decomposition import PCA
#flatten into a one dimensional array
def vectorize_data(data):
    return data.flatten()

#perform pca analysis on it
def do_pca(train, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(train)
    new_train = pca.transform(train)
    return new_train

#turn it into a single big method
def pca(train, num_components):
    train = do_pca(train, num_components)
    return train
