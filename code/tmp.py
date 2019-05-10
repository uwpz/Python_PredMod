


from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)
#KFold(n_splits=2, random_state=None, shuffle=False)
#for train_index, test_index in kf.split(X):
a = kf.split(X)
train_index, test_index = next(a)
print("TRAIN:", train_index, "TEST:", test_index)
