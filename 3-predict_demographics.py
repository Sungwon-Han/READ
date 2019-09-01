import itertools
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from dataloader import *
from parameters import *
import xgboost as xgb

def main(args):
    trainX, trainy, testX, testy = prepare_dataset(args.idx)
    pca_dim = find_parameters(args.idx, trainX, trainy)
    print("idx:{}-pca_dim:{}".format(args.idx, pca_dim))
    
    poly = PolynomialFeatures(interaction_only = False, include_bias = False)
    pca = train_pca(trainX, components = pca_dim)
    reduced_trainX = dimensional_reduction(trainX, pca)
    reduced_testX = dimensional_reduction(testX, pca)
    trainX_interact = poly.fit_transform(reduced_trainX)
    testX_interact = poly.fit_transform(reduced_testX)

    mse, r2_val = linear_regression(trainX_interact, np.log(np.array(trainy)), testX_interact, np.log(np.array(testy)))   
    print("Final Mean squared error: %.4f" % (mse))
    print("Final R2 score: %.4f" % (r2_val))


def prepare_dataset(idx):
    train_district = GPSReducedDataset(metadata = "./data/train/metadata.csv", root_dir = "./data/train/reduced/", predict_y_idx = idx)
    test_district = GPSReducedDataset(metadata = "./data/test/metadata.csv", root_dir = "./data/test/reduced/", predict_y_idx = idx)
    trainX, trainy, testX, testy = [], [], [], []
    for i in range(len(train_district)):
        trainX.append(train_district[i]['images'])
        trainy.append(train_district[i]['y'].item())

    for i in range(len(test_district)):
        testX.append(test_district[i]['images'])
        testy.append(test_district[i]['y'].item())
    
    return trainX, trainy, testX, testy


def train_pca(X, components = 3):
    first = True
    for images in X:
        if first:
            train_for_pca = images
            first = False
        else:
            train_for_pca = np.concatenate([train_for_pca, images])    

    pca = PCA(n_components = components)
    pca.fit(train_for_pca)
    return pca


def dimensional_reduction(X, pca):
    reduced_X = []
    for images in X:
        train_pca = pca.transform(images)
        train_x = np.append(np.concatenate([np.mean(train_pca, axis = 0), np.std(train_pca, axis = 0)]), len(images))
        colnum = train_pca.shape[1]
        for subset in itertools.combinations(range(colnum), 2):
            train_x = np.append(train_x, np.corrcoef(train_pca[:, subset[0]], train_pca[:, subset[1]])[0][1])
            
        reduced_X.append(train_x)        
    reduced_X = np.array(reduced_X)
    return reduced_X


def linear_regression(X, y, test_X, test_y):
    regr = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    regr.fit(X, y)
    y_pred = regr.predict(test_X)
    mean_squared = mean_squared_error(test_y, y_pred)
    r2_val = r2_score(test_y, y_pred)
    return mean_squared, r2_val


def find_parameters(idx, trainX, trainy):
    kf = KFold(n_splits = 4, shuffle = True)
    kf.get_n_splits(trainX)

    trainX_list, validX_list, trainy_list, validy_list = [], [], [], []
    for train_index, valid_index in kf.split(trainX):
        tX, ty, vX, vy = [], [], [], []
        for i in train_index:
            tX.append(trainX[i])
            ty.append(np.array(trainy)[i])

        for i in valid_index:
            vX.append(trainX[i])
            vy.append(np.array(trainy)[i])

        trainX_list.append(np.array(tX))
        validX_list.append(np.array(vX))
        trainy_list.append(np.array(ty))
        validy_list.append(np.array(vy))
        
    poly = PolynomialFeatures(interaction_only = False, include_bias = False)
    maximum = -float('inf')
    for pca_dimension in range(1, 11):
        final_mean_squared = []
        final_r2_val = []
        for split in range(4):
            tX = trainX_list[split]
            vX = validX_list[split]
            ty = trainy_list[split]
            vy = validy_list[split]
            pca = train_pca(tX, components = pca_dimension)
            reduced_trainX = dimensional_reduction(tX, pca)
            reduced_validX = dimensional_reduction(vX, pca)

            trainX_interact = poly.fit_transform(reduced_trainX)
            validX_interact = poly.fit_transform(reduced_validX)
            _, r2_val = linear_regression(trainX_interact, np.log(ty), validX_interact, np.log(vy))
            final_r2_val.append(r2_val)

        final_r2_val = np.array(final_r2_val)
        performance = np.mean(final_r2_val)
        if maximum < performance:
            maximum = performance
            maximum_dim = pca_dimension
            
    return maximum_dim


if __name__ == '__main__':
    args = predict_demographics_parser()
    main(args)


