import numpy as np
import pandas as pd

CATEGORIES = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
N = 5


class NaiveBayesClassifier:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        init assigning the train and test datasets
        """
        self.train = train
        self.test = test

    def separate_dataset(self, df: pd.DataFrame) -> dict:
        """
        ARGUMENTS: df => DataFrame consisting original dataset
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: separates the dataset based on class
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns a dictionary with key=class name &
        value=DataFrame consisting of all relavent rows
        """
        cats = dict()
        for item in CATEGORIES:
            cats[item] = df[df['class'] == item]
        return cats

    def summerize(self, cats: pd.DataFrame) -> dict:
        """
        ARGUMENTS: cats: a dict with key-value pairs for each class
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the mean and standard deviation for each class
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns a dictionary with the calculated stats
        key=class name & value=stats
        """
        summaries = dict()
        for item in CATEGORIES:
            summaries[item] = cats[item].agg([np.mean, np.std])
        return summaries

    def calc_prob_distro(self, x: float, mean: float, std: float) -> float:
        """
        ARGUMENTS: mean, std: mean and standard deviation of a class
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the probability using gaussian density function
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns the probability of a class
        """
        exponent = np.exp(-(float(x) - float(mean))**2 / (2 * std**2))
        return (1 / (std * np.sqrt(2 * np.pi))) * exponent

    def calc_class_prob(self, train: pd.DataFrame, summaries: dict, row: pd.Series) -> dict:
        """
        ARGUMENTS:
        train => the training data (dataframe)
        summaries => statistics(mead/std)
        row => row from the dataset
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the probability of a sample
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns the probability that a sample belongs to a class
        """
        total_rows = np.sum([len(train[item].index) for item in CATEGORIES])
        probabilities = {
            item: len(train[item].index) / total_rows for item in CATEGORIES}
        for item in CATEGORIES:
            for i in range(len(summaries[item].columns)):
                mean, std = summaries[item].iloc[:, i]
                probabilities[item] += self.calc_prob_distro(row[i], mean, std)
        return probabilities

    def predict(self, train: pd.DataFrame, summaries: dict, row: pd.Series) -> str:
        """
        ARGUMENTS:
        train => the training data (dataframe)
        summaries => statistics(mead/std)
        row => row from the dataset
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the class sample belongs to
        by using the probabilities for each class
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns the class name the sample might belong to
        """
        probabilities = self.calc_class_prob(train, summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if (best_label is None) or (probability > best_prob):
                best_prob = probability
                best_label = class_value
        return best_label

    def accuracy_metric(self, actual: pd.Series, predicted: pd.Series) -> float:
        """
        ARGUMENTS:
        actual => the class, a sample actually belongs to
        predicted => the class predicted for the sample
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the percentage for each correctly predicted sample
        ---------------------------------------------------------------------------------------------------------------------------
        RETURNS: returns the score calculated
        """
        correct = 0
        for i in range(len(actual)):
            if actual.iat[i] == predicted.iat[i]:
                correct += 1
        return correct * 100 / float(len(actual))

    def naive_bayes(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        """
        AURGUMENTS: train and test datasets
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: the main naive bayes algorithm for prediction
        ---------------------------------------------------------------------------------------------------------------------------
        RETURNS: returns the list of predictions for the dataset
        """
        train_sep = self.separate_dataset(train)
        summaries = self.summerize(train_sep)
        predictions = []
        predictions = test.apply(lambda row: self.predict(
            train_sep, summaries, row), axis=1)
        return predictions

    def cross_validate(self, dataset: pd.DataFrame) -> list:
        """
        ARGUMENTS: dataset (DataFrame)
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: splits the dataset into N folds for cross validation
        ---------------------------------------------------------------------------------------------------------------------------
        RETURNS: returns a list of dataframes containg each fold
        """
        d_split = []
        d_copy = dataset.copy(deep=True)
        fold_size = len(dataset) // N
        for _ in range(N):
            fold = pd.DataFrame()
            while len(fold) < fold_size:
                idx = np.random.choice(d_copy.index, replace=False)
                fold = fold.append(d_copy.loc[idx, :])
                d_copy.drop(idx, inplace=True)
            d_split.append(fold)
        return d_split

    def eval_algo(self) -> float:
        """
        DESCRIPTION: start point to test and predict datasets
        ---------------------------------------------------------------------------------------------------------------------------
        RETURNS: the accuracy rates for the test and train datasets
        """
        train_folds = self.cross_validate(self.train)
        scores = []

        for i in range(N):
            train_folds_copy = train_folds.copy()
            test_set = train_folds_copy.pop(i)
            train_set = pd.concat(train_folds_copy)
            predicted = self.naive_bayes(train_set, test_set)
            actual = test_set['class'].copy(deep=True)
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)

        xVal_accuracy = sum(scores) / len(scores)
        test_pred = self.naive_bayes(self.train, self.test)
        test_actual = self.test['class'].copy(deep=True)
        test_accuracy = self.accuracy_metric(test_actual, test_pred)
        return xVal_accuracy, test_accuracy


if __name__ == "__main__":

    # LOAD THE CSV FILES INTO A DATAFRAME
    train_data = pd.read_csv('train.csv')
    train_data.rename(columns={'4': 'class'}, inplace=True)
    test_data = pd.read_csv('test.csv')
    test_data.rename(columns={'4': 'class'}, inplace=True)

    classifier = NaiveBayesClassifier(train_data, test_data)
    xval_acc, test_acc = classifier.eval_algo()
    print(f'{N}-fold cross validation accuracy: {xval_acc}')
    print(f'Test set accuracy: {test_acc}')
