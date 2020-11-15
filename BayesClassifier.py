import numpy as np
import pandas as pd

CATEGORIES = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']


class BayesClassifier:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self.train = train
        self.test = test

    def separate_dataset(self, df: pd.DataFrame) -> dict:
        """
        ARGUMENTS: df => DataFrame consisting original dataset
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: separates the dataset based on class
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns a dictionary with key=class name &
        value=DataFrame consisting all relavent rows
        """
        separate = {}
        for item in CATEGORIES:
            separate[item] = df[df['class'] == item]
        return separate

    def summarize(self) -> dict:
        """
        DESCRIPTION: calculates the mean and covariance matrix for each class
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns a dictionary with the calculated stats
        key=class name & value=stats
        """
        separate = self.separate_dataset(self.train)
        summarize = dict()
        for item in CATEGORIES:
            temp = [i[:-1].astype(float) for i in separate[item].values]
            summarize[item] = [separate[item].agg(
                [np.mean]), np.cov(temp, rowvar=False)]
        return summarize

    def calc_prob_sample(self, x: np.array, mean: np.array, covar: np.array) -> float:
        """
        ARGUMENTS:
        x => sample
        mean, covar: mean and covariance matric of a class
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the probability assumint multivariate normal distribution
        with given aprior = 1/3 for all classes
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns the probability of a sample belonging to a class
        """
        x_m = x - mean
        probability = (1. / (np.sqrt((2 * np.pi)**len(covar) * np.linalg.det(covar))) * np.exp(-(
            np.linalg.solve(np.array(covar, dtype='float'), np.array(x_m, dtype='float')).T.dot(x_m)) / 2))
        return probability

    def calc_prob_class(self, x: np.array, summary: dict) -> dict:
        """
        ARGUMENTS: x: sample
        ---------------------------------------------------------------------------------------------------------------------------
        DESCRIPTION: calculates the probability of a sample
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns the probabilities of a sample beloging to each class
        """
        probabilities = {}
        for item in CATEGORIES:
            p = self.calc_prob_sample(
                x, summary[item][0].values[0], summary[item][1].astype(float))
            probabilities[item] = p / 3
        return probabilities

    def predict(self) -> list:
        """
        DESCRIPTION: calculates the class sample belongs to
        by using the probabilities for each class
        ---------------------------------------------------------------------------------------------------------------------------
        RETURN: returns the class name the sample might belong to
        """
        predicted = []
        actual = []
        summary = self.summarize()
        for row in self.test.values:
            actual.append(row[-1])
            row = np.delete(row, -1)
            class_row = ''
            maximum = -1
            for item, probability in self.calc_prob_class(row, summary).items():
                if probability > maximum:
                    maximum = probability
                    class_row = item
            predicted.append(class_row)
        return predicted, actual

    def accuracy_meter(self) -> float:
        """
        DESCRIPTION: calculates the percentage for each correctly predicted sample
        ---------------------------------------------------------------------------------------------------------------------------
        RETURNS: returns the score calculated
        """
        predicted, actual = self.predict()
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct * 100 / float(len(actual))


if __name__ == "__main__":
    # LOAD THE CSV FILES INTO A DATAFRAME
    train_data = pd.read_csv('train.csv')
    train_data.rename(columns={'4': 'class'}, inplace=True)
    test_data = pd.read_csv('test.csv')
    test_data.rename(columns={'4': 'class'}, inplace=True)

    classifier = BayesClassifier(train_data, test_data)
    accuracy = classifier.accuracy_meter()
    print(f'Model accuracy for test data: {accuracy}')
