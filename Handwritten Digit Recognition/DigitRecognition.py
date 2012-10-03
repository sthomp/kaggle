import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy

def cross_validation():
    dataset = numpy.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
    target=numpy.array([x[0] for x in dataset])
    train=numpy.array([x[1:] for x in dataset])

    rf = RandomForestClassifier(n_estimators=100, n_jobs=4)

    cv = sklearn.cross_validation.KFold(len(train), k=5, indices=False)

    results=[]
    for traincv, testcv in cv:
        model = rf.fit(train[traincv], target[traincv])
        predictions = model.predict(train[testcv])
        result = predictions==target[testcv]
        results.append(sum(result)/len(result))

    print numpy.array(results).mean()

def submission():
    dataset = numpy.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]

    target=numpy.array([x[0] for x in dataset])
    train=numpy.array([x[1:] for x in dataset])
    test = numpy.genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]

    rf = RandomForestClassifier(n_estimators=100, n_jobs=4)
    rf.fit(train, target)
    predictions = rf.predict(test)

    numpy.savetxt('submission.csv', predictions, delimiter=',', fmt='%i')

if __name__=="__main__":
    submission()