import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy
import logloss
import matplotlib

def main():
    dataset = numpy.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
    target=numpy.array([x[0] for x in dataset])
    train=numpy.array([x[1:] for x in dataset])

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=4)

    cv = sklearn.cross_validation.KFold(len(train), k=5, indices=False)

    results=[]
    for traincv, testcv in cv:
        probas = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append(logloss.llfun(target[testcv], [x[1] for x in probas]))

    print results

    numpy.savetxt('submission.csv',predicted_probs, delimiter=',',fmt='%f')



if __name__=="__main__":
    main()