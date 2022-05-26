class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        ret = self.clf.predict(X)
        return ret

    def predict_proba(self, X):
        if len(X) == 1:
            self.probas_ = self.clf.predict_proba(X)[0]
            sums_to = sum(self.probas_)
            new_probs = [x / sums_to for x in self.probas_]
            return new_probs
        else:
            self.probas_ = self.clf.predict_proba(X)
            print(self.probas_)
            ret_list = []
            for list_of_probs in self.probas_:
                sums_to = sum(list_of_probs)
                print(sums_to)
                new_probs = [x / sums_to for x in list_of_probs]
                ret_list.append(np.asarray(new_probs))
            return np.asarray(ret_list)



the_model = MultiLabelProbClassifier(model)
pipe = Pipeline([('text2vec', Text2Vec()), ('model', the_model)])
pipe.fit(X_train, Y_train)

pred = pipe.predict(X_val)


te = TextExplainer(random_state=42, n_samples=300, position_dependent=True)

def explain_pred(sentence):
    te.fit(sentence, pipe.predict_proba)
    t_pred = te.explain_prediction()
    #t_pred = te.explain_prediction(top = 20, target_names=["ANB", "CAP", "ECON", "EDU", "ENV", "EX", "FED", "HEG", "NAT", "POL", "TOP", "ORI", "QER","COL","MIL", "ARMS", "THE", "INTHEG", "ABL", "FEM", "POST", "PHIL", "ANAR", "OTHR"])
    txt = format_as_text(t_pred)
    html = format_as_html(t_pred)
    html_file = open("latest_prediction.html", "a+")
    html_file.write(html)
    html_file.close()
    print(te.metrics_)
